#!/usr/bin/env python3
"""Render screenshot-style DPG-HPT trace panels for the BMVC appendix."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont

PANEL_SIZE = (1550, 1080)
PAGE_SIZE = (2400, 3300)
FIG1B_STYLE_SIZE = (3200, 1600)
PANEL_GAP = 36
PAGE_MARGIN = 40

BG = "#f3f5f8"
CARD = "#ffffff"
INK = "#18202a"
MUTED = "#5a6472"
RULE = "#d8dde6"
BLUE = "#2563eb"
GREEN = "#059669"
AMBER = "#b45309"
RED = "#dc2626"
DARK = "#111827"
CODE_BG = "#f8fafc"


def _font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
        ]
    elif bold:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


TITLE_FONT = _font(38, bold=True)
TAG_FONT = _font(23, bold=True)
SECTION_FONT = _font(25, bold=True)
BODY_FONT = _font(22)
CODE_FONT = _font(20, mono=True)
SMALL_FONT = _font(18)
PAGE_TITLE_FONT = _font(50, bold=True)
PAGE_SUBTITLE_FONT = _font(26)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {} if payload is None else payload


def _short(value: Any, length: int = 12) -> str:
    text = "" if value is None else str(value)
    return text[:length]


def _format_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(str(item) for item in value) + "]"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)


def _wrap(text: str, width: int) -> list[str]:
    wrapped: list[str] = []
    for raw in str(text).splitlines() or [""]:
        if not raw:
            wrapped.append("")
            continue
        wrapped.extend(textwrap.wrap(raw, width=width, break_long_words=False))
    return wrapped


def _pill(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: str) -> int:
    x, y = xy
    padding_x = 16
    padding_y = 8
    bbox = draw.textbbox((0, 0), text, font=TAG_FONT)
    w = bbox[2] - bbox[0] + padding_x * 2
    h = bbox[3] - bbox[1] + padding_y * 2
    draw.rounded_rectangle((x, y, x + w, y + h), radius=12, fill=color)
    draw.text((x + padding_x, y + padding_y - 2), text, font=TAG_FONT, fill="white")
    return w


def _draw_rows(
    draw: ImageDraw.ImageDraw,
    rows: list[tuple[str, str]],
    x: int,
    y: int,
    width_chars: int,
    max_y: int,
) -> None:
    cursor = y
    for kind, text in rows:
        if cursor > max_y:
            draw.text((x, cursor), "...", font=BODY_FONT, fill=MUTED)
            return
        if kind == "section":
            cursor += 12
            draw.text((x, cursor), text.upper(), font=SECTION_FONT, fill=BLUE)
            cursor += 35
            continue
        if kind == "code":
            lines = _wrap(text, width_chars)
            line_height = 28
            block_h = len(lines) * line_height + 18
            draw.rounded_rectangle(
                (x - 8, cursor - 6, x + PANEL_SIZE[0] - 110, cursor + block_h),
                radius=10,
                fill=CODE_BG,
                outline=RULE,
            )
            for line in lines:
                draw.text((x + 4, cursor), line, font=CODE_FONT, fill=DARK)
                cursor += line_height
            cursor += 16
            continue
        color = MUTED if kind == "muted" else INK
        font = SMALL_FONT if kind == "muted" else BODY_FONT
        for line in _wrap(text, width_chars):
            draw.text((x, cursor), line, font=font, fill=color)
            cursor += 31 if kind != "muted" else 26
        cursor += 4


def _draw_panel(
    title: str,
    tag: str,
    tag_color: str,
    rows: list[tuple[str, str]],
) -> Image.Image:
    image = Image.new("RGB", PANEL_SIZE, BG)
    draw = ImageDraw.Draw(image)
    margin = 36
    draw.rounded_rectangle(
        (margin, margin, PANEL_SIZE[0] - margin, PANEL_SIZE[1] - margin),
        radius=18,
        fill=CARD,
        outline=RULE,
        width=2,
    )
    _pill(draw, (margin + 26, margin + 24), tag, tag_color)
    draw.text((margin + 26, margin + 82), title, font=TITLE_FONT, fill=INK)
    draw.line(
        (margin + 26, margin + 142, PANEL_SIZE[0] - margin - 26, margin + 142),
        fill=RULE,
        width=2,
    )
    _draw_rows(
        draw,
        rows,
        x=margin + 36,
        y=margin + 172,
        width_chars=92,
        max_y=PANEL_SIZE[1] - margin - 50,
    )
    return image


def _prompt_rows(context: dict[str, Any]) -> list[tuple[str, str]]:
    summary = context["target_profile"]["summary"]
    routes = context["reference_profiles"]["routes"]
    route_text = ", ".join(f"{r['dataset']}:{r['route']}" for r in routes)
    return [
        ("section", "Frozen context"),
        (
            "code",
            "\n".join(
                [
                    f"target_dataset: {summary['dataset_id']}",
                    f"domain: {summary['domain']}",
                    (
                        f"unit: {summary['coordinate_unit']} / "
                        f"inferred {summary['inferred_unit']}"
                    ),
                    f"pair_type: {summary['pair_type']}",
                    (
                        f"pose: {summary.get('pose_direction')} | "
                        f"{summary.get('pose_transform_shapes')}"
                    ),
                    f"reference_routes: {route_text}",
                ]
            ),
        ),
        ("section", "Prompt constraints"),
        (
            "text",
            "Read only frozen dataset profiles, public route evidence, "
            "model route cards, and transfer rules.",
        ),
        (
            "text",
            "Fill conservative/default/aggressive candidates only for "
            "allowlisted fields; keep locked fields fixed.",
        ),
        (
            "text",
            "For every candidate field, provide value, owner/status, "
            "source route, conversion rule, and evidence path.",
        ),
        (
            "code",
            "\n".join(
                [
                    f"candidate_limit: {context.get('candidate_limit')}",
                    f"allowed_parameters: {len(context.get('allowed_parameters', []))}",
                    f"locked_parameters: {len(context.get('locked_parameters', []))}",
                    "used_official_test_feedback: false",
                ]
            ),
        ),
    ]


def _proposal_rows(proposal: dict[str, Any]) -> list[tuple[str, str]]:
    selected = proposal.get("reference_selection", {}).get("selected", [])
    rejected = proposal.get("reference_selection", {}).get("rejected", [])
    selected_text = ", ".join(str(item.get("dataset")) for item in selected) or "none"
    rejected_text = ", ".join(str(item.get("dataset")) for item in rejected) or "none"
    candidates = proposal.get("candidates", {})
    candidate = candidates.get("default") or next(iter(candidates.values()))
    params = candidate.get("params", {})
    lines = []
    for field_name, payload in params.items():
        value = _format_value(payload.get("value"))
        status = payload.get("status")
        evidence_count = len(payload.get("evidence", []))
        lines.append(f"{field_name}: {value}  [{status}, evidence={evidence_count}]")
    return [
        ("section", "Reference decision"),
        ("code", f"selected: {selected_text}\nrejected: {rejected_text}"),
        ("section", "Default candidate excerpt"),
        ("code", "\n".join(lines[:12])),
        (
            "muted",
            "The proposal is a structured argument. It is not executable "
            "until deterministic validation produces candidate configs.",
        ),
    ]


def _validator_rows(
    validation: dict[str, Any],
    trace: dict[str, Any],
    candidate_validation: dict[str, Any] | None,
) -> list[tuple[str, str]]:
    summary = trace.get("validation_summary", {})
    digest_lines = [
        f"context_digest: {_short(trace.get('digests', {}).get('context_digest'))}",
        f"proposal_digest: {_short(trace.get('digests', {}).get('proposal_digest'))}",
        (
            "candidate_bundle: "
            f"{_short(trace.get('digests', {}).get('candidate_bundle_digest'))}"
        ),
    ]
    smoke_lines = [
        "gate_task: build runtime configs; optional run_eval",
        "reject: config failure, crash, OOM, missing metrics, collapse",
        "candidate_validation: not retained",
    ]
    if candidate_validation is not None:
        smoke_lines = [
            "gate_task: build runtime configs; optional run_eval",
            "reject: config failure, crash, OOM, missing metrics, collapse",
            f"execute_eval: {candidate_validation.get('execute_eval')}",
            "results:",
        ]
        for item in candidate_validation.get("results", [])[:4]:
            smoke_lines.append(
                f"  - {item.get('candidate')}: "
                f"{item.get('status')} ({item.get('reason')})"
            )
    return [
        ("section", "Proposal validator"),
        (
            "code",
            "\n".join(
                [
                    f"passed: {validation.get('passed')}",
                    f"errors: {len(validation.get('errors', []))}",
                    f"warnings: {len(validation.get('warnings', []))}",
                    f"candidate_count: {validation.get('candidate_count')}",
                    f"validated_candidates: {validation.get('validated_candidates')}",
                    f"trace_passed: {summary.get('passed')}",
                ]
            ),
        ),
        ("section", "Validated gates"),
        (
            "text",
            "Schema, route comparisons, allowlist, locked fields, evidence "
            "paths, owner labels, unit/pose, normalization, and test firewall.",
        ),
        ("section", "Runtime smoke sanity"),
        ("code", "\n".join(smoke_lines)),
        ("section", "Provenance digests"),
        ("code", "\n".join(digest_lines)),
    ]


def _file_rows(run_dir: Path) -> list[tuple[str, str]]:
    paths = [
        "context/context_pack.json",
        "context/context_pack.md",
        "proposal/agent_proposal.yaml",
        "validated/proposal_validation.json",
        "validated/transfer_trace.json",
        "validated/candidate_configs.yaml",
        "candidate_validation/validation_summary.json",
        "report/transfer_report.md",
        "promoted/promoted_default.yaml",
    ]
    tree_lines = []
    for rel in paths:
        marker = "OK" if (run_dir / rel).exists() else "--"
        tree_lines.append(f"{marker}  {rel}")
    return [
        ("section", "Frozen output overview"),
        ("code", "\n".join(tree_lines)),
        ("section", "What this panel proves"),
        (
            "text",
            "The prompt context, model response, validator outputs, normalized "
            "candidates, and human-readable report are separate files.",
        ),
        (
            "text",
            "A reviewer can rerun validation from the proposal and context "
            "instead of trusting a hidden conversation transcript.",
        ),
    ]


def _combine(panels: list[Image.Image]) -> Image.Image:
    w, h = PANEL_SIZE
    canvas = Image.new(
        "RGB",
        (PAGE_MARGIN * 2 + w * 2 + PANEL_GAP, PAGE_MARGIN * 2 + h * 2 + PANEL_GAP),
        BG,
    )
    positions = [
        (PAGE_MARGIN, PAGE_MARGIN),
        (PAGE_MARGIN + w + PANEL_GAP, PAGE_MARGIN),
        (PAGE_MARGIN, PAGE_MARGIN + h + PANEL_GAP),
        (PAGE_MARGIN + w + PANEL_GAP, PAGE_MARGIN + h + PANEL_GAP),
    ]
    for panel, pos in zip(panels, positions, strict=True):
        canvas.paste(panel, pos)
    return canvas


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str = "#64748b",
) -> None:
    draw.line((start[0], start[1], end[0], end[1]), fill=color, width=6)
    ex, ey = end
    sx, sy = start
    if abs(ex - sx) >= abs(ey - sy):
        direction = 1 if ex >= sx else -1
        points = [
            (ex, ey),
            (ex - 22 * direction, ey - 14),
            (ex - 22 * direction, ey + 14),
        ]
    else:
        direction = 1 if ey >= sy else -1
        points = [
            (ex, ey),
            (ex - 14, ey - 22 * direction),
            (ex + 14, ey - 22 * direction),
        ]
    draw.polygon(points, fill=color)


def _draw_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    lines: list[str] | None = None,
    *,
    font: ImageFont.FreeTypeFont | None = None,
    title_fill: str = INK,
    line_fill: str = MUTED,
    width: int = 4,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill="white", outline="black", width=width)
    title_font = font or TITLE_FONT
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_w = title_bbox[2] - title_bbox[0]
    draw.text(
        (x0 + (x1 - x0 - title_w) / 2, y0 + 36),
        title,
        font=title_font,
        fill=title_fill,
    )
    if not lines:
        return
    y = y0 + 120
    for line in lines:
        for wrapped in _wrap(line, max(24, int((x1 - x0) / 18))):
            bbox = draw.textbbox((0, 0), wrapped, font=SMALL_FONT)
            line_w = bbox[2] - bbox[0]
            draw.text(
                (x0 + (x1 - x0 - line_w) / 2, y),
                wrapped,
                font=SMALL_FONT,
                fill=line_fill,
            )
            y += 30


def _draw_plus(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    draw.text((x, y), "+", font=TITLE_FONT, fill="black")


def _draw_robot(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    draw.rounded_rectangle((x, y, x + 92, y + 62), radius=14, outline="black", width=4)
    draw.line((x + 46, y - 22, x + 46, y), fill="black", width=4)
    draw.ellipse((x + 38, y - 34, x + 54, y - 18), outline="black", width=4)
    draw.ellipse((x + 24, y + 22, x + 34, y + 32), fill="black")
    draw.ellipse((x + 58, y + 22, x + 68, y + 32), fill="black")
    draw.arc((x + 30, y + 25, x + 64, y + 52), 20, 160, fill="black", width=3)
    draw.rounded_rectangle(
        (x - 18, y + 16, x, y + 46),
        radius=8,
        outline="black",
        width=4,
    )
    draw.rounded_rectangle(
        (x + 92, y + 16, x + 110, y + 46),
        radius=8,
        outline="black",
        width=4,
    )


def _draw_page_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    tag: str,
    title: str,
    rows: list[tuple[str, str]],
    tag_color: str,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=24, fill=CARD, outline=RULE, width=3)
    _pill(draw, (x0 + 26, y0 + 24), tag, tag_color)
    draw.text((x0 + 28, y0 + 82), title, font=TITLE_FONT, fill=INK)
    draw.line((x0 + 28, y0 + 140, x1 - 28, y0 + 140), fill=RULE, width=2)
    _draw_rows(
        draw,
        rows,
        x=x0 + 36,
        y=y0 + 166,
        width_chars=max(42, int((x1 - x0) / 17)),
        max_y=y1 - 34,
    )


def _candidate_excerpt(proposal: dict[str, Any]) -> list[str]:
    candidates = proposal.get("candidates", {})
    candidate = candidates.get("default") or next(iter(candidates.values()))
    params = candidate.get("params", {})
    keep_fields = [
        "data.num_points",
        "preprocess.normalize_mode",
        "model.private_normalization_route",
        "data.voxel_size",
        "model.matching_radius",
        "losses.r_p",
        "losses.r_n",
        "eval.acceptance_radius",
    ]
    lines = []
    for field_name in keep_fields:
        if field_name not in params:
            continue
        payload = params[field_name]
        value = _format_value(payload.get("value"))
        status = payload.get("status")
        evidence_count = len(payload.get("evidence", []))
        lines.append(f"{field_name}: {value} [{status}, ev={evidence_count}]")
    return lines


def _rationale_excerpt(proposal: dict[str, Any]) -> list[str]:
    candidates = proposal.get("candidates", {})
    candidate = candidates.get("default") or next(iter(candidates.values()))
    params = candidate.get("params", {})
    field_summaries = {
        "data.voxel_size": (
            "3DMatch first_subsampling_dl=0.025 scaled by 100x -> 2.5 "
            "mm_like for the C3VD metric route."
        ),
        "model.matching_radius": (
            "3DMatch overlap radius=0.0375 scaled by 100x -> 3.75 "
            "mm_like; used as correspondence radius."
        ),
        "losses.r_p": (
            "Keep RegTR relation r_p=8*voxel_size -> 20.0; "
            "r_n keeps the 16* relation -> 40.0."
        ),
        "eval.acceptance_radius": (
            "Scaled checkpoint-selection threshold only; official RR/RTE "
            "thresholds are unchanged."
        ),
    }
    excerpts = []
    for field_name, summary in field_summaries.items():
        payload = params.get(field_name, {})
        evidence = payload.get("evidence", [])
        evidence_path = evidence[0].get("path", "missing") if evidence else "missing"
        excerpts.append(f"{field_name}: {summary} Evidence: {evidence_path}")
    return excerpts


def _appendix_page_rows(
    context: dict[str, Any],
    proposal: dict[str, Any],
    validation: dict[str, Any],
    trace: dict[str, Any],
    candidate_validation: dict[str, Any] | None,
    run_dir: Path,
) -> list[tuple[str, str, list[tuple[str, str]], str]]:
    summary = context["target_profile"]["summary"]
    selected = proposal.get("reference_selection", {}).get("selected", [])
    rejected = proposal.get("reference_selection", {}).get("rejected", [])
    selected_text = ", ".join(str(item.get("dataset")) for item in selected) or "none"
    rejected_text = ", ".join(str(item.get("dataset")) for item in rejected) or "none"
    smoke_status = "not retained"
    if candidate_validation is not None:
        smoke_status = (
            f"execute_eval={candidate_validation.get('execute_eval')}; "
            + ", ".join(
                f"{item.get('candidate')}:{item.get('status')}"
                for item in candidate_validation.get("results", [])
            )
        )
    file_lines = []
    for rel in (
        "context/context_pack.json",
        "proposal/agent_proposal.yaml",
        "validated/proposal_validation.json",
        "validated/transfer_trace.json",
        "validated/candidate_configs.yaml",
        "candidate_validation/validation_summary.json",
        "report/transfer_report.md",
    ):
        marker = "OK" if (run_dir / rel).exists() else "--"
        file_lines.append(f"{marker} {rel}")
    return [
        (
            "A",
            "Frozen Evidence Inputs",
            [
                ("code", "\n".join(
                    [
                        f"C3VD: {summary['dataset_id']}",
                        (
                            f"unit/pair: {summary['coordinate_unit']} | "
                            f"{summary['pair_type']}"
                        ),
                        f"pose: {summary.get('pose_direction')} | 4x4 homogeneous",
                        f"routes: {selected_text} selected; {rejected_text} rejected",
                    ]
                )),
                ("text", "Inputs are files, not a hidden chat history."),
                (
                    "code",
                    "\n".join(
                        [
                            "context/context_pack.json",
                            "reference_profiles/regtr_reference_profiles.json",
                            "configs/benchmark/hparam_transfer/transfer_rules.yaml",
                        ]
                    ),
                ),
            ],
            BLUE,
        ),
        (
            "B",
            "Prompt Package Sent to LLM",
            [
                (
                    "text",
                    "Generated from context pack, route cards, reference "
                    "profiles, and transfer rules.",
                ),
                (
                    "code",
                    "\n".join(
                        [
                            "task: write structured YAML proposal",
                            "must include: selected/rejected routes",
                            "must include: per-field evidence + owner label",
                            f"candidate_limit: {context.get('candidate_limit')}",
                            (
                                "allowed_fields: "
                                f"{len(context.get('allowed_parameters', []))}"
                            ),
                            (
                                "locked_fields: "
                                f"{len(context.get('locked_parameters', []))}"
                            ),
                            "firewall: used_official_test_feedback=false",
                        ]
                    ),
                ),
                (
                    "text",
                    "This panel is the prompt package before the LLM call; the "
                    "raw context remains separately auditable.",
                ),
            ],
            BLUE,
        ),
        (
            "C",
            "Model Response: Proposal YAML",
            [
                ("code", "\n".join(_candidate_excerpt(proposal))),
                (
                    "text",
                    "This is the model answer retained for audit; it is not "
                    "directly promoted.",
                ),
                (
                    "code",
                    "proposal/agent_proposal.yaml\n"
                    "test_set_firewall.used_official_test_feedback: false",
                ),
            ],
            GREEN,
        ),
        (
            "D",
            "Auditable Rationale Fields",
            [
                ("code", "\n".join(_rationale_excerpt(proposal))),
                (
                    "text",
                    "The reproducible reasoning is the explicit "
                    "evidence/rationale stored in proposal fields, not a private "
                    "scratch transcript.",
                ),
            ],
            GREEN,
        ),
        (
            "E",
            "Deterministic Validator",
            [
                ("code", "\n".join(
                    [
                        f"passed: {validation.get('passed')}",
                        f"errors: {len(validation.get('errors', []))}",
                        f"warnings: {len(validation.get('warnings', []))}",
                        (
                            "validated_candidates: "
                            f"{validation.get('validated_candidates')}"
                        ),
                    ]
                )),
                (
                    "text",
                    "Checks schema, allowlist, locked fields, ownership, "
                    "evidence, unit/pose, normalization, and firewall.",
                ),
                (
                    "code",
                    "\n".join(
                        [
                            "reject unknown fields",
                            "reject locked-field edits",
                            "reject missing route comparisons",
                            "reject official test feedback",
                        ]
                    ),
                ),
            ],
            AMBER,
        ),
        (
            "F",
            "Smoke Gate and Frozen Files",
            [
                ("code", "\n".join(
                    [
                        "smoke task: build runtime config; optional run_eval",
                        "reject: crash/OOM/missing metrics/all-fail collapse",
                        f"retained status: {smoke_status}",
                    ]
                )),
                ("code", "\n".join(file_lines)),
                (
                    "muted",
                    "Audit basis: prompt package, proposal YAML, validation "
                    "trace, candidate configs, and retained output files.",
                ),
            ],
            RED,
        ),
    ]


def _draw_appendix_page(
    context: dict[str, Any],
    proposal: dict[str, Any],
    validation: dict[str, Any],
    trace: dict[str, Any],
    candidate_validation: dict[str, Any] | None,
    run_dir: Path,
) -> Image.Image:
    image = Image.new("RGB", PAGE_SIZE, BG)
    draw = ImageDraw.Draw(image)
    margin = 70
    draw.text(
        (margin, 62),
        "One Complete DPG-HPT Trace: RegTR -> C3VD",
        font=PAGE_TITLE_FONT,
        fill=INK,
    )
    subtitle = (
        "Prompt package, structured model response, deterministic validation, "
        "smoke gate, and frozen files."
    )
    draw.text((margin, 128), subtitle, font=PAGE_SUBTITLE_FONT, fill=MUTED)
    draw.line((margin, 178, PAGE_SIZE[0] - margin, 178), fill=RULE, width=3)

    rows = _appendix_page_rows(
        context,
        proposal,
        validation,
        trace,
        candidate_validation,
        run_dir,
    )
    card_w = 1070
    card_h = 820
    x_left = margin
    x_right = PAGE_SIZE[0] - margin - card_w
    y_positions = [230, 1140, 2050]
    boxes = [
        (x_left, y_positions[0], x_left + card_w, y_positions[0] + card_h),
        (x_right, y_positions[0], x_right + card_w, y_positions[0] + card_h),
        (x_left, y_positions[1], x_left + card_w, y_positions[1] + card_h),
        (x_right, y_positions[1], x_right + card_w, y_positions[1] + card_h),
        (x_left, y_positions[2], x_left + card_w, y_positions[2] + card_h),
        (x_right, y_positions[2], x_right + card_w, y_positions[2] + card_h),
    ]
    for (tag, title, content, color), box in zip(rows, boxes, strict=True):
        _draw_page_card(draw, box, tag, title, content, color)

    def mid_y(index: int) -> int:
        return (boxes[index][1] + boxes[index][3]) // 2

    arrow_pairs = [
        (
            (boxes[0][2] + 16, mid_y(0)),
            (boxes[1][0] - 18, mid_y(1)),
        ),
        (
            (boxes[1][0] + card_w // 2, boxes[1][3] + 28),
            (boxes[3][0] + card_w // 2, boxes[3][1] - 28),
        ),
        (
            (boxes[3][0] - 18, mid_y(3)),
            (boxes[2][2] + 16, mid_y(2)),
        ),
        (
            (boxes[2][0] + card_w // 2, boxes[2][3] + 28),
            (boxes[4][0] + card_w // 2, boxes[4][1] - 28),
        ),
        (
            (boxes[4][2] + 16, mid_y(4)),
            (boxes[5][0] - 18, mid_y(5)),
        ),
    ]
    for start, end in arrow_pairs:
        _draw_arrow(draw, start, end)
    return image


def _draw_figure1b_style_trace(
    context: dict[str, Any],
    proposal: dict[str, Any],
    validation: dict[str, Any],
    trace: dict[str, Any],
    candidate_validation: dict[str, Any] | None,
) -> Image.Image:
    image = Image.new("RGB", FIG1B_STYLE_SIZE, "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(48, bold=True)
    box_title_font = _font(42)
    small_title_font = _font(33)
    note_font = _font(22)
    mono = _font(21, mono=True)

    draw.text(
        (90, 58),
        "(b) DPG-HPT Complete Trace: RegTR -> C3VD",
        font=title_font,
        fill="black",
    )
    draw.line((90, 132, 3060, 132), fill="black", width=2)

    input_box = (100, 220, 880, 1120)
    draw.rectangle(input_box, fill="white", outline="black", width=4)
    draw.text((155, 255), "INPUT", font=small_title_font, fill="black")

    inner_x0, inner_x1 = 165, 815
    census_box = (inner_x0, 340, inner_x1, 520)
    route_box = (inner_x0, 610, inner_x1, 790)
    rules_box = (inner_x0, 880, inner_x1, 1060)

    summary = context["target_profile"]["summary"]
    routes = context["reference_profiles"]["routes"]
    route_label = ", ".join(str(r["dataset"]) for r in routes) or "none"
    _draw_box(
        draw,
        census_box,
        "Dataset census",
        [
            f"C3VD: {summary['coordinate_unit']}, {summary['pair_type']}",
            f"pose: {summary.get('pose_direction')}, 4x4",
        ],
        font=small_title_font,
        width=3,
    )
    _draw_plus(draw, 485, 535)
    _draw_box(
        draw,
        route_box,
        "Model route audit",
        [
            f"RegTR public route: {route_label}",
            "reject ModelNet40 / KITTI",
        ],
        font=small_title_font,
        width=3,
    )
    _draw_plus(draw, 485, 805)
    _draw_box(
        draw,
        rules_box,
        "Transfer rules",
        [
            f"allowlist={len(context.get('allowed_parameters', []))}",
            f"locked={len(context.get('locked_parameters', []))}, firewall=false",
        ],
        font=small_title_font,
        width=3,
    )

    proposal_box = (1000, 560, 1540, 830)
    validator_box = (1645, 560, 2145, 830)
    smoke_box = (2345, 585, 2545, 805)
    final_box = (2645, 560, 2995, 830)

    _draw_arrow(draw, (880, 655), (996, 655), "black")
    draw.rounded_rectangle((875, 570, 990, 632), radius=10, fill="white")
    draw.text((885, 574), "PROMPT", font=SECTION_FONT, fill=BLUE)
    draw.text((885, 602), "pre-LLM", font=note_font, fill=MUTED)
    _draw_box(
        draw,
        proposal_box,
        "Evidence-\nconstrained\nagent proposal",
        None,
        font=box_title_font,
        width=4,
    )
    _draw_robot(draw, 1395, 735)
    draw.text(
        (1030, 855),
        "LLM ANSWER: proposal/agent_proposal.yaml",
        font=note_font,
        fill=MUTED,
    )
    draw.text(
        (1030, 885),
        "select 3DMatch; reject ModelNet40/KITTI",
        font=note_font,
        fill=MUTED,
    )

    _draw_arrow(draw, (1540, 695), (1640, 695), "black")
    _draw_box(
        draw,
        validator_box,
        "Deterministic\nvalidation",
        None,
        font=box_title_font,
        width=4,
    )
    validator_lines = [
        (
            f"passed={validation.get('passed')}, "
            f"errors={len(validation.get('errors', []))}"
        ),
        "schema | allowlist | locked | unit/pose | evidence",
    ]
    y = 852
    for line in validator_lines:
        draw.text((1655, y), line, font=note_font, fill=MUTED)
        y += 30

    # Fail loop, matching Fig. 1b.
    top_y = 420
    draw.line((1900, 560, 1900, top_y, 1270, top_y, 1270, 560), fill="black", width=4)
    _draw_arrow(draw, (1270, top_y + 3), (1270, 558), "black")
    draw.text((1510, top_y - 40), "fail", font=small_title_font, fill="black")

    _draw_arrow(draw, (2145, 695), (2338, 695), "black")
    draw.text((2195, 650), "pass", font=small_title_font, fill="black")
    _draw_box(
        draw,
        smoke_box,
        "Smoke\nsanity",
        None,
        font=small_title_font,
        width=4,
    )
    smoke_status = "dry-run retained"
    if candidate_validation is not None:
        smoke_status = f"execute_eval={candidate_validation.get('execute_eval')}"
    draw.text((2280, 842), smoke_status, font=note_font, fill=MUTED)
    draw.text((2230, 872), "reject crash/OOM/collapse", font=note_font, fill=MUTED)

    _draw_arrow(draw, (2545, 695), (2640, 695), "black")
    _draw_box(
        draw,
        final_box,
        "Final\nbenchmark\nconfig",
        None,
        font=box_title_font,
        width=4,
    )
    draw.text(
        (2625, 855),
        "OUTPUT FILES:",
        font=note_font,
        fill=MUTED,
    )
    draw.text(
        (2625, 885),
        "configs + trace + report",
        font=note_font,
        fill=MUTED,
    )

    # Bottom audit strip with concrete artifacts attached to the Fig. 1b nodes.
    strip = (100, 1210, 3100, 1535)
    draw.rounded_rectangle(strip, radius=16, fill="#f8fafc", outline=RULE, width=2)
    col_w = 970
    headers = [
        ("Prompt package", BLUE),
        ("Model response excerpt", GREEN),
        ("Frozen output overview", AMBER),
    ]
    for col, (header, color) in enumerate(headers):
        draw.text((140 + col * col_w, 1235), header, font=SECTION_FONT, fill=color)

    candidates = proposal.get("candidates", {})
    candidate = candidates.get("default") or next(iter(candidates.values()))
    params = candidate.get("params", {})
    prompt_lines = [
        f"target={summary['dataset_id']}; unit={summary['coordinate_unit']}",
        f"allowed={len(context.get('allowed_parameters', []))}; "
        f"locked={len(context.get('locked_parameters', []))}; "
        f"limit={context.get('candidate_limit')}",
        "firewall: used_official_test_feedback=false",
        "source: context/context_pack.json",
    ]
    proposal_lines = [
        "selected route: 3DMatch indoor metric scene",
        (
            f"voxel={params['data.voxel_size']['value']}; "
            f"radius={params['model.matching_radius']['value']}"
        ),
        (
            f"r_p={params['losses.r_p']['value']}; "
            f"r_n={params['losses.r_n']['value']}"
        ),
        (
            f"points={params['data.num_points']['value']}; "
            f"norm={params['preprocess.normalize_mode']['value']}"
        ),
        "owner label + evidence path stored per field",
    ]
    output_lines = [
        f"proposal_validation.json: passed={validation.get('passed')}",
        f"candidate_configs.yaml: {validation.get('candidate_count')} candidates",
        "transfer_trace.json: context/proposal/config digests",
        "validation_summary.json: execute_eval=false retained",
        "transfer_report.md + promoted/promoted_default.yaml",
    ]
    for row, line in enumerate(prompt_lines):
        draw.text((140, 1285 + row * 36), line, font=mono, fill=DARK)
    for row, line in enumerate(proposal_lines):
        draw.text((140 + col_w, 1285 + row * 36), line, font=mono, fill=DARK)
    for row, line in enumerate(output_lines):
        draw.text((140 + 2 * col_w, 1285 + row * 36), line, font=mono, fill=DARK)
    digests = trace.get("digests", {})
    digest_text = (
        f"context={_short(digests.get('context_digest'))} | "
        f"proposal={_short(digests.get('proposal_digest'))} | "
        f"candidate={_short(digests.get('candidate_bundle_digest'))}"
    )
    draw.text((140, 1500), digest_text, font=note_font, fill=MUTED)
    return image


def build_panels(run_dir: Path, out_dir: Path) -> dict[str, str]:
    context = _load_json(run_dir / "context/context_pack.json")
    proposal = _load_yaml(run_dir / "proposal/agent_proposal.yaml")
    validation = _load_json(run_dir / "validated/proposal_validation.json")
    trace = _load_json(run_dir / "validated/transfer_trace.json")
    candidate_validation_path = run_dir / "candidate_validation/validation_summary.json"
    candidate_validation = None
    if candidate_validation_path.exists():
        candidate_validation = _load_json(candidate_validation_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        (
            "hpt_trace_regtr_prompt.png",
            _draw_panel(
                "A. Frozen Prompt Input",
                "PROMPT",
                BLUE,
                _prompt_rows(context),
            ),
        ),
        (
            "hpt_trace_regtr_response.png",
            _draw_panel(
                "B. Structured Model Response",
                "LLM ANSWER",
                GREEN,
                _proposal_rows(proposal),
            ),
        ),
        (
            "hpt_trace_regtr_validator.png",
            _draw_panel(
                "C. Validator and Smoke Gate",
                "VALIDATOR",
                AMBER,
                _validator_rows(validation, trace, candidate_validation),
            ),
        ),
        (
            "hpt_trace_regtr_files.png",
            _draw_panel(
                "D. Frozen Output Files",
                "FILES",
                RED,
                _file_rows(run_dir),
            ),
        ),
    ]
    written: dict[str, str] = {}
    panels = []
    for filename, image in specs:
        path = out_dir / filename
        image.save(path)
        written[filename] = str(path)
        panels.append(image)

    combined = _combine(panels)
    combined_path = out_dir / "hpt_trace_regtr_flow.png"
    combined.save(combined_path)
    written["hpt_trace_regtr_flow.png"] = str(combined_path)

    appendix_page = _draw_appendix_page(
        context=context,
        proposal=proposal,
        validation=validation,
        trace=trace,
        candidate_validation=candidate_validation,
        run_dir=run_dir,
    )
    appendix_page_path = out_dir / "hpt_trace_regtr_appendix_page.png"
    appendix_page.save(appendix_page_path)
    written["hpt_trace_regtr_appendix_page.png"] = str(appendix_page_path)

    figure1b = _draw_figure1b_style_trace(
        context=context,
        proposal=proposal,
        validation=validation,
        trace=trace,
        candidate_validation=candidate_validation,
    )
    figure1b_path = out_dir / "hpt_trace_regtr_figure1b_style.png"
    figure1b.save(figure1b_path)
    written["hpt_trace_regtr_figure1b_style.png"] = str(figure1b_path)

    manifest_path = out_dir / "hpt_trace_regtr_manifest.json"
    manifest = {
        "source_run_dir": str(run_dir),
        "generated_files": written,
        "source_artifacts": [
            "context/context_pack.json",
            "proposal/agent_proposal.yaml",
            "validated/proposal_validation.json",
            "validated/transfer_trace.json",
            "validated/candidate_configs.yaml",
            "candidate_validation/validation_summary.json",
            "report/transfer_report.md",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    written["manifest"] = str(manifest_path)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/benchmark/hparam_transfer/regtr_measured_run"),
        help="DPG-HPT measured run directory, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("BMVC2026_Linzhe/images/hpt_trace_panels"),
        help="Output directory for panel PNG files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    run_dir = args.run_dir if args.run_dir.is_absolute() else repo_root / args.run_dir
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo_root / args.out_dir
    written = build_panels(run_dir=run_dir, out_dir=out_dir)
    for name, path in written.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
