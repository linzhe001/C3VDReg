#!/usr/bin/env python3
"""Export separate DPG-HPT trace materials for appendix figure assembly."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Iterable

import yaml
from PIL import Image, ImageDraw, ImageFont

CANVAS_W = 2400
MARGIN = 72
SECTION_GAP = 34
LINE_H = 30
BG = "#f3f5f8"
CARD = "#ffffff"
INK = "#111827"
MUTED = "#4b5563"
BLUE = "#2563eb"
GREEN = "#059669"
AMBER = "#b45309"
RED = "#dc2626"
RULE = "#d7dde7"
CODE_BG = "#f8fafc"


def _font(
    size: int,
    *,
    bold: bool = False,
    mono: bool = False,
) -> ImageFont.FreeTypeFont:
    candidates: list[str]
    if mono:
        candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ]
    elif bold:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


TITLE = _font(44, bold=True)
SUBTITLE = _font(24)
HEADER = _font(28, bold=True)
BODY = _font(24)
CODE = _font(22, mono=True)
SMALL = _font(19)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_json(path: Path) -> dict:
    return json.loads(_read(path))


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(_read(path))
    return {} if payload is None else payload


def _wrap_text(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for raw in str(text).splitlines() or [""]:
        if not raw:
            lines.append("")
            continue
        lines.extend(
            textwrap.wrap(
                raw,
                width=width,
                replace_whitespace=False,
                drop_whitespace=False,
                break_long_words=False,
            )
        )
    return lines


def _numbered_excerpt(path: Path, start: int, end: int) -> str:
    lines = _read(path).splitlines()
    selected = lines[start - 1 : end]
    return "\n".join(f"{idx:04d}: {line}" for idx, line in enumerate(selected, start))


def _value_text(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _geometry_table(profile: dict) -> str:
    geom = profile.get("geometry", {})
    rows = []
    metric_specs = [
        ("source_point_count", "source_point_count"),
        ("target_point_count", "target_point_count"),
        ("source_bbox_diag", "bbox_diag"),
        ("target_bbox_diag", "bbox_diag"),
        ("source_nn_spacing", "nearest_neighbor_spacing"),
        ("target_nn_spacing", "nearest_neighbor_spacing"),
    ]
    for label, key in metric_specs:
        stats = geom.get(key, {})
        if key in {"bbox_diag", "nearest_neighbor_spacing"}:
            prefix = "source" if label.startswith("source") else "target"
            stats = {
                "p10": stats.get(f"{prefix}_p10"),
                "p50": stats.get(f"{prefix}_p50"),
                "p90": stats.get(f"{prefix}_p90"),
            }
        rows.append(
            f"{label:22s} p10={_value_text(stats.get('p10')):>8s} "
            f"p50={_value_text(stats.get('p50')):>8s} "
            f"p90={_value_text(stats.get('p90')):>8s}"
        )
    return "\n".join(rows)


def _write_markdown(
    out_dir: Path,
    filename: str,
    title: str,
    sources: Iterable[str],
    sections: list[tuple[str, str]],
) -> Path:
    lines = [f"# {title}", "", "## Sources"]
    lines.extend(f"- `{source}`" for source in sources)
    for heading, body in sections:
        lines.extend(["", f"## {heading}", "", "```text", body.rstrip(), "```"])
    path = out_dir / filename
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _measure_sections(sections: list[tuple[str, str]]) -> int:
    y = 150
    for _, body in sections:
        y += 46
        for line in body.splitlines() or [""]:
            wrapped = _wrap_text(line, 138)
            y += max(1, len(wrapped)) * LINE_H
        y += SECTION_GAP
    return max(900, y + MARGIN)


def _render_panel(
    out_path: Path,
    title: str,
    subtitle: str,
    sections: list[tuple[str, str]],
    accent: str = BLUE,
) -> Path:
    height = _measure_sections(sections)
    image = Image.new("RGB", (CANVAS_W, height), BG)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (MARGIN, MARGIN, CANVAS_W - MARGIN, height - MARGIN),
        radius=22,
        fill=CARD,
        outline=RULE,
        width=3,
    )
    draw.text((MARGIN + 34, MARGIN + 30), title, font=TITLE, fill=INK)
    draw.text((MARGIN + 34, MARGIN + 90), subtitle, font=SUBTITLE, fill=MUTED)
    draw.line(
        (MARGIN + 34, MARGIN + 128, CANVAS_W - MARGIN - 34, MARGIN + 128),
        fill=RULE,
        width=3,
    )
    y = MARGIN + 158
    box_x0 = MARGIN + 34
    box_x1 = CANVAS_W - MARGIN - 34
    for heading, body in sections:
        draw.text((box_x0, y), heading.upper(), font=HEADER, fill=accent)
        y += 44
        wrapped_lines: list[str] = []
        for raw_line in body.splitlines() or [""]:
            wrapped = _wrap_text(raw_line, 138)
            wrapped_lines.extend(wrapped or [""])
        block_h = max(1, len(wrapped_lines)) * LINE_H + 26
        draw.rounded_rectangle(
            (box_x0, y - 8, box_x1, y + block_h),
            radius=12,
            fill=CODE_BG,
            outline=RULE,
            width=2,
        )
        for line in wrapped_lines:
            draw.text((box_x0 + 18, y), line, font=CODE, fill=INK)
            y += LINE_H
        y += SECTION_GAP
    image.save(out_path)
    return out_path


def _render_file_pages(
    out_dir: Path,
    stem: str,
    source_path: Path,
    title: str,
    lines_per_page: int = 72,
) -> list[Path]:
    lines = _read(source_path).splitlines()
    outputs: list[Path] = []
    for page_idx, start in enumerate(range(0, len(lines), lines_per_page), start=1):
        end = min(start + lines_per_page, len(lines))
        page_lines = [
            f"{idx + 1:04d}: {line}" for idx, line in enumerate(lines[start:end], start)
        ]
        page_path = out_dir / f"{stem}_p{page_idx:02d}.png"
        _render_panel(
            page_path,
            f"{title} ({page_idx}/{(len(lines) - 1) // lines_per_page + 1})",
            str(source_path),
            [("file screenshot", "\n".join(page_lines))],
            accent=GREEN,
        )
        outputs.append(page_path)
    return outputs


def _proposal_prompt_contract(context: dict) -> str:
    summary = context["target_profile"]["summary"]
    routes = context["reference_profiles"]["routes"]
    route_lines = [
        f"- {route['dataset']}: {route['route']} ({route['config_path']})"
        for route in routes
    ]
    return "\n".join(
        [
            "ROLE: evidence-constrained DPG-HPT proposal author",
            "",
            "INPUTS:",
            f"- target: {summary['dataset_id']} | {summary['domain']}",
            f"- unit/pair: {summary['coordinate_unit']} | {summary['pair_type']}",
            f"- pose: {summary.get('pose_direction')} | "
            f"{summary.get('pose_transform_shapes')}",
            *route_lines,
            f"- candidate_limit: {context.get('candidate_limit')}",
            f"- allowed_parameters: {len(context.get('allowed_parameters', []))}",
            f"- locked_parameters: {len(context.get('locked_parameters', []))}",
            "",
            "INSTRUCTION:",
            "Read only the frozen context pack, dataset profiles, model route cards,",
            "public config evidence, and transfer rules. Select/reject routes with",
            "evidence. Fill conservative/default/aggressive candidates only for",
            "allowlisted fields. For every field, write value, owner/status, source",
            "route, conversion rule, and evidence path. Preserve",
            "used_official_test_feedback=false. Abstain rather than invent evidence.",
        ]
    )


def build_materials(repo_root: Path, out_dir: Path) -> dict[str, list[str]]:
    run_dir = repo_root / "outputs/benchmark/hparam_transfer/regtr_measured_run"
    src_hpt = repo_root / "src/benchmarking/hparam_transfer"
    c3vd_profile_path = src_hpt / "dataset_profiles/c3vd_raycasting_v1.json"
    match_profile_path = src_hpt / "dataset_profiles/3DMatch.json"
    regtr_conf_path = repo_root / "baselines/RegTR/src/conf/3dmatch.yaml"
    train_script_path = repo_root / "baselines/RegTR/src/train.py"
    routes_path = repo_root / "configs/benchmark/hparam_transfer/baseline_routes.yaml"
    rules_path = repo_root / "configs/benchmark/hparam_transfer/transfer_rules.yaml"
    skill_path = src_hpt / "SKILL.md"
    context_path = run_dir / "context/context_pack.json"
    proposal_path = run_dir / "proposal/agent_proposal.yaml"
    validation_path = run_dir / "validated/proposal_validation.json"
    trace_path = run_dir / "validated/transfer_trace.json"
    candidate_path = run_dir / "validated/candidate_configs.yaml"
    candidate_validation_path = run_dir / "candidate_validation/validation_summary.json"
    smoke_log_path = (
        run_dir
        / "train_default/train_bridge_logs/c3vd/"
        / "260430_210548_benchmark_full_regtr/log.txt"
    )
    train_metrics_path = run_dir / "train_default/train_bridge_metrics/regtr_train.json"
    promoted_path = run_dir / "promoted/promoted_default.yaml"
    eval_config_path = run_dir / "configs/regtr_eval_full_8192_best_perturbed.yaml"

    out_dir.mkdir(parents=True, exist_ok=True)
    c3vd = _load_json(c3vd_profile_path)
    match = _load_json(match_profile_path)
    context = _load_json(context_path)
    proposal = _load_yaml(proposal_path)
    validation = _load_json(validation_path)
    candidate_validation = _load_json(candidate_validation_path)
    train_metrics = _load_json(train_metrics_path)
    trace = _load_json(trace_path)

    generated: dict[str, list[str]] = {}

    def add(name: str, paths: list[Path]) -> None:
        generated[name] = [str(path) for path in paths]

    c3vd_summary = c3vd["data"]
    c3vd_pose = c3vd["pose"]
    c3vd_sections = [
        (
            "target profile summary",
            "\n".join(
                [
                    f"dataset_id: {c3vd['dataset_id']}",
                    f"domain: {c3vd_summary['domain']}",
                    f"pair_type: {c3vd_summary['pair_type']}",
                    f"coordinate_unit: {c3vd_summary['coordinate_unit']}",
                    f"inferred_unit: {c3vd_summary['inferred_unit']}",
                    f"split_policy: {c3vd_summary.get('split_policy')}",
                    f"split/full_test_pairs: {c3vd['coverage'].get('split')} / "
                    f"{c3vd['coverage'].get('split_pair_count_full')}",
                ]
            ),
        ),
        ("measured geometry", _geometry_table(c3vd)),
        (
            "pose contract",
            "\n".join(
                [
                    f"storage_field: {c3vd_pose.get('storage_field')}",
                    f"pose_source: {c3vd_pose.get('pose_source')}",
                    f"transform_format: {c3vd_pose.get('transform_format')}",
                    f"transform_shapes: {c3vd_pose.get('transform_shapes')}",
                    f"direction: {c3vd_pose.get('direction')}",
                    f"translation_unit: {c3vd_pose.get('translation_unit')}",
                    f"valid_se3_fraction: {c3vd_pose.get('valid_se3_fraction')}",
                ]
            ),
        ),
    ]
    add(
        "data_census_c3vd",
        [
            _write_markdown(
                out_dir,
                "01_data_census_c3vd.md",
                "Data Census: Target C3VD",
                [str(c3vd_profile_path)],
                c3vd_sections,
            ),
            _render_panel(
                out_dir / "01_data_census_c3vd.png",
                "Data Census: Target C3VD",
                str(c3vd_profile_path),
                c3vd_sections,
                accent=BLUE,
            ),
        ],
    )

    match_sections = [
        (
            "source reference profile",
            "\n".join(
                [
                    f"dataset_id: {match['dataset_id']}",
                    f"domain: {match['data']['domain']}",
                    f"pair_type: {match['data']['pair_type']}",
                    f"coordinate_unit: {match['data']['coordinate_unit']}",
                    f"inferred_unit: {match['data']['inferred_unit']}",
                    "split: 46 train / 8 val / 8 test scenes (registry note)",
                    "overlap: commonly >30% registration-pair route",
                ]
            ),
        ),
        ("registry geometry availability", _geometry_table(match)),
        (
            "RegTR 3DMatch scale evidence",
            "\n".join(
                [
                    "dataset.overlap_radius: 0.0375",
                    "kpconv_options.first_subsampling_dl: 0.025",
                    "losses.r_p / r_n: 0.2 / 0.4",
                    "validation.reg_success_thresh_trans: 0.1",
                ]
            ),
        ),
    ]
    add(
        "data_census_3dmatch",
        [
            _write_markdown(
                out_dir,
                "02_data_census_3dmatch.md",
                "Data Census: Source 3DMatch",
                [str(match_profile_path), str(regtr_conf_path)],
                match_sections,
            ),
            _render_panel(
                out_dir / "02_data_census_3dmatch.png",
                "Data Census: Source 3DMatch",
                f"{match_profile_path} + {regtr_conf_path}",
                match_sections,
                accent=BLUE,
            ),
        ],
    )

    route_sections = [
        (
            "route card evidence",
            "\n".join(
                [
                    "model: regtr",
                    "known_route: 3DMatch -> indoor_metric_scene",
                    "config: baselines/RegTR/src/conf/3dmatch.yaml",
                    "preferred_for_c3vd: 3DMatch, 3DLoMatch",
                    "reject_for_c3vd_by_default: ModelNet40, KITTI",
                ]
            ),
        ),
        ("vendor 3DMatch config excerpt", _numbered_excerpt(regtr_conf_path, 1, 36)),
        ("loss/eval scale fields", _numbered_excerpt(regtr_conf_path, 88, 106)),
        ("training entrypoint excerpt", _numbered_excerpt(train_script_path, 1, 48)),
    ]
    add(
        "model_route_audit",
        [
            _write_markdown(
                out_dir,
                "03_model_route_audit_regtr_3dmatch.md",
                "Model Route Audit: RegTR 3DMatch",
                [str(routes_path), str(regtr_conf_path), str(train_script_path)],
                route_sections,
            ),
            _render_panel(
                out_dir / "03_model_route_audit_regtr_3dmatch.png",
                "Model Route Audit: RegTR 3DMatch",
                "Route card + vendor config + training entrypoint",
                route_sections,
                accent=AMBER,
            ),
        ],
    )

    rules_sections = [
        (
            "transfer rules",
            "\n".join(
                [
                    "candidate_limit: 3",
                    "allowed_groups: length_like, density_like, normalization_route,",
                    "  runtime_sanity, heuristic_control",
                    "locked_groups: architecture, optimization, benchmark_policy",
                    "firewall: use_official_test_feedback=false",
                    "proposal_policy: require evidence, route rationale,",
                    "  cross-profile compatibility, reject locked/unknown fields",
                ]
            ),
        ),
        ("rule file excerpt", _numbered_excerpt(rules_path, 1, 85)),
        ("skill workflow excerpt", _numbered_excerpt(skill_path, 1, 120)),
    ]
    add(
        "transfer_rules",
        [
            _write_markdown(
                out_dir,
                "04_transfer_rules.md",
                "Transfer Rules",
                [str(rules_path), str(skill_path)],
                rules_sections,
            ),
            _render_panel(
                out_dir / "04_transfer_rules.png",
                "Transfer Rules",
                "transfer_rules.yaml + hparam-transfer/SKILL.md",
                rules_sections,
                accent=AMBER,
            ),
        ],
    )

    prompt_sections = [
        ("pre-LLM prompt package", _proposal_prompt_contract(context)),
        (
            "auditable exchange note",
            "\n".join(
                [
                    "The retained reproducible chain is prompt package ->",
                    "agent_proposal.yaml -> validator outputs -> candidate configs.",
                    "No hidden private reasoning transcript is required or available;",
                    "field-level rationale and evidence paths are explicit in YAML.",
                ]
            ),
        ),
    ]
    add(
        "agent_prompt_chain",
        [
            _write_markdown(
                out_dir,
                "05_agent_prompt_chain.md",
                "Agent Proposal: Prompt Chain",
                [str(context_path), str(proposal_path)],
                prompt_sections,
            ),
            _render_panel(
                out_dir / "05_agent_prompt_chain.png",
                "Agent Proposal: Prompt Chain",
                "Frozen pre-LLM package reconstructed from context_pack.json",
                prompt_sections,
                accent=GREEN,
            ),
        ],
    )

    default_candidate = proposal["candidates"]["default"]["params"]
    proposal_summary = [
        (
            "proposal decision summary",
            "\n".join(
                [
                    "selected: 3DMatch indoor metric scene",
                    (
                        "rejected: ModelNet40 normalized-object route; "
                        "KITTI outdoor-lidar route"
                    ),
                    f"data.num_points: {default_candidate['data.num_points']['value']}",
                    (
                        "normalize_mode: "
                        f"{default_candidate['preprocess.normalize_mode']['value']}"
                    ),
                    f"voxel_size: {default_candidate['data.voxel_size']['value']}",
                    (
                        "matching_radius: "
                        f"{default_candidate['model.matching_radius']['value']}"
                    ),
                    f"losses.r_p/r_n: {default_candidate['losses.r_p']['value']} / "
                    f"{default_candidate['losses.r_n']['value']}",
                    f"eval.acceptance_radius: "
                    f"{default_candidate['eval.acceptance_radius']['value']}",
                ]
            ),
        )
    ]
    proposal_summary_paths = [
        _write_markdown(
            out_dir,
            "06_agent_proposal_summary.md",
            "Agent Proposal: Output Summary",
            [str(proposal_path)],
            proposal_summary,
        ),
        _render_panel(
            out_dir / "06_agent_proposal_summary.png",
            "Agent Proposal: Output Summary",
            str(proposal_path),
            proposal_summary,
            accent=GREEN,
        ),
    ]
    proposal_pages = _render_file_pages(
        out_dir,
        "06_agent_proposal_yaml",
        proposal_path,
        "Agent Proposal YAML",
        lines_per_page=72,
    )
    add("agent_proposal_output", proposal_summary_paths + proposal_pages)

    validation_sections = [
        (
            "deterministic validation pseudo-code",
            "\n".join(
                [
                    "proposal = load_agent_proposal(agent_proposal.yaml)",
                    "context = load_context_pack(context_pack.json)",
                    "check schema_version, model, target_dataset",
                    "require used_official_test_feedback == false",
                    "require candidate_count <= candidate_limit",
                    "validate selected/rejected route rationales",
                    "validate cross-profile route comparisons + evidence",
                    "for each candidate field:",
                    "  reject locked fields and fields outside allowlist",
                    "  require allowed owner/status label",
                    "  require evidence path and transfer_basis",
                    "if passed: normalize candidate configs and write trace digests",
                ]
            ),
        ),
        (
            "RegTR validation output",
            "\n".join(
                [
                    f"passed: {validation['passed']}",
                    f"errors: {len(validation['errors'])}",
                    f"warnings: {len(validation['warnings'])}",
                    f"candidate_count: {validation['candidate_count']}",
                    f"validated_candidates: {validation['validated_candidates']}",
                    f"context_digest: {trace['digests']['context_digest'][:16]}...",
                    f"proposal_digest: {trace['digests']['proposal_digest'][:16]}...",
                    f"candidate_bundle_digest: "
                    f"{trace['digests']['candidate_bundle_digest'][:16]}...",
                ]
            ),
        ),
        ("validation JSON excerpt", _numbered_excerpt(validation_path, 1, 72)),
    ]
    add(
        "deterministic_validation",
        [
            _write_markdown(
                out_dir,
                "07_deterministic_validation.md",
                "Deterministic Validation",
                [
                    str(
                        repo_root
                        / "src/benchmarking/hparam_transfer/proposal_validation.py"
                    ),
                    str(validation_path),
                    str(trace_path),
                ],
                validation_sections,
            ),
            _render_panel(
                out_dir / "07_deterministic_validation.png",
                "Deterministic Validation",
                "proposal_validation.py + proposal_validation.json",
                validation_sections,
                accent=RED,
            ),
        ],
    )

    smoke_sections = [
        (
            "smoke sanity pseudo-code",
            "\n".join(
                [
                    "bundle = load_candidate_bundle(candidate_configs.yaml)",
                    "runtime = load_benchmark_config(runtime_config.yaml)",
                    "for each candidate:",
                    "  deep-merge candidate overrides into runtime config",
                    "  set output_dir and hparam_transfer_candidate metadata",
                    "  if execute_eval:",
                    "    run_eval(config)",
                    "    reject crash/OOM/missing metrics/all-fail collapse",
                    "  else:",
                    "    record dry_run with built output path",
                ]
            ),
        ),
        (
            "candidate-validation output",
            "\n".join(
                [
                    f"execute_eval: {candidate_validation['execute_eval']}",
                    "conservative/default/aggressive: dry_run (execute_eval=false)",
                    f"passed_candidates: {candidate_validation['passed_candidates']}",
                    f"failed_candidates: {candidate_validation['failed_candidates']}",
                    "This retained DPG-HPT gate is config-build/dry-run evidence,",
                    "not official C3VD test-score selection.",
                ]
            ),
        ),
        (
            "RegTR train smoke task output",
            "\n".join(
                [
                    "script: src/benchmarking/bridges/train_regtr_c3vd.py",
                    "config: regtr_smoke_bridge_config.yaml",
                    "task: create loaders/model, run 1 validation sanity step",
                    f"wrapper_return_code: {train_metrics.get('return_code')}",
                    f"torch_peak_allocated_mb: "
                    f"{train_metrics.get('torch_peak_allocated_mb')}",
                    "command and output evidence from log:",
                    _numbered_excerpt(smoke_log_path, 1, 22),
                    _numbered_excerpt(smoke_log_path, 42, 58),
                ]
            ),
        ),
    ]
    add(
        "smoke_sanity",
        [
            _write_markdown(
                out_dir,
                "08_smoke_sanity.md",
                "Smoke Sanity",
                [
                    str(
                        repo_root
                        / "src/benchmarking/hparam_transfer/candidate_validation.py"
                    ),
                    str(candidate_validation_path),
                    str(train_metrics_path),
                    str(smoke_log_path),
                ],
                smoke_sections,
            ),
            _render_panel(
                out_dir / "08_smoke_sanity.png",
                "Smoke Sanity",
                "candidate_validation.py + retained RegTR smoke log",
                smoke_sections,
                accent=RED,
            ),
        ],
    )

    final_sections = [
        (
            "final benchmark config summary",
            "\n".join(
                [
                    "promoted candidate: default",
                    "benchmark: c3vd_raycasting_v1 | point_unit=mm_like",
                    "model: regtr",
                    "normalize_mode: none",
                    "num_points_override: 8192 in final eval config",
                    "overrides: voxel=2.5, radius=3.75, r_p=20.0, r_n=40.0,",
                    "  checkpoint-selection translation threshold=10.0",
                    "official RR/RTE thresholds remain benchmark-owned.",
                ]
            ),
        ),
        ("promoted_default excerpt", _numbered_excerpt(promoted_path, 1, 76)),
        ("final eval config excerpt", _numbered_excerpt(eval_config_path, 1, 80)),
    ]
    final_paths = [
        _write_markdown(
            out_dir,
            "09_final_benchmark_config.md",
            "Final Benchmark Config",
            [str(promoted_path), str(eval_config_path), str(candidate_path)],
            final_sections,
        ),
        _render_panel(
            out_dir / "09_final_benchmark_config.png",
            "Final Benchmark Config",
            "promoted_default.yaml + regtr_eval_full_8192_best_perturbed.yaml",
            final_sections,
            accent=BLUE,
        ),
    ]
    final_paths.extend(
        _render_file_pages(
            out_dir,
            "09_promoted_default_yaml",
            promoted_path,
            "Promoted Default YAML",
            lines_per_page=76,
        )
    )
    final_paths.extend(
        _render_file_pages(
            out_dir,
            "09_final_eval_config_yaml",
            eval_config_path,
            "Final Eval Config YAML",
            lines_per_page=80,
        )
    )
    add("final_benchmark_config", final_paths)

    manifest_path = out_dir / "materials_manifest.json"
    manifest_path.write_text(
        json.dumps(generated, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    generated["manifest"] = [str(manifest_path)]
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("BMVC2026_Linzhe/images/hpt_trace_materials"),
        help="Output directory for separate material screenshots.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo_root / args.out_dir
    generated = build_materials(repo_root=repo_root, out_dir=out_dir)
    for key, paths in generated.items():
        print(f"{key}:")
        for path in paths:
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
