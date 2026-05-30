#!/usr/bin/env python3
"""Export RegTR DPG-HPT agent proposal YAML as Markdown materials."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {} if payload is None else payload


def _dump_yaml(payload: Any) -> str:
    return yaml.safe_dump(
        payload,
        sort_keys=False,
        allow_unicode=True,
        width=100,
    ).rstrip()


def _evidence_lines(evidence: list[dict[str, Any]] | None) -> list[str]:
    lines: list[str] = []
    for item in evidence or []:
        path = item.get("path", "missing")
        field = item.get("field", "missing")
        lines.append(f"  - `{path}` :: `{field}`")
    return lines or ["  - missing"]


def _write_full_markdown(
    proposal: dict[str, Any],
    source_path: Path,
    out_path: Path,
) -> None:
    lines = [
        "# RegTR DPG-HPT Agent Proposal YAML",
        "",
        f"- source: `{source_path}`",
        f"- model: `{proposal.get('model')}`",
        f"- target_dataset: `{proposal.get('target_dataset')}`",
        f"- agent: `{proposal.get('agent', {}).get('name')}`",
        f"- run_id: `{proposal.get('agent', {}).get('run_id')}`",
        f"- used_official_test_feedback: "
        f"`{proposal.get('test_set_firewall', {}).get('used_official_test_feedback')}`",
        "",
        "```yaml",
        _dump_yaml(proposal),
        "```",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _field_row(field_name: str, payload: dict[str, Any]) -> str:
    value = payload.get("value")
    status = payload.get("status")
    basis = str(payload.get("transfer_basis", "")).replace("\n", " ")
    evidence = payload.get("evidence", [])
    evidence_text = "<br>".join(
        f"`{item.get('path')}` :: `{item.get('field')}`" for item in evidence
    )
    return f"| `{field_name}` | `{value}` | `{status}` | {basis} | {evidence_text} |"


def _write_reasoning_chain(
    proposal: dict[str, Any],
    source_path: Path,
    out_path: Path,
) -> None:
    lines: list[str] = [
        "# RegTR DPG-HPT Auditable Rationale Chain",
        "",
        f"- source: `{source_path}`",
        "- scope: extracted from explicit YAML fields only",
        "- note: this is not a hidden chain-of-thought transcript; it is the retained "
        "rationale/evidence chain stored for reproducibility.",
        "",
        "## 1. Firewall And Task Binding",
        "",
        f"- model: `{proposal.get('model')}`",
        f"- target_dataset: `{proposal.get('target_dataset')}`",
        f"- used_official_test_feedback: "
        f"`{proposal.get('test_set_firewall', {}).get('used_official_test_feedback')}`",
        "",
        "## 2. Route Selection Rationale",
        "",
    ]

    selection = proposal.get("reference_selection", {})
    for bucket_title, bucket_key in [
        ("Selected Routes", "selected"),
        ("Rejected Routes", "rejected"),
    ]:
        lines.extend([f"### {bucket_title}", ""])
        for entry in selection.get(bucket_key, []):
            lines.extend(
                [
                    f"#### `{entry.get('dataset')}`",
                    "",
                    str(entry.get("reason", "")).strip(),
                    "",
                    "Evidence:",
                    *_evidence_lines(entry.get("evidence")),
                    "",
                ]
            )

    cross = proposal.get("cross_profile_compatibility", {})
    target = cross.get("target_profile", {})
    lines.extend(
        [
            "## 3. Target-Profile Interpretation",
            "",
            f"- dataset: `{target.get('dataset')}`",
            f"- domain: `{target.get('domain')}`",
            f"- source_modality: `{target.get('source_modality')}`",
            f"- target_modality: `{target.get('target_modality')}`",
            f"- pair_type: `{target.get('pair_type')}`",
            f"- coordinate_unit: `{target.get('coordinate_unit')}`",
            f"- inferred_unit: `{target.get('inferred_unit')}`",
            f"- normalization_route: `{target.get('normalization_route')}`",
            f"- point_count_profile: `{target.get('point_count_profile')}`",
            f"- bbox_spacing_profile: `{target.get('bbox_spacing_profile')}`",
            "",
            "## 4. Cross-Profile Route Comparison",
            "",
        ]
    )
    for entry in cross.get("route_comparisons", []):
        lines.extend(
            [
                f"### `{entry.get('dataset')}` / `{entry.get('route')}`",
                "",
                f"- status: `{entry.get('status')}`",
                f"- reason: {entry.get('reason')}",
                "- evidence:",
                *_evidence_lines(entry.get("evidence")),
                "",
            ]
        )

    lines.extend(["## 5. Train-Hparam Decision Policy", ""])
    for item in cross.get("train_hparam_decision_policy", []):
        lines.append(f"- {item}")
    lines.append("")

    default_params = proposal.get("candidates", {}).get("default", {}).get("params", {})
    lines.extend(
        [
            "## 6. Default Candidate Field Decisions",
            "",
            "| Field | Value | Owner/status | Transfer basis | Evidence |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for field_name, payload in default_params.items():
        lines.append(_field_row(field_name, payload))
    lines.append("")

    lines.extend(["## 7. Candidate Scale Variants", ""])
    for candidate_name, candidate in proposal.get("candidates", {}).items():
        params = candidate.get("params", {})
        voxel = params.get("data.voxel_size", {}).get("value")
        radius = params.get("model.matching_radius", {}).get("value")
        rp = params.get("losses.r_p", {}).get("value")
        rn = params.get("losses.r_n", {}).get("value")
        status = params.get("data.voxel_size", {}).get("status")
        lines.append(
            f"- `{candidate_name}`: voxel=`{voxel}`, matching_radius=`{radius}`, "
            f"r_p/r_n=`{rp}/{rn}`, status=`{status}`"
        )
    lines.append("")

    lines.extend(["## 8. Risks And Notes", ""])
    for risk in proposal.get("risks", []):
        lines.extend(
            [
                f"### `{risk.get('id')}`",
                "",
                f"- severity: `{risk.get('severity')}`",
                f"- note: {risk.get('note')}",
                "",
            ]
        )
    lines.extend(["Notes:", ""])
    for note in proposal.get("notes_for_agent", []):
        lines.append(f"- {note}")
    lines.append("")

    lines.extend(
        [
            "## 9. Appendix Figure Use",
            "",
            "For the agent-proposal panel, use Sections 2, 4, and 6 as the visible "
            "content: route decision, cross-profile comparison, and default-candidate "
            "field decisions. The full YAML can be cited as the retained model output.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--proposal",
        type=Path,
        default=Path("outputs/benchmark/hparam_transfer/regtr_measured_run/proposal/agent_proposal.yaml"),
        help="Path to agent_proposal.yaml.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("BMVC2026_Linzhe/appendix_materials/hpt"),
        help="Output directory for Markdown materials.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    proposal_path = args.proposal.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    proposal = _load_yaml(proposal_path)

    full_path = out_dir / "regtr_agent_proposal_full.md"
    reasoning_path = out_dir / "regtr_agent_proposal_reasoning_chain.md"
    _write_full_markdown(proposal, proposal_path, full_path)
    _write_reasoning_chain(proposal, proposal_path, reasoning_path)

    print(f"full_markdown: {full_path}")
    print(f"reasoning_chain: {reasoning_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
