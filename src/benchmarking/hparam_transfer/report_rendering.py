"""Human-readable transfer report rendering for DPG-HPT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def _load_structured_payload(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return {} if payload is None else payload
    raise ValueError(f"Unsupported payload format: {path.suffix}")


def load_validation_payload(path: str | Path) -> dict[str, Any]:
    """Load validation JSON or YAML."""

    payload = _load_structured_payload(path)
    if not isinstance(payload, dict):
        raise ValueError("Validation payload must deserialize to a mapping.")
    return payload


def _selection_lines(
    selection: dict[str, Any],
    bucket: str,
) -> list[str]:
    lines = [f"## {bucket.title()} References", ""]
    entries = selection.get(bucket, [])
    if not entries:
        lines.append("- none")
        lines.append("")
        return lines
    for entry in entries:
        dataset = entry.get("dataset", "unknown_dataset")
        reason = entry.get("reason", "no rationale provided")
        lines.append(f"- `{dataset}`: {reason}")
        evidence = entry.get("evidence", [])
        for item in evidence:
            lines.append(
                "  "
                f"- evidence: `{item.get('path')}` :: `{item.get('field')}`"
            )
    lines.append("")
    return lines


def _candidate_lines(proposal: dict[str, Any]) -> list[str]:
    lines = ["## Candidate Parameters", ""]
    candidates = proposal.get("candidates", {})
    if not candidates:
        lines.append("- none")
        lines.append("")
        return lines
    for candidate_name, candidate_payload in candidates.items():
        lines.extend(
            [
                f"### {candidate_name}",
                "",
                "| Field | Status | Value | Evidence Count |",
                "| --- | --- | --- | ---: |",
            ]
        )
        params = candidate_payload.get("params", {})
        if not params:
            lines.append("| none | none | none | 0 |")
        for field_name, field_payload in params.items():
            evidence = field_payload.get("evidence", [])
            value = json.dumps(field_payload.get("value"), ensure_ascii=True)
            lines.append(
                f"| {field_name} | {field_payload.get('status')} | "
                f"{value} | {len(evidence)} |"
            )
        lines.append("")
    return lines


def _risk_lines(
    context_pack: dict[str, Any],
    proposal: dict[str, Any],
) -> list[str]:
    lines = ["## Risks", ""]
    risks = list(context_pack.get("known_risks", [])) + list(proposal.get("risks", []))
    if not risks:
        lines.append("- none")
        lines.append("")
        return lines
    seen: set[str] = set()
    for risk in risks:
        risk_id = str(risk.get("id", "unknown_risk"))
        if risk_id in seen:
            continue
        seen.add(risk_id)
        note = risk.get("note", risk.get("status", "unspecified"))
        severity = risk.get("severity", "unspecified")
        lines.append(f"- `{risk_id}` [{severity}]: {note}")
    lines.append("")
    return lines


def _validation_lines(validation: dict[str, Any]) -> list[str]:
    lines = [
        "## Validation",
        "",
        f"- passed: `{validation.get('passed')}`",
        f"- candidate_count: `{validation.get('candidate_count')}`",
        f"- validated_candidates: `{validation.get('validated_candidates')}`",
        "",
        "### Errors",
        "",
    ]
    errors = validation.get("errors", [])
    if errors:
        lines.extend(f"- {error}" for error in errors)
    else:
        lines.append("- none")
    lines.extend(["", "### Warnings", ""])
    warnings = validation.get("warnings", [])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")
    lines.append("")
    return lines


def render_transfer_report(
    context_pack: dict[str, Any],
    proposal: dict[str, Any],
    validation: dict[str, Any],
) -> str:
    """Render a human-readable transfer report."""

    lines = [
        (
            "# Transfer Report: "
            f"{context_pack['model']} -> {context_pack['target_dataset']}"
        ),
        "",
        "## Summary",
        "",
        f"- protocol_id: `{context_pack.get('protocol_id')}`",
        (
            "- target_profile_digest: "
            f"`{context_pack['digests']['target_profile_digest']}`"
        ),
        (
            "- reference_profile_digest: "
            f"`{context_pack['digests']['reference_profile_digest']}`"
        ),
        f"- context_digest: `{context_pack['digests']['context_digest']}`",
        (
            "- use_official_test_feedback: "
            f"`{context_pack['test_set_firewall'].get('use_official_test_feedback')}`"
        ),
        "",
        "## Target Profile Snapshot",
        "",
        f"- domain: `{context_pack['target_profile']['summary']['domain']}`",
        (
            "- coordinate_unit: "
            f"`{context_pack['target_profile']['summary']['coordinate_unit']}`"
        ),
        (
            "- inferred_unit: "
            f"`{context_pack['target_profile']['summary']['inferred_unit']}`"
        ),
        f"- pair_type: `{context_pack['target_profile']['summary']['pair_type']}`",
        (
            "- pose_direction: "
            f"`{context_pack['target_profile']['summary'].get('pose_direction')}`"
        ),
        (
            "- pose_transform_shapes: "
            f"`{context_pack['target_profile']['summary'].get('pose_transform_shapes')}`"
        ),
        (
            "- pose_source: "
            f"`{context_pack['target_profile']['summary'].get('pose_source')}`"
        ),
        "",
    ]
    lines.extend(_selection_lines(proposal.get("reference_selection", {}), "selected"))
    lines.extend(_selection_lines(proposal.get("reference_selection", {}), "rejected"))
    lines.extend(_candidate_lines(proposal))
    lines.extend(
        [
            "## Locked Parameters",
            "",
            f"- count: `{len(context_pack.get('locked_parameters', []))}`",
        ]
    )
    lines.extend(
        f"- `{field_name}`" for field_name in context_pack.get("locked_parameters", [])
    )
    lines.extend([""])
    lines.extend(_risk_lines(context_pack, proposal))
    lines.extend(_validation_lines(validation))
    next_action = (
        "Proceed to candidate runtime validation."
        if validation.get("passed")
        else "Fix proposal errors before candidate generation."
    )
    lines.extend(["## Next Action", "", f"- {next_action}", ""])
    return "\n".join(lines)


def render_validation_summary(validation: dict[str, Any]) -> str:
    """Render a compact validation-only summary."""

    lines = [
        "# Validation Summary",
        "",
        f"- passed: `{validation.get('passed')}`",
        f"- error_count: `{len(validation.get('errors', []))}`",
        f"- warning_count: `{len(validation.get('warnings', []))}`",
        "",
    ]
    return "\n".join(lines)


def write_transfer_report(
    report_markdown: str,
    output_dir: str | Path,
) -> Path:
    """Write transfer_report.md to output_dir."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "transfer_report.md"
    report_path.write_text(report_markdown, encoding="utf-8")
    return report_path
