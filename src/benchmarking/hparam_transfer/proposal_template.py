"""Agent proposal template generation for DPG-HPT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def _route_parameter_name(field_spec: Any) -> str | None:
    if isinstance(field_spec, str):
        return field_spec
    if isinstance(field_spec, dict):
        parameter = field_spec.get("parameter") or field_spec.get("field")
        return str(parameter) if parameter else None
    return None


def _preferred_reference_selection(
    context_pack: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    route_card = context_pack.get("route_card", {})
    preferred = set(route_card.get("preferred_for_c3vd", []))
    rejected = set(route_card.get("reject_for_c3vd_by_default", []))
    selected_entries: list[dict[str, Any]] = []
    rejected_entries: list[dict[str, Any]] = []

    for route in context_pack.get("reference_profiles", {}).get("routes", []):
        dataset = route.get("dataset")
        evidence = [{"path": route.get("config_path"), "field": route.get("route")}]
        entry = {
            "dataset": dataset,
            "reason": (
                "Pre-filled from route card preference; replace with "
                "evidence-backed rationale after reading baseline config and code."
            ),
            "evidence": evidence,
        }
        if dataset in preferred:
            selected_entries.append(entry)
        elif dataset in rejected:
            rejected_entries.append(entry)
    return selected_entries, rejected_entries


def _cross_profile_template(
    context_pack: dict[str, Any],
    selected_entries: list[dict[str, Any]],
    rejected_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = {entry.get("dataset") for entry in selected_entries}
    rejected = {entry.get("dataset") for entry in rejected_entries}
    target_profile = context_pack.get("target_profile", {})
    target_summary = target_profile.get("summary", {})
    route_comparisons: list[dict[str, Any]] = []
    for route in context_pack.get("reference_profiles", {}).get("routes", []):
        dataset = route.get("dataset")
        if dataset in selected:
            status = "preferred"
        elif dataset in rejected:
            status = "rejected"
        else:
            status = "usable_with_risk"
        route_comparisons.append(
            {
                "dataset": dataset,
                "route": route.get("route"),
                "status": status,
                "reason": (
                    "Replace with C3VD-vs-route comparison across domain, "
                    "pair type, unit/scale, normalization, pose format/direction, "
                    "density, overlap, and model support."
                ),
                "evidence": [
                    {
                        "path": route.get("config_path"),
                        "field": route.get("route"),
                    }
                ],
            }
        )
    return {
        "target_profile": {
            "dataset": target_summary.get("dataset_id"),
            "digest": target_profile.get("digest"),
            "pose_direction": target_summary.get("pose_direction"),
            "pose_transform_shapes": target_summary.get("pose_transform_shapes"),
            "pose_source": target_summary.get("pose_source"),
            "pose_translation_unit": target_summary.get("pose_translation_unit"),
            "pose_storage_field_present_fraction": target_summary.get(
                "pose_storage_field_present_fraction"
            ),
        },
        "route_comparisons": route_comparisons,
        "train_hparam_decision_policy": [
            "Use only model-supported routes that are compatible with C3VD.",
            (
                "Keep benchmark-owned C3VD raw point budget at 8192 "
                "unless explicitly changed."
            ),
            (
                "Record per-field owner, source route/profile, "
                "conversion rule, and evidence."
            ),
            (
                "Reject train-ready status if unit, normalization, or "
                "pose-shape/direction train/eval protocol mismatches remain."
            ),
        ],
    }


def build_agent_proposal_template(
    context_pack: dict[str, Any],
    agent_name: str = "codex_or_claude_code",
) -> dict[str, Any]:
    """Build a structured proposal skeleton from a frozen context pack."""

    selected_entries, rejected_entries = _preferred_reference_selection(context_pack)
    expected_candidates = context_pack.get("candidate_policy", {}).get(
        "expected_candidates",
        ["default"],
    )
    route_card = context_pack.get("route_card", {})
    review_queue: list[str] = []
    for field_spec in route_card.get("scale_sensitive_params", []):
        parameter_name = _route_parameter_name(field_spec)
        if (
            parameter_name is not None
            and parameter_name in context_pack.get("allowed_parameters", [])
        ):
            review_queue.append(parameter_name)
    proposal: dict[str, Any] = {
        "schema_version": 1,
        "model": context_pack.get("model"),
        "target_dataset": context_pack.get("target_dataset"),
        "agent": {
            "name": agent_name,
            "run_id": None,
        },
        "test_set_firewall": {
            "used_official_test_feedback": False,
        },
        "reference_selection": {
            "selected": selected_entries,
            "rejected": rejected_entries,
        },
        "cross_profile_compatibility": _cross_profile_template(
            context_pack,
            selected_entries,
            rejected_entries,
        ),
        "review_queue": review_queue,
        "candidates": {
            candidate_name: {"params": {}}
            for candidate_name in expected_candidates
        },
        "risks": list(context_pack.get("known_risks", [])),
        "notes_for_agent": [
            "Replace pre-filled reasons with evidence-backed rationales.",
            "Complete cross_profile_compatibility before filling C3VD train hparams.",
            "Fill candidate params only for allowlisted fields.",
            "Move code-change requirements into risks instead of candidate params.",
            "Preserve used_official_test_feedback=false.",
        ],
        "digests": {
            "context_digest": context_pack.get("digests", {}).get("context_digest"),
            "template_digest": None,
        },
    }
    proposal["digests"]["template_digest"] = json.dumps(
        proposal,
        sort_keys=True,
        separators=(",", ":"),
    )
    return proposal


def write_agent_proposal_template(
    proposal: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    """Write `agent_proposal.yaml` into a target directory."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    proposal_path = output_dir / "agent_proposal.yaml"
    proposal_path.write_text(
        yaml.safe_dump(
            proposal,
            sort_keys=False,
            allow_unicode=False,
        ),
        encoding="utf-8",
    )
    return proposal_path
