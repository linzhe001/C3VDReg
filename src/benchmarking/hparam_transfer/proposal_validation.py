"""Proposal validation and candidate normalization for DPG-HPT."""

from __future__ import annotations

import hashlib
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


def load_agent_proposal(path: str | Path) -> dict[str, Any]:
    """Load one agent proposal from YAML or JSON."""

    payload = _load_structured_payload(path)
    if not isinstance(payload, dict):
        raise ValueError("Agent proposal must deserialize to a mapping.")
    return payload


def load_context_pack(path: str | Path) -> dict[str, Any]:
    """Load one context pack from JSON or YAML."""

    payload = _load_structured_payload(path)
    if not isinstance(payload, dict):
        raise ValueError("Context pack must deserialize to a mapping.")
    return payload


def _proposal_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _candidate_names(proposal: dict[str, Any]) -> list[str]:
    candidates = proposal.get("candidates", {})
    if not isinstance(candidates, dict):
        return []
    return list(candidates.keys())


def _validate_reference_selection(
    proposal: dict[str, Any],
    require_rationale: bool,
) -> list[str]:
    errors: list[str] = []
    selection = proposal.get("reference_selection", {})
    if not isinstance(selection, dict):
        return ["reference_selection must be a mapping."]
    for bucket in ("selected", "rejected"):
        entries = selection.get(bucket, [])
        if not isinstance(entries, list):
            errors.append(f"reference_selection.{bucket} must be a list.")
            continue
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                errors.append(
                    f"reference_selection.{bucket}[{index}] must be a mapping."
                )
                continue
            if require_rationale and not entry.get("reason"):
                errors.append(
                    f"reference_selection.{bucket}[{index}] is missing reason."
                )
    return errors


def _route_key(route: dict[str, Any]) -> tuple[Any, Any]:
    return route.get("dataset"), route.get("route")


def _validate_cross_profile_compatibility(
    proposal: dict[str, Any],
    context_pack: dict[str, Any],
    require_cross_profile: bool,
    require_evidence: bool,
) -> list[str]:
    if not require_cross_profile and "cross_profile_compatibility" not in proposal:
        return []

    errors: list[str] = []
    payload = proposal.get("cross_profile_compatibility")
    if not isinstance(payload, dict):
        return ["cross_profile_compatibility must be a mapping."]

    target_profile = payload.get("target_profile", {})
    if not isinstance(target_profile, dict):
        errors.append("cross_profile_compatibility.target_profile must be a mapping.")
    else:
        expected_target = context_pack.get("target_dataset")
        if target_profile.get("dataset") not in {expected_target, None}:
            errors.append(
                "cross_profile_compatibility.target_profile.dataset does not match "
                "context_pack.target_dataset."
            )
        expected_summary = context_pack.get("target_profile", {}).get("summary", {})
        expected_pose_direction = expected_summary.get("pose_direction")
        if (
            expected_pose_direction is not None
            and target_profile.get("pose_direction") != expected_pose_direction
        ):
            errors.append(
                "cross_profile_compatibility.target_profile.pose_direction does not "
                "match context_pack target pose direction."
            )
        expected_pose_shapes = expected_summary.get("pose_transform_shapes")
        if (
            expected_pose_shapes
            and target_profile.get("pose_transform_shapes") != expected_pose_shapes
        ):
            errors.append(
                "cross_profile_compatibility.target_profile.pose_transform_shapes "
                "does not match context_pack target pose shapes."
            )
        expected_pose_source = expected_summary.get("pose_source")
        if (
            expected_pose_source is not None
            and target_profile.get("pose_source") != expected_pose_source
        ):
            errors.append(
                "cross_profile_compatibility.target_profile.pose_source does not "
                "match context_pack target pose source."
            )
        expected_pose_unit = expected_summary.get("pose_translation_unit")
        if (
            expected_pose_unit is not None
            and target_profile.get("pose_translation_unit") != expected_pose_unit
        ):
            errors.append(
                "cross_profile_compatibility.target_profile.pose_translation_unit "
                "does not match context_pack target pose unit."
            )
        expected_present_fraction = expected_summary.get(
            "pose_storage_field_present_fraction"
        )
        if (
            expected_present_fraction is not None
            and target_profile.get("pose_storage_field_present_fraction")
            != expected_present_fraction
        ):
            errors.append(
                "cross_profile_compatibility.target_profile."
                "pose_storage_field_present_fraction does not match context_pack."
            )

    comparisons = payload.get("route_comparisons", [])
    if not isinstance(comparisons, list):
        errors.append("cross_profile_compatibility.route_comparisons must be a list.")
        return errors
    if require_cross_profile and not comparisons:
        errors.append("cross_profile_compatibility.route_comparisons is empty.")

    allowed_statuses = {
        "preferred",
        "usable_with_risk",
        "density_only",
        "runtime_only",
        "rejected",
    }
    seen_routes: set[tuple[Any, Any]] = set()
    for index, entry in enumerate(comparisons):
        if not isinstance(entry, dict):
            errors.append(
                f"cross_profile_compatibility.route_comparisons[{index}] "
                "must be a mapping."
            )
            continue
        route_key = _route_key(entry)
        seen_routes.add(route_key)
        status = entry.get("status")
        if status not in allowed_statuses:
            errors.append(
                f"cross_profile_compatibility.route_comparisons[{index}] "
                f"uses invalid status '{status}'."
            )
        if not entry.get("reason"):
            errors.append(
                f"cross_profile_compatibility.route_comparisons[{index}] "
                "is missing reason."
            )
        if require_evidence and not entry.get("evidence"):
            errors.append(
                f"cross_profile_compatibility.route_comparisons[{index}] "
                "is missing evidence."
            )

    expected_routes = {
        _route_key(route)
        for route in context_pack.get("reference_profiles", {}).get("routes", [])
    }
    missing_routes = sorted(expected_routes - seen_routes)
    for dataset, route in missing_routes:
        errors.append(
            "cross_profile_compatibility.route_comparisons is missing "
            f"model route {dataset}:{route}."
        )
    return errors


def _validate_candidate_params(
    proposal: dict[str, Any],
    context_pack: dict[str, Any],
) -> tuple[list[str], list[str], dict[str, dict[str, Any]]]:
    errors: list[str] = []
    warnings: list[str] = []
    normalized: dict[str, dict[str, Any]] = {}
    candidates = proposal.get("candidates", {})
    if not isinstance(candidates, dict):
        return ["candidates must be a mapping."], warnings, normalized

    allowed_parameters = set(context_pack.get("allowed_parameters", []))
    locked_parameters = set(context_pack.get("locked_parameters", []))
    allowed_labels = set(
        context_pack.get("candidate_policy", {}).get("allowed_labels", [])
    )
    require_evidence = bool(
        context_pack.get("proposal_policy", {}).get("require_evidence", False)
    )

    for candidate_name, candidate_payload in candidates.items():
        if not isinstance(candidate_payload, dict):
            errors.append(f"candidates.{candidate_name} must be a mapping.")
            continue
        params = candidate_payload.get("params", {})
        if not isinstance(params, dict):
            errors.append(f"candidates.{candidate_name}.params must be a mapping.")
            continue
        normalized_params: dict[str, Any] = {}
        for field_name, field_payload in params.items():
            if field_name in locked_parameters:
                errors.append(
                    f"candidates.{candidate_name}.{field_name} modifies a locked field."
                )
            if field_name not in allowed_parameters:
                errors.append(
                    f"candidates.{candidate_name}.{field_name} is not in allowlist."
                )
                continue
            if not isinstance(field_payload, dict):
                errors.append(
                    f"candidates.{candidate_name}.{field_name} must be a mapping."
                )
                continue
            status = field_payload.get("status")
            if status not in allowed_labels:
                errors.append(
                    f"candidates.{candidate_name}.{field_name} uses invalid status "
                    f"'{status}'."
                )
            evidence = field_payload.get("evidence", [])
            if require_evidence and not evidence:
                errors.append(
                    f"candidates.{candidate_name}.{field_name} is missing evidence."
                )
            if status == "unsupported_requires_code_change":
                errors.append(
                    f"candidates.{candidate_name}.{field_name} must be moved to risks, "
                    "not candidate params."
                )
            if status == "requires_user_approval":
                warnings.append(
                    f"candidates.{candidate_name}.{field_name} requires user approval."
                )
            normalized_params[field_name] = {
                "value": field_payload.get("value"),
                "status": status,
                "evidence": evidence,
                "transfer_basis": field_payload.get("transfer_basis"),
            }
        normalized[candidate_name] = normalized_params
    return errors, warnings, normalized


def validate_proposal(
    proposal: dict[str, Any],
    context_pack: dict[str, Any],
) -> dict[str, Any]:
    """Validate one agent proposal against a frozen context pack."""

    errors: list[str] = []
    warnings: list[str] = []

    if proposal.get("schema_version") != 1:
        errors.append("schema_version must equal 1.")
    if proposal.get("model") != context_pack.get("model"):
        errors.append("proposal.model does not match context_pack.model.")
    if proposal.get("target_dataset") != context_pack.get("target_dataset"):
        errors.append("proposal.target_dataset does not match context pack.")

    firewall = proposal.get("test_set_firewall", {})
    if firewall.get("used_official_test_feedback") is not False:
        errors.append("Proposal must declare used_official_test_feedback=false.")

    candidate_limit = int(context_pack.get("candidate_limit", 0))
    candidate_names = _candidate_names(proposal)
    if len(candidate_names) > candidate_limit:
        errors.append(
            f"Candidate count {len(candidate_names)} exceeds limit {candidate_limit}."
        )

    expected_candidates = context_pack.get("candidate_policy", {}).get(
        "expected_candidates",
        [],
    )
    unexpected = [
        candidate_name
        for candidate_name in candidate_names
        if expected_candidates and candidate_name not in expected_candidates
    ]
    for candidate_name in unexpected:
        warnings.append(f"Unexpected candidate name: {candidate_name}")

    require_rationale = bool(
        context_pack.get("proposal_policy", {}).get(
            "require_reference_selection_rationale",
            False,
        )
    )
    require_evidence = bool(
        context_pack.get("proposal_policy", {}).get("require_evidence", False)
    )
    require_cross_profile = bool(
        context_pack.get("proposal_policy", {}).get(
            "require_cross_profile_compatibility",
            False,
        )
    )
    errors.extend(_validate_reference_selection(proposal, require_rationale))
    errors.extend(
        _validate_cross_profile_compatibility(
            proposal,
            context_pack,
            require_cross_profile,
            require_evidence,
        )
    )
    param_errors, param_warnings, normalized = _validate_candidate_params(
        proposal,
        context_pack,
    )
    errors.extend(param_errors)
    warnings.extend(param_warnings)

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "candidate_count": len(candidate_names),
        "validated_candidates": sorted(normalized.keys()),
        "normalized_params": normalized,
    }


def normalize_candidate_configs(
    proposal: dict[str, Any],
    context_pack: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    """Normalize validated candidate params into generated config bundle."""

    if not validation.get("passed", False):
        raise ValueError("Cannot normalize candidate configs from a failed proposal.")

    normalized_params = validation.get("normalized_params", {})
    candidate_bundle: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": context_pack.get("protocol_id"),
        "model": proposal.get("model"),
        "target_dataset": proposal.get("target_dataset"),
        "candidates": {},
    }
    for candidate_name, params in normalized_params.items():
        metadata_only_statuses = {"benchmark_owned", "model_private"}
        overrides = {
            field_name: payload["value"]
            for field_name, payload in params.items()
            if payload["status"] not in metadata_only_statuses
        }
        candidate_bundle["candidates"][candidate_name] = {
            "overrides": overrides,
            "field_metadata": params,
            "provenance": {
                "context_digest": context_pack["digests"]["context_digest"],
                "target_profile_digest": context_pack["digests"][
                    "target_profile_digest"
                ],
                "reference_profile_digest": context_pack["digests"][
                    "reference_profile_digest"
                ],
                "proposal_digest": _proposal_digest(proposal),
            },
        }
    candidate_bundle["digests"] = {
        "candidate_bundle_digest": _proposal_digest(candidate_bundle),
    }
    return candidate_bundle


def build_transfer_trace(
    proposal: dict[str, Any],
    context_pack: dict[str, Any],
    validation: dict[str, Any],
    candidate_configs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact trace payload for validation outputs."""

    trace: dict[str, Any] = {
        "schema_version": 1,
        "model": context_pack.get("model"),
        "target_dataset": context_pack.get("target_dataset"),
        "digests": {
            "context_digest": context_pack["digests"]["context_digest"],
            "proposal_digest": _proposal_digest(proposal),
            "validation_digest": _proposal_digest(validation),
        },
        "validation_summary": {
            "passed": validation.get("passed"),
            "error_count": len(validation.get("errors", [])),
            "warning_count": len(validation.get("warnings", [])),
        },
    }
    if candidate_configs is not None:
        trace["digests"]["candidate_bundle_digest"] = candidate_configs["digests"][
            "candidate_bundle_digest"
        ]
    return trace


def write_validation_outputs(
    validation: dict[str, Any],
    candidate_configs: dict[str, Any] | None,
    trace: dict[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path | None, Path]:
    """Write validation JSON, candidate YAML, and transfer trace JSON."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = output_dir / "proposal_validation.json"
    trace_path = output_dir / "transfer_trace.json"
    candidate_path: Path | None = None

    validation_path.write_text(
        json.dumps(validation, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    trace_path.write_text(
        json.dumps(trace, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    if candidate_configs is not None:
        candidate_path = output_dir / "candidate_configs.yaml"
        candidate_path.write_text(
            yaml.safe_dump(
                candidate_configs,
                sort_keys=False,
                allow_unicode=False,
            ),
            encoding="utf-8",
        )
    return validation_path, candidate_path, trace_path
