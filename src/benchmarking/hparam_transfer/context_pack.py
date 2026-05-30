"""Context pack assembly for DPG-HPT."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _flatten_rule_groups(groups: dict[str, list[str]]) -> list[str]:
    flattened: list[str] = []
    for _, fields in groups.items():
        for field in fields:
            if field not in flattened:
                flattened.append(field)
    return flattened


def _source_refs(reference_profiles: dict[str, Any]) -> dict[str, list[Any]]:
    config_refs: list[dict[str, Any]] = []
    code_paths: list[str] = []
    for route in reference_profiles.get("routes", []):
        config_refs.append(
            {
                "dataset": route["dataset"],
                "route": route["route"],
                "config_path": route["config_path"],
                "confidence": route.get("confidence"),
            }
        )
        config_path = route["config_path"]
        if config_path not in code_paths:
            code_paths.append(config_path)
        for field in route.get("fields", []):
            if field["config_path"] not in code_paths:
                code_paths.append(field["config_path"])
    return {
        "configs": config_refs,
        "code_paths": code_paths,
    }


def _known_risks(
    route_card: dict[str, Any],
    reference_profiles: dict[str, Any],
) -> list[dict[str, Any]]:
    risks = list(route_card.get("known_risks", []))
    bundle_risks = reference_profiles.get("known_risks", [])
    for risk in bundle_risks:
        if risk not in risks:
            risks.append(risk)
    return risks


def build_context_pack(
    model_id: str,
    target_dataset: str,
    target_profile: dict[str, Any],
    reference_profiles: dict[str, Any],
    route_card: dict[str, Any],
    transfer_rules: dict[str, Any],
) -> dict[str, Any]:
    """Build a frozen agent-facing context pack."""

    allowed_parameters = _flatten_rule_groups(
        transfer_rules.get("allowed_groups", {})
    )
    locked_parameters = _flatten_rule_groups(
        transfer_rules.get("locked_groups", {})
    )
    target_pose = target_profile.get("pose", {})
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": transfer_rules.get("protocol_id"),
        "model": model_id,
        "target_dataset": target_dataset,
        "target_profile": {
            "digest": target_profile["digests"]["profile_digest"],
            "summary": {
                "dataset_id": target_profile["dataset_id"],
                "domain": target_profile["data"]["domain"],
                "coordinate_unit": target_profile["data"]["coordinate_unit"],
                "inferred_unit": target_profile["data"]["inferred_unit"],
                "pair_type": target_profile["data"]["pair_type"],
                "split_policy": target_profile["data"]["split_policy"],
                "pose_direction": target_pose.get("direction"),
                "pose_transform_shapes": target_pose.get("transform_shapes", []),
                "pose_source": target_pose.get("pose_source"),
                "pose_translation_unit": target_pose.get("translation_unit"),
                "pose_valid_se3_fraction": target_pose.get("valid_se3_fraction"),
                "pose_storage_field_present_fraction": target_pose.get(
                    "storage_field_present_fraction"
                ),
            },
            "coverage": target_profile.get("coverage", {}),
            "pose": target_pose,
        },
        "reference_profiles": {
            "digest": reference_profiles["digests"]["reference_profile_digest"],
            "route_count": reference_profiles["route_count"],
            "routes": reference_profiles["routes"],
        },
        "route_card": route_card,
        "allowed_parameters": allowed_parameters,
        "locked_parameters": locked_parameters,
        "candidate_limit": transfer_rules.get("candidate_limit", 0),
        "test_set_firewall": transfer_rules.get("firewall", {}),
        "candidate_policy": transfer_rules.get("candidate_policy", {}),
        "proposal_policy": transfer_rules.get("proposal_policy", {}),
        "report_policy": transfer_rules.get("report_policy", {}),
        "source_refs": _source_refs(reference_profiles),
        "known_risks": _known_risks(route_card, reference_profiles),
    }
    payload["digests"] = {
        "context_digest": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "target_profile_digest": target_profile["digests"]["profile_digest"],
        "reference_profile_digest": reference_profiles["digests"][
            "reference_profile_digest"
        ],
    }
    return payload


def render_context_pack_markdown(context_pack: dict[str, Any]) -> str:
    """Render a compact markdown summary for an agent-facing context pack."""

    target_profile = context_pack["target_profile"]
    reference_profiles = context_pack["reference_profiles"]
    target_summary = target_profile["summary"]
    pose_shapes = ", ".join(target_summary.get("pose_transform_shapes") or ["unknown"])
    lines = [
        f"# Context Pack: {context_pack['model']} -> {context_pack['target_dataset']}",
        "",
        "## Target Profile",
        "",
        f"- dataset_id: `{target_summary['dataset_id']}`",
        f"- domain: `{target_summary['domain']}`",
        f"- coordinate_unit: `{target_summary['coordinate_unit']}`",
        f"- inferred_unit: `{target_summary['inferred_unit']}`",
        f"- pair_type: `{target_summary['pair_type']}`",
        f"- split_policy: `{target_summary['split_policy']}`",
        f"- pose_direction: `{target_summary.get('pose_direction')}`",
        f"- pose_transform_shapes: `{pose_shapes}`",
        f"- pose_source: `{target_summary.get('pose_source')}`",
        f"- pose_translation_unit: `{target_summary.get('pose_translation_unit')}`",
        (
            "- pose_storage_field_present_fraction: "
            f"`{target_summary.get('pose_storage_field_present_fraction')}`"
        ),
        (
            "- pose_valid_se3_fraction: "
            f"`{target_summary.get('pose_valid_se3_fraction')}`"
        ),
        f"- target_profile_digest: `{target_profile['digest']}`",
        "",
        "## Reference Routes",
        "",
        f"- route_count: `{reference_profiles['route_count']}`",
        "",
        "| Dataset | Route | Confidence | Resolved Fields | Missing Fields |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for route in reference_profiles["routes"]:
        lines.append(
            "| "
            f"{route['dataset']} | "
            f"{route['route']} | "
            f"{route['confidence']} | "
            f"{route['summary']['resolved_count']} | "
            f"{route['summary']['missing_count']} |"
        )
    lines.extend(
        [
            "",
            "## Policy",
            "",
            f"- candidate_limit: `{context_pack['candidate_limit']}`",
            f"- allowed_parameter_count: `{len(context_pack['allowed_parameters'])}`",
            f"- locked_parameter_count: `{len(context_pack['locked_parameters'])}`",
            (
                "- use_official_test_feedback: "
                f"`{context_pack['test_set_firewall'].get('use_official_test_feedback')}`"
            ),
            "",
            "## Known Risks",
            "",
        ]
    )
    if context_pack["known_risks"]:
        for risk in context_pack["known_risks"]:
            lines.append(
                "- "
                f"{risk.get('id', 'unknown_risk')}: "
                f"{risk.get('note', risk.get('status', 'unspecified'))}"
            )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Digests",
            "",
            f"- context_digest: `{context_pack['digests']['context_digest']}`",
            (
                "- reference_profile_digest: "
                f"`{context_pack['digests']['reference_profile_digest']}`"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def write_context_pack_outputs(
    context_pack: dict[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown outputs for a context pack."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "context_pack.json"
    md_path = output_dir / "context_pack.md"
    json_path.write_text(
        json.dumps(context_pack, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(
        render_context_pack_markdown(context_pack),
        encoding="utf-8",
    )
    return json_path, md_path


def load_transfer_rules(path: str | Path) -> dict[str, Any]:
    """Load transfer rules from YAML."""

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_route_card(path: str | Path, model_id: str) -> dict[str, Any]:
    """Load one model route card from baseline_routes.yaml."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return payload["models"][model_id]
