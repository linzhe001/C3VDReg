"""Promotion gate for DPG-HPT candidate configs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from src.benchmarking.config.loader import load_benchmark_config
from src.benchmarking.config.schema import validate_config
from src.benchmarking.hparam_transfer.candidate_validation import (
    load_candidate_bundle,
)


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = _deep_merge(base[key], value)
        else:
            merged[key] = value
    return merged


def _load_structured_payload(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Structured payload does not exist: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return {} if payload is None else payload
    raise ValueError(f"Unsupported payload format: {path.suffix}")


def _payload_digest(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _candidate_status(
    validation_summary: dict[str, Any] | None,
    candidate_name: str,
) -> str | None:
    if validation_summary is None:
        return None
    for item in validation_summary.get("results", []):
        if item.get("candidate") == candidate_name:
            return str(item.get("status"))
    return None


def build_promoted_model_config(
    candidate_bundle: dict[str, Any],
    base_config: dict[str, Any],
    candidate_name: str,
    candidate_bundle_path: str | Path | None = None,
    base_config_path: str | Path | None = None,
    validation_summary: dict[str, Any] | None = None,
    validation_summary_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one promoted durable model config and its provenance manifest."""

    candidates = candidate_bundle.get("candidates", {})
    if candidate_name not in candidates:
        raise KeyError(f"Candidate {candidate_name!r} not found in candidate bundle.")

    candidate_payload = candidates[candidate_name]
    candidate_status = _candidate_status(validation_summary, candidate_name)
    if validation_summary is not None and candidate_status != "passed":
        raise ValueError(
            "Only candidates marked as 'passed' in validation_summary can be "
            f"promoted. Candidate {candidate_name!r} has status {candidate_status!r}."
        )

    promoted = json.loads(json.dumps(base_config))
    promoted.setdefault("model", {})
    promoted["model"]["id"] = candidate_bundle["model"]
    promoted["model"].setdefault("overrides", {})
    promoted["model"]["overrides"] = _deep_merge(
        promoted["model"]["overrides"],
        candidate_payload.get("overrides", {}),
    )
    promoted = validate_config(promoted).to_dict()

    manifest = {
        "schema_version": 1,
        "protocol_id": candidate_bundle.get("protocol_id"),
        "model": candidate_bundle.get("model"),
        "target_dataset": candidate_bundle.get("target_dataset"),
        "promoted_candidate": candidate_name,
        "candidate_status": candidate_status,
        "source_artifacts": {
            "candidate_bundle_path": (
                str(Path(candidate_bundle_path).resolve())
                if candidate_bundle_path is not None
                else None
            ),
            "base_config_path": (
                str(Path(base_config_path).resolve())
                if base_config_path is not None
                else None
            ),
            "validation_summary_path": (
                str(Path(validation_summary_path).resolve())
                if validation_summary_path is not None
                else None
            ),
        },
        "digests": {
            "candidate_bundle_digest": candidate_bundle.get("digests", {}).get(
                "candidate_bundle_digest"
            ),
            "candidate_provenance_digest": _payload_digest(
                candidate_payload.get("provenance", {})
            ),
            "promoted_config_digest": _payload_digest(promoted),
        },
        "selected_overrides": dict(candidate_payload.get("overrides", {})),
        "field_metadata": candidate_payload.get("field_metadata", {}),
    }
    return promoted, manifest


def write_promoted_artifacts(
    promoted_config: dict[str, Any],
    manifest: dict[str, Any],
    output_dir: str | Path,
    candidate_name: str,
    target_config_path: str | Path | None = None,
    execute: bool = False,
) -> dict[str, str | None]:
    """Write preview artifacts and optionally promote to a durable config path."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_config_path = output_dir / f"promoted_{candidate_name}.yaml"
    preview_manifest_path = output_dir / f"promoted_{candidate_name}.json"
    preview_config_path.write_text(
        yaml.safe_dump(promoted_config, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    preview_manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    resolved_target_config = None
    resolved_target_manifest = None
    if execute:
        if target_config_path is None:
            raise ValueError("execute=True requires target_config_path.")
        target_path = Path(target_config_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_manifest_path = target_path.with_suffix(".hparam_transfer.json")
        target_path.write_text(
            yaml.safe_dump(promoted_config, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
        target_manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        resolved_target_config = str(target_path.resolve())
        resolved_target_manifest = str(target_manifest_path.resolve())

    return {
        "preview_config_path": str(preview_config_path.resolve()),
        "preview_manifest_path": str(preview_manifest_path.resolve()),
        "target_config_path": resolved_target_config,
        "target_manifest_path": resolved_target_manifest,
    }


def load_validation_summary(path: str | Path | None) -> dict[str, Any] | None:
    """Load candidate validation summary from JSON or YAML."""

    payload = _load_structured_payload(path)
    if payload is not None and not isinstance(payload, dict):
        raise ValueError("Validation summary must deserialize to a mapping.")
    return payload


def load_base_model_config(path: str | Path) -> dict[str, Any]:
    """Load one durable benchmark config for promotion merge."""

    return load_benchmark_config(path)


def load_candidate_bundle_payload(path: str | Path) -> dict[str, Any]:
    """Load one candidate bundle."""

    return load_candidate_bundle(path)
