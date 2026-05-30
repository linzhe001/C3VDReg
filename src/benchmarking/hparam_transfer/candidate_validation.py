"""Runtime sanity validation for DPG-HPT candidate configs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.benchmarking.runners.eval_runner import run_eval


def _load_structured_payload(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return {} if payload is None else payload
    raise ValueError(f"Unsupported payload format: {path.suffix}")


def load_candidate_bundle(path: str | Path) -> dict[str, Any]:
    """Load one candidate config bundle from YAML or JSON."""

    payload = _load_structured_payload(path)
    if not isinstance(payload, dict):
        raise ValueError("Candidate bundle must deserialize to a mapping.")
    return payload


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = _deep_merge(base[key], value)
        else:
            merged[key] = value
    return merged


def _build_candidate_eval_config(
    runtime_config: dict[str, Any],
    candidate_bundle: dict[str, Any],
    candidate_name: str,
    candidate_payload: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    config = json.loads(json.dumps(runtime_config))
    config.setdefault("model", {})
    config["model"]["id"] = candidate_bundle["model"]
    config["model"].setdefault("overrides", {})
    config["model"]["overrides"] = _deep_merge(
        config["model"]["overrides"],
        candidate_payload.get("overrides", {}),
    )
    config.setdefault("runtime", {})
    config["runtime"]["output_dir"] = str((output_root / candidate_name).resolve())
    config["runtime"]["export_html"] = False
    config.setdefault("analysis", {})
    config["analysis"].setdefault("export", {})
    config["analysis"]["export"]["html"] = False
    config["analysis"].setdefault("qualitative", {})
    config["analysis"]["qualitative"]["export_failure_gallery"] = False
    config["model"]["overrides"].setdefault(
        "hparam_transfer_candidate",
        candidate_name,
    )
    config["model"]["overrides"].setdefault(
        "hparam_transfer_protocol",
        candidate_bundle.get("protocol_id"),
    )
    return config


def _status_from_exception(exc: Exception) -> tuple[str, str]:
    message = str(exc)
    lowered = message.lower()
    if "out of memory" in lowered or "oom" in lowered:
        return "oom", message
    return "failed", message


def validate_candidate_configs(
    candidate_bundle: dict[str, Any],
    runtime_config: dict[str, Any],
    output_root: str | Path,
    execute_eval: bool = True,
) -> dict[str, Any]:
    """Validate generated candidates with dry-run or real eval execution."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    candidates = candidate_bundle.get("candidates", {})

    for candidate_name, candidate_payload in candidates.items():
        eval_config = _build_candidate_eval_config(
            runtime_config=runtime_config,
            candidate_bundle=candidate_bundle,
            candidate_name=candidate_name,
            candidate_payload=candidate_payload,
            output_root=output_root,
        )
        result: dict[str, Any] = {
            "candidate": candidate_name,
            "status": "dry_run",
            "reason": "execute_eval=false",
            "output_dir": eval_config["runtime"]["output_dir"],
            "metrics": {},
        }
        if execute_eval:
            try:
                summary = run_eval(eval_config)
                overall = summary.get("overall", {})
                recall = float(
                    overall.get("registration_recall@rre_5deg_rte_5mm", 0.0)
                )
                result["metrics"] = {
                    "sample_count": summary.get("sample_count"),
                    "registration_recall@rre_5deg_rte_5mm": recall,
                    "rre_deg_mean": overall.get("rre_deg_mean"),
                    "rte_mm_mean": overall.get("rte_mm_mean"),
                    "latency_ms_mean": overall.get("latency_ms_mean"),
                }
                if recall <= 0.0:
                    result["status"] = "collapse"
                    result["reason"] = "all-fail collapse on validation subset"
                else:
                    result["status"] = "passed"
                    result["reason"] = "runtime sanity passed"
            except Exception as exc:  # noqa: BLE001
                status, reason = _status_from_exception(exc)
                result["status"] = status
                result["reason"] = reason
        results.append(result)

    summary = {
        "schema_version": 1,
        "protocol_id": candidate_bundle.get("protocol_id"),
        "model": candidate_bundle.get("model"),
        "target_dataset": candidate_bundle.get("target_dataset"),
        "execute_eval": execute_eval,
        "results": results,
        "passed_candidates": [
            item["candidate"] for item in results if item["status"] == "passed"
        ],
        "failed_candidates": [
            item["candidate"] for item in results if item["status"] != "passed"
        ],
    }
    return summary


def render_validation_summary(summary: dict[str, Any]) -> str:
    """Render a human-readable candidate validation summary."""

    lines = [
        "# Candidate Validation Summary",
        "",
        f"- model: `{summary['model']}`",
        f"- target_dataset: `{summary['target_dataset']}`",
        f"- execute_eval: `{summary['execute_eval']}`",
        "",
        "| Candidate | Status | Reason |",
        "| --- | --- | --- |",
    ]
    for item in summary.get("results", []):
        lines.append(
            f"| {item['candidate']} | {item['status']} | {item['reason']} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_validation_summary(
    summary: dict[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write candidate validation summary JSON and Markdown."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "validation_summary.json"
    md_path = output_dir / "validation_summary.md"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(
        render_validation_summary(summary),
        encoding="utf-8",
    )
    return json_path, md_path
