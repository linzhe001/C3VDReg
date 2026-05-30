"""Helpers for benchmark eval rollout auditing and config generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml

from src.benchmarking.config.schema import validate_config
from src.benchmarking.diagnostics.compatibility import (
    assert_baseline_repo_clean,
    assert_runtime_policy_compatible,
)
from src.benchmarking.registry.model_registry import ModelRegistry, ModelSpec

WEIGHTS_CONFIG_PATH = Path("src/unified_testing/WEIGHTS_CONFIG.yaml")
WEIGHTS_MODEL_ALIASES = {
    "dcp": "dcp",
    "mamba3d": "pointnetlk_c3vd",
    "pointnetlk": "pointnetlk",
    "pointnetlk_revisited": "pointnetlk_revisited",
}
CHECKPOINT_CANDIDATES = {
    "dcp": (
        "experiments/checkpoints/c3vd_dcp_v2/models/model_best.pth",
        "baselines/dcp/pretrained/dcp_v2.t7",
    ),
    "mamba3d": (
        "experiments/checkpoints/c3vd_pointnetlk_c3vd/"
        "mamba3d_pointlk_resume_0706_0435_model_best.pth",
        "experiments/checkpoints/c3vd_mamba3d_unified/mamba3d_pointlk_model_best.pth",
    ),
    "mamba3d_mamba2": (
        "experiments/checkpoints/c3vd_mamba3d_mamba2_smoke/"
        "mamba3d_pointlk_model_best.pth",
    ),
    "mamba3d_mamba2_direct": (
        "experiments/checkpoints/c3vd_mamba3d_mamba2_direct_smoke/"
        "mamba3d_pointlk_model_best.pth",
    ),
    "mambanetlk": (
        "experiments/checkpoints/c3vd_mambanetlk_smoke/"
        "mamba3d_pointlk_model_best.pth",
    ),
    "pointnetlk": (
        "experiments/checkpoints/c3vd_pointnetlk/c3vd_pointnetlk_model_model_best.pth",
    ),
    "pointnetlk_revisited": (
        "experiments/checkpoints/c3vd_pointnetlk_revisited/"
        "pointnetlk_c3vd_model_best.pth",
    ),
}
CHECKPOINT_GLOBS = {
    "bufferx": (
        "outputs/benchmark/hparam_transfer/bufferx*/**/train_bridge/Pose/best.pth",
        "outputs/benchmark/hparam_transfer/bufferx*/**/train_bridge/Pose/*.pth",
        "outputs/benchmark/**/bufferx*/**/train_bridge/Pose/best.pth",
        "outputs/benchmark/**/bufferx*/**/train_bridge/Pose/*.pth",
    ),
    "geotransformer": (
        "experiments/checkpoints/geotransformer/**/*.pth",
        "baselines/GeoTransformer/**/*.pth",
    ),
    "regtr": (
        "experiments/checkpoints/c3vd_regtr/**/*.pth",
        "experiments/checkpoints/c3vd_regtr/**/*.ckpt",
        "baselines/RegTR/**/*.pth",
        "baselines/RegTR/**/*.ckpt",
    ),
}


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {} if payload is None else dict(payload)


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = _deep_merge(base[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_repo_path(repo_root: Path, candidate: str | Path) -> Path:
    path = Path(candidate)
    if path.is_absolute():
        return path
    return repo_root / path


def _load_weights_catalog(repo_root: Path) -> dict[str, Any]:
    config_path = repo_root / WEIGHTS_CONFIG_PATH
    if not config_path.exists():
        return {}
    return _load_yaml(config_path)


def _default_device_for_spec(spec: ModelSpec, requested_device: str) -> str:
    if spec.runtime_policy == "cpu_only":
        return "cpu"
    if requested_device.startswith("cuda"):
        return requested_device
    return "cuda:0"


def _looks_like_checkpoint(candidate: Path) -> bool:
    suffix = candidate.suffix.lower()
    if suffix == ".ckpt":
        return True
    if suffix not in {".pth", ".pt", ".t7"}:
        return False

    name = candidate.name.lower()
    checkpoint_tokens = (
        "best",
        "checkpoint",
        "ckpt",
        "epoch",
        "last",
        "model",
        "snap",
        "weight",
    )
    return any(token in name for token in checkpoint_tokens)


def _iter_registry_model_ids(registry: ModelRegistry) -> list[str]:
    return sorted(registry._specs.keys())


def resolve_checkpoint_candidates(
    repo_root: Path,
    model_id: str,
    configured_checkpoint: str | None = None,
) -> list[Path]:
    """Return ordered checkpoint candidates for a benchmark model."""

    repo_root = repo_root.resolve()
    seen: set[Path] = set()
    ordered: list[Path] = []

    def _append(path_like: str | Path) -> None:
        candidate = _normalize_repo_path(repo_root, path_like).resolve()
        if candidate in seen:
            return
        seen.add(candidate)
        ordered.append(candidate)

    if configured_checkpoint:
        _append(configured_checkpoint)

    weights_catalog = _load_weights_catalog(repo_root)
    weights_key = WEIGHTS_MODEL_ALIASES.get(model_id)
    if weights_key is not None:
        dataset_section = weights_catalog.get(weights_key, {}).get("c3vd", {})
        model_path = dataset_section.get("model_path")
        if isinstance(model_path, str) and model_path.strip():
            _append(model_path)

    for candidate in CHECKPOINT_CANDIDATES.get(model_id, ()):
        _append(candidate)

    for pattern in CHECKPOINT_GLOBS.get(model_id, ()):
        for candidate in sorted(repo_root.glob(pattern)):
            if candidate.is_file() and _looks_like_checkpoint(candidate):
                _append(candidate)

    return ordered


def resolve_checkpoint_path(
    repo_root: Path,
    model_id: str,
    configured_checkpoint: str | None = None,
) -> Path | None:
    """Return the first existing checkpoint path for a model."""

    for candidate in resolve_checkpoint_candidates(
        repo_root=repo_root,
        model_id=model_id,
        configured_checkpoint=configured_checkpoint,
    ):
        if candidate.exists():
            return candidate
    return None


def build_eval_config(
    repo_root: Path,
    runtime_config_path: Path,
    output_root: Path,
    spec: ModelSpec,
) -> dict[str, Any]:
    """Build a normalized eval config for one model from durable templates."""

    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    runtime_payload = _load_yaml(runtime_config_path)
    model_payload = _load_yaml(
        repo_root / "configs" / "benchmark" / "models" / f"{spec.model_id}.yaml"
    )
    merged = _deep_merge(runtime_payload, model_payload)

    merged.setdefault("runtime", {})
    requested_device = str(merged["runtime"].get("device", "cuda:0"))
    merged["runtime"]["device"] = _default_device_for_spec(spec, requested_device)
    merged["runtime"]["output_dir"] = str(output_root / "runs" / spec.model_id)

    configured_checkpoint = merged.get("model", {}).get("checkpoint_path")
    resolved_checkpoint = resolve_checkpoint_path(
        repo_root=repo_root,
        model_id=spec.model_id,
        configured_checkpoint=configured_checkpoint,
    )
    if resolved_checkpoint is not None:
        merged.setdefault("model", {})
        merged["model"]["checkpoint_path"] = str(resolved_checkpoint)

    return validate_config(merged).to_dict()


def build_eval_rollout_markdown(report: dict[str, Any]) -> str:
    """Render a compact markdown summary for the rollout audit."""

    lines = [
        "# Eval Rollout Audit",
        "",
        f"- Runtime config: `{report['runtime_config']}`",
        f"- CUDA available: `{report['environment']['cuda_available']}`",
        f"- CUDA device count: `{report['environment']['cuda_device_count']}`",
        "",
        "| model | ready | checkpoint | device | normalize | blockers |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in report["models"]:
        blockers = "; ".join(entry["blockers"]) if entry["blockers"] else "-"
        checkpoint = entry["checkpoint_path"] or entry["checkpoint_status"]
        lines.append(
            f"| {entry['model_id']} | {entry['ready_to_run']} | "
            f"{checkpoint} | {entry['runtime_device']} | "
            f"{entry['effective_normalize_mode']} | {blockers} |"
        )
    lines.append("")
    return "\n".join(lines)


def generate_eval_rollout_audit(
    repo_root: Path,
    runtime_config_path: Path,
    output_root: Path,
    model_ids: Iterable[str] | None = None,
    write_configs: bool = True,
) -> dict[str, Any]:
    """Audit per-model eval readiness and optionally emit runnable configs."""

    repo_root = repo_root.resolve()
    runtime_config_path = runtime_config_path.resolve()
    output_root = output_root.resolve()
    config_dir = output_root / "configs"
    report: dict[str, Any] = {
        "repo_root": str(repo_root),
        "runtime_config": str(runtime_config_path),
        "environment": {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
        },
        "models": [],
    }

    registry = ModelRegistry()
    requested_ids = (
        list(model_ids)
        if model_ids is not None
        else _iter_registry_model_ids(registry)
    )

    if write_configs:
        config_dir.mkdir(parents=True, exist_ok=True)

    for model_id in requested_ids:
        spec = registry.get(model_id)
        blockers: list[str] = []
        baseline_repo_clean = None
        checkpoint_status = "missing"
        checkpoint_path: Path | None = None
        generated_config_path: Path | None = None
        config_payload: dict[str, Any] | None = None

        try:
            config_payload = build_eval_config(
                repo_root=repo_root,
                runtime_config_path=runtime_config_path,
                output_root=output_root,
                spec=spec,
            )
        except Exception as exc:  # pragma: no cover - exercised via audit status
            blockers.append(f"config: {exc}")

        if config_payload is not None:
            runtime_device = str(config_payload["runtime"]["device"])
            checkpoint_value = config_payload["model"].get("checkpoint_path")
            checkpoint_candidates = [
                str(path)
                for path in resolve_checkpoint_candidates(
                    repo_root=repo_root,
                    model_id=model_id,
                    configured_checkpoint=checkpoint_value,
                )
            ]
            checkpoint_path = (
                Path(checkpoint_value).resolve() if checkpoint_value else None
            )
            if spec.model_id == "icp":
                checkpoint_status = "not_required"
            elif checkpoint_path is not None and checkpoint_path.exists():
                checkpoint_status = "resolved"
            else:
                checkpoint_status = "missing"
                blockers.append("checkpoint: no C3VD checkpoint resolved")

            try:
                assert_runtime_policy_compatible(spec, runtime_device)
                runtime_ready = True
            except Exception as exc:
                runtime_ready = False
                blockers.append(f"runtime: {exc}")

            if spec.source_kind == "vendor_readonly":
                try:
                    assert_baseline_repo_clean(repo_root=repo_root, spec=spec)
                    baseline_repo_clean = True
                except Exception as exc:
                    baseline_repo_clean = False
                    blockers.append(f"baseline: {exc}")
            else:
                baseline_repo_clean = None

            if write_configs:
                generated_config_path = config_dir / f"{model_id}.yaml"
                generated_config_path.write_text(
                    yaml.safe_dump(config_payload, sort_keys=False),
                    encoding="utf-8",
                )
        else:
            runtime_device = None
            checkpoint_candidates = [
                str(path)
                for path in resolve_checkpoint_candidates(
                    repo_root=repo_root,
                    model_id=model_id,
                )
            ]
            runtime_ready = False

        ready_to_run = (
            config_payload is not None
            and checkpoint_status in {"resolved", "not_required"}
            and runtime_ready
            and baseline_repo_clean in {True, None}
        )

        report["models"].append(
            {
                "model_id": model_id,
                "source_kind": spec.source_kind,
                "runtime_policy": spec.runtime_policy,
                "baseline_normalization_policy_id": (
                    spec.baseline_normalization_policy_id
                ),
                "baseline_normalization_reference": (
                    spec.baseline_normalization_reference
                ),
                "effective_normalize_mode": (
                    config_payload["model"]["overrides"].get("normalize_mode")
                    if config_payload is not None
                    else spec.default_eval_normalize_mode
                ),
                "runtime_device": runtime_device,
                "checkpoint_status": checkpoint_status,
                "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
                "checkpoint_candidates": checkpoint_candidates,
                "baseline_repo_clean": baseline_repo_clean,
                "generated_config_path": (
                    str(generated_config_path) if generated_config_path else None
                ),
                "runner_command": (
                    f"python scripts/runners/eval_benchmark.py --config "
                    f"{generated_config_path}"
                    if generated_config_path
                    else None
                ),
                "ready_to_run": bool(ready_to_run),
                "blockers": blockers,
            }
        )

    return report


def write_eval_rollout_report(
    report: dict[str, Any],
    output_root: Path,
) -> tuple[Path, Path]:
    """Write the rollout audit to JSON and markdown under one output root."""

    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    json_path = output_root / "rollout_audit.json"
    markdown_path = output_root / "rollout_audit.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(build_eval_rollout_markdown(report), encoding="utf-8")
    return json_path, markdown_path
