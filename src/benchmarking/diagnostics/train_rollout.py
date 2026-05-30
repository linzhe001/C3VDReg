"""Helpers for benchmark train rollout auditing and config generation."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import yaml

import src.benchmarking.runners.train_runner as train_runner
from src.benchmarking.config.schema import validate_config
from src.benchmarking.diagnostics.compatibility import (
    assert_baseline_repo_clean,
    assert_runtime_policy_compatible,
)
from src.benchmarking.registry.model_registry import ModelRegistry, ModelSpec

TRAIN_BRIDGE_BUILDERS: dict[str, Callable[[dict[str, Any], Path], dict[str, Any]]] = {
    "bufferx": train_runner._build_bufferx_bridge_config,
    "dcp": train_runner._build_dcp_bridge_config,
    "pointnetlk": train_runner._build_pointnetlk_bridge_config,
    "pointnetlk_revisited": train_runner._build_pointnetlk_revisited_bridge_config,
    "mamba3d": train_runner._build_mamba3d_bridge_config,
    "mamba3d_true": train_runner._build_mamba3d_bridge_config,
    "mamba3d_mamba2": train_runner._build_mamba3d_bridge_config,
    "mamba3d_mamba2_direct": train_runner._build_mamba3d_bridge_config,
    "mambanetlk": train_runner._build_mamba3d_bridge_config,
    "regtr": train_runner._build_regtr_bridge_config,
    "geotransformer": train_runner._build_geotransformer_bridge_config,
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


def _default_device_for_spec(spec: ModelSpec, requested_device: str) -> str:
    if spec.runtime_policy == "cpu_only":
        return "cpu"
    if requested_device.startswith("cuda"):
        return requested_device
    return "cuda:0"


def _requested_train_mode(runtime_config_path: Path) -> str:
    runtime_payload = _load_yaml(runtime_config_path)
    train_mode = str(runtime_payload.get("runtime", {}).get("train_mode", "smoke"))
    if train_mode not in {"smoke", "full"}:
        raise ValueError(
            "Train rollout audit only supports runtime.train_mode "
            f"'smoke' or 'full'. Received {train_mode!r}."
        )
    return train_mode


def _iter_registry_model_ids(registry: ModelRegistry) -> list[str]:
    return sorted(registry._specs.keys())


def _summarize_train_limits(
    model_id: str,
    bridge_config: dict[str, Any],
) -> dict[str, Any]:
    if model_id == "dcp":
        training = bridge_config["training"]
        summary = {
            "num_epochs": int(training["num_epochs"]),
            "num_points": int(bridge_config["dataset"]["num_points"]),
        }
        if "max_train_steps" in training:
            summary["max_train_steps"] = int(training["max_train_steps"])
        if "max_test_steps" in training:
            summary["max_test_steps"] = int(training["max_test_steps"])
        return summary
    if model_id == "pointnetlk":
        training = bridge_config["training"]
        summary = {
            "classifier_epochs": int(training["classifier"]["epochs"]),
            "pointnetlk_epochs": int(training["pointnetlk"]["epochs"]),
            "num_points": int(bridge_config["dataset"]["num_points"]),
        }
        if "max_train_steps" in training["classifier"]:
            summary["classifier_max_train_steps"] = int(
                training["classifier"]["max_train_steps"]
            )
        if "max_train_steps" in training["pointnetlk"]:
            summary["pointnetlk_max_train_steps"] = int(
                training["pointnetlk"]["max_train_steps"]
            )
        if "max_test_steps" in training["pointnetlk"]:
            summary["pointnetlk_max_test_steps"] = int(
                training["pointnetlk"]["max_test_steps"]
            )
        return summary
    if model_id == "pointnetlk_revisited":
        training = bridge_config["training"]
        summary = {
            "max_epochs": int(training["max_epochs"]),
            "num_points": int(bridge_config["dataset"]["num_points"]),
        }
        if "max_train_steps" in training:
            summary["max_train_steps"] = int(training["max_train_steps"])
        if "max_val_steps" in training:
            summary["max_val_steps"] = int(training["max_val_steps"])
        return summary
    if model_id in {
        "mamba3d",
        "mamba3d_true",
        "mamba3d_mamba2",
        "mamba3d_mamba2_direct",
        "mambanetlk",
    }:
        training = bridge_config["training"]
        summary = {
            "epochs": int(training["epochs"]),
            "num_points": int(bridge_config["dataset"]["num_points"]),
        }
        if "max_train_steps" in training:
            summary["max_train_steps"] = int(training["max_train_steps"])
        if "max_test_steps" in training:
            summary["max_test_steps"] = int(training["max_test_steps"])
        return summary
    if model_id == "regtr":
        return {
            "niter": int(bridge_config["train_options"]["niter"]),
            "train_batch_size": int(bridge_config["dataset"]["train_batch_size"]),
            "val_batch_size": int(bridge_config["dataset"]["val_batch_size"]),
            "num_points": int(bridge_config["dataset"]["num_points"]),
        }
    if model_id == "geotransformer":
        training = bridge_config["training"]
        summary = {
            "max_epochs": int(training["max_epochs"]),
            "batch_size": int(training["batch_size"]),
            "num_points": int(bridge_config["dataset"]["num_points"]),
        }
        if "max_train_steps" in training and training["max_train_steps"] is not None:
            summary["max_train_steps"] = int(training["max_train_steps"])
        if "max_val_steps" in training and training["max_val_steps"] is not None:
            summary["max_val_steps"] = int(training["max_val_steps"])
        return summary
    if model_id == "bufferx":
        training = bridge_config["bufferx"]["train"]
        return {
            "epoch": int(training["epoch"]),
            "max_iter": int(training["max_iter"]),
            "batch_size": int(training["batch_size"]),
            "num_points": int(bridge_config["dataset"]["num_points"]),
            "heuristic_mode": str(bridge_config["dataset"]["heuristic_mode"]),
            "stages": list(training["all_stage"]),
        }
    return {}


@contextmanager
def _patched_train_runner_repo_root(repo_root: Path):
    original = train_runner._repo_root
    train_runner._repo_root = lambda: repo_root
    try:
        yield
    finally:
        train_runner._repo_root = original


def build_train_config(
    repo_root: Path,
    runtime_config_path: Path,
    output_root: Path,
    spec: ModelSpec,
) -> dict[str, Any]:
    """Build a normalized train config for one model from durable templates."""

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

    return validate_config(merged).to_dict()


def build_train_rollout_markdown(report: dict[str, Any]) -> str:
    """Render a compact markdown summary for the train rollout audit."""

    lines = [
        "# Train Rollout Audit",
        "",
        f"- Runtime config: `{report['runtime_config']}`",
        f"- Stable train runner mode: `{report['stable_train_runner_mode']}`",
        f"- Requested train mode: `{report['requested_train_mode']}`",
        f"- Full-train runtime config exists: `{report['full_train_runtime_exists']}`",
        f"- CUDA available: `{report['environment']['cuda_available']}`",
        "",
        "| model | train supported | normalize | smoke ready | full ready | blockers |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in report["models"]:
        blockers = "; ".join(entry["blockers"]) if entry["blockers"] else "-"
        lines.append(
            f"| {entry['model_id']} | {entry['supports_train']} | "
            f"{entry['effective_normalize_mode']} | {entry['smoke_train_ready']} | "
            f"{entry['full_train_ready']} | "
            f"{blockers} |"
        )
    lines.append("")
    return "\n".join(lines)


def generate_train_rollout_audit(
    repo_root: Path,
    runtime_config_path: Path,
    output_root: Path,
    model_ids: Iterable[str] | None = None,
    write_configs: bool = True,
) -> dict[str, Any]:
    """Audit per-model train readiness and optionally emit smoke configs."""

    repo_root = repo_root.resolve()
    runtime_config_path = runtime_config_path.resolve()
    output_root = output_root.resolve()
    config_dir = output_root / "configs"
    bridge_dir = output_root / "bridge_configs"
    runtime_dir = repo_root / "configs" / "benchmark" / "runtime"
    full_train_runtime_path = runtime_dir / "full_train.yaml"
    requested_train_mode = _requested_train_mode(runtime_config_path)

    report: dict[str, Any] = {
        "repo_root": str(repo_root),
        "runtime_config": str(runtime_config_path),
        "stable_train_runner_mode": "configurable",
        "requested_train_mode": requested_train_mode,
        "full_train_runtime_exists": full_train_runtime_path.exists(),
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
        bridge_dir.mkdir(parents=True, exist_ok=True)

    for model_id in requested_ids:
        spec = registry.get(model_id)
        blockers: list[str] = []
        baseline_repo_clean = None
        generated_config_path: Path | None = None
        bridge_config_path: Path | None = None
        bridge_config: dict[str, Any] | None = None
        config_payload: dict[str, Any] | None = None

        try:
            config_payload = build_train_config(
                repo_root=repo_root,
                runtime_config_path=runtime_config_path,
                output_root=output_root,
                spec=spec,
            )
        except Exception as exc:  # pragma: no cover - surfaced in report
            blockers.append(f"config: {exc}")

        runtime_device = (
            str(config_payload["runtime"]["device"]) if config_payload else None
        )

        if config_payload is not None:
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
            runtime_ready = False

        supports_train = bool(spec.capabilities.supports_train)
        bridge_builder = TRAIN_BRIDGE_BUILDERS.get(model_id)
        if not supports_train:
            blockers.append("train: model is eval-only in the current registry")
        elif bridge_builder is None:
            blockers.append("train: bridge builder is not implemented")

        if supports_train and bridge_builder is not None and config_payload is not None:
            try:
                with _patched_train_runner_repo_root(repo_root):
                    bridge_config = bridge_builder(
                        config_payload,
                        output_root / "bridge_runs" / model_id,
                    )
            except Exception as exc:
                blockers.append(f"bridge: {exc}")
            else:
                if write_configs:
                    bridge_config_path = bridge_dir / f"{model_id}.yaml"
                    bridge_config_path.write_text(
                        yaml.safe_dump(bridge_config, sort_keys=False),
                        encoding="utf-8",
                    )

        requested_mode_ready = (
            supports_train
            and bridge_builder is not None
            and config_payload is not None
            and bridge_config is not None
            and runtime_ready
            and baseline_repo_clean in {True, None}
        )

        smoke_train_ready = requested_train_mode == "smoke" and requested_mode_ready
        full_train_ready = requested_train_mode == "full" and requested_mode_ready
        if supports_train:
            if (
                requested_train_mode == "full"
                and not report["full_train_runtime_exists"]
            ):
                blockers.append(
                    "full-train: configs/benchmark/runtime/full_train.yaml is missing"
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
                "supports_train": supports_train,
                "train_bridge": spec.train_bridge,
                "effective_normalize_mode": (
                    config_payload["model"]["overrides"].get("normalize_mode")
                    if config_payload is not None
                    else spec.default_eval_normalize_mode
                ),
                "runtime_device": runtime_device,
                "baseline_repo_clean": baseline_repo_clean,
                "generated_config_path": (
                    str(generated_config_path) if generated_config_path else None
                ),
                "bridge_config_path": (
                    str(bridge_config_path) if bridge_config_path else None
                ),
                "train_limits": (
                    _summarize_train_limits(model_id, bridge_config)
                    if bridge_config is not None
                    else None
                ),
                "runner_command": (
                    f"python scripts/runners/train_benchmark.py --config "
                    f"{generated_config_path}"
                    if generated_config_path
                    else None
                ),
                "smoke_train_ready": bool(smoke_train_ready),
                "full_train_ready": bool(full_train_ready),
                "ready_for_requested_mode": bool(requested_mode_ready),
                "blockers": blockers,
            }
        )

    return report


def write_train_rollout_report(
    report: dict[str, Any],
    output_root: Path,
) -> tuple[Path, Path]:
    """Write the train rollout audit to JSON and markdown under one root."""

    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    json_path = output_root / "train_rollout_audit.json"
    markdown_path = output_root / "train_rollout_audit.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        build_train_rollout_markdown(report),
        encoding="utf-8",
    )
    return json_path, markdown_path


def build_train_launch_plan(
    report: dict[str, Any],
    ready_only: bool = True,
) -> list[dict[str, Any]]:
    """Collect launchable train entries from a rollout audit report."""

    plan: list[dict[str, Any]] = []
    for entry in report["models"]:
        if not entry["supports_train"]:
            continue
        if ready_only and not entry["ready_for_requested_mode"]:
            continue
        if entry["generated_config_path"] is None or entry["runner_command"] is None:
            continue
        plan.append(
            {
                "model_id": entry["model_id"],
                "config_path": entry["generated_config_path"],
                "bridge_config_path": entry["bridge_config_path"],
                "runner_command": entry["runner_command"],
                "runtime_device": entry["runtime_device"],
                "train_limits": entry["train_limits"],
                "ready_for_requested_mode": entry["ready_for_requested_mode"],
                "blockers": list(entry["blockers"]),
            }
        )
    return plan
