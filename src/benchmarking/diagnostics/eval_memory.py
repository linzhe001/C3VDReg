"""Helpers for benchmarking eval-time CUDA memory across model/point-count sweeps."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import torch

from src.benchmarking.datasets.c3vd_manifest_dataset import C3VDManifestDataset
from src.benchmarking.diagnostics.compatibility import (
    assert_baseline_repo_clean,
    assert_model_preprocess_compatibility,
    assert_runtime_policy_compatible,
)
from src.benchmarking.diagnostics.eval_rollout import build_eval_config
from src.benchmarking.preprocess.pipeline import PreprocessPipeline
from src.benchmarking.preprocess.registry import PreprocessRegistry
from src.benchmarking.registry.model_registry import ModelRegistry
from src.benchmarking.runners.eval_runner import _instantiate_adapter


def _iter_registry_model_ids(registry: ModelRegistry) -> list[str]:
    return sorted(registry._specs.keys())


def _gpu_index(device: str) -> int:
    if ":" not in device:
        return 0
    return int(device.rsplit(":", 1)[-1])


def _prepare_cuda_device(device_index: int) -> None:
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device index {device_index} is invalid for "
            f"{torch.cuda.device_count()} visible device(s)."
        )
    torch.cuda.set_device(device_index)


def _memory_mb(value: int) -> float:
    return float(value) / (1024.0 * 1024.0)


def _build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Eval Memory Profile",
        "",
        f"- Runtime config: `{report['runtime_config']}`",
        f"- Samples per case: `{report['samples_per_case']}`",
        f"- CUDA available: `{report['environment']['cuda_available']}`",
        "",
        "| model | num_points | status | load alloc MB | predict alloc MB | blocker |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in report["entries"]:
        blocker = entry["blocker"] or "-"
        load_alloc = (
            f"{entry['load_peak_allocated_mb']:.2f}"
            if entry["load_peak_allocated_mb"] is not None
            else "-"
        )
        predict_alloc = (
            f"{entry['predict_peak_allocated_mb_max']:.2f}"
            if entry["predict_peak_allocated_mb_max"] is not None
            else "-"
        )
        lines.append(
            f"| {entry['model_id']} | {entry['num_points']} | {entry['status']} | "
            f"{load_alloc} | {predict_alloc} | {blocker} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_eval_memory_report(
    report: dict[str, Any],
    output_root: Path,
) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "eval_memory_profile.json"
    markdown_path = output_root / "eval_memory_profile.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_build_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def profile_eval_memory(
    repo_root: Path,
    runtime_config_path: Path,
    output_root: Path,
    num_points_list: Iterable[int],
    model_ids: Iterable[str] | None = None,
    samples_per_case: int = 1,
    manifest_path: str | None = None,
    subset_config_path: str | None = None,
    dataset_root: str | None = None,
    perturbation_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Profile eval-time CUDA memory for model/point-count combinations."""

    repo_root = repo_root.resolve()
    runtime_config_path = runtime_config_path.resolve()
    output_root = output_root.resolve()
    registry = ModelRegistry()
    preprocess_registry = PreprocessRegistry()
    perturbation_overrides = dict(perturbation_overrides or {})

    requested_ids = (
        list(model_ids)
        if model_ids is not None
        else _iter_registry_model_ids(registry)
    )
    point_counts = [int(value) for value in num_points_list]

    report: dict[str, Any] = {
        "repo_root": str(repo_root),
        "runtime_config": str(runtime_config_path),
        "samples_per_case": int(samples_per_case),
        "environment": {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
        },
        "entries": [],
    }

    for model_id in requested_ids:
        spec = registry.get(model_id)
        for num_points in point_counts:
            config = build_eval_config(
                repo_root=repo_root,
                runtime_config_path=runtime_config_path,
                output_root=output_root / "runs" / f"{model_id}_{num_points}",
                spec=spec,
            )
            config["preprocess"]["num_points_override"] = int(num_points)
            if manifest_path is not None:
                config["data"]["manifest_path"] = manifest_path
            if subset_config_path is not None:
                config["data"]["subset_config_path"] = subset_config_path
            if dataset_root is not None:
                config["data"]["dataset_root"] = dataset_root
            if perturbation_overrides:
                merged_perturbation = dict(config.get("perturbation", {}))
                merged_perturbation.update(perturbation_overrides)
                config["perturbation"] = merged_perturbation

            entry: dict[str, Any] = {
                "model_id": model_id,
                "num_points": int(num_points),
                "runtime_device": str(config["runtime"]["device"]),
                "status": "blocked",
                "blocker": None,
                "load_peak_allocated_mb": None,
                "load_peak_reserved_mb": None,
                "predict_peak_allocated_mb_max": None,
                "predict_peak_reserved_mb_max": None,
                "predict_time_ms_mean": None,
                "samples_profiled": 0,
            }

            try:
                spec = assert_model_preprocess_compatibility(
                    registry=registry,
                    model_id=model_id,
                    preprocess_profile_id=str(config["preprocess"]["profile"]),
                )
                assert_runtime_policy_compatible(spec, str(config["runtime"]["device"]))
                assert_baseline_repo_clean(repo_root=repo_root, spec=spec)
            except Exception as exc:
                entry["blocker"] = str(exc)
                report["entries"].append(entry)
                continue

            if not str(config["runtime"]["device"]).startswith("cuda"):
                entry["blocker"] = (
                    f"Model '{model_id}' is not running on CUDA; GPU memory profiling "
                    "is not applicable."
                )
                report["entries"].append(entry)
                continue

            device_index = _gpu_index(str(config["runtime"]["device"]))
            _prepare_cuda_device(device_index)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device_index)
            adapter_config = dict(config["model"].get("overrides", {}))
            adapter_config.setdefault("device", config["runtime"]["device"])
            adapter_config["checkpoint_path"] = config["model"].get("checkpoint_path")
            adapter = _instantiate_adapter(spec, adapter_config)
            torch.cuda.synchronize(device_index)

            entry["load_peak_allocated_mb"] = _memory_mb(
                torch.cuda.max_memory_allocated(device_index)
            )
            entry["load_peak_reserved_mb"] = _memory_mb(
                torch.cuda.max_memory_reserved(device_index)
            )

            dataset = C3VDManifestDataset(
                manifest_path=config["data"]["manifest_path"],
                split=str(config["benchmark"]["split"]),
                preprocess_pipeline=PreprocessPipeline(preprocess_registry),
                preprocess_profile_id=str(config["preprocess"]["profile"]),
                seed=int(config["preprocess"]["seed"]),
                preprocess_overrides={
                    "sampling_override": config["preprocess"].get("sampling_override"),
                    "num_points_override": config["preprocess"].get(
                        "num_points_override"
                    ),
                },
                perturbation_config=config.get("perturbation", {}),
                subset_config_path=config["data"].get("subset_config_path"),
                subset_name=config["benchmark"].get("subset_name"),
                dataset_root=config["data"].get("dataset_root"),
            )

            predict_peak_allocated: list[float] = []
            predict_peak_reserved: list[float] = []
            predict_time_ms: list[float] = []
            for index, sample in enumerate(dataset):
                if index >= samples_per_case:
                    break
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device_index)
                torch.cuda.synchronize(device_index)
                source_points = sample["source_points"]
                target_points = sample["target_points"]
                start = perf_counter()
                adapter.predict(source_points, target_points)
                torch.cuda.synchronize(device_index)
                predict_time_ms.append((perf_counter() - start) * 1000.0)
                predict_peak_allocated.append(
                    _memory_mb(torch.cuda.max_memory_allocated(device_index))
                )
                predict_peak_reserved.append(
                    _memory_mb(torch.cuda.max_memory_reserved(device_index))
                )

            entry["samples_profiled"] = len(predict_time_ms)
            if predict_time_ms:
                entry["status"] = "ok"
                entry["predict_peak_allocated_mb_max"] = max(predict_peak_allocated)
                entry["predict_peak_reserved_mb_max"] = max(predict_peak_reserved)
                entry["predict_time_ms_mean"] = sum(predict_time_ms) / len(
                    predict_time_ms
                )
            else:
                entry["status"] = "blocked"
                entry["blocker"] = (
                    "Dataset yielded zero samples for the requested split/subset."
                )
            report["entries"].append(entry)

    return report
