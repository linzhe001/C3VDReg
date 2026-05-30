"""Transformer-like token stress profiling for train-time memory."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml

from src.benchmarking.diagnostics.compatibility import (
    assert_baseline_repo_clean,
    assert_runtime_policy_compatible,
)
from src.benchmarking.diagnostics.train_memory import (
    _build_case_config_and_bridge,
    _effective_micro_batch_size,
    _gpu_index,
    _run_profile_bridge_commands,
)
from src.benchmarking.registry.model_registry import ModelRegistry

DEFAULT_TOKEN_STRESS_MODELS = ("dcp", "geotransformer")
DEFAULT_TOKEN_LENGTHS = (128, 256, 512, 768, 1024, 1536)
DEFAULT_GEOTRANSFORMER_SPATIAL_SCALES = (
    15.0,
    20.0,
    24.0,
    28.0,
    32.0,
    36.0,
    40.0,
    50.0,
    60.0,
    80.0,
    100.0,
)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _probe_geotransformer_tokens(
    bridge_config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    from src.benchmarking.bridges.train_geotransformer_c3vd import (
        _make_loader,
        _make_vendor_cfg,
    )

    cfg = _make_vendor_cfg(bridge_config, output_dir)
    loader = _make_loader(
        bridge_config,
        split="train",
        cfg=cfg,
        neighbor_limits=[
            int(value)
            for value in bridge_config["training"].get(
                "neighbor_limits",
                [38, 36, 36, 38],
            )
        ],
        max_pairs_key="max_train_pairs",
    )
    batch = next(iter(loader))
    level_lengths = [
        [int(value) for value in level.tolist()] for level in batch["lengths"]
    ]
    coarse_lengths = level_lengths[-1]
    return {
        "level_lengths": level_lengths,
        "coarse_ref_tokens": int(coarse_lengths[0]),
        "coarse_src_tokens": int(coarse_lengths[1]),
        "actual_token_length": int(sum(coarse_lengths)),
    }


def _choose_geotransformer_scale(
    bridge_config: dict[str, Any],
    output_dir: Path,
    target_token_length: int,
    spatial_scales: Iterable[float],
) -> tuple[float, dict[str, Any]]:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for scale in spatial_scales:
        candidate_config = yaml.safe_load(yaml.safe_dump(bridge_config))
        candidate_config["model"]["spatial_scale"] = float(scale)
        token_info = _probe_geotransformer_tokens(candidate_config, output_dir)
        candidates.append((float(scale), token_info))
    return min(
        candidates,
        key=lambda item: abs(
            int(item[1]["actual_token_length"]) - int(target_token_length)
        ),
    )


def _base_entry(
    model_id: str,
    target_token_length: int,
    active_unit_name: str,
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "target_token_length": int(target_token_length),
        "active_unit_name": active_unit_name,
        "actual_token_length": None,
        "raw_points": None,
        "status": "blocked",
        "safe_under_limit": None,
        "peak_gpu_memory_mb": None,
        "torch_peak_allocated_mb": None,
        "duration_ms": None,
        "effective_batch_size": None,
        "spatial_scale": None,
        "blocker": None,
    }


def _profile_dcp_token_case(
    repo_root: Path,
    runtime_config_path: Path,
    output_root: Path,
    target_token_length: int,
    runtime_device: str,
    memory_limit_mb: float,
    micro_batch_size: int,
    steps_per_case: int,
    timeout_seconds: float | None,
) -> dict[str, Any]:
    entry = _base_entry("dcp", target_token_length, "attention_points")
    config, bridge_config, bridge_config_path = _build_case_config_and_bridge(
        repo_root=repo_root,
        runtime_config_path=runtime_config_path,
        output_root=output_root / f"dcp_token_{target_token_length}",
        model_id="dcp",
        num_points=int(target_token_length),
        device=runtime_device,
        micro_batch_size=micro_batch_size,
        steps_per_case=steps_per_case,
        manifest_path=None,
        subset_config_path=None,
        dataset_root=None,
    )
    result = _run_profile_bridge_commands(
        repo_root=repo_root,
        model_id="dcp",
        config=config,
        bridge_config=bridge_config,
        bridge_config_path=bridge_config_path,
        device_index=_gpu_index(runtime_device),
        poll_interval_sec=0.1,
        timeout_seconds=timeout_seconds,
    )
    peak_mb = result["peak_used_memory_mb"]
    entry.update(
        {
            "actual_token_length": int(target_token_length),
            "raw_points": int(target_token_length),
            "status": result["status"],
            "safe_under_limit": (
                bool(peak_mb <= memory_limit_mb)
                if result["status"] == "ok" and peak_mb is not None
                else None
            ),
            "peak_gpu_memory_mb": peak_mb,
            "torch_peak_allocated_mb": result["torch_peak_allocated_mb"],
            "duration_ms": result["duration_ms"],
            "effective_batch_size": _effective_micro_batch_size(
                "dcp",
                bridge_config,
            ),
            "blocker": result["blocker"],
        }
    )
    return entry


def _profile_geotransformer_token_case(
    repo_root: Path,
    runtime_config_path: Path,
    output_root: Path,
    target_token_length: int,
    raw_points: int,
    runtime_device: str,
    memory_limit_mb: float,
    micro_batch_size: int,
    steps_per_case: int,
    timeout_seconds: float | None,
    spatial_scales: Iterable[float],
) -> dict[str, Any]:
    entry = _base_entry("geotransformer", target_token_length, "coarse_superpoints")
    config, bridge_config, bridge_config_path = _build_case_config_and_bridge(
        repo_root=repo_root,
        runtime_config_path=runtime_config_path,
        output_root=output_root / f"geotransformer_token_{target_token_length}",
        model_id="geotransformer",
        num_points=int(raw_points),
        device=runtime_device,
        micro_batch_size=micro_batch_size,
        steps_per_case=steps_per_case,
        manifest_path=None,
        subset_config_path=None,
        dataset_root=None,
    )
    scale, token_info = _choose_geotransformer_scale(
        bridge_config,
        output_root / f"geotransformer_token_{target_token_length}" / "probe",
        target_token_length,
        spatial_scales,
    )
    bridge_config["model"]["spatial_scale"] = float(scale)
    _write_yaml(bridge_config_path, bridge_config)
    result = _run_profile_bridge_commands(
        repo_root=repo_root,
        model_id="geotransformer",
        config=config,
        bridge_config=bridge_config,
        bridge_config_path=bridge_config_path,
        device_index=_gpu_index(runtime_device),
        poll_interval_sec=0.1,
        timeout_seconds=timeout_seconds,
    )
    peak_mb = result["peak_used_memory_mb"]
    entry.update(
        {
            "actual_token_length": token_info["actual_token_length"],
            "raw_points": int(raw_points),
            "status": result["status"],
            "safe_under_limit": (
                bool(peak_mb <= memory_limit_mb)
                if result["status"] == "ok" and peak_mb is not None
                else None
            ),
            "peak_gpu_memory_mb": peak_mb,
            "torch_peak_allocated_mb": result["torch_peak_allocated_mb"],
            "duration_ms": result["duration_ms"],
            "effective_batch_size": _effective_micro_batch_size(
                "geotransformer",
                bridge_config,
            ),
            "spatial_scale": float(scale),
            "blocker": result["blocker"],
            **token_info,
        }
    )
    return entry


def profile_token_stress(
    repo_root: Path,
    runtime_config_path: Path,
    output_root: Path,
    token_lengths: Iterable[int] = DEFAULT_TOKEN_LENGTHS,
    model_ids: Iterable[str] = DEFAULT_TOKEN_STRESS_MODELS,
    geotransformer_raw_points: int = 65536,
    geotransformer_spatial_scales: Iterable[float] = (
        DEFAULT_GEOTRANSFORMER_SPATIAL_SCALES
    ),
    device: str | None = None,
    memory_limit_gb: float = 32.0,
    safety_ratio: float = 0.90,
    micro_batch_size: int = 1,
    steps_per_case: int = 1,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    runtime_config_path = runtime_config_path.resolve()
    output_root = output_root.resolve()
    runtime_payload = yaml.safe_load(runtime_config_path.read_text(encoding="utf-8"))
    runtime_device = str(
        device or runtime_payload.get("runtime", {}).get("device", "cuda:0")
    )
    usable_limit_mb = float(memory_limit_gb) * 1024.0 * float(safety_ratio)
    registry = ModelRegistry()
    entries: list[dict[str, Any]] = []

    report = {
        "repo_root": str(repo_root),
        "runtime_config": str(runtime_config_path),
        "token_lengths": [int(value) for value in token_lengths],
        "memory_limit_gb": float(memory_limit_gb),
        "safety_ratio": float(safety_ratio),
        "usable_limit_mb": usable_limit_mb,
        "geotransformer_raw_points": int(geotransformer_raw_points),
        "entries": entries,
    }

    for model_id in model_ids:
        spec = registry.get(model_id)
        for target in report["token_lengths"]:
            entry = _base_entry(model_id, target, "unsupported")
            if model_id not in {"dcp", "geotransformer"}:
                entry["blocker"] = (
                    "Token stress probe is implemented only for dcp and "
                    "geotransformer in the current benchmark harness."
                )
                entries.append(entry)
                continue
            if not torch.cuda.is_available():
                entry["blocker"] = "CUDA unavailable: torch.cuda.is_available() is False."
                entries.append(entry)
                continue
            try:
                assert_runtime_policy_compatible(spec, runtime_device)
                assert_baseline_repo_clean(repo_root=repo_root, spec=spec)
                if model_id == "dcp":
                    entry = _profile_dcp_token_case(
                        repo_root=repo_root,
                        runtime_config_path=runtime_config_path,
                        output_root=output_root,
                        target_token_length=target,
                        runtime_device=runtime_device,
                        memory_limit_mb=usable_limit_mb,
                        micro_batch_size=micro_batch_size,
                        steps_per_case=steps_per_case,
                        timeout_seconds=timeout_seconds,
                    )
                else:
                    entry = _profile_geotransformer_token_case(
                        repo_root=repo_root,
                        runtime_config_path=runtime_config_path,
                        output_root=output_root,
                        target_token_length=target,
                        raw_points=geotransformer_raw_points,
                        runtime_device=runtime_device,
                        memory_limit_mb=usable_limit_mb,
                        micro_batch_size=micro_batch_size,
                        steps_per_case=steps_per_case,
                        timeout_seconds=timeout_seconds,
                        spatial_scales=geotransformer_spatial_scales,
                    )
            except Exception as exc:
                entry["status"] = "failed"
                entry["blocker"] = str(exc)
            entries.append(entry)
    return report


def _build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Transformer-Like Token Stress Test",
        "",
        f"- Runtime config: `{report['runtime_config']}`",
        f"- Memory budget: `{report['memory_limit_gb']}GB * "
        f"{report['safety_ratio']} = {report['usable_limit_mb']:.2f} MB`",
        "- DCP token length is input attention points.",
        "- GeoTransformer token length is coarse ref+src superpoints; "
        "spatial_scale is searched to approximate each target.",
        "",
        "| model | target tokens | actual tokens | active unit | raw points | "
        "status | safe | peak GPU MB | time ms | spatial scale | blocker |",
        "| --- | ---: | ---: | --- | ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for entry in report["entries"]:
        peak = entry["peak_gpu_memory_mb"]
        duration = entry["duration_ms"]
        scale = entry["spatial_scale"]
        peak_text = f"{float(peak):.2f}" if peak is not None else "-"
        duration_text = f"{float(duration):.1f}" if duration is not None else "-"
        scale_text = f"{float(scale):.3g}" if scale is not None else "-"
        lines.append(
            f"| {entry['model_id']} | {entry['target_token_length']} | "
            f"{entry['actual_token_length'] or '-'} | {entry['active_unit_name']} | "
            f"{entry['raw_points'] or '-'} | {entry['status']} | "
            f"{entry['safe_under_limit']} | {peak_text} | {duration_text} | "
            f"{scale_text} | {entry['blocker'] or '-'} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_token_stress_report(
    report: dict[str, Any],
    output_root: Path,
) -> tuple[Path, Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "transformer_token_stress.json"
    csv_path = output_root / "transformer_token_stress.csv"
    markdown_path = output_root / "transformer_token_stress.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_csv(csv_path, report["entries"])
    markdown_path.write_text(_build_markdown(report), encoding="utf-8")
    return json_path, csv_path, markdown_path
