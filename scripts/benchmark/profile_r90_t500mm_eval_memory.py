#!/usr/bin/env python3
"""Profile retained R90/T500mm eval GPU memory with 8192 points and batch size 1."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmarking.datasets.c3vd_manifest_dataset import C3VDManifestDataset  # noqa: E402
from src.benchmarking.diagnostics.compatibility import (  # noqa: E402
    assert_baseline_repo_clean,
    assert_model_preprocess_compatibility,
    assert_runtime_policy_compatible,
)
from src.benchmarking.preprocess.pipeline import PreprocessPipeline  # noqa: E402
from src.benchmarking.preprocess.registry import PreprocessRegistry  # noqa: E402
from src.benchmarking.registry.model_registry import ModelRegistry  # noqa: E402
from src.benchmarking.runners.eval_runner import _instantiate_adapter  # noqa: E402


DEFAULT_EVAL_CONFIGS: tuple[tuple[str, str], ...] = (
    (
        "geotransformer",
        "outputs/benchmark/r90_t500mm_protocol/geotransformer/eval_test/"
        "normalized_eval_config.json",
    ),
    (
        "regtr_fixed_latency_rerun",
        "outputs/benchmark/r90_t500mm_from_scratch_fixed_regtr_dcp/regtr/"
        "eval_test_latency_rerun2/normalized_eval_config.json",
    ),
    (
        "mamba3d_mamba2_direct_sort_xyz",
        "outputs/benchmark/mamba2_followup_point_order_pair_initializer/"
        "direct_sort_xyz_e5/eval_test_maxiter10/normalized_eval_config.json",
    ),
    (
        "pointnetlk_revisited_protocol",
        "outputs/benchmark/r90_t500mm_protocol/pointnetlk_revisited/eval_test/"
        "normalized_eval_config.json",
    ),
    (
        "pointnetlk",
        "outputs/benchmark/r90_t500mm_protocol/pointnetlk/eval_test/"
        "normalized_eval_config.json",
    ),
    (
        "dcp_fixed",
        "outputs/benchmark/r90_t500mm_from_scratch_fixed_regtr_dcp/dcp/"
        "eval_test/normalized_eval_config.json",
    ),
)


def _memory_mb(value: int) -> float:
    return float(value) / (1024.0 * 1024.0)


def _gpu_index(device: str) -> int:
    if ":" not in device:
        return 0
    return int(device.rsplit(":", 1)[-1])


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def _load_eval_config(
    path: Path,
    output_root: Path,
    run_id: str,
    num_points: int,
) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    config.setdefault("preprocess", {})
    config["preprocess"]["num_points_override"] = int(num_points)
    config.setdefault("runtime", {})
    config["runtime"]["batch_size"] = 1
    config["runtime"]["num_workers"] = 0
    config["runtime"]["output_dir"] = str(output_root / "runs" / run_id)
    return config


def _parse_eval_config_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        run_id, path = value.split("=", 1)
        return run_id.strip(), _resolve_repo_path(path.strip())
    path = _resolve_repo_path(value)
    return path.parent.name, path


def _default_eval_configs() -> list[tuple[str, Path]]:
    return [(run_id, _resolve_repo_path(path)) for run_id, path in DEFAULT_EVAL_CONFIGS]


def _select_configs(
    eval_config_args: list[str] | None,
    selected_models: set[str] | None,
) -> list[tuple[str, Path]]:
    configs = (
        [_parse_eval_config_arg(value) for value in eval_config_args]
        if eval_config_args
        else _default_eval_configs()
    )
    if selected_models is None:
        return configs

    selected: list[tuple[str, Path]] = []
    for run_id, path in configs:
        model_id = None
        if path.exists():
            try:
                model_id = json.loads(path.read_text(encoding="utf-8"))["model"]["id"]
            except Exception:
                model_id = None
        if run_id in selected_models or model_id in selected_models:
            selected.append((run_id, path))
    return selected


def _build_dataset(config: dict[str, Any], preprocess_registry: PreprocessRegistry) -> Any:
    return C3VDManifestDataset(
        manifest_path=config["data"]["manifest_path"],
        split=str(config["benchmark"]["split"]),
        preprocess_pipeline=PreprocessPipeline(preprocess_registry),
        preprocess_profile_id=str(config["preprocess"]["profile"]),
        seed=int(config["preprocess"]["seed"]),
        preprocess_overrides={
            "sampling_override": config["preprocess"].get("sampling_override"),
            "num_points_override": config["preprocess"].get("num_points_override"),
        },
        perturbation_config=config.get("perturbation", {}),
        subset_config_path=config["data"].get("subset_config_path"),
        subset_name=config["benchmark"].get("subset_name"),
        dataset_root=config["data"].get("dataset_root"),
    )


def _cleanup_cuda(device_index: int) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device_index)


def _profile_one(
    run_id: str,
    config_path: Path,
    output_root: Path,
    num_points: int,
    samples: int,
    warmup_samples: int,
    skip_clean_check: bool,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "run_id": run_id,
        "config_path": str(config_path),
        "model_id": None,
        "checkpoint_path": None,
        "num_points": int(num_points),
        "batch_size": 1,
        "warmup_samples": int(warmup_samples),
        "samples_requested": int(samples),
        "samples_profiled": 0,
        "runtime_device": None,
        "status": "blocked",
        "blocker": None,
        "load_peak_allocated_mb": None,
        "load_peak_reserved_mb": None,
        "predict_peak_allocated_mb_max": None,
        "predict_peak_allocated_mb_mean": None,
        "predict_peak_reserved_mb_max": None,
        "predict_peak_reserved_mb_mean": None,
        "predict_time_ms_mean": None,
        "predict_time_ms_p90": None,
    }
    if not config_path.exists():
        entry["blocker"] = f"Missing eval config: {config_path}"
        return entry
    if not torch.cuda.is_available():
        entry["blocker"] = "CUDA is not available."
        return entry

    config = _load_eval_config(config_path, output_root, run_id, num_points)
    model_id = str(config["model"]["id"])
    checkpoint_path = config["model"].get("checkpoint_path")
    device = str(config["runtime"]["device"])
    entry.update(
        {
            "model_id": model_id,
            "checkpoint_path": checkpoint_path,
            "runtime_device": device,
        }
    )
    if not device.startswith("cuda"):
        entry["blocker"] = f"Runtime device is not CUDA: {device}"
        return entry

    registry = ModelRegistry()
    preprocess_registry = PreprocessRegistry()
    try:
        spec = assert_model_preprocess_compatibility(
            registry=registry,
            model_id=model_id,
            preprocess_profile_id=str(config["preprocess"]["profile"]),
        )
        assert_runtime_policy_compatible(spec, device)
        if not skip_clean_check:
            assert_baseline_repo_clean(repo_root=REPO_ROOT, spec=spec)
    except Exception as exc:
        entry["blocker"] = str(exc)
        return entry

    device_index = _gpu_index(device)
    if device_index >= torch.cuda.device_count():
        entry["blocker"] = (
            f"CUDA device index {device_index} is invalid for "
            f"{torch.cuda.device_count()} visible device(s)."
        )
        return entry

    adapter = None
    dataset = None
    try:
        torch.cuda.set_device(device_index)
        _cleanup_cuda(device_index)
        torch.cuda.reset_peak_memory_stats(device_index)

        adapter_config = dict(config["model"].get("overrides", {}))
        adapter_config.setdefault("device", device)
        adapter_config["checkpoint_path"] = checkpoint_path
        adapter = _instantiate_adapter(spec, adapter_config)
        torch.cuda.synchronize(device_index)
        entry["load_peak_allocated_mb"] = _memory_mb(
            torch.cuda.max_memory_allocated(device_index)
        )
        entry["load_peak_reserved_mb"] = _memory_mb(
            torch.cuda.max_memory_reserved(device_index)
        )

        dataset = _build_dataset(config, preprocess_registry)
        predict_allocated: list[float] = []
        predict_reserved: list[float] = []
        predict_time_ms: list[float] = []
        total_needed = max(0, warmup_samples) + max(0, samples)

        for index, sample in enumerate(dataset):
            if index >= total_needed:
                break

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device_index)
            torch.cuda.synchronize(device_index)

            source_points = np.asarray(sample["source_points"], dtype=np.float64)
            target_points = np.asarray(sample["target_points"], dtype=np.float64)
            start = perf_counter()
            prediction = adapter.predict(source_points, target_points)
            torch.cuda.synchronize(device_index)
            elapsed_ms = (perf_counter() - start) * 1000.0
            del prediction

            if index < warmup_samples:
                continue
            predict_time_ms.append(elapsed_ms)
            predict_allocated.append(
                _memory_mb(torch.cuda.max_memory_allocated(device_index))
            )
            predict_reserved.append(
                _memory_mb(torch.cuda.max_memory_reserved(device_index))
            )

        entry["samples_profiled"] = len(predict_time_ms)
        if not predict_time_ms:
            entry["blocker"] = "Dataset yielded no measured samples."
            return entry

        entry["status"] = "ok"
        entry["predict_peak_allocated_mb_max"] = max(predict_allocated)
        entry["predict_peak_allocated_mb_mean"] = mean(predict_allocated)
        entry["predict_peak_reserved_mb_max"] = max(predict_reserved)
        entry["predict_peak_reserved_mb_mean"] = mean(predict_reserved)
        entry["predict_time_ms_mean"] = mean(predict_time_ms)
        entry["predict_time_ms_p90"] = _percentile(predict_time_ms, 90.0)
        return entry
    except Exception as exc:
        entry["blocker"] = repr(exc)
        return entry
    finally:
        del dataset
        del adapter
        _cleanup_cuda(device_index)


def _format_float(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.1f}"


def _write_reports(report: dict[str, Any], output_root: Path) -> tuple[Path, Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "r90_t500mm_eval_memory_bsz1_8192.json"
    csv_path = output_root / "r90_t500mm_eval_memory_bsz1_8192.csv"
    markdown_path = output_root / "r90_t500mm_eval_memory_bsz1_8192.md"

    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    fields = [
        "run_id",
        "model_id",
        "status",
        "num_points",
        "batch_size",
        "warmup_samples",
        "samples_profiled",
        "load_peak_reserved_mb",
        "load_peak_allocated_mb",
        "predict_peak_reserved_mb_max",
        "predict_peak_reserved_mb_mean",
        "predict_peak_allocated_mb_max",
        "predict_peak_allocated_mb_mean",
        "predict_time_ms_mean",
        "predict_time_ms_p90",
        "checkpoint_path",
        "config_path",
        "blocker",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in report["entries"]:
            writer.writerow({field: entry.get(field) for field in fields})

    lines = [
        "# R90/T500mm Eval Memory Profile",
        "",
        f"- Num points: `{report['num_points']}`",
        "- Batch size: `1` pair per `adapter.predict` call",
        f"- Warmup samples per model: `{report['warmup_samples']}`",
        f"- Measured samples per model: `{report['samples_per_model']}`",
        "- Memory is eval-time CUDA peak after model load; warmup samples are excluded.",
        "",
        "| Run | Model | Status | Load Reserved MB | Load Alloc MB | "
        "Predict Reserved Max MB | Predict Alloc Max MB | Predict ms Mean | "
        "Samples | Blocker |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for entry in report["entries"]:
        lines.append(
            f"| `{entry['run_id']}` | `{entry.get('model_id') or '-'}` | "
            f"{entry['status']} | {_format_float(entry['load_peak_reserved_mb'])} | "
            f"{_format_float(entry['load_peak_allocated_mb'])} | "
            f"{_format_float(entry['predict_peak_reserved_mb_max'])} | "
            f"{_format_float(entry['predict_peak_allocated_mb_max'])} | "
            f"{_format_float(entry['predict_time_ms_mean'])} | "
            f"{entry['samples_profiled']} | {entry['blocker'] or '-'} |"
        )
    lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, csv_path, markdown_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-config",
        action="append",
        default=None,
        help=(
            "Eval config to profile. Use run_id=path or path. "
            "Defaults to the retained R90/T500mm summary-table configs."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional run_id or model_id subset.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=8192,
        help="Point count override for each eval config.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Measured samples per model.",
    )
    parser.add_argument(
        "--warmup-samples",
        type=int,
        default=1,
        help="Warmup samples per model, excluded from the reported peak/time.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/benchmark/r90_t500mm_eval_memory_bsz1_8192",
        help="Output directory for JSON/CSV/Markdown reports.",
    )
    parser.add_argument(
        "--skip-clean-check",
        action="store_true",
        help="Skip vendor baseline git-clean checks for local diagnostic runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = _resolve_repo_path(args.output_root)
    selected_models = set(args.models) if args.models else None
    configs = _select_configs(args.eval_config, selected_models)
    report: dict[str, Any] = {
        "repo_root": str(REPO_ROOT),
        "output_root": str(output_root),
        "num_points": int(args.num_points),
        "batch_size": 1,
        "samples_per_model": int(args.samples),
        "warmup_samples": int(args.warmup_samples),
        "entries": [],
    }

    for run_id, config_path in configs:
        print(f"[profile] {run_id}: {config_path}", flush=True)
        report["entries"].append(
            _profile_one(
                run_id=run_id,
                config_path=config_path,
                output_root=output_root,
                num_points=args.num_points,
                samples=args.samples,
                warmup_samples=args.warmup_samples,
                skip_clean_check=args.skip_clean_check,
            )
        )

    json_path, csv_path, markdown_path = _write_reports(report, output_root)
    ok_count = sum(1 for entry in report["entries"] if entry["status"] == "ok")
    print(
        f"R90/T500mm eval memory profile complete: "
        f"{ok_count}/{len(report['entries'])} ok. "
        f"JSON={json_path} CSV={csv_path} MD={markdown_path}",
        flush=True,
    )
    return 0 if ok_count == len(report["entries"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
