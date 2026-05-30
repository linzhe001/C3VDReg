#!/usr/bin/env python3
# ruff: noqa: E402,I001
"""Build post-hoc overlap and trajectory-axis diagnostics for the hard protocol.

The script intentionally reads existing benchmark result JSONL files instead of
rerunning model inference. It computes GT point-cloud overlap from the evaluated
dataset samples and joins that geometry signal with per-model recall outcomes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")
for _font_path in (
    "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Bold.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Italic.otf",
):
    if Path(_font_path).exists():
        font_manager.fontManager.addfont(_font_path)
matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Nimbus Roman"],
        "mathtext.fontset": "stix",
    }
)

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarking.datasets.c3vd_manifest_dataset import C3VDManifestDataset
from src.benchmarking.metrics.units import point_unit_to_mm_scale
from src.benchmarking.preprocess.pipeline import PreprocessPipeline
from src.benchmarking.preprocess.registry import PreprocessRegistry

DEFAULT_PROTOCOL_ROOT = Path("outputs/benchmark/r25_90_t100_500mm_protocol")
DEFAULT_FIGURE_DIR = Path("BMVC2026_Linzhe/images")
DEFAULT_OUT_DIR = DEFAULT_PROTOCOL_ROOT / "error_analysis"
DEFAULT_CONFIG = (
    DEFAULT_PROTOCOL_ROOT / "geotransformer/eval_test/normalized_eval_config.json"
)

RUN_SPECS = {
    "DCP": "dcp/eval_test/results.jsonl",
    "GeoTransformer": "geotransformer/eval_test/results.jsonl",
    "ICP": "icp/eval_test/results.jsonl",
    "PointNetLK-Mamba": "mamba2_direct/eval_test/results.jsonl",
    "PointNetLK": "pointnetlk/eval_test/results.jsonl",
    "PointNetLK-Revisited": "pointnetlk_revisited/eval_test/results.jsonl",
    "RegTR": "regtr/eval_test/results.jsonl",
}

PLOT_METHOD_LABELS = {
    "DCP": "DCP",
    "GeoTransformer": "GeoT",
    "ICP": "ICP",
    "PointNetLK-Mamba": "PNLK-M",
    "PointNetLK": "PNLK",
    "PointNetLK-Revisited": "PNLK-R",
    "RegTR": "RegTR",
}

OVERLAP_BINS = (
    (0.0, 0.25, "[0,25)"),
    (0.25, 0.50, "[25,50)"),
    (0.50, 0.75, "[50,75)"),
    (0.75, 1.01, "[75,100]"),
)


@dataclass(frozen=True)
class OverlapRecord:
    sample_id: str
    scene_id: str
    frame_id: int
    overlap_ratio: float
    source_overlap: float
    target_overlap: float
    visible_nn_median_mm: float | None
    visible_nn_p90_mm: float | None


def _load_json_or_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(handle)
        return json.load(handle)


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{key}' in {config!r} to be a mapping.")
    return value


def _build_dataset(config_path: Path) -> tuple[C3VDManifestDataset, str]:
    config = _load_json_or_yaml(config_path)
    benchmark_cfg = _require_mapping(config, "benchmark")
    data_cfg = _require_mapping(config, "data")
    preprocess_cfg = _require_mapping(config, "preprocess")

    dataset = C3VDManifestDataset(
        manifest_path=data_cfg["manifest_path"],
        split=benchmark_cfg.get("split", "test"),
        preprocess_pipeline=PreprocessPipeline(PreprocessRegistry()),
        preprocess_profile_id=preprocess_cfg["profile"],
        seed=int(preprocess_cfg.get("seed", 42)),
        preprocess_overrides={
            "sampling_override": preprocess_cfg.get("sampling_override"),
            "num_points_override": preprocess_cfg.get("num_points_override"),
        },
        perturbation_config=config.get("perturbation", {}),
        subset_config_path=data_cfg.get("subset_config_path"),
        subset_name=benchmark_cfg.get("subset_name"),
        dataset_root=data_cfg.get("dataset_root"),
    )
    return dataset, str(benchmark_cfg.get("point_unit", "m"))


def _sample_points(points: np.ndarray, sample_count: int) -> np.ndarray:
    if points.shape[0] <= sample_count:
        return points
    indices = np.linspace(0, points.shape[0] - 1, sample_count, dtype=np.int64)
    return points[indices]


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _overlap_fraction(query: np.ndarray, reference: np.ndarray, radius: float) -> float:
    if query.size == 0 or reference.size == 0:
        return 0.0
    tree = cKDTree(reference)
    distances, _ = tree.query(query, k=1, workers=-1)
    return float(np.mean(distances <= radius))


def compute_overlap_records(
    dataset: C3VDManifestDataset,
    *,
    point_unit: str,
    radius_mm: float,
    sample_count: int,
) -> list[OverlapRecord]:
    unit_scale = point_unit_to_mm_scale(point_unit)
    radius_raw = radius_mm / unit_scale

    records: list[OverlapRecord] = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        record = sample["record"]
        source = _sample_points(np.asarray(sample["source_points"]), sample_count)
        target = _sample_points(np.asarray(sample["target_points"]), sample_count)
        gt_transform = np.asarray(sample["gt_transform"], dtype=np.float64)

        aligned_source = _transform_points(source, gt_transform)
        source_overlap = _overlap_fraction(aligned_source, target, radius_raw)
        target_overlap = _overlap_fraction(target, aligned_source, radius_raw)
        overlap_ratio = 0.5 * (source_overlap + target_overlap)

        visible_nn_median_mm = record.metadata.get("visible_nn_median_mm")
        visible_nn_p90_mm = record.metadata.get("visible_nn_p90_mm")
        records.append(
            OverlapRecord(
                sample_id=record.sample_id,
                scene_id=record.scene_id,
                frame_id=record.frame_id,
                overlap_ratio=overlap_ratio,
                source_overlap=source_overlap,
                target_overlap=target_overlap,
                visible_nn_median_mm=float(visible_nn_median_mm)
                if visible_nn_median_mm is not None
                else None,
                visible_nn_p90_mm=float(visible_nn_p90_mm)
                if visible_nn_p90_mm is not None
                else None,
            )
        )
    return records


def write_overlap_csv(records: Iterable[OverlapRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "scene_id",
        "frame_id",
        "gt_overlap_ratio",
        "source_overlap",
        "target_overlap",
        "visible_nn_median_mm",
        "visible_nn_p90_mm",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "scene_id": record.scene_id,
                    "frame_id": record.frame_id,
                    "gt_overlap_ratio": f"{record.overlap_ratio:.6f}",
                    "source_overlap": f"{record.source_overlap:.6f}",
                    "target_overlap": f"{record.target_overlap:.6f}",
                    "visible_nn_median_mm": ""
                    if record.visible_nn_median_mm is None
                    else f"{record.visible_nn_median_mm:.6f}",
                    "visible_nn_p90_mm": ""
                    if record.visible_nn_p90_mm is None
                    else f"{record.visible_nn_p90_mm:.6f}",
                }
            )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _bucket_name(overlap: float) -> str:
    for lo, hi, label in OVERLAP_BINS:
        if lo <= overlap < hi:
            return label
    return "[75,100]" if overlap >= 1.0 else "[0,25)"


def build_bucket_summary(
    overlap_records: list[OverlapRecord],
    protocol_root: Path,
) -> list[dict[str, Any]]:
    overlap_by_id = {
        record.sample_id: record.overlap_ratio for record in overlap_records
    }
    rows: list[dict[str, Any]] = []

    for method, rel_path in RUN_SPECS.items():
        result_path = protocol_root / rel_path
        if not result_path.exists():
            continue
        results = _load_jsonl(result_path)
        for label in [item[2] for item in OVERLAP_BINS]:
            bucket_results = [
                row
                for row in results
                if row.get("sample_id") in overlap_by_id
                and _bucket_name(overlap_by_id[str(row["sample_id"])]) == label
            ]
            if not bucket_results:
                rows.append(
                    {
                        "method": method,
                        "overlap_bucket": label,
                        "n": 0,
                        "mean_overlap": np.nan,
                        "rr5": np.nan,
                        "rre_mean": np.nan,
                        "rte_mean": np.nan,
                        "rte_p90": np.nan,
                    }
                )
                continue
            rte_values = np.asarray([float(row["rte_mm"]) for row in bucket_results])
            rre_values = np.asarray([float(row["rre_deg"]) for row in bucket_results])
            successes = np.asarray(
                [
                    float(row["rte_mm"]) <= 5.0 and float(row["rre_deg"]) <= 5.0
                    for row in bucket_results
                ],
                dtype=np.float64,
            )
            overlaps = np.asarray(
                [overlap_by_id[str(row["sample_id"])] for row in bucket_results],
                dtype=np.float64,
            )
            rows.append(
                {
                    "method": method,
                    "overlap_bucket": label,
                    "n": len(bucket_results),
                    "mean_overlap": float(np.mean(overlaps)),
                    "rr5": float(np.mean(successes)),
                    "rre_mean": float(np.mean(rre_values)),
                    "rte_mean": float(np.mean(rte_values)),
                    "rte_p90": float(np.percentile(rte_values, 90)),
                }
            )
    return rows


def _quantile_assignments(
    overlap_records: list[OverlapRecord],
    quantile_count: int = 4,
) -> tuple[dict[str, str], dict[str, str]]:
    sorted_records = sorted(overlap_records, key=lambda item: item.overlap_ratio)
    assignments: dict[str, str] = {}
    ranges: dict[str, str] = {}

    for quantile_idx in range(quantile_count):
        start = (len(sorted_records) * quantile_idx) // quantile_count
        stop = (len(sorted_records) * (quantile_idx + 1)) // quantile_count
        bucket_records = sorted_records[start:stop]
        if not bucket_records:
            continue
        label = f"Q{quantile_idx + 1}"
        lo = bucket_records[0].overlap_ratio
        hi = bucket_records[-1].overlap_ratio
        ranges[label] = f"{100.0 * lo:.1f}-{100.0 * hi:.1f}"
        for record in bucket_records:
            assignments[record.sample_id] = label
    return assignments, ranges


def build_quantile_summary(
    overlap_records: list[OverlapRecord],
    protocol_root: Path,
    *,
    quantile_count: int = 4,
) -> list[dict[str, Any]]:
    overlap_by_id = {
        record.sample_id: record.overlap_ratio for record in overlap_records
    }
    quantile_by_id, quantile_ranges = _quantile_assignments(
        overlap_records,
        quantile_count=quantile_count,
    )
    labels = [f"Q{idx + 1}" for idx in range(quantile_count)]
    rows: list[dict[str, Any]] = []

    for method, rel_path in RUN_SPECS.items():
        result_path = protocol_root / rel_path
        if not result_path.exists():
            continue
        results = _load_jsonl(result_path)
        for label in labels:
            quantile_results = [
                row
                for row in results
                if row.get("sample_id") in quantile_by_id
                and quantile_by_id[str(row["sample_id"])] == label
            ]
            if not quantile_results:
                rows.append(
                    {
                        "method": method,
                        "overlap_quantile": label,
                        "overlap_range_pct": quantile_ranges.get(label, ""),
                        "n": 0,
                        "mean_overlap": np.nan,
                        "rr5": np.nan,
                        "rre_mean": np.nan,
                        "rte_mean": np.nan,
                        "rte_p90": np.nan,
                    }
                )
                continue
            rte_values = np.asarray([float(row["rte_mm"]) for row in quantile_results])
            rre_values = np.asarray([float(row["rre_deg"]) for row in quantile_results])
            successes = np.asarray(
                [
                    float(row["rte_mm"]) <= 5.0 and float(row["rre_deg"]) <= 5.0
                    for row in quantile_results
                ],
                dtype=np.float64,
            )
            overlaps = np.asarray(
                [overlap_by_id[str(row["sample_id"])] for row in quantile_results],
                dtype=np.float64,
            )
            rows.append(
                {
                    "method": method,
                    "overlap_quantile": label,
                    "overlap_range_pct": quantile_ranges.get(label, ""),
                    "n": len(quantile_results),
                    "mean_overlap": float(np.mean(overlaps)),
                    "rr5": float(np.mean(successes)),
                    "rre_mean": float(np.mean(rre_values)),
                    "rte_mean": float(np.mean(rte_values)),
                    "rte_p90": float(np.percentile(rte_values, 90)),
                }
            )
    return rows


def write_bucket_summary(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "overlap_bucket",
        "n",
        "mean_overlap",
        "rr5",
        "rre_mean",
        "rte_mean",
        "rte_p90",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ""
                    if isinstance(value := row[key], float) and np.isnan(value)
                    else f"{value:.6f}"
                    if isinstance(value, float)
                    else value
                    for key in fieldnames
                }
            )


def write_quantile_summary(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "overlap_quantile",
        "overlap_range_pct",
        "n",
        "mean_overlap",
        "rr5",
        "rre_mean",
        "rte_mean",
        "rte_p90",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ""
                    if isinstance(value := row[key], float) and np.isnan(value)
                    else f"{value:.6f}"
                    if isinstance(value, float)
                    else value
                    for key in fieldnames
                }
            )


def _plot_bucket_summary(rows: list[dict[str, Any]], figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [item[2] for item in OVERLAP_BINS]
    x = np.arange(len(labels))

    plt.figure(figsize=(4.15, 3.00))
    for method in RUN_SPECS:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        y = []
        for label in labels:
            match = next(
                (row for row in method_rows if row["overlap_bucket"] == label),
                None,
            )
            y.append(np.nan if match is None else match["rr5"] * 100.0)
        if np.all(np.isnan(y)):
            continue
        plt.plot(
            x,
            y,
            marker="o",
            linewidth=1.8,
            markersize=4.2,
            label=PLOT_METHOD_LABELS.get(method, method),
        )

    plt.xticks(x, labels, fontsize=14.0)
    plt.yticks(fontsize=14.0)
    plt.xlabel("GT overlap ratio (%)", fontsize=16.2)
    plt.ylabel("Strict RR@5 (%)", fontsize=16.2)
    plt.ylim(-2, 102)
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    plt.legend(fontsize=13.4, ncol=2, frameon=False)
    plt.tight_layout(pad=0.25)
    plt.savefig(figure_path)
    plt.close()


def _plot_quantile_summary(rows: list[dict[str, Any]], figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    labels = sorted({row["overlap_quantile"] for row in rows})
    ranges = {
        row["overlap_quantile"]: row["overlap_range_pct"]
        for row in rows
        if row.get("overlap_range_pct")
    }
    x = np.arange(len(labels))

    plt.figure(figsize=(4.15, 3.00))
    for method in RUN_SPECS:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        y = []
        for label in labels:
            match = next(
                (row for row in method_rows if row["overlap_quantile"] == label),
                None,
            )
            y.append(np.nan if match is None else match["rr5"] * 100.0)
        if np.all(np.isnan(y)):
            continue
        plt.plot(
            x,
            y,
            marker="o",
            linewidth=1.8,
            markersize=4.2,
            label=PLOT_METHOD_LABELS.get(method, method),
        )

    tick_labels = [f"{label}\n{ranges.get(label, '')}" for label in labels]
    plt.xticks(x, tick_labels, fontsize=12.8)
    plt.yticks(fontsize=14.0)
    plt.xlabel("GT overlap quantile (%)", fontsize=16.2)
    plt.ylabel("Strict RR@5 (%)", fontsize=16.2)
    plt.ylim(-2, 102)
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    plt.legend(fontsize=13.4, ncol=2, frameon=False)
    plt.tight_layout(pad=0.25)
    plt.savefig(figure_path)
    plt.close()


def _parse_pose_file(path: Path) -> tuple[np.ndarray, str]:
    rows: list[np.ndarray] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            values = [
                float(item)
                for item in line.replace(" ", ",").split(",")
                if item.strip()
            ]
            if len(values) != 16:
                continue
            rows.append(np.asarray(values, dtype=np.float64).reshape(4, 4))

    if not rows:
        raise ValueError(f"No 4x4 poses found in {path}")

    poses = np.stack(rows, axis=0)
    row_centers = poses[:, 3, :3]
    col_centers = poses[:, :3, 3]
    row_step = float(np.median(np.linalg.norm(np.diff(row_centers, axis=0), axis=1)))
    col_step = float(np.median(np.linalg.norm(np.diff(col_centers, axis=0), axis=1)))
    if row_step >= col_step:
        return row_centers, "row_translation"
    return col_centers, "column_translation"


def _axis_at_frame(centers: np.ndarray, frame_id: int) -> np.ndarray:
    if frame_id <= 0:
        direction = centers[1] - centers[0]
    elif frame_id >= len(centers) - 1:
        direction = centers[-1] - centers[-2]
    else:
        direction = centers[frame_id + 1] - centers[frame_id - 1]
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    return direction / norm


def write_trajectory_axes(dataset: C3VDManifestDataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pose_cache: dict[Path, tuple[np.ndarray, str]] = {}
    fieldnames = [
        "sample_id",
        "scene_id",
        "frame_id",
        "center_x",
        "center_y",
        "center_z",
        "axial_x",
        "axial_y",
        "axial_z",
        "pose_convention",
        "pose_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(len(dataset)):
            sample = dataset[idx]
            record = sample["record"]
            pose_path_raw = record.metadata.get(
                "target_pose_path"
            ) or record.metadata.get("reference_pose_path")
            if not pose_path_raw:
                continue
            pose_path = Path(str(pose_path_raw))
            if pose_path not in pose_cache:
                pose_cache[pose_path] = _parse_pose_file(pose_path)
            centers, convention = pose_cache[pose_path]
            if record.frame_id < 0 or record.frame_id >= len(centers):
                continue
            axis = _axis_at_frame(centers, record.frame_id)
            center = centers[record.frame_id]
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "scene_id": record.scene_id,
                    "frame_id": record.frame_id,
                    "center_x": f"{center[0]:.6f}",
                    "center_y": f"{center[1]:.6f}",
                    "center_z": f"{center[2]:.6f}",
                    "axial_x": f"{axis[0]:.8f}",
                    "axial_y": f"{axis[1]:.8f}",
                    "axial_z": f"{axis[2]:.8f}",
                    "pose_convention": convention,
                    "pose_path": str(pose_path),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--radius-mm", type=float, default=5.0)
    parser.add_argument("--sample-count", type=int, default=2048)
    args = parser.parse_args()

    dataset, point_unit = _build_dataset(args.config)
    overlap_records = compute_overlap_records(
        dataset,
        point_unit=point_unit,
        radius_mm=args.radius_mm,
        sample_count=args.sample_count,
    )

    overlap_csv = args.out_dir / "gt_overlap_by_sample.csv"
    write_overlap_csv(overlap_records, overlap_csv)

    bucket_rows = build_bucket_summary(overlap_records, args.protocol_root)
    bucket_csv = args.out_dir / "gt_overlap_bucket_summary.csv"
    write_bucket_summary(bucket_rows, bucket_csv)

    quantile_rows = build_quantile_summary(overlap_records, args.protocol_root)
    quantile_csv = args.out_dir / "gt_overlap_quantile_summary.csv"
    write_quantile_summary(quantile_rows, quantile_csv)

    axes_csv = args.out_dir / "trajectory_axial_vectors.csv"
    write_trajectory_axes(dataset, axes_csv)

    figure_pdf = args.figure_dir / "hard_overlap_rr_curve.pdf"
    _plot_bucket_summary(bucket_rows, figure_pdf)

    quantile_figure_pdf = args.figure_dir / "hard_overlap_quantile_rr_curve.pdf"
    _plot_quantile_summary(quantile_rows, quantile_figure_pdf)

    print(f"Wrote {overlap_csv}")
    print(f"Wrote {bucket_csv}")
    print(f"Wrote {quantile_csv}")
    print(f"Wrote {axes_csv}")
    print(f"Wrote {figure_pdf}")
    print(f"Wrote {quantile_figure_pdf}")


if __name__ == "__main__":
    main()
