#!/usr/bin/env python3
# ruff: noqa: E402
"""Build axial/radial translation-error diagnostics from exported transforms."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmarking.metrics.units import point_unit_to_mm_scale

TRANSFORM_ROOT = Path(
    "outputs/benchmark/r25_90_t100_500mm_protocol/pose_transform_exports"
)
ERROR_ANALYSIS_ROOT = Path(
    "outputs/benchmark/r25_90_t100_500mm_protocol/error_analysis"
)
AXES_CSV = ERROR_ANALYSIS_ROOT / "trajectory_axial_vectors.csv"
OUT_DIR = ERROR_ANALYSIS_ROOT
FIGURE_DIR = Path("BMVC2026_Linzhe/images")
METHOD_LABELS = {
    "dcp": "DCP",
    "geotransformer": "GeoT",
    "icp": "ICP",
    "mamba2_direct": "PNLK-M",
    "pointnetlk": "PNLK",
    "pointnetlk_revisited": "PNLK-R",
    "regtr": "RegTR",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_axes(path: Path) -> dict[str, np.ndarray]:
    axes: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            axis = np.asarray(
                [
                    float(row["axial_x"]),
                    float(row["axial_y"]),
                    float(row["axial_z"]),
                ],
                dtype=np.float64,
            )
            norm = np.linalg.norm(axis)
            if norm > 1e-12:
                axes[row["sample_id"]] = axis / norm
    return axes


def _iter_transform_files(root: Path) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for model_dir in sorted(root.iterdir() if root.exists() else []):
        transform_path = model_dir / "eval_test" / "pose_transforms.jsonl"
        if transform_path.exists():
            files.append((model_dir.name, transform_path))
    return files


def _decompose_row(row: dict[str, Any], axis: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(row["pred_transform"], dtype=np.float64)
    gt = np.asarray(row["gt_transform"], dtype=np.float64)
    point_unit = str(row.get("point_unit", "m"))
    scale = point_unit_to_mm_scale(point_unit)
    translation_error_mm = (pred[:3, 3] - gt[:3, 3]) * scale
    rte_from_vector = float(np.linalg.norm(translation_error_mm))
    axial_signed = float(np.dot(translation_error_mm, axis))
    axial_abs = abs(axial_signed)
    radial = math.sqrt(max(rte_from_vector**2 - axial_abs**2, 0.0))
    axial_fraction = axial_abs / rte_from_vector if rte_from_vector > 1e-12 else 0.0
    axial_energy_fraction = (
        (axial_abs**2) / (rte_from_vector**2) if rte_from_vector > 1e-12 else 0.0
    )
    return {
        "sample_id": row["sample_id"],
        "scene_id": row["scene_id"],
        "frame_id": int(row["frame_id"]),
        "model_id": row["model_id"],
        "rre_deg": float(row["rre_deg"]),
        "rte_mm": float(row["rte_mm"]),
        "rte_from_vector_mm": rte_from_vector,
        "axial_signed_mm": axial_signed,
        "axial_abs_mm": axial_abs,
        "radial_mm": radial,
        "axial_fraction": axial_fraction,
        "axial_energy_fraction": axial_energy_fraction,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in rows], dtype=np.float64)

    axial = values("axial_abs_mm")
    radial = values("radial_mm")
    rte = values("rte_mm")
    frac = values("axial_fraction")
    energy = values("axial_energy_fraction")
    vector_rte = values("rte_from_vector_mm")
    return {
        "n": len(rows),
        "rte_mean_mm": float(np.mean(rte)),
        "rte_p90_mm": float(np.percentile(rte, 90)),
        "rte_vector_max_abs_diff_mm": float(np.max(np.abs(rte - vector_rte))),
        "axial_abs_mean_mm": float(np.mean(axial)),
        "axial_abs_median_mm": float(np.median(axial)),
        "axial_abs_p90_mm": float(np.percentile(axial, 90)),
        "radial_mean_mm": float(np.mean(radial)),
        "radial_median_mm": float(np.median(radial)),
        "radial_p90_mm": float(np.percentile(radial, 90)),
        "axial_fraction_mean": float(np.mean(frac)),
        "axial_energy_fraction_mean": float(np.mean(energy)),
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.6f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def _plot_summary(summary_rows: list[dict[str, Any]], figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["method"] for row in summary_rows]
    x = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(4.25, 3.00))
    plt.bar(
        x - width / 2,
        [float(row["axial_abs_p90_mm"]) for row in summary_rows],
        width,
        label="Axial 90th percentile",
        color="#4c78a8",
    )
    plt.bar(
        x + width / 2,
        [float(row["radial_p90_mm"]) for row in summary_rows],
        width,
        label="Radial 90th percentile",
        color="#f58518",
    )
    plt.xticks(x, labels, rotation=25, ha="right", fontsize=14.0)
    plt.yticks(fontsize=14.0)
    plt.ylabel("90th-percentile error (mm)", fontsize=16.2)
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    plt.legend(frameon=False, fontsize=13.4)
    plt.tight_layout(pad=0.25)
    plt.savefig(figure_path)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transforms-root", type=Path, default=TRANSFORM_ROOT)
    parser.add_argument("--axes-csv", type=Path, default=AXES_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    axes = _read_axes(args.axes_csv)
    per_sample_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for model_key, transform_path in _iter_transform_files(args.transforms_root):
        method_rows = []
        for row in _read_jsonl(transform_path):
            axis = axes.get(str(row["sample_id"]))
            if axis is None:
                continue
            decomposed = _decompose_row(row, axis)
            decomposed["method"] = METHOD_LABELS.get(model_key, model_key)
            method_rows.append(decomposed)
        if not method_rows:
            continue
        summary = _summarize(method_rows)
        summary_rows.append(
            {
                "method": METHOD_LABELS.get(model_key, model_key),
                "model_key": model_key,
                **summary,
            }
        )
        per_sample_rows.extend(method_rows)

    if not summary_rows:
        raise SystemExit(
            f"No pose_transforms.jsonl files found under {args.transforms_root}."
        )

    summary_rows.sort(key=lambda row: float(row["rte_p90_mm"]))
    per_sample_rows.sort(key=lambda row: (str(row["method"]), str(row["sample_id"])))

    _write_csv(per_sample_rows, args.out_dir / "axial_radial_by_sample.csv")
    _write_csv(summary_rows, args.out_dir / "axial_radial_summary.csv")
    _plot_summary(
        summary_rows,
        args.figure_dir / "hard_axial_radial_translation_p90.pdf",
    )
    print(f"Wrote {args.out_dir / 'axial_radial_by_sample.csv'}")
    print(f"Wrote {args.out_dir / 'axial_radial_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
