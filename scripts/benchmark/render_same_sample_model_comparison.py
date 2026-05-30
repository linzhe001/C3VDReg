#!/usr/bin/env python3
# ruff: noqa: E402,I001
"""Render a same-sample qualitative comparison across learned baselines."""

from __future__ import annotations

import json
import shutil
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
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark.render_bmvc_qualitative_panels import (  # noqa: E402
    _apply_transform,
    _camera_direction_to_mpl_view,
    _choose_error_view,
    _load_json,
    _load_sample,
    _sample_evenly,
    _style_axis,
)
from src.benchmarking.metrics.units import (  # noqa: E402
    canonicalize_point_unit,
    point_unit_to_mm_scale,
)

PROTOCOL_ROOT = REPO_ROOT / "outputs" / "benchmark" / "r25_90_t100_500mm_protocol"
TRANSFORM_ROOT = PROTOCOL_ROOT / "pose_transform_exports"
PAPER_FIGURE_DIR = REPO_ROOT / "outputs" / "benchmark" / "figures" / "paper"
BMVC_IMAGE_DIR = REPO_ROOT / "BMVC2026_Linzhe" / "images"
SAMPLE_ID = "cecum_t3_a:0133"
OUTPUT_NAME = "same_sample_model_comparison.pdf"

MODELS = (
    ("geotransformer", "GeoT"),
    ("mamba2_direct", "PNLK-M"),
    ("regtr", "RegTR"),
    ("pointnetlk_revisited", "PNLK-R"),
    ("pointnetlk", "PNLK"),
    ("dcp", "DCP"),
)


def _read_result_row(model_key: str, sample_id: str) -> dict[str, Any]:
    path = PROTOCOL_ROOT / model_key / "eval_test" / "results.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("sample_id")) == sample_id:
                return row
    raise KeyError(f"{sample_id!r} not found in {path}")


def _read_transform(model_key: str, sample_id: str) -> np.ndarray:
    path = TRANSFORM_ROOT / model_key / "eval_test" / "pose_transforms.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("sample_id")) == sample_id:
                return np.asarray(row["pred_transform"], dtype=np.float64)
    raise KeyError(f"{sample_id!r} not found in {path}")


def main() -> int:
    config = _load_json(
        PROTOCOL_ROOT / "geotransformer" / "eval_test" / "normalized_eval_config.json"
    )
    sample = _load_sample(config, SAMPLE_ID)
    source = np.asarray(sample["source_points"], dtype=np.float64)
    target = np.asarray(sample["target_points"], dtype=np.float64)
    point_unit = canonicalize_point_unit(
        str(config["benchmark"].get("point_unit", "m"))
    )
    mm_scale = point_unit_to_mm_scale(point_unit)

    source_small = _sample_evenly(source, 2200)
    target_small = _sample_evenly(target, 2200)
    results = {
        model_key: _read_result_row(model_key, SAMPLE_ID)
        for model_key, _ in MODELS
    }
    predictions: dict[str, np.ndarray] = {}
    distances: dict[str, np.ndarray] = {}
    tree = cKDTree(target_small)
    for model_key, _ in MODELS:
        pred_source = _apply_transform(
            source_small,
            _read_transform(model_key, SAMPLE_ID),
        )
        predictions[model_key] = pred_source
        distances[model_key] = tree.query(pred_source)[0] * mm_scale

    camera_direction = _choose_error_view(
        predictions["geotransformer"],
        distances["geotransformer"],
    )
    elev, azim = _camera_direction_to_mpl_view(camera_direction)
    finite_distances = np.concatenate(
        [values[np.isfinite(values)] for values in distances.values()]
    )
    color_max = max(10.0, min(60.0, float(np.percentile(finite_distances, 92.0))))

    fig = plt.figure(figsize=(11.8, 5.95), dpi=190)
    grid = fig.add_gridspec(2, 3, wspace=0.01, hspace=0.17)
    axes = []
    rendered = None
    for index, (model_key, label) in enumerate(MODELS):
        ax = fig.add_subplot(grid[index // 3, index % 3], projection="3d")
        axes.append(ax)
        pred_source = predictions[model_key]
        ax.scatter(
            target_small[:, 0],
            target_small[:, 1],
            target_small[:, 2],
            s=3.3,
            c="#c3cad5",
            alpha=0.32,
        )
        rendered = ax.scatter(
            pred_source[:, 0],
            pred_source[:, 1],
            pred_source[:, 2],
            s=5.5,
            c=distances[model_key],
            cmap="inferno",
            vmin=0.0,
            vmax=color_max,
            alpha=0.90,
        )
        _style_axis(ax, np.vstack([target_small, pred_source]), elev, azim)
        row = results[model_key]
        ax.set_title(
            f"{label}: {float(row['rre_deg']):.1f}deg / "
            f"{float(row['rte_mm']):.1f}mm",
            fontsize=19.5,
            pad=0.0,
        )

    cax = fig.add_axes([0.925, 0.205, 0.014, 0.585])
    if rendered is not None:
        colorbar = fig.colorbar(rendered, cax=cax)
        colorbar.set_label("NN distance (mm)", fontsize=18.0, labelpad=5)
        colorbar.ax.tick_params(labelsize=16.5, length=2)
    fig.subplots_adjust(left=0.010, right=0.910, top=0.925, bottom=0.025)

    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    BMVC_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = PAPER_FIGURE_DIR / OUTPUT_NAME
    fig.savefig(pdf_path)
    plt.close(fig)
    shutil.copy2(pdf_path, BMVC_IMAGE_DIR / OUTPUT_NAME)

    summary_path = PAPER_FIGURE_DIR / "same_sample_model_comparison.json"
    summary = {
        "sample_id": SAMPLE_ID,
        "point_unit": point_unit,
        "color_max_mm": color_max,
        "pdf": str(pdf_path.relative_to(REPO_ROOT)),
        "models": {
            label: {
                "model_key": model_key,
                "rre_deg": float(results[model_key]["rre_deg"]),
                "rte_mm": float(results[model_key]["rte_mm"]),
                "rr5": int(results[model_key]["success_5deg_5mm"]),
                "visible_nn_mean_mm": float(results[model_key]["visible_nn_mean_mm"]),
            }
            for model_key, label in MODELS
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
