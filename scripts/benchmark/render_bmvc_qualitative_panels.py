#!/usr/bin/env python3
# ruff: noqa: E402,I001
"""Render deterministic BMVC qualitative alignment panels."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
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

from src.benchmarking.datasets.c3vd_manifest_dataset import C3VDManifestDataset  # noqa: E402
from src.benchmarking.metrics.units import (  # noqa: E402
    canonicalize_point_unit,
    point_unit_to_mm_scale,
)
from src.benchmarking.preprocess.pipeline import PreprocessPipeline  # noqa: E402
from src.benchmarking.preprocess.registry import PreprocessRegistry  # noqa: E402
from src.benchmarking.registry.model_registry import ModelRegistry  # noqa: E402

HARD_SELECTION = (
    REPO_ROOT / "outputs" / "benchmark" / "r25_90_t100_500mm_protocol"
    / "figures" / "qualitative_selection_manifest.json"
)
LEGACY_SELECTION = (
    REPO_ROOT / "outputs" / "benchmark" / "figures" / "paper"
    / "qualitative_selection_manifest.json"
)
DEFAULT_SELECTION = HARD_SELECTION if HARD_SELECTION.exists() else LEGACY_SELECTION
PAPER_FIGURE_DIR = REPO_ROOT / "outputs" / "benchmark" / "figures" / "paper"
FAILURE_CASE_DIR = REPO_ROOT / "outputs" / "benchmark" / "figures" / "failure_cases"
BMVC_IMAGE_DIR = REPO_ROOT / "BMVC2026_Linzhe" / "images"
TRANSFORM_EXPORT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "benchmark"
    / "r25_90_t100_500mm_protocol"
    / "pose_transform_exports"
)
DEFAULT_CAMERA_DIRECTION = np.array([1.55, 1.18, 0.92], dtype=np.float64)
DEFAULT_CAMERA_DIRECTION = DEFAULT_CAMERA_DIRECTION / np.linalg.norm(
    DEFAULT_CAMERA_DIRECTION
)
VIEW_CANDIDATE_COUNT = 48


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_symbol(path: str) -> Any:
    module_name, symbol_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def _instantiate_adapter(config: dict[str, Any]) -> Any:
    registry = ModelRegistry()
    spec = registry.get(config["model"]["id"])
    adapter_config = dict(config["model"].get("overrides", {}))
    adapter_config.setdefault("device", config["runtime"]["device"])
    adapter_config["checkpoint_path"] = config["model"].get("checkpoint_path")
    adapter_class = _load_symbol(spec.adapter_path)
    adapter = adapter_class(SimpleNamespace(**adapter_config))
    adapter.load_model(config["model"].get("checkpoint_path"))
    return adapter


def _prediction_to_transform(prediction: Any) -> np.ndarray:
    if isinstance(prediction, tuple) and len(prediction) == 2:
        rotation, translation = prediction
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
        transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
        return transform
    transform = np.asarray(prediction, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Unsupported prediction shape {transform.shape}.")
    return transform


def _load_exported_transform(case: dict[str, Any]) -> np.ndarray | None:
    transform_path = (
        TRANSFORM_EXPORT_ROOT
        / str(case["model_key"])
        / "eval_test"
        / "pose_transforms.jsonl"
    )
    if not transform_path.exists():
        return None
    sample_id = str(case["sample_id"])
    with transform_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("sample_id")) == sample_id:
                return np.asarray(row["pred_transform"], dtype=np.float64)
    return None


def _apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (transform[:3, :3] @ points.T).T + transform[:3, 3]


def _sample_evenly(points: np.ndarray, sample_count: int) -> np.ndarray:
    if len(points) <= sample_count:
        return points.astype(np.float64, copy=False)
    indices = np.linspace(0, len(points) - 1, num=sample_count, dtype=int)
    return points[indices].astype(np.float64, copy=False)


def _sanitize(text: str) -> str:
    return "".join(
        char if char.isalnum() or char in "._-" else "_" for char in text
    )


def _trim_image_horizontally(
    image: np.ndarray,
    *,
    threshold: float = 0.985,
    margin_fraction: float = 0.015,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[1] <= 1:
        return image
    rgb = image[..., :3]
    alpha = image[..., 3] if image.shape[2] == 4 else np.ones(image.shape[:2])
    content = (np.min(rgb, axis=2) < threshold) | (alpha < 0.99)
    columns = np.where(np.any(content, axis=0))[0]
    if columns.size == 0:
        return image
    margin = max(10, int(round(image.shape[1] * margin_fraction)))
    left = max(0, int(columns[0]) - margin)
    right = min(image.shape[1], int(columns[-1]) + margin + 1)
    return image[:, left:right, :]


def _remove_case_title_band(
    image: np.ndarray,
    *,
    top_fraction: float = 0.070,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[0] <= 1:
        return image
    top = min(image.shape[0] - 1, max(0, int(round(image.shape[0] * top_fraction))))
    return image[top:, :, :]


def _case_title(case: dict[str, Any]) -> str:
    return (
        f"{case['label']} | {case['display_name']} | {case['sample_id']} | "
        f"RRE={float(case['rre_deg']):.2f}deg, "
        f"RTE={float(case['rte_mm']):.2f}mm"
    )


def _load_sample(config: dict[str, Any], sample_id: str) -> dict[str, Any]:
    dataset = C3VDManifestDataset(
        manifest_path=config["data"]["manifest_path"],
        split=config["benchmark"]["split"],
        preprocess_pipeline=PreprocessPipeline(PreprocessRegistry()),
        preprocess_profile_id=config["preprocess"]["profile"],
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
    for sample in dataset:
        if str(sample["sample_id"]) == sample_id:
            return sample
    raise KeyError(
        f"sample_id {sample_id!r} not found in "
        f"{config['data']['manifest_path']}"
    )


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        return DEFAULT_CAMERA_DIRECTION.copy()
    return vector / norm


def _camera_basis_from_direction(
    camera_direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    camera_direction = _normalize_vector(camera_direction)
    up_reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(camera_direction, up_reference))) > 0.96:
        up_reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = _normalize_vector(np.cross(up_reference, camera_direction))
    up = _normalize_vector(np.cross(camera_direction, right))
    return right, up


def _fibonacci_sphere_directions(count: int) -> np.ndarray:
    indices = np.arange(count, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - (2.0 * indices + 1.0) / count
    radius = np.sqrt(np.clip(1.0 - y * y, 0.0, 1.0))
    theta = phi * indices
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    return np.stack([x, y, z], axis=1)


def _choose_error_view(points: np.ndarray, distances_mm: np.ndarray) -> np.ndarray:
    finite = np.isfinite(distances_mm) & np.all(np.isfinite(points), axis=1)
    if not np.any(finite):
        return DEFAULT_CAMERA_DIRECTION.copy()
    points = points[finite]
    distances_mm = distances_mm[finite]
    center = np.mean(points, axis=0)
    best_score = -np.inf
    best_direction = DEFAULT_CAMERA_DIRECTION.copy()
    for candidate in _fibonacci_sphere_directions(VIEW_CANDIDATE_COUNT):
        right, up = _camera_basis_from_direction(candidate)
        rel = points - center
        proj_x = rel @ right
        proj_y = rel @ up
        area = max(float(np.ptp(proj_x) * np.ptp(proj_y)), 1.0e-6)
        depth = rel @ candidate
        visible = np.argsort(depth)[-max(1, len(depth) // 3):]
        visible_error = float(np.percentile(distances_mm[visible], 90.0))
        score = visible_error + 0.02 * area
        if score > best_score:
            best_score = score
            best_direction = candidate
    return _normalize_vector(best_direction)


def _camera_direction_to_mpl_view(camera_direction: np.ndarray) -> tuple[float, float]:
    camera_direction = _normalize_vector(camera_direction)
    elev = float(np.degrees(np.arcsin(np.clip(camera_direction[2], -1.0, 1.0))))
    azim = float(np.degrees(np.arctan2(camera_direction[1], camera_direction[0])))
    return elev, azim


def _set_equal_axes(
    ax: Any,
    points: np.ndarray,
    *,
    padding: float = 0.015,
    zoom: float = 1.0,
    clip_percentiles: tuple[float, float] = (3.0, 97.0),
) -> tuple[float, float, float] | None:
    finite = np.asarray(points, dtype=np.float64)
    finite = finite[np.all(np.isfinite(finite), axis=1)]
    if len(finite) == 0:
        return None
    if len(finite) >= 16:
        low, high = clip_percentiles
        mins = np.percentile(finite, low, axis=0)
        maxs = np.percentile(finite, high, axis=0)
    else:
        mins = finite.min(axis=0)
        maxs = finite.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 1.0e-6)
    radius *= max(0.38, zoom) * (1.0 + padding)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    return (1.0, 1.0, 1.0)


def _set_data_axes(
    ax: Any,
    points: np.ndarray,
    *,
    padding: float = 0.025,
    zoom: float = 1.0,
    clip_percentiles: tuple[float, float] = (2.0, 98.0),
) -> tuple[float, float, float] | None:
    finite = np.asarray(points, dtype=np.float64)
    finite = finite[np.all(np.isfinite(finite), axis=1)]
    if len(finite) == 0:
        return None
    if len(finite) >= 16:
        low, high = clip_percentiles
        mins = np.percentile(finite, low, axis=0)
        maxs = np.percentile(finite, high, axis=0)
    else:
        mins = finite.min(axis=0)
        maxs = finite.max(axis=0)
    center = (mins + maxs) / 2.0
    half_spans = np.maximum((maxs - mins) / 2.0, 1.0e-6)
    half_spans *= max(0.55, zoom) * (1.0 + padding)
    ax.set_xlim(center[0] - half_spans[0], center[0] + half_spans[0])
    ax.set_ylim(center[1] - half_spans[1], center[1] + half_spans[1])
    ax.set_zlim(center[2] - half_spans[2], center[2] + half_spans[2])
    return (1.0, 1.0, 1.0)


def _style_axis(
    ax: Any,
    view_points: np.ndarray,
    elev: float,
    azim: float,
    *,
    zoom: float = 1.0,
    clip_percentiles: tuple[float, float] = (3.0, 97.0),
    equal_axes: bool = True,
) -> None:
    if equal_axes:
        aspect = _set_equal_axes(
            ax,
            view_points,
            zoom=zoom,
            clip_percentiles=clip_percentiles,
        )
    else:
        aspect = _set_data_axes(
            ax,
            view_points,
            zoom=zoom,
            clip_percentiles=clip_percentiles,
        )
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_box_aspect(aspect or (1.0, 1.0, 1.0))
    ax.set_axis_off()
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 1.0))


def _trimmed_chamfer_mm(
    pred_source: np.ndarray,
    target_points: np.ndarray,
    mm_scale: float,
) -> float:
    pred_to_target = cKDTree(target_points).query(pred_source)[0]
    target_to_pred = cKDTree(pred_source).query(target_points)[0]
    chamfer = np.concatenate([pred_to_target, target_to_pred]) * mm_scale
    cutoff = np.percentile(chamfer, 90.0)
    trimmed = chamfer[chamfer <= cutoff]
    return float(np.mean(trimmed)) if trimmed.size else float("nan")


def _render_case_panel(
    case: dict[str, Any],
    output_dir: Path,
    config_cache: dict[str, dict[str, Any]],
    adapter_cache: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    cache_key = str(case["result_dir"])
    config = config_cache.setdefault(
        cache_key,
        _load_json(REPO_ROOT / case["result_dir"] / "normalized_eval_config.json"),
    )
    sample = _load_sample(config, str(case["sample_id"]))
    source = np.asarray(sample["source_points"], dtype=np.float64)
    target = np.asarray(sample["target_points"], dtype=np.float64)
    pred_transform = _load_exported_transform(case)
    if pred_transform is None:
        if cache_key not in adapter_cache:
            adapter_cache[cache_key] = _instantiate_adapter(config)
        adapter = adapter_cache[cache_key]
        pred_transform = _prediction_to_transform(adapter.predict(source, target))
    gt_transform = np.asarray(sample["gt_transform"], dtype=np.float64)
    point_unit = canonicalize_point_unit(
        str(config["benchmark"].get("point_unit", "m"))
    )
    mm_scale = point_unit_to_mm_scale(point_unit)

    source_small = _sample_evenly(source, 2048)
    target_small = _sample_evenly(target, 2048)
    pred_source = _apply_transform(source_small, pred_transform)
    gt_source = _apply_transform(source_small, gt_transform)
    distances_mm = cKDTree(target_small).query(pred_source)[0] * mm_scale
    trim_cd_mm = _trimmed_chamfer_mm(pred_source, target_small, mm_scale)
    color_max = max(
        float(np.percentile(distances_mm[np.isfinite(distances_mm)], 95.0)),
        float(case.get("visible_nn_mean_mm", 0.0)) * 3.0,
        1.0,
    )
    camera_direction = _choose_error_view(pred_source, distances_mm)
    elev, azim = _camera_direction_to_mpl_view(camera_direction)

    fig = plt.figure(figsize=(13.8, 3.18), dpi=190)
    axes = [
        fig.add_axes([0.000, 0.075, 0.255, 0.785], projection="3d"),
        fig.add_axes([0.312, 0.075, 0.245, 0.785], projection="3d"),
        fig.add_axes([0.590, 0.085, 0.205, 0.775], projection="3d"),
    ]
    hist_ax = fig.add_axes([0.833, 0.185, 0.160, 0.675])
    colorbar_ax = fig.add_axes([0.280, 0.275, 0.010, 0.475])
    titles = ("Initial", "Prediction error", "Ground truth")

    axes[0].scatter(
        target_small[:, 0],
        target_small[:, 1],
        target_small[:, 2],
        s=5.8,
        c="#b8c0cc",
        alpha=0.42,
    )
    axes[0].scatter(
        source_small[:, 0],
        source_small[:, 1],
        source_small[:, 2],
        s=7.0,
        c="#2563a9",
        alpha=0.82,
    )

    axes[1].scatter(
        target_small[:, 0],
        target_small[:, 1],
        target_small[:, 2],
        s=4.8,
        c="#c9ced6",
        alpha=0.28,
    )
    rendered = axes[1].scatter(
        pred_source[:, 0],
        pred_source[:, 1],
        pred_source[:, 2],
        s=7.4,
        c=distances_mm,
        cmap="inferno",
        vmin=0.0,
        vmax=color_max,
        alpha=0.9,
    )

    axes[2].scatter(
        target_small[:, 0],
        target_small[:, 1],
        target_small[:, 2],
        s=4.8,
        c="#b8c0cc",
        alpha=0.35,
    )
    axes[2].scatter(
        gt_source[:, 0],
        gt_source[:, 1],
        gt_source[:, 2],
        s=7.0,
        c="#1f8a5b",
        alpha=0.82,
    )

    for ax, title in zip(axes, titles, strict=True):
        if title == "Initial":
            view_points = np.vstack([source_small, target_small])
            zoom = 1.0
            clip_percentiles = (0.0, 100.0)
            equal_axes = False
        elif title == "Prediction error":
            view_points = np.vstack([pred_source, target_small])
            zoom = 0.60
            clip_percentiles = (4.0, 96.0)
            equal_axes = True
        else:
            view_points = np.vstack([gt_source, target_small])
            zoom = 0.60
            clip_percentiles = (4.0, 96.0)
            equal_axes = True
        _style_axis(
            ax,
            view_points,
            elev,
            azim,
            zoom=zoom,
            clip_percentiles=clip_percentiles,
            equal_axes=equal_axes,
        )
        ax.set_title(title, fontsize=24.0, pad=1.0)

    colorbar = fig.colorbar(
        rendered,
        cax=colorbar_ax,
    )
    colorbar.ax.set_ylabel("")
    colorbar.ax.yaxis.set_ticks_position("left")
    colorbar.ax.tick_params(labelsize=16.0, length=2, pad=1)

    ordered = np.sort(distances_mm[np.isfinite(distances_mm)])
    if ordered.size:
        cdf = np.linspace(0.0, 1.0, ordered.size)
        hist_ax.plot(ordered, cdf, color="#222222", linewidth=1.8)
        hist_ax.axvline(5.0, color="#2f6fbb", linewidth=1.0, linestyle="--")
        hist_ax.axvline(10.0, color="#bf6f2a", linewidth=1.0, linestyle="--")
        hist_ax.set_xlim(0.0, max(float(np.percentile(ordered, 98.0)), 10.0))
        hist_ax.set_ylim(0.0, 1.0)
    hist_ax.grid(alpha=0.22, linewidth=0.8)
    hist_ax.set_title("Residual CDF", fontsize=24.0, pad=2)
    hist_ax.set_xlabel("NN distance (mm)", fontsize=20.0, labelpad=0)
    hist_ax.set_ylabel("")
    hist_ax.tick_params(axis="both", labelsize=18.0, pad=1)
    hist_ax.text(
        0.02,
        0.05,
        f"mean {float(np.mean(ordered)) if ordered.size else float('nan'):.2f} mm\n"
        f"p90 "
        f"{float(np.percentile(ordered, 90.0)) if ordered.size else float('nan'):.2f}"
        " mm\n"
        f"Trim CD {trim_cd_mm:.2f} mm",
        transform=hist_ax.transAxes,
        fontsize=16.5,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.85},
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    base = _sanitize(f"{case['label']}_{case['model_key']}_{case['sample_id']}")
    pdf_path = output_dir / f"{base}.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)

    metadata = {
        **case,
        "panel_pdf": str(pdf_path.relative_to(REPO_ROOT)),
        "render_sample_count": len(source_small),
        "point_unit": point_unit,
        "prediction_color_p95_distance_mm": color_max,
        "trimmed_chamfer_mm": trim_cd_mm,
        "camera_direction": camera_direction.tolist(),
        "axis_scale_strategy": (
            "initial data-bounded crop; prediction and ground-truth robust crops"
        ),
    }
    return pdf_path, metadata


def _build_contact_sheet(
    panel_paths: list[Path],
    metadata: list[dict[str, Any]],
) -> None:
    panel_letters = "abcdefghijklmnopqrstuvwxyz"
    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for row_index, panel_path in enumerate(panel_paths):
        letter = panel_letters[row_index]
        row_pdf_path = PAPER_FIGURE_DIR / f"qualitative_case_{letter}.pdf"
        shutil.copy2(panel_path, row_pdf_path)
        if BMVC_IMAGE_DIR.exists():
            shutil.copy2(row_pdf_path, BMVC_IMAGE_DIR / row_pdf_path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selection = _load_json(args.selection)
    ordered_labels = [
        "easy_success",
        "plausible_but_wrong",
        "translation_heavy_failure",
        "severe_failure",
        "hard_geometry_failure",
    ]
    panel_paths: list[Path] = []
    metadata: list[dict[str, Any]] = []
    config_cache: dict[str, dict[str, Any]] = {}
    adapter_cache: dict[str, Any] = {}
    for label in ordered_labels:
        if label not in selection:
            continue
        path, row = _render_case_panel(
            selection[label],
            FAILURE_CASE_DIR,
            config_cache,
            adapter_cache,
        )
        panel_paths.append(path)
        metadata.append(row)

    _build_contact_sheet(panel_paths, metadata)
    manifest_path = PAPER_FIGURE_DIR / "qualitative_alignment_panels_manifest.json"
    manifest_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "cases": len(metadata),
                "case_pdfs": [
                    str(path.relative_to(REPO_ROOT))
                    for path in sorted(
                        PAPER_FIGURE_DIR.glob("qualitative_case_*.pdf")
                    )
                ],
                "manifest": str(manifest_path.relative_to(REPO_ROOT)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
