"""Static figure exporters for benchmark reports."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from src.benchmarking.analysis.curve_metrics import (
    build_success_latency_pareto,
    compute_multithreshold_recall,
)
from src.benchmarking.metrics.units import (
    canonicalize_point_unit,
    point_unit_to_mm_scale,
)

LOGGER = logging.getLogger(__name__)

QUALITATIVE_BACKGROUND_RGBA = (1.0, 1.0, 1.0, 1.0)
QUALITATIVE_BACKGROUND_HEX = "#ffffff"
ALIGNMENT_ANIMATION_FRAME_COUNT = 28
MAX_ERROR_VIEW_CANDIDATE_COUNT = 96
MAX_ERROR_VIEW_PROJECTION_RESOLUTION = 196
DEFAULT_CAMERA_DIRECTION = np.array([1.55, 1.18, 0.92], dtype=np.float64)
DEFAULT_CAMERA_DIRECTION = DEFAULT_CAMERA_DIRECTION / np.linalg.norm(
    DEFAULT_CAMERA_DIRECTION
)


def export_curve_figures(
    records: list[dict[str, object]],
    output_dir: str | Path,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    curves_dir = output_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    recall_rows = compute_multithreshold_recall(records)
    pareto_rows = build_success_latency_pareto(records)

    rr_path = curves_dir / "rr_multithreshold.png"
    plt.figure(figsize=(6, 4))
    plt.plot(
        [row["rotation_deg"] for row in recall_rows],
        [row["recall"] for row in recall_rows],
        marker="o",
    )
    plt.xlabel("Rotation Threshold (deg)")
    plt.ylabel("Recall")
    plt.title("Multi-threshold Registration Recall")
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    plt.savefig(rr_path)
    plt.close()

    pareto_path = curves_dir / "success_latency_pareto.png"
    plt.figure(figsize=(6, 4))
    plt.scatter(
        [row["latency_ms"] for row in pareto_rows],
        [row["success_rate"] for row in pareto_rows],
    )
    for row in pareto_rows:
        plt.annotate(row["label"], (row["latency_ms"], row["success_rate"]))
    plt.xlabel("Latency (ms)")
    plt.ylabel("RR@5deg_5mm")
    plt.title("Success-Latency Pareto")
    plt.tight_layout()
    plt.savefig(pareto_path)
    plt.close()

    return {
        "rr_multithreshold": str(rr_path),
        "success_latency_pareto": str(pareto_path),
    }


def export_geometry_figures(
    records: list[dict[str, object]],
    output_dir: str | Path,
    render_case: dict[str, object] | None = None,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    geometry_dir = output_dir / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)

    distances = [
        float(record["visible_nn_mean_mm"])
        for record in records
        if record.get("visible_nn_mean_mm") is not None
    ]
    if not distances:
        distances = [0.0]

    hist_path = geometry_dir / "visible_distance_hist.png"
    plt.figure(figsize=(6, 4))
    plt.hist(distances, bins=min(20, len(distances)))
    plt.xlabel("Visible NN Mean Distance (mm)")
    plt.ylabel("Count")
    plt.title("Visible Distance Histogram")
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()

    cdf_path = geometry_dir / "visible_distance_cdf.png"
    ordered = sorted(distances)
    cdf = [(index + 1) / len(ordered) for index in range(len(ordered))]
    plt.figure(figsize=(6, 4))
    plt.plot(ordered, cdf)
    plt.xlabel("Visible NN Mean Distance (mm)")
    plt.ylabel("CDF")
    plt.title("Visible Distance CDF")
    plt.tight_layout()
    plt.savefig(cdf_path)
    plt.close()

    outputs = {
        "visible_distance_hist": str(hist_path),
        "visible_distance_cdf": str(cdf_path),
    }
    if render_case is not None:
        outputs.update(export_pointcloud_distance_render(render_case, output_dir))
        outputs.update(
            export_pointcloud_alignment_transition_animation(
                render_case,
                output_dir,
            )
        )
    return outputs


def _apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (transform[:3, :3] @ points.T).T + transform[:3, 3]


def _deterministic_sample(points: np.ndarray, sample_count: int) -> np.ndarray:
    if len(points) <= sample_count:
        return points.astype(np.float64, copy=False)
    indices = np.linspace(0, len(points) - 1, num=sample_count, dtype=int)
    return points[indices].astype(np.float64, copy=False)


def _set_equal_axes(ax, points: np.ndarray, padding: float = 0.08) -> None:
    finite = np.asarray(points, dtype=np.float64)
    finite = finite[np.all(np.isfinite(finite), axis=1)]
    if len(finite) == 0:
        return
    if len(finite) >= 16:
        mins = np.percentile(finite, 1.0, axis=0)
        maxs = np.percentile(finite, 99.0, axis=0)
    else:
        mins = finite.min(axis=0)
        maxs = finite.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 1.0e-6)
    radius *= 1.0 + padding
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _style_3d_axis(ax) -> None:
    ax.set_facecolor(QUALITATIVE_BACKGROUND_HEX)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_axis_off()
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(QUALITATIVE_BACKGROUND_RGBA)


def _sanitize_sample_id(sample_id: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id)).strip("_") or "sample"


def _alignment_animation_progress() -> np.ndarray:
    frame_positions = np.linspace(
        0.0,
        1.0,
        num=ALIGNMENT_ANIMATION_FRAME_COUNT,
        dtype=np.float64,
    )
    return 0.5 - 0.5 * np.cos(np.pi * frame_positions)


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
    right = np.cross(up_reference, camera_direction)
    right = _normalize_vector(right)
    up = np.cross(camera_direction, right)
    up = _normalize_vector(up)
    return right, up


def _fibonacci_sphere_directions(count: int) -> np.ndarray:
    if count <= 1:
        return DEFAULT_CAMERA_DIRECTION[None, :].copy()
    indices = np.arange(count, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - (2.0 * indices + 1.0) / count
    radius = np.sqrt(np.clip(1.0 - y * y, 0.0, 1.0))
    theta = phi * indices
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    return np.stack([x, y, z], axis=1)


def _choose_max_error_camera_direction(
    points: np.ndarray,
    distances_mm: np.ndarray,
) -> np.ndarray:
    if len(points) == 0 or len(distances_mm) == 0:
        return DEFAULT_CAMERA_DIRECTION.copy()

    finite_mask = np.isfinite(distances_mm)
    if not np.any(finite_mask):
        return DEFAULT_CAMERA_DIRECTION.copy()

    points = points[finite_mask]
    distances_mm = distances_mm[finite_mask]
    center = np.mean(points, axis=0)
    candidates = _fibonacci_sphere_directions(MAX_ERROR_VIEW_CANDIDATE_COUNT)

    best_score = -np.inf
    best_direction = DEFAULT_CAMERA_DIRECTION.copy()
    for candidate in candidates:
        right, up = _camera_basis_from_direction(candidate)
        rel = points - center
        proj_x = rel @ right
        proj_y = rel @ up
        depth = rel @ candidate

        x_span = max(float(proj_x.max() - proj_x.min()), 1.0e-6)
        y_span = max(float(proj_y.max() - proj_y.min()), 1.0e-6)
        grid_size = MAX_ERROR_VIEW_PROJECTION_RESOLUTION - 1
        x_idx = np.clip(
            ((proj_x - proj_x.min()) / x_span * grid_size).astype(np.int32),
            0,
            grid_size,
        )
        y_idx = np.clip(
            ((proj_y - proj_y.min()) / y_span * grid_size).astype(np.int32),
            0,
            grid_size,
        )
        linear = y_idx * MAX_ERROR_VIEW_PROJECTION_RESOLUTION + x_idx
        order = np.argsort(depth)[::-1]
        visible_mask = np.ones(len(order), dtype=bool)
        visible_mask[1:] = linear[order][1:] != linear[order][:-1]
        visible_indices = order[visible_mask]
        visible_distances = distances_mm[visible_indices]
        if len(visible_distances) == 0:
            continue

        topk = max(1, int(np.ceil(len(visible_distances) * 0.1)))
        topk_mean = float(
            np.mean(np.partition(visible_distances, -topk)[-topk:])
        )
        score = topk_mean
        if score > best_score:
            best_score = score
            best_direction = candidate

    return _normalize_vector(best_direction)


def _camera_direction_to_mpl_view(camera_direction: np.ndarray) -> tuple[float, float]:
    camera_direction = _normalize_vector(camera_direction)
    elev = float(np.degrees(np.arcsin(np.clip(camera_direction[2], -1.0, 1.0))))
    azim = float(np.degrees(np.arctan2(camera_direction[1], camera_direction[0])))
    return elev, azim


def _prepare_distance_render_payload(
    render_case: dict[str, object],
) -> dict[str, object]:
    point_unit = canonicalize_point_unit(str(render_case.get("point_unit", "m")))
    mm_scale = point_unit_to_mm_scale(point_unit)
    source_points = _deterministic_sample(
        np.asarray(render_case["source_points"], dtype=np.float64),
        sample_count=2048,
    )
    target_points = _deterministic_sample(
        np.asarray(render_case["target_points"], dtype=np.float64),
        sample_count=2048,
    )
    pred_transform = np.asarray(render_case["pred_transform"], dtype=np.float64)
    aligned_source = _apply_transform(source_points, pred_transform)

    tree = cKDTree(target_points)
    initial_distance_mm = tree.query(source_points)[0] * mm_scale
    pointwise_distance_mm = tree.query(aligned_source)[0] * mm_scale
    static_view_points = np.vstack([source_points, target_points, aligned_source])
    camera_direction = _choose_max_error_camera_direction(
        aligned_source,
        pointwise_distance_mm,
    )
    transition_view_sets = [target_points]
    for progress in _alignment_animation_progress():
        transition_view_sets.append(
            _apply_transform(
                source_points,
                _interpolate_pred_transform(pred_transform, progress),
            )
        )
    transition_view_points = np.vstack(transition_view_sets)
    finite_distances = np.concatenate(
        [
            initial_distance_mm[np.isfinite(initial_distance_mm)],
            pointwise_distance_mm[np.isfinite(pointwise_distance_mm)],
        ]
    )
    color_max = max(
        float(np.percentile(finite_distances, 95.0)) if finite_distances.size else 0.0,
        float(render_case.get("visible_nn_mean_mm", 0.0)) * 3.0,
        1.0,
    )

    return {
        "sample_id": render_case["sample_id"],
        "scene_id": render_case["scene_id"],
        "visible_nn_mean_mm": render_case["visible_nn_mean_mm"],
        "rre_deg": render_case["rre_deg"],
        "rte_mm": render_case["rte_mm"],
        "point_unit": point_unit,
        "point_unit_to_mm_scale": mm_scale,
        "source_points": source_points,
        "target_points": target_points,
        "pred_transform": pred_transform,
        "aligned_source": aligned_source,
        "initial_distance_mm": initial_distance_mm,
        "pointwise_distance_mm": pointwise_distance_mm,
        "static_view_points": static_view_points,
        "transition_view_points": transition_view_points,
        "camera_direction": camera_direction,
        "camera_strategy": "max_visible_error",
        "color_max": color_max,
    }


def _build_distance_render_figure(payload: dict[str, object]) -> plt.Figure:
    source_points = np.asarray(payload["source_points"], dtype=np.float64)
    target_points = np.asarray(payload["target_points"], dtype=np.float64)
    aligned_source = np.asarray(payload["aligned_source"], dtype=np.float64)
    pointwise_distance_mm = np.asarray(
        payload["pointwise_distance_mm"], dtype=np.float64
    )
    camera_direction = np.asarray(payload["camera_direction"], dtype=np.float64)
    elev, azim = _camera_direction_to_mpl_view(camera_direction)
    before_view_points = np.vstack([source_points, target_points])
    after_view_points = np.vstack([aligned_source, target_points])

    figure = plt.figure(figsize=(12, 5))
    figure.patch.set_facecolor(QUALITATIVE_BACKGROUND_HEX)

    ax_before = figure.add_subplot(1, 2, 1, projection="3d")
    _style_3d_axis(ax_before)
    ax_before.scatter(
        target_points[:, 0],
        target_points[:, 1],
        target_points[:, 2],
        s=4,
        c="#9aa0a6",
        alpha=0.35,
        label="target",
    )
    ax_before.scatter(
        source_points[:, 0],
        source_points[:, 1],
        source_points[:, 2],
        s=4,
        c="#1f77b4",
        alpha=0.6,
        label="source",
    )
    ax_before.set_title("Before Alignment")
    legend = ax_before.legend(loc="upper right")
    if legend is not None:
        legend.get_frame().set_facecolor(QUALITATIVE_BACKGROUND_HEX)
        legend.get_frame().set_edgecolor("#c8d1db")
    _set_equal_axes(ax_before, before_view_points)

    ax_after = figure.add_subplot(1, 2, 2, projection="3d")
    _style_3d_axis(ax_after)
    ax_after.scatter(
        target_points[:, 0],
        target_points[:, 1],
        target_points[:, 2],
        s=4,
        c="#d0d4d9",
        alpha=0.25,
    )
    rendered = ax_after.scatter(
        aligned_source[:, 0],
        aligned_source[:, 1],
        aligned_source[:, 2],
        s=5,
        c=pointwise_distance_mm,
        cmap="inferno",
        vmin=0.0,
        vmax=float(payload["color_max"]),
        alpha=0.9,
    )
    ax_after.set_title("Predicted Alignment (NN Distance)")
    _set_equal_axes(ax_after, after_view_points)
    colorbar = figure.colorbar(rendered, ax=ax_after, fraction=0.046, pad=0.04)
    colorbar.set_label("Nearest-target distance (mm)")

    for axis in (ax_before, ax_after):
        axis.view_init(elev=elev, azim=azim)

    figure.suptitle(
        "Worst Visible-Distance Case\n"
        f"sample_id={payload['sample_id']} | "
        f"visible_nn_mean_mm={float(payload['visible_nn_mean_mm']):.2f} | "
        f"rre_deg={float(payload['rre_deg']):.2f} | "
        f"rte_mm={float(payload['rte_mm']):.2f} | "
        f"view={payload['camera_strategy']}",
        fontsize=11,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    return figure


def _interpolate_pred_transform(
    pred_transform: np.ndarray,
    progress: float,
) -> np.ndarray:
    progress = float(np.clip(progress, 0.0, 1.0))
    key_times = np.array([0.0, 1.0], dtype=np.float64)
    key_rotations = Rotation.from_matrix(
        np.stack([np.eye(3, dtype=np.float64), pred_transform[:3, :3]], axis=0)
    )
    slerp = Slerp(key_times, key_rotations)
    rotation = slerp([progress]).as_matrix()[0]
    translation = pred_transform[:3, 3] * progress

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _build_transition_render_figure(
    payload: dict[str, object],
    progress: float,
) -> plt.Figure:
    source_points = np.asarray(payload["source_points"], dtype=np.float64)
    target_points = np.asarray(payload["target_points"], dtype=np.float64)
    pred_transform = np.asarray(payload["pred_transform"], dtype=np.float64)
    camera_direction = np.asarray(payload["camera_direction"], dtype=np.float64)
    elev, azim = _camera_direction_to_mpl_view(camera_direction)

    current_transform = _interpolate_pred_transform(pred_transform, progress)
    current_source = _apply_transform(source_points, current_transform)
    pointwise_distance_mm = (
        cKDTree(target_points).query(current_source)[0]
        * float(payload["point_unit_to_mm_scale"])
    )
    before_view_points = np.vstack([source_points, target_points])
    current_view_points = np.vstack([current_source, target_points])

    figure = plt.figure(figsize=(12, 5))
    figure.patch.set_facecolor(QUALITATIVE_BACKGROUND_HEX)

    ax_before = figure.add_subplot(1, 2, 1, projection="3d")
    _style_3d_axis(ax_before)
    ax_before.scatter(
        target_points[:, 0],
        target_points[:, 1],
        target_points[:, 2],
        s=4,
        c="#9aa0a6",
        alpha=0.35,
        label="target",
    )
    ax_before.scatter(
        source_points[:, 0],
        source_points[:, 1],
        source_points[:, 2],
        s=4,
        c="#1f77b4",
        alpha=0.6,
        label="source",
    )
    ax_before.set_title("Before Alignment")
    legend = ax_before.legend(loc="upper right")
    if legend is not None:
        legend.get_frame().set_facecolor(QUALITATIVE_BACKGROUND_HEX)
        legend.get_frame().set_edgecolor("#c8d1db")
    _set_equal_axes(ax_before, before_view_points)

    ax_transition = figure.add_subplot(1, 2, 2, projection="3d")
    _style_3d_axis(ax_transition)
    ax_transition.scatter(
        target_points[:, 0],
        target_points[:, 1],
        target_points[:, 2],
        s=4,
        c="#d0d4d9",
        alpha=0.25,
    )
    rendered = ax_transition.scatter(
        current_source[:, 0],
        current_source[:, 1],
        current_source[:, 2],
        s=5,
        c=pointwise_distance_mm,
        cmap="inferno",
        vmin=0.0,
        vmax=float(payload["color_max"]),
        alpha=0.9,
    )
    ax_transition.set_title(
        f"Alignment Transition ({int(round(progress * 100.0))}%)"
    )
    _set_equal_axes(ax_transition, current_view_points)
    colorbar = figure.colorbar(rendered, ax=ax_transition, fraction=0.046, pad=0.04)
    colorbar.set_label("Nearest-target distance (mm)")

    for axis in (ax_before, ax_transition):
        axis.view_init(elev=elev, azim=azim)

    figure.suptitle(
        "Before-to-Predicted Alignment Transition\n"
        f"sample_id={payload['sample_id']} | "
        f"visible_nn_mean_mm={float(payload['visible_nn_mean_mm']):.2f} | "
        f"rre_deg={float(payload['rre_deg']):.2f} | "
        f"rte_mm={float(payload['rte_mm']):.2f} | "
        f"view={payload['camera_strategy']}",
        fontsize=11,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    return figure


def _figure_to_rgb_array(figure: plt.Figure) -> np.ndarray:
    figure.canvas.draw()
    return np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()


def _save_gif(
    frames: list[np.ndarray],
    output_path: str | Path,
    duration_ms: int,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [
        Image.fromarray(frame.astype(np.uint8))
        for frame in frames
    ]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )


def _write_open3d_payload(
    payload: dict[str, object],
    path: str | Path,
) -> None:
    output_path = Path(path)
    np.savez_compressed(
        output_path,
        sample_id=np.asarray(str(payload["sample_id"])),
        scene_id=np.asarray(str(payload["scene_id"])),
        visible_nn_mean_mm=np.asarray(float(payload["visible_nn_mean_mm"])),
        rre_deg=np.asarray(float(payload["rre_deg"])),
        rte_mm=np.asarray(float(payload["rte_mm"])),
        point_unit=np.asarray(str(payload["point_unit"])),
        point_unit_to_mm_scale=np.asarray(float(payload["point_unit_to_mm_scale"])),
        source_points=np.asarray(payload["source_points"], dtype=np.float64),
        target_points=np.asarray(payload["target_points"], dtype=np.float64),
        pred_transform=np.asarray(payload["pred_transform"], dtype=np.float64),
        initial_distance_mm=np.asarray(
            payload["initial_distance_mm"], dtype=np.float64
        ),
        pointwise_distance_mm=np.asarray(
            payload["pointwise_distance_mm"], dtype=np.float64
        ),
        static_view_points=np.asarray(payload["static_view_points"], dtype=np.float64),
        transition_view_points=np.asarray(
            payload["transition_view_points"], dtype=np.float64
        ),
        camera_direction=np.asarray(payload["camera_direction"], dtype=np.float64),
        color_max=np.asarray(float(payload["color_max"])),
    )


def _run_open3d_render_worker(
    payload: dict[str, object],
    output_path: str | Path,
    mode: str,
) -> bool:
    worker_path = Path(__file__).with_name("open3d_renderer.py")
    if not worker_path.exists():
        return False

    env = os.environ.copy()
    env.setdefault("XDG_CACHE_HOME", "/tmp")
    env.setdefault("EGL_PLATFORM", "surfaceless")
    env.setdefault("OPEN3D_CPU_RENDERING", "true")
    dri_path = Path("/usr/lib/x86_64-linux-gnu/dri")
    system_libstdcpp = Path("/lib/x86_64-linux-gnu/libstdc++.so.6")
    if dri_path.exists():
        env.setdefault("LIBGL_DRIVERS_PATH", str(dri_path))
        env.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "swrast")
    if system_libstdcpp.exists() and not env.get("LD_PRELOAD"):
        env["LD_PRELOAD"] = str(system_libstdcpp)

    with TemporaryDirectory(prefix="open3d_render_") as temp_dir:
        payload_path = Path(temp_dir) / "payload.npz"
        _write_open3d_payload(payload, payload_path)
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(worker_path),
                    mode,
                    "--payload",
                    str(payload_path),
                    "--output",
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
                timeout=180,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            LOGGER.warning("Open3D qualitative render failed: %s", exc)
            return False

    if completed.returncode == 0 and Path(output_path).exists():
        return True

    log_excerpt = (completed.stderr or completed.stdout).strip()
    if log_excerpt:
        LOGGER.warning("Open3D qualitative render fallback: %s", log_excerpt)
    return False


def export_pointcloud_distance_render(
    render_case: dict[str, object],
    output_dir: str | Path,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    qualitative_dir = output_dir / "qualitative" / "distance_render"
    qualitative_dir.mkdir(parents=True, exist_ok=True)

    payload = _prepare_distance_render_payload(render_case)

    sample_id = _sanitize_sample_id(payload["sample_id"])
    image_path = qualitative_dir / f"{sample_id}_worst_visible_distance.png"
    metadata_path = qualitative_dir / f"{sample_id}_worst_visible_distance.json"

    render_backend = "open3d"
    if not _run_open3d_render_worker(payload, image_path, mode="distance"):
        render_backend = "matplotlib"
        figure = _build_distance_render_figure(payload)
        figure.savefig(
            image_path,
            dpi=180,
            facecolor=figure.get_facecolor(),
        )
        plt.close(figure)

    metadata = {
        "sample_id": payload["sample_id"],
        "scene_id": payload["scene_id"],
        "visible_nn_mean_mm": payload["visible_nn_mean_mm"],
        "rre_deg": payload["rre_deg"],
        "rte_mm": payload["rte_mm"],
        "point_unit": payload["point_unit"],
        "render_backend": render_backend,
        "camera_view_strategy": payload["camera_strategy"],
        "color_max_mm": payload["color_max"],
        "distance_render_path": str(image_path),
    }
    metadata_path.write_text(
        json_dumps(metadata),
        encoding="utf-8",
    )
    return {
        "worst_visible_distance_render": str(image_path),
    }


def export_pointcloud_alignment_transition_animation(
    render_case: dict[str, object],
    output_dir: str | Path,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    qualitative_dir = output_dir / "qualitative" / "distance_render"
    qualitative_dir.mkdir(parents=True, exist_ok=True)

    payload = _prepare_distance_render_payload(render_case)
    sample_id = _sanitize_sample_id(payload["sample_id"])
    gif_path = qualitative_dir / f"{sample_id}_alignment_transition.gif"

    if _run_open3d_render_worker(payload, gif_path, mode="animation"):
        return {
            "alignment_transition": str(gif_path),
        }

    frames: list[np.ndarray] = []
    for progress in _alignment_animation_progress():
        figure = _build_transition_render_figure(
            payload,
            progress=float(progress),
        )
        frames.append(_figure_to_rgb_array(figure))
        plt.close(figure)

    _save_gif(frames, gif_path, duration_ms=80)
    return {
        "alignment_transition": str(gif_path),
    }


def json_dumps(payload: dict[str, object]) -> str:
    import json

    return json.dumps(payload, indent=2) + "\n"
