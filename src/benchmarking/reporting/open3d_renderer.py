"""Open3D-backed qualitative renderer worker."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("OPEN3D_CPU_RENDERING", "true")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

RENDER_WIDTH = 900
RENDER_HEIGHT = 720
RENDER_FOV_DEG = 32.0
BACKGROUND_RGBA = [1.0, 1.0, 1.0, 1.0]
BACKGROUND_HEX = "#ffffff"
ALIGNMENT_ANIMATION_FRAME_COUNT = 28
TARGET_COLOR = np.array([0.75, 0.79, 0.84], dtype=np.float64)
SOURCE_COLOR = np.array([0.18, 0.44, 0.74], dtype=np.float64)
DEFAULT_CAMERA_DIRECTION = np.array([1.55, 1.18, 0.92], dtype=np.float64)
DEFAULT_CAMERA_DIRECTION = DEFAULT_CAMERA_DIRECTION / np.linalg.norm(
    DEFAULT_CAMERA_DIRECTION
)


def _load_payload(path: str | Path) -> dict[str, object]:
    archive = np.load(Path(path), allow_pickle=False)
    return {
        "sample_id": str(archive["sample_id"].item()),
        "scene_id": str(archive["scene_id"].item()),
        "visible_nn_mean_mm": float(archive["visible_nn_mean_mm"].item()),
        "rre_deg": float(archive["rre_deg"].item()),
        "rte_mm": float(archive["rte_mm"].item()),
        "point_unit": str(archive["point_unit"].item())
        if "point_unit" in archive
        else "m",
        "point_unit_to_mm_scale": float(archive["point_unit_to_mm_scale"].item())
        if "point_unit_to_mm_scale" in archive
        else 1000.0,
        "source_points": archive["source_points"].astype(np.float64, copy=False),
        "target_points": archive["target_points"].astype(np.float64, copy=False),
        "pred_transform": archive["pred_transform"].astype(np.float64, copy=False),
        "pointwise_distance_mm": archive["pointwise_distance_mm"].astype(
            np.float64, copy=False
        )
        if "pointwise_distance_mm" in archive
        else None,
        "static_view_points": archive["static_view_points"].astype(
            np.float64, copy=False
        ),
        "transition_view_points": archive["transition_view_points"].astype(
            np.float64, copy=False
        ),
        "camera_direction": archive["camera_direction"].astype(
            np.float64, copy=False
        )
        if "camera_direction" in archive
        else DEFAULT_CAMERA_DIRECTION.copy(),
        "color_max": float(archive["color_max"].item()),
    }


def _apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (transform[:3, :3] @ points.T).T + transform[:3, 3]


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


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        return DEFAULT_CAMERA_DIRECTION.copy()
    return vector / norm


def _alignment_animation_progress() -> np.ndarray:
    frame_positions = np.linspace(
        0.0,
        1.0,
        num=ALIGNMENT_ANIMATION_FRAME_COUNT,
        dtype=np.float64,
    )
    return 0.5 - 0.5 * np.cos(np.pi * frame_positions)


def _build_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def _fit_camera(
    all_points: np.ndarray,
    camera_direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = ((mins + maxs) / 2.0).astype(np.float32)
    extent = np.maximum(maxs - mins, 1.0e-3)
    radius = max(float(np.linalg.norm(extent)), 0.12)
    camera_direction = _normalize_vector(camera_direction).astype(np.float32)
    eye = center + camera_direction * (radius * 1.75)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(camera_direction, up))) > 0.96:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return center, eye.astype(np.float32), up


def _adaptive_point_size(count: int, base: float) -> float:
    if count <= 32:
        return base * 3.0
    if count <= 128:
        return base * 2.0
    if count <= 512:
        return base * 1.4
    return base


def _render_scene(
    renderer: o3d.visualization.rendering.OffscreenRenderer,
    target_points: np.ndarray,
    source_points: np.ndarray,
    source_colors: np.ndarray,
    camera_state: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    renderer.scene.clear_geometry()
    renderer.scene.set_background(BACKGROUND_RGBA)
    renderer.scene.show_skybox(False)
    renderer.scene.view.set_post_processing(False)

    target_cloud = _build_point_cloud(
        target_points,
        np.repeat(TARGET_COLOR[None, :], len(target_points), axis=0),
    )
    source_cloud = _build_point_cloud(source_points, source_colors)

    target_material = o3d.visualization.rendering.MaterialRecord()
    target_material.shader = "defaultUnlit"
    target_material.point_size = _adaptive_point_size(len(target_points), base=5.0)

    source_material = o3d.visualization.rendering.MaterialRecord()
    source_material.shader = "defaultUnlit"
    source_material.point_size = _adaptive_point_size(len(source_points), base=7.0)

    renderer.scene.add_geometry("target", target_cloud, target_material)
    renderer.scene.add_geometry("source", source_cloud, source_material)

    center, eye, up = camera_state
    renderer.setup_camera(RENDER_FOV_DEG, center, eye, up)
    return np.asarray(renderer.render_to_image())


def _distance_colors(distances_mm: np.ndarray, color_max: float) -> np.ndarray:
    color_max = max(float(color_max), 1.0e-6)
    normalized = np.clip(distances_mm / color_max, 0.0, 1.0)
    return plt.get_cmap("inferno")(normalized)[:, :3]


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


def _compose_two_panel_figure(
    left_image: np.ndarray,
    right_image: np.ndarray,
    payload: dict[str, object],
    right_title: str,
) -> plt.Figure:
    figure = plt.figure(figsize=(14.5, 6.4), dpi=160)
    grid = figure.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.04), wspace=0.04)

    left_axis = figure.add_subplot(grid[0, 0])
    right_axis = figure.add_subplot(grid[0, 1])
    colorbar_axis = figure.add_subplot(grid[0, 2])

    for axis, image, title in (
        (left_axis, left_image, "Initial Pose"),
        (right_axis, right_image, right_title),
    ):
        axis.set_facecolor(BACKGROUND_HEX)
        axis.imshow(image)
        axis.set_title(title, fontsize=15, pad=10)
        axis.axis("off")

    color_max = max(float(payload["color_max"]), 1.0e-6)
    scalar = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=0.0, vmax=color_max),
        cmap="inferno",
    )
    colorbar = figure.colorbar(scalar, cax=colorbar_axis)
    colorbar.set_label("Nearest-target distance (mm)")

    figure.suptitle(
        "Worst Visible-Distance Case\n"
        f"sample_id={payload['sample_id']} | "
        f"visible_nn_mean_mm={float(payload['visible_nn_mean_mm']):.2f} | "
        f"rre_deg={float(payload['rre_deg']):.2f} | "
        f"rte_mm={float(payload['rte_mm']):.2f}",
        fontsize=16,
        y=0.98,
    )
    figure.patch.set_facecolor(BACKGROUND_HEX)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return figure


def render_distance_case(
    payload: dict[str, object],
    output_path: str | Path,
) -> None:
    renderer = o3d.visualization.rendering.OffscreenRenderer(
        RENDER_WIDTH,
        RENDER_HEIGHT,
    )
    try:
        source_points = np.asarray(payload["source_points"], dtype=np.float64)
        target_points = np.asarray(payload["target_points"], dtype=np.float64)
        pred_transform = np.asarray(payload["pred_transform"], dtype=np.float64)
        camera_direction = np.asarray(payload["camera_direction"], dtype=np.float64)

        aligned_source = _apply_transform(source_points, pred_transform)
        pointwise_distance_mm = payload.get("pointwise_distance_mm")
        if pointwise_distance_mm is None:
            pointwise_distance_mm = (
                cKDTree(target_points).query(aligned_source)[0]
                * float(payload["point_unit_to_mm_scale"])
            )
        pointwise_distance_mm = np.asarray(pointwise_distance_mm, dtype=np.float64)
        before_camera_state = _fit_camera(
            np.vstack([source_points, target_points]),
            camera_direction,
        )
        after_camera_state = _fit_camera(
            np.vstack([aligned_source, target_points]),
            camera_direction,
        )

        before_image = _render_scene(
            renderer,
            target_points=target_points,
            source_points=source_points,
            source_colors=np.repeat(SOURCE_COLOR[None, :], len(source_points), axis=0),
            camera_state=before_camera_state,
        )
        after_image = _render_scene(
            renderer,
            target_points=target_points,
            source_points=aligned_source,
            source_colors=_distance_colors(
                pointwise_distance_mm,
                color_max=float(payload["color_max"]),
            ),
            camera_state=after_camera_state,
        )

        figure = _compose_two_panel_figure(
            before_image,
            after_image,
            payload,
            right_title="Predicted Alignment (NN Distance)",
        )
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(
                output_path,
                dpi=160,
                facecolor=figure.get_facecolor(),
            )
        finally:
            plt.close(figure)
    finally:
        del renderer


def render_transition_animation(
    payload: dict[str, object],
    output_path: str | Path,
) -> None:
    renderer = o3d.visualization.rendering.OffscreenRenderer(
        RENDER_WIDTH,
        RENDER_HEIGHT,
    )
    try:
        source_points = np.asarray(payload["source_points"], dtype=np.float64)
        target_points = np.asarray(payload["target_points"], dtype=np.float64)
        pred_transform = np.asarray(payload["pred_transform"], dtype=np.float64)
        transition_view_points = np.asarray(
            payload["transition_view_points"], dtype=np.float64
        )
        camera_direction = np.asarray(payload["camera_direction"], dtype=np.float64)
        color_max = float(payload["color_max"])
        tree = cKDTree(target_points)
        eased_progress = _alignment_animation_progress()
        camera_state = _fit_camera(transition_view_points, camera_direction)

        before_image = _render_scene(
            renderer,
            target_points=target_points,
            source_points=source_points,
            source_colors=np.repeat(SOURCE_COLOR[None, :], len(source_points), axis=0),
            camera_state=camera_state,
        )

        frames: list[np.ndarray] = []

        for progress in eased_progress:
            current_transform = _interpolate_pred_transform(pred_transform, progress)
            current_source = _apply_transform(source_points, current_transform)
            distances_mm = (
                tree.query(current_source)[0]
                * float(payload["point_unit_to_mm_scale"])
            )

            transition_image = _render_scene(
                renderer,
                target_points=target_points,
                source_points=current_source,
                source_colors=_distance_colors(distances_mm, color_max=color_max),
                camera_state=camera_state,
            )
            figure = _compose_two_panel_figure(
                before_image,
                transition_image,
                payload,
                right_title=f"Alignment Transition ({int(round(progress * 100.0))}%)",
            )
            try:
                frames.append(_figure_to_rgb_array(figure))
            finally:
                plt.close(figure)

        _save_gif(frames, output_path, duration_ms=70)
    finally:
        del renderer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("distance", "animation"),
        help="Type of qualitative asset to render.",
    )
    parser.add_argument("--payload", required=True, help="Path to payload .npz file.")
    parser.add_argument("--output", required=True, help="Output asset path.")
    args = parser.parse_args()

    payload = _load_payload(args.payload)
    output_path = Path(args.output)
    if args.mode == "distance":
        render_distance_case(payload, output_path)
    else:
        render_transition_animation(payload, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
