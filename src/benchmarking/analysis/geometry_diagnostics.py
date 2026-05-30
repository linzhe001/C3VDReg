"""Geometry diagnostics for overlap-aware benchmark analysis."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from src.benchmarking.metrics.units import distance_to_millimeters


def _deterministic_sample(points: np.ndarray, sample_count: int) -> np.ndarray:
    if len(points) <= sample_count:
        return points.astype(np.float64, copy=False)
    indices = np.linspace(0, len(points) - 1, num=sample_count, dtype=int)
    return points[indices].astype(np.float64, copy=False)


def _apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (transform[:3, :3] @ points.T).T + transform[:3, 3]


def _resolution_scale(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    tree = cKDTree(points)
    distances = tree.query(points, k=2)[0][:, 1]
    return float(np.median(distances))


def _gt_overlap_mask(
    source_points: np.ndarray,
    target_points: np.ndarray,
    gt_transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gt_source = _apply_transform(source_points, gt_transform)
    target_tree = cKDTree(target_points)
    gt_to_target = target_tree.query(gt_source)[0]
    if gt_to_target.size == 0:
        return np.ones(len(source_points), dtype=bool), gt_to_target

    resolution = max(_resolution_scale(gt_source), _resolution_scale(target_points))
    cutoff = max(resolution * 4.0, 1.0e-6)
    mask = gt_to_target <= cutoff
    if not np.any(mask):
        mask[int(np.argmin(gt_to_target))] = True
    return mask, gt_to_target


def compute_visible_distance_stats(
    source_points: np.ndarray,
    target_points: np.ndarray,
    pred_transform: np.ndarray,
    gt_transform: np.ndarray,
    sample_count: int,
    distance_mode: str,
    point_unit: str = "m",
) -> dict[str, float]:
    """Compute deterministic visible/overlap-aware distance summaries in millimeters."""

    source_points = _deterministic_sample(np.asarray(source_points), sample_count)
    target_points = _deterministic_sample(np.asarray(target_points), sample_count)
    pred_source = _apply_transform(
        source_points, np.asarray(pred_transform, dtype=np.float64)
    )
    overlap_mask, gt_to_target = _gt_overlap_mask(
        source_points,
        target_points,
        np.asarray(gt_transform, dtype=np.float64),
    )

    target_tree = cKDTree(target_points)
    pred_to_target = target_tree.query(pred_source)[0]
    pred_tree = cKDTree(pred_source)
    target_to_pred = pred_tree.query(target_points)[0]
    overlap_active = pred_to_target[overlap_mask]
    visible_cutoff = (
        float(np.percentile(gt_to_target, 95)) if gt_to_target.size else 0.0
    )
    visible_mask = (
        gt_to_target <= visible_cutoff if gt_to_target.size else overlap_mask
    )
    visible_active = pred_to_target[visible_mask]

    if distance_mode == "visible_only":
        active = visible_active
    elif distance_mode == "overlap_only":
        active = overlap_active
    elif distance_mode == "visible_overlap_preferred":
        active = overlap_active if overlap_active.size else visible_active
    else:
        raise ValueError(f"Unsupported distance_mode '{distance_mode}'.")

    if active.size == 0:
        active = pred_to_target

    chamfer = np.concatenate([pred_to_target, target_to_pred])
    trimmed_cutoff = np.percentile(chamfer, 90)
    trimmed = chamfer[chamfer <= trimmed_cutoff]
    overlap_only = overlap_active if overlap_active.size else pred_to_target

    return {
        "visible_nn_mean_mm": distance_to_millimeters(np.mean(active), point_unit),
        "visible_nn_median_mm": distance_to_millimeters(np.median(active), point_unit),
        "visible_nn_p90_mm": distance_to_millimeters(
            np.percentile(active, 90), point_unit
        ),
        "trimmed_chamfer_mm": distance_to_millimeters(np.mean(trimmed), point_unit),
        "overlap_only_distance_mm": distance_to_millimeters(
            np.mean(overlap_only), point_unit
        ),
    }


def build_distance_heatmap_manifest(
    records: list[dict[str, object]],
    topk: int,
) -> dict[str, object]:
    ranked = sorted(
        records,
        key=lambda item: float(
            item.get("visible_nn_mean_mm") or item.get("rre_deg") or 0.0
        ),
        reverse=True,
    )
    return {
        "topk": [
            {
                "sample_id": record.get("sample_id"),
                "scene_id": record.get("scene_id"),
                "visible_nn_mean_mm": record.get("visible_nn_mean_mm"),
                "failure_tags": record.get("failure_tags", []),
            }
            for record in ranked[:topk]
        ]
    }
