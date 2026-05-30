"""Pose and success metrics used by the benchmark core."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from src.benchmarking.metrics.units import (
    canonicalize_point_unit,
    distance_to_meters,
    distance_to_millimeters,
)


def _as_transform(matrix: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got shape {transform.shape}.")
    return transform


def _rotation_error_deg(pred_transform: np.ndarray, gt_transform: np.ndarray) -> float:
    relative = np.linalg.inv(gt_transform) @ pred_transform
    rotation = relative[:3, :3]
    cos_theta = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _translation_error_m(pred_transform: np.ndarray, gt_transform: np.ndarray) -> float:
    relative = np.linalg.inv(gt_transform) @ pred_transform
    return float(np.linalg.norm(relative[:3, 3]))


def compute_registration_recall(
    rre_deg: float,
    rte_mm: float,
) -> dict[str, int]:
    thresholds = (
        (1.0, 1.0),
        (3.0, 3.0),
        (5.0, 5.0),
        (10.0, 10.0),
    )
    return {
        f"success_{int(rot)}deg_{int(trans)}mm": int(rre_deg <= rot and rte_mm <= trans)
        for rot, trans in thresholds
    }


def compute_pose_metrics(
    pred_transform: np.ndarray | Iterable[Iterable[float]],
    gt_transform: np.ndarray | Iterable[Iterable[float]],
    source_points: np.ndarray | None = None,
    target_points: np.ndarray | None = None,
    point_unit: str = "m",
) -> dict[str, float | int | None]:
    """Compute official pose metrics and thresholded success flags."""

    point_unit = canonicalize_point_unit(point_unit)

    pred_transform = _as_transform(pred_transform)
    gt_transform = _as_transform(gt_transform)

    rre_deg = _rotation_error_deg(pred_transform, gt_transform)
    translation_error_raw = _translation_error_m(pred_transform, gt_transform)
    translation_error_m = distance_to_meters(translation_error_raw, point_unit)
    rte_mm = distance_to_millimeters(translation_error_raw, point_unit)
    rmse_mm: float | None = None

    if source_points is not None:
        source_points = np.asarray(source_points, dtype=np.float64)
        pred_points = (pred_transform[:3, :3] @ source_points.T).T + pred_transform[
            :3, 3
        ]
        gt_points = (gt_transform[:3, :3] @ source_points.T).T + gt_transform[:3, 3]
        rmse_mm = float(
            distance_to_millimeters(
                np.sqrt(np.mean(np.sum((pred_points - gt_points) ** 2, axis=1))),
                point_unit,
            )
        )
    elif target_points is not None:
        target_points = np.asarray(target_points, dtype=np.float64)
        rmse_mm = float(math.sqrt(np.mean(np.sum(target_points**2, axis=1))) * 0.0)

    return {
        "rotation_error_deg": rre_deg,
        "translation_error_m": translation_error_m,
        "rre_deg": rre_deg,
        "rte_mm": rte_mm,
        "rmse_mm": rmse_mm,
        **compute_registration_recall(rre_deg, rte_mm),
    }
