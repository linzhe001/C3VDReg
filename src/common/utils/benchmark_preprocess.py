"""Shared point-cloud sampling and perturbation helpers for benchmark train/eval."""

from __future__ import annotations

import contextlib
from typing import Iterator, Literal

import numpy as np
from scipy.spatial.transform import Rotation

from .sampling import random_resample, voxel_down_sample

SamplingMode = Literal["none", "random", "voxel", "fps"]
NoiseApplyMode = Literal["source", "both"]
NormalizeMode = Literal["none", "unit_cube", "joint"]


@contextlib.contextmanager
def seeded_numpy(seed: int | None) -> Iterator[None]:
    if seed is None:
        yield
        return
    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        yield
    finally:
        np.random.set_state(state)


def farthest_point_sample(points: np.ndarray, num_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if num_points >= len(points):
        return random_resample(points, num_points)

    sampled = np.empty((num_points, 3), dtype=np.float32)
    centroid = points.mean(axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    farthest = int(np.argmax(distances))
    min_dist = np.full(len(points), np.inf, dtype=np.float64)

    for index in range(num_points):
        sampled[index] = points[farthest]
        delta = points - points[farthest]
        min_dist = np.minimum(min_dist, np.sum(delta * delta, axis=1))
        farthest = int(np.argmax(min_dist))

    return sampled


def sample_point_cloud(
    points: np.ndarray,
    sampling: SamplingMode,
    num_points: int | None,
    seed: int | None = None,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if sampling == "none" or num_points is None:
        return points.astype(np.float32, copy=False)

    with seeded_numpy(seed):
        if sampling == "random":
            return random_resample(points, int(num_points))
        if sampling == "voxel":
            return voxel_down_sample(points, int(num_points))
        if sampling == "fps":
            return farthest_point_sample(points, int(num_points))
    raise ValueError(f"Unsupported sampling mode '{sampling}'.")


def normalization_transform(center: np.ndarray, scale: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] /= scale
    transform[:3, 3] = -np.asarray(center, dtype=np.float64) / scale
    return transform


def unit_cube_normalize(
    points: np.ndarray,
) -> tuple[np.ndarray, dict[str, object], np.ndarray]:
    """Apply PointNetLK-style unit-cube normalization to one point cloud.

    The legacy baseline path first scales by the largest bbox side length and
    then zero-centers the result by subtracting the point-cloud mean. This is
    equivalent to `(points - mean(points)) / max_bbox_range`.
    """

    points = np.asarray(points, dtype=np.float32)
    min_bounds = points.min(axis=0)
    max_bounds = points.max(axis=0)
    center = points.mean(axis=0)
    scale = float(np.max(max_bounds - min_bounds))
    if scale < 1e-8:
        scale = 1.0

    transform = normalization_transform(center=center, scale=scale)
    normalized = apply_transform_matrix(points, transform)
    return (
        normalized,
        {
            "normalize_center": center.tolist(),
            "normalize_scale": scale,
        },
        transform,
    )


def joint_bbox_normalize_pair(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], np.ndarray]:
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    all_points = np.vstack([source, target])
    min_bounds = all_points.min(axis=0)
    max_bounds = all_points.max(axis=0)
    center = (min_bounds + max_bounds) / 2.0
    scale = float(np.max(max_bounds - min_bounds))
    if scale < 1e-8:
        scale = 1.0
    return (
        ((source - center) / scale).astype(np.float32, copy=False),
        ((target - center) / scale).astype(np.float32, copy=False),
        {
            "normalize_center": center.tolist(),
            "normalize_scale": scale,
        },
        normalization_transform(center, scale),
    )


def normalize_point_cloud_pair(
    source: np.ndarray,
    target: np.ndarray,
    normalize_mode: NormalizeMode,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], np.ndarray, np.ndarray]:
    """Normalize a source/target pair while preserving a rigid pose contract.

    For `unit_cube`, the target/template cloud defines the PointNetLK-style
    unit-cube transform and the same transform is applied to both clouds. This
    mirrors ModelNet tracking baselines where one normalized template is
    transformed into the source, and avoids introducing a scale ratio between
    source and target.
    """

    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    if normalize_mode == "none":
        identity = np.eye(4, dtype=np.float64)
        return (
            source.copy(),
            target.copy(),
            {
                "normalize_mode": "none",
                "source_normalization": None,
                "target_normalization": None,
            },
            identity,
            identity,
        )

    if normalize_mode == "joint":
        source_norm, target_norm, shared_meta, shared_transform = (
            joint_bbox_normalize_pair(source, target)
        )
        return (
            source_norm,
            target_norm,
            {
                "normalize_mode": "joint",
                "source_normalization": dict(shared_meta),
                "target_normalization": dict(shared_meta),
            },
            shared_transform,
            shared_transform,
        )

    if normalize_mode == "unit_cube":
        target_norm, target_meta, target_transform = unit_cube_normalize(target)
        source_norm = apply_transform_matrix(source, target_transform)
        return (
            source_norm,
            target_norm,
            {
                "normalize_mode": "unit_cube",
                "unit_cube_reference": "target",
                "source_normalization": dict(target_meta),
                "target_normalization": dict(target_meta),
            },
            target_transform,
            target_transform,
        )

    raise ValueError(f"Unsupported normalize_mode '{normalize_mode}'.")


def apply_transform_matrix(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    transform = np.asarray(transform, dtype=np.float64)
    return (
        (transform[:3, :3] @ points.astype(np.float64).T).T + transform[:3, 3]
    ).astype(np.float32, copy=False)


def recover_raw_transform(
    normalized_transform: np.ndarray,
    source_norm_transform: np.ndarray,
    target_norm_transform: np.ndarray,
) -> np.ndarray:
    """Lift a normalized-space rigid transform back to raw point coordinates."""

    normalized_transform = np.asarray(normalized_transform, dtype=np.float64)
    source_norm_transform = np.asarray(source_norm_transform, dtype=np.float64)
    target_norm_transform = np.asarray(target_norm_transform, dtype=np.float64)
    return (
        np.linalg.inv(target_norm_transform)
        @ normalized_transform
        @ source_norm_transform
    )


def sample_rigid_transform(
    rotation_deg: float,
    translation_m: float,
    *,
    min_rotation_deg: float = 0.0,
    min_translation_m: float = 0.0,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng(seed)
    transform = np.eye(4, dtype=np.float64)
    if min_rotation_deg < 0.0 or min_translation_m < 0.0:
        raise ValueError("Minimum perturbation magnitudes must be non-negative.")
    if min_rotation_deg > rotation_deg:
        raise ValueError("min_rotation_deg must be <= rotation_deg.")
    if min_translation_m > translation_m:
        raise ValueError("min_translation_m must be <= translation_m.")
    if rotation_deg > 0.0:
        axis = rng.normal(size=3)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-8:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            axis = axis / axis_norm
        if min_rotation_deg > 0.0:
            angle_abs = float(
                rng.uniform(np.deg2rad(min_rotation_deg), np.deg2rad(rotation_deg))
            )
            angle_rad = angle_abs * float(rng.choice([-1.0, 1.0]))
        else:
            angle_rad = float(
                rng.uniform(-np.deg2rad(rotation_deg), np.deg2rad(rotation_deg))
            )
        transform[:3, :3] = Rotation.from_rotvec(axis * angle_rad).as_matrix()
    if translation_m > 0.0:
        if min_translation_m > 0.0:
            direction = rng.normal(size=3)
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm < 1e-8:
                direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            else:
                direction = direction / direction_norm
            magnitude = float(rng.uniform(min_translation_m, translation_m))
            transform[:3, 3] = direction * magnitude
        else:
            transform[:3, 3] = rng.uniform(-translation_m, translation_m, size=3)
    return transform


def sample_gaussian_noise(
    shape: tuple[int, ...],
    sigma: float,
    clip: float,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=shape)
    if clip > 0.0:
        noise = np.clip(noise, -clip, clip)
    return noise.astype(np.float32)


def apply_pair_perturbation(
    source: np.ndarray,
    target: np.ndarray,
    *,
    rotation_deg: float,
    translation_m: float,
    min_rotation_deg: float = 0.0,
    min_translation_m: float = 0.0,
    noise_sigma: float = 0.0,
    noise_clip: float = 0.0,
    apply_noise_to: NoiseApplyMode = "source",
    gt_transform: np.ndarray | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, object]]:
    rng = rng or np.random.default_rng(seed)
    rigid_transform = sample_rigid_transform(
        rotation_deg=rotation_deg,
        translation_m=translation_m,
        min_rotation_deg=min_rotation_deg,
        min_translation_m=min_translation_m,
        rng=rng,
    )
    source_perturbed = apply_transform_matrix(source, rigid_transform)
    target_perturbed = np.asarray(target, dtype=np.float32).copy()

    if noise_sigma > 0.0:
        source_perturbed = source_perturbed + sample_gaussian_noise(
            source_perturbed.shape,
            sigma=noise_sigma,
            clip=noise_clip,
            rng=rng,
        )
        if apply_noise_to == "both":
            target_perturbed = target_perturbed + sample_gaussian_noise(
                target_perturbed.shape,
                sigma=noise_sigma,
                clip=noise_clip,
                rng=rng,
            )

    updated_gt = None
    if gt_transform is not None:
        updated_gt = np.asarray(gt_transform, dtype=np.float64) @ np.linalg.inv(
            rigid_transform
        )

    metadata = {
        "rotation_deg": float(rotation_deg),
        "translation_m": float(translation_m),
        "min_rotation_deg": float(min_rotation_deg),
        "min_translation_m": float(min_translation_m),
        "sampled_rotation_deg": float(
            np.degrees(
                Rotation.from_matrix(rigid_transform[:3, :3]).magnitude()
            )
        ),
        "sampled_translation_m": float(np.linalg.norm(rigid_transform[:3, 3])),
        "noise_sigma": float(noise_sigma),
        "noise_clip": float(noise_clip),
        "apply_noise_to": str(apply_noise_to),
        "rigid_transform": rigid_transform.tolist(),
    }
    return (
        source_perturbed.astype(np.float32, copy=False),
        target_perturbed.astype(np.float32, copy=False),
        None if updated_gt is None else updated_gt.astype(np.float64, copy=False),
        metadata,
    )
