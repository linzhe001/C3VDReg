"""Preprocess profile declarations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SamplingMode = Literal["none", "random", "voxel", "fps"]
NormalizeMode = Literal["none", "joint_bbox", "unit_cube", "model_private"]


@dataclass(frozen=True)
class PreprocessProfile:
    profile_id: str
    clean_invalid: bool
    sampling: SamplingMode
    num_points: int | None
    estimate_normals: bool
    normalize: NormalizeMode
    keep_metric_unit: bool
    deterministic: bool
    leaderboard_track: str


DEFAULT_PROFILES: tuple[PreprocessProfile, ...] = (
    PreprocessProfile(
        profile_id="canonical_v1",
        clean_invalid=True,
        sampling="voxel",
        num_points=2048,
        estimate_normals=False,
        normalize="none",
        keep_metric_unit=True,
        deterministic=True,
        leaderboard_track="main",
    ),
    PreprocessProfile(
        profile_id="normals_v1",
        clean_invalid=True,
        sampling="voxel",
        num_points=2048,
        estimate_normals=True,
        normalize="none",
        keep_metric_unit=True,
        deterministic=True,
        leaderboard_track="normals",
    ),
    PreprocessProfile(
        profile_id="legacy_joint_norm_v1",
        clean_invalid=True,
        sampling="voxel",
        num_points=1024,
        estimate_normals=False,
        normalize="joint_bbox",
        keep_metric_unit=False,
        deterministic=True,
        leaderboard_track="legacy",
    ),
    PreprocessProfile(
        profile_id="debug_raw_v1",
        clean_invalid=True,
        sampling="none",
        num_points=None,
        estimate_normals=False,
        normalize="none",
        keep_metric_unit=True,
        deterministic=True,
        leaderboard_track="debug",
    ),
)
