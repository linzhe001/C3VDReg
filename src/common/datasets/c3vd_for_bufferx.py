"""C3VD adapter utilities for BUFFER-X bridges."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData

from src.benchmarking.manifest_schema import validate_manifest_record
from src.common.utils.benchmark_preprocess import (
    apply_pair_perturbation,
    normalize_point_cloud_pair,
    sample_point_cloud,
)
from src.common.utils.sampling import clean_point_cloud, random_resample


@dataclass(frozen=True)
class BufferXC3VDConfig:
    manifest_path: str
    dataset_root: str | None
    split: str
    train_scenes: tuple[str, ...] = ()
    val_scenes: tuple[str, ...] = ()
    test_scenes: tuple[str, ...] = ()
    num_points: int = 8192
    sampling_mode: str = "voxel"
    normalize_mode: str = "none"
    random_seed: int = 42
    max_pairs: int | None = None
    first_downsample: float = 0.02
    second_downsample: float = 0.035
    max_num_points: int = 30000
    heuristic_mode: str = "native"
    num_fps: int = 1500
    num_points_radius_estimate: int = 2000
    perturbation_enabled: bool = False
    rotation_deg: float = 0.0
    translation_m: float = 0.0
    noise_sigma: float = 0.0
    noise_clip: float = 0.0
    apply_noise_to: str = "source"


def _read_ply_xyz(path: Path) -> np.ndarray:
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    return np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)


def _resolve_path(raw_path: str, dataset_root: Path | None) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if dataset_root is None:
        return path.resolve()
    return (dataset_root / path).resolve()


def _voxel_down_sample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    down = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size=float(voxel_size))
    return np.asarray(down.points, dtype=np.float32)


def _pca_sphericity(points: np.ndarray) -> tuple[float, np.ndarray]:
    centered = np.asarray(points, dtype=np.float64) - np.mean(points, axis=0)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order].T
    largest = float(max(eigvals[0], 1e-12))
    return float(eigvals[-1] / largest), eigvecs


def _sphericity_based_voxel_analysis(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[float, float, bool]:
    src_sphericity, src_components = _pca_sphericity(source)
    tgt_sphericity, tgt_components = _pca_sphericity(target)
    if len(source) > len(target):
        ref_points = source
        sphericity = src_sphericity
        components = src_components
    else:
        ref_points = target
        sphericity = tgt_sphericity
        components = tgt_components

    transformed = (
        np.asarray(ref_points, dtype=np.float64) - ref_points.mean(axis=0)
    ) @ components.T
    z_range = float(transformed[:, 2].max() - transformed[:, 2].min())
    alpha = 1.0 if sphericity < 0.05 else 1.5
    voxel_size = max(np.sqrt(max(z_range, 0.0)) / 100.0 * alpha, 0.001)

    src_z = src_components[-1] / np.linalg.norm(src_components[-1])
    tgt_z = tgt_components[-1] / np.linalg.norm(tgt_components[-1])
    src_global = abs(float(np.dot(src_z, np.array([0.0, 0.0, 1.0])))) > 0.98
    tgt_global = abs(float(np.dot(tgt_z, np.array([0.0, 0.0, 1.0])))) > 0.98
    same_direction = float(np.dot(src_z, tgt_z)) > 0.96
    return (
        round(float(voxel_size), 4),
        float(sphericity),
        bool(src_global and tgt_global and same_direction),
    )


def _ensure_point_count(points: np.ndarray, min_points: int, seed: int) -> np.ndarray:
    if len(points) >= min_points:
        return points.astype(np.float32, copy=False)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        return random_resample(points, min_points)
    finally:
        np.random.set_state(state)


def _maybe_limit_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) <= max_points:
        return points.astype(np.float32, copy=False)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), int(max_points), replace=False)
    return points[indices].astype(np.float32, copy=False)


def prepare_bufferx_pair(
    source: np.ndarray,
    target: np.ndarray,
    gt_transform: np.ndarray,
    config: BufferXC3VDConfig,
    *,
    index: int,
    split: str,
    sample_id: str = "",
    scene_id: str = "",
) -> dict[str, Any]:
    if config.heuristic_mode not in {"native", "fixed"}:
        raise ValueError(
            "BUFFER-X heuristic_mode must be either 'native' or 'fixed'. "
            f"Received {config.heuristic_mode!r}."
        )
    seed = int(config.random_seed) + int(index) * 9973
    source = clean_point_cloud(source, min_points=min(len(source), 8))
    target = clean_point_cloud(target, min_points=min(len(target), 8))
    source = sample_point_cloud(source, config.sampling_mode, config.num_points, seed)
    target = sample_point_cloud(
        target,
        config.sampling_mode,
        config.num_points,
        seed + 1,
    )
    source, target, _, _, _ = normalize_point_cloud_pair(
        source,
        target,
        config.normalize_mode,
    )

    if config.perturbation_enabled:
        source, target, gt_updated, _ = apply_pair_perturbation(
            source=source,
            target=target,
            gt_transform=gt_transform,
            rotation_deg=config.rotation_deg,
            translation_m=config.translation_m,
            noise_sigma=config.noise_sigma,
            noise_clip=config.noise_clip,
            apply_noise_to=config.apply_noise_to,
            seed=None if split == "train" else seed + 2,
        )
        relt_pose = np.asarray(gt_updated, dtype=np.float32)
    else:
        relt_pose = np.asarray(gt_transform, dtype=np.float32)

    if split == "test" and config.heuristic_mode == "native":
        first_downsample, sphericity, aligned_to_global_z = (
            _sphericity_based_voxel_analysis(source, target)
        )
    else:
        first_downsample = float(config.first_downsample)
        sphericity = 0.0
        aligned_to_global_z = False

    src_fds = _voxel_down_sample(source, first_downsample)
    tgt_fds = _voxel_down_sample(target, first_downsample)
    required_fds = max(int(config.num_fps), int(config.num_points_radius_estimate), 1)
    src_fds = _ensure_point_count(src_fds, required_fds, seed + 3)
    tgt_fds = _ensure_point_count(tgt_fds, required_fds, seed + 4)

    src_sds = _voxel_down_sample(source, config.second_downsample)
    tgt_sds = _voxel_down_sample(target, config.second_downsample)
    src_sds = _maybe_limit_points(src_sds, int(config.max_num_points), seed + 5)
    tgt_sds = _maybe_limit_points(tgt_sds, int(config.max_num_points), seed + 6)

    return {
        "src_fds_pts": src_fds.astype(np.float32, copy=False),
        "tgt_fds_pts": tgt_fds.astype(np.float32, copy=False),
        "relt_pose": relt_pose.astype(np.float32, copy=False),
        "src_sds_pts": src_sds.astype(np.float32, copy=False),
        "tgt_sds_pts": tgt_sds.astype(np.float32, copy=False),
        "src_id": sample_id or f"{scene_id}/source_{index:06d}",
        "tgt_id": sample_id or f"{scene_id}/target_{index:06d}",
        "voxel_size": float(config.second_downsample),
        "dataset_name": "C3VD",
        "scene_name": scene_id,
        "sphericity": float(sphericity),
        "is_aligned_to_global_z": bool(aligned_to_global_z),
        "first_downsample": float(first_downsample),
    }


def collate_bufferx_pair(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("BUFFER-X bridge only supports batch_size=1.")
    item = batch[0]
    return {
        "src_fds_pcd": torch.tensor(item["src_fds_pts"], dtype=torch.float32),
        "tgt_fds_pcd": torch.tensor(item["tgt_fds_pts"], dtype=torch.float32),
        "src_sds_pcd": torch.tensor(item["src_sds_pts"][:, :3], dtype=torch.float32),
        "tgt_sds_pcd": torch.tensor(item["tgt_sds_pts"][:, :3], dtype=torch.float32),
        "relt_pose": torch.tensor(item["relt_pose"], dtype=torch.float32),
        "src_id": item["src_id"],
        "tgt_id": item["tgt_id"],
        "scene_name": item["scene_name"],
        "sensor": "",
        "voxel_sizes": torch.tensor([item["voxel_size"]], dtype=torch.float32),
        "dataset_names": [item["dataset_name"]],
        "sphericity": torch.tensor([item["sphericity"]], dtype=torch.float32),
        "is_aligned_to_global_z": item["is_aligned_to_global_z"],
    }


class C3VDForBufferX(torch.utils.data.Dataset):
    """Manifest-backed C3VD dataset in BUFFER-X training format."""

    def __init__(self, config: BufferXC3VDConfig) -> None:
        self.config = config
        self.manifest_path = Path(config.manifest_path).resolve()
        self.dataset_root = (
            Path(config.dataset_root).resolve() if config.dataset_root else None
        )
        self.records = self._load_records()

    def _active_scenes(self) -> set[str] | None:
        if self.config.split == "train" and self.config.train_scenes:
            return set(self.config.train_scenes)
        if self.config.split == "val" and self.config.val_scenes:
            return set(self.config.val_scenes)
        if self.config.split == "test" and self.config.test_scenes:
            return set(self.config.test_scenes)
        return None

    def _load_records(self) -> list[Any]:
        active_scenes = self._active_scenes()
        records: list[Any] = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = validate_manifest_record(json.loads(line))
                if record.split != self.config.split:
                    continue
                if active_scenes is not None and record.scene_id not in active_scenes:
                    continue
                records.append(record)
                if (
                    self.config.max_pairs is not None
                    and len(records) >= self.config.max_pairs
                ):
                    break
        if not records:
            raise ValueError(
                f"No C3VD manifest records found for split={self.config.split!r}."
            )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        source = _read_ply_xyz(_resolve_path(record.source_path, self.dataset_root))
        target = _read_ply_xyz(_resolve_path(record.target_path, self.dataset_root))
        return prepare_bufferx_pair(
            source=source,
            target=target,
            gt_transform=np.asarray(record.gt_transform, dtype=np.float64),
            config=self.config,
            index=index,
            split=self.config.split,
            sample_id=record.sample_id,
            scene_id=record.scene_id,
        )
