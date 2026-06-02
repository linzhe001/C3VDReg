"""C3VD dataset adapter for GeoTransformer training."""

from __future__ import annotations

from typing import Optional

import numpy as np

from common.datasets.c3vd_base import C3VDDatasetBase
from common.utils.benchmark_preprocess import (
    apply_pair_perturbation,
    apply_transform_matrix,
    normalize_point_cloud_pair,
    sample_point_cloud,
    sample_rigid_transform,
)
from common.utils.sampling import clean_point_cloud


class C3VDForGeoTransformer(C3VDDatasetBase):
    """Convert C3VD source/target pairs into GeoTransformer stack-mode records."""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        num_points: Optional[int] = None,
        sampling_mode: str = "voxel",
        normalize_mode: str = "none",
        perturbation_enabled: bool = False,
        rotation_deg: float = 0.0,
        translation_m: float = 0.0,
        noise_sigma: float = 0.0,
        noise_clip: float = 0.0,
        apply_noise_to: str = "source",
        synthetic_rotation_deg: float = 45.0,
        synthetic_translation_m: float = 0.5,
        train_ratio: float = 0.7,
        random_seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(
            data_root=data_root,
            split=split,
            pair_mode="one_to_one",
            scene_split=True,
            train_ratio=train_ratio,
            random_seed=random_seed,
            **kwargs,
        )
        self.num_points = num_points
        self.sampling_mode = sampling_mode
        self.normalize_mode = normalize_mode
        self.perturbation_enabled = bool(perturbation_enabled)
        self.rotation_deg = float(rotation_deg)
        self.translation_m = float(translation_m)
        self.noise_sigma = float(noise_sigma)
        self.noise_clip = float(noise_clip)
        self.apply_noise_to = apply_noise_to
        self.synthetic_rotation_deg = float(synthetic_rotation_deg)
        self.synthetic_translation_m = float(synthetic_translation_m)

        print("C3VD-GeoTransformer Adapter initialized:")
        print(f"  Num points: {num_points if num_points is not None else 'All'}")
        print(f"  Sampling mode: {sampling_mode}")
        print(f"  Normalize mode: {normalize_mode}")

    def __getitem__(self, index: int) -> dict[str, object]:
        data = super().__getitem__(index)
        source = clean_point_cloud(data["source"], min_points=100)
        target = clean_point_cloud(data["target"], min_points=100)

        sample_seed = self.random_seed + index * 9973
        if self.num_points is not None:
            source = sample_point_cloud(
                source,
                sampling=self.sampling_mode,
                num_points=self.num_points,
                seed=sample_seed,
            )
            target = sample_point_cloud(
                target,
                sampling=self.sampling_mode,
                num_points=self.num_points,
                seed=sample_seed + 1,
            )

        source, target, _, _, _ = normalize_point_cloud_pair(
            source,
            target,
            self.normalize_mode,
        )

        identity = np.eye(4, dtype=np.float64)
        if self.perturbation_enabled:
            perturb_seed = None if self.split == "train" else sample_seed + 2
            source, target, gt_transform, _ = apply_pair_perturbation(
                source,
                target,
                rotation_deg=self.rotation_deg,
                translation_m=self.translation_m,
                noise_sigma=self.noise_sigma,
                noise_clip=self.noise_clip,
                apply_noise_to=self.apply_noise_to,
                gt_transform=identity,
                seed=perturb_seed,
            )
            transform = gt_transform if gt_transform is not None else identity
        else:
            transform_seed = None if self.split == "train" else sample_seed + 2
            source_to_perturbed = sample_rigid_transform(
                rotation_deg=self.synthetic_rotation_deg,
                translation_m=self.synthetic_translation_m,
                seed=transform_seed,
            )
            source = apply_transform_matrix(source, source_to_perturbed)
            transform = np.linalg.inv(source_to_perturbed)

        return {
            "scene_name": data["scene"],
            "ref_frame": data["target_id"],
            "src_frame": data["source_id"],
            "overlap": 1.0,
            "ref_points": target.astype(np.float32, copy=False),
            "src_points": source.astype(np.float32, copy=False),
            "ref_feats": np.ones((target.shape[0], 1), dtype=np.float32),
            "src_feats": np.ones((source.shape[0], 1), dtype=np.float32),
            # GeoTransformer expects transform to map src_points into ref_points.
            "transform": np.asarray(transform, dtype=np.float32),
        }
