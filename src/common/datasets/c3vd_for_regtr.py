"""
C3VD Dataset Adapter for RegTR

This adapter converts C3VD data format to RegTR's expected format.
RegTR expects dictionary with overlap masks and correspondences.

Author: Claude
Date: 2025-10-18
"""

from typing import Dict, Optional

import numpy as np
import torch

from common.utils.benchmark_preprocess import (
    apply_pair_perturbation,
    apply_transform_matrix,
    normalize_point_cloud_pair,
    sample_point_cloud,
)
from common.utils.overlap_utils import compute_overlap
from common.utils.sampling import clean_point_cloud
from common.utils.transform_utils import apply_transform, random_se3_transform

from .c3vd_base import C3VDDatasetBase


class C3VDForRegTR(C3VDDatasetBase):
    """
    C3VD Dataset adapter for RegTR

    RegTR expects dict with:
        - src_xyz: torch.Tensor (N, 3)
        - tgt_xyz: torch.Tensor (M, 3)
        - src_overlap: torch.BoolTensor (N,)
        - tgt_overlap: torch.BoolTensor (M,)
        - correspondences: torch.LongTensor (2, K)
        - pose: torch.Tensor (3, 4) SE(3) matrix
        - overlap_p: float, overlap percentage
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        num_points: Optional[int] = None,  # None = use all points
        overlap_radius: float = 0.0375,  # RegTR default
        rot_factor: float = 4.0,
        trans_mag: float = 0.5,
        sampling_mode: str = "voxel",
        normalize_mode: str = "none",
        perturbation_enabled: bool = False,
        rotation_deg: float = 0.0,
        translation_m: float = 0.0,
        noise_sigma: float = 0.0,
        noise_clip: float = 0.0,
        apply_noise_to: str = "source",
        **kwargs,
    ):
        """
        Args:
            data_root: Path to C3VD dataset
            split: 'train' or 'test'
            num_points: Number of points to sample (None = use all)
            overlap_radius: Search radius for overlap computation (meters)
            rot_factor: Rotation range factor (range = ±π/rot_factor)
            trans_mag: Translation magnitude (meters)
            **kwargs: Additional arguments for C3VDDatasetBase
        """
        super().__init__(data_root, split=split, **kwargs)

        self.num_points = num_points
        self.overlap_radius = overlap_radius
        self.rot_factor = rot_factor
        self.trans_mag = trans_mag
        self.sampling_mode = sampling_mode
        self.normalize_mode = normalize_mode
        self.perturbation_enabled = bool(perturbation_enabled)
        self.rotation_deg = float(rotation_deg)
        self.translation_m = float(translation_m)
        self.noise_sigma = float(noise_sigma)
        self.noise_clip = float(noise_clip)
        self.apply_noise_to = apply_noise_to

        print("\nC3VD-RegTR Adapter initialized:")
        print(f"  Num points: {num_points if num_points else 'All'}")
        print(f"  Overlap radius: {overlap_radius}")
        print(f"  Rotation factor: {rot_factor} (range: ±{180 / rot_factor:.1f}°)")
        print(f"  Translation magnitude: {trans_mag}")
        print(f"  Sampling mode: {sampling_mode}")
        print(f"  Normalize mode: {normalize_mode}")

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a data pair in RegTR format

        Returns:
            Dictionary with RegTR-format data
        """
        # Get base data from C3VDDatasetBase
        data = super().__getitem__(idx)
        source = data["source"]  # [N, 3]
        target = data["target"]  # [M, 3]

        # Step 1: Clean point clouds (remove NaN/Inf)
        # Same as PointNetLK_c3vd Mamba3D training
        source = clean_point_cloud(source, min_points=100)
        target = clean_point_cloud(target, min_points=100)

        # Step 2: VoxelGrid downsampling (spatial uniform sampling) if num_points is set
        # Same as PointNetLK_c3vd Mamba3D training
        sample_seed = self.random_seed + idx * 9973
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

        # Step 3: Baseline-aware model-private normalization
        source, target, _, _, _ = normalize_point_cloud_pair(
            source,
            target,
            self.normalize_mode,
        )

        # Set random seed for test set (for reproducibility)
        if self.split != "train":
            np.random.seed(idx)

        if self.perturbation_enabled:
            perturb_seed = None if self.split == "train" else sample_seed + 2
            transformed_source, target, _, perturb_meta = apply_pair_perturbation(
                source,
                target,
                rotation_deg=self.rotation_deg,
                translation_m=self.translation_m,
                noise_sigma=self.noise_sigma,
                noise_clip=self.noise_clip,
                apply_noise_to=self.apply_noise_to,
                seed=perturb_seed,
            )
            source_to_target = np.linalg.inv(
                np.asarray(perturb_meta["rigid_transform"], dtype=np.float64)
            )
            pose = source_to_target[:3, :].astype(np.float32, copy=False)
        else:
            # Generate random SE(3) transformation
            # In C3VD, source and target are already aligned (GT = Identity)
            # We apply random transform to create registration problem
            R_ab, t_ab, _ = random_se3_transform(
                rot_factor=self.rot_factor, trans_mag=self.trans_mag
            )

            # Apply transformation: transformed_source = R @ source + t
            # This simulates misalignment
            transformed_source = apply_transform(source, R_ab, t_ab)

            # RegTR expects pose to map the observed source back to target space.
            R_ba = R_ab.T
            t_ba = -R_ba @ t_ab
            pose = np.zeros((3, 4), dtype=np.float32)
            pose[:3, :3] = R_ba
            pose[:3, 3] = t_ba

        # Compute overlap regions and correspondences in the aligned frame, as
        # RegTR's 3DMatch loader does. The returned indices still refer to the
        # observed source point order.
        pose_4x4 = np.eye(4, dtype=np.float64)
        pose_4x4[:3, :] = pose
        aligned_source = apply_transform_matrix(transformed_source, pose_4x4)
        src_overlap_mask, tgt_overlap_mask, correspondences = compute_overlap(
            aligned_source, target, self.overlap_radius
        )

        # Compute overlap percentage
        overlap_p = (
            np.sum(src_overlap_mask) / len(src_overlap_mask)
            + np.sum(tgt_overlap_mask) / len(tgt_overlap_mask)
        ) / 2.0

        # Prepare RegTR-format dictionary
        data_pair = {
            "src_xyz": torch.from_numpy(transformed_source).float(),  # (N, 3)
            "tgt_xyz": torch.from_numpy(target).float(),  # (M, 3)
            "src_overlap": torch.from_numpy(src_overlap_mask),  # (N,) bool
            "tgt_overlap": torch.from_numpy(tgt_overlap_mask),  # (M,) bool
            "correspondences": torch.from_numpy(correspondences).long(),  # (2, K)
            "pose": torch.from_numpy(pose).float(),  # (3, 4)
            "overlap_p": overlap_p,  # float
            "idx": idx,
            "src_path": data["source_file"],
            "tgt_path": data["target_file"],
            "scene": data["scene"],
        }

        return data_pair


if __name__ == "__main__":
    """Test C3VDForRegTR adapter"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=str, required=True, help="Path to C3VD dataset"
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=None,
        help="Number of points to sample (default: use all)",
    )
    args = parser.parse_args()

    print("Testing C3VD-RegTR Adapter...")
    print("=" * 60)

    # Create dataset
    dataset = C3VDForRegTR(
        data_root=args.data_root,
        split="train",
        num_points=args.num_points,
        overlap_radius=0.0375,
        rot_factor=4.0,
        trans_mag=0.5,
    )

    print(f"\nDataset size: {len(dataset)}")

    # Load first sample
    print("\nLoading sample 0...")
    sample = dataset[0]

    # Print sample info
    print("\nSample 0 data:")
    print(f"  src_xyz shape: {sample['src_xyz'].shape} (expected [N, 3])")
    print(f"  tgt_xyz shape: {sample['tgt_xyz'].shape} (expected [M, 3])")
    print(f"  src_overlap shape: {sample['src_overlap'].shape} (expected [N])")
    print(f"  tgt_overlap shape: {sample['tgt_overlap'].shape} (expected [M])")
    print(
        f"  correspondences shape: {sample['correspondences'].shape} (expected [2, K])"
    )
    print(f"  pose shape: {sample['pose'].shape} (expected [3, 4])")

    print("\n  Overlap statistics:")
    src_overlap_count = sample["src_overlap"].sum()
    tgt_overlap_count = sample["tgt_overlap"].sum()
    print(
        f"    Source overlap: {src_overlap_count}/{len(sample['src_overlap'])} "
        f"({100 * sample['src_overlap'].float().mean():.1f}%)"
    )
    print(
        f"    Target overlap: {tgt_overlap_count}/{len(sample['tgt_overlap'])} "
        f"({100 * sample['tgt_overlap'].float().mean():.1f}%)"
    )
    print(f"    Correspondences: {sample['correspondences'].shape[1]}")
    print(f"    Overlap ratio: {sample['overlap_p']:.3f}")

    print(f"\n  Scene: {sample['scene']}")
    print(f"  Source: {sample['src_path']}")
    print(f"  Target: {sample['tgt_path']}")

    # Verify pose matrix
    pose = sample["pose"].numpy()
    R = pose[:3, :3]
    t = pose[:3, 3]

    # Check if rotation matrix is valid (orthogonal)
    should_be_identity = R @ R.T
    ortho_error = np.abs(should_be_identity - np.eye(3)).max()
    det_R = np.linalg.det(R)

    print("\n  Pose validation:")
    print(f"    R orthogonality error: {ortho_error:.6f} (should be ~0)")
    print(f"    det(R): {det_R:.6f} (should be ~1)")
    print(f"    ||t||: {np.linalg.norm(t):.3f}")

    if ortho_error < 1e-5 and abs(det_R - 1.0) < 1e-5:
        print("    ✓ Valid SE(3) transformation")
    else:
        print("    ✗ Invalid transformation!")

    # Test batch loading
    print("\nTesting batch loading (5 samples)...")
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=5, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    print("  Batch shapes:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"    {key}: {value.shape}")

    print("\n✓ Successfully loaded 5 samples")

    print("\n" + "=" * 60)
    print("RegTR Adapter test completed!")
