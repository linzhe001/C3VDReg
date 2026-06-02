"""
C3VD Dataset Adapter for DCP (Deep Closest Point)

This adapter converts C3VD dataset to DCP's expected format.
"""

import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

# Add parent directory to path for imports
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from typing import Tuple

from common.datasets.c3vd_base import C3VDDatasetBase
from common.utils.benchmark_preprocess import (
    apply_pair_perturbation,
    normalize_point_cloud_pair,
    sample_point_cloud,
)
from common.utils.sampling import clean_point_cloud
from common.utils.transform_utils import (
    apply_transform,
    jitter_pointcloud,
    random_se3_transform,
)


class C3VDForDCP(C3VDDatasetBase):
    """
    C3VD dataset adapter for DCP.

    DCP expects data in the following format:
    Returns:
        Tuple of:
        - pointcloud1: [3, N] template point cloud
        - pointcloud2: [3, N] source point cloud (transformed)
        - R_ab: [3, 3] rotation matrix (template -> source)
        - translation_ab: [3] translation vector (template -> source)
        - R_ba: [3, 3] inverse rotation matrix (source -> template)
        - translation_ba: [3] inverse translation vector (source -> template)
        - euler_ab: [3] Euler angles (z, y, x) for forward transform
        - euler_ba: [3] Euler angles for inverse transform
    """

    def __init__(
        self,
        data_root: str,
        num_points: int = 1024,
        split: str = "train",
        gaussian_noise: bool = False,
        rot_factor: float = 4.0,
        trans_mag: float = 0.5,
        sampling_mode: str = "voxel",
        normalize_mode: str = "unit_cube",
        perturbation_enabled: bool = False,
        rotation_deg: float = 0.0,
        translation_m: float = 0.0,
        noise_sigma: float = 0.0,
        noise_clip: float = 0.0,
        apply_noise_to: str = "source",
        train_ratio: float = 0.7,
        random_seed: int = 42,
        val_scenes=None,
        **kwargs,
    ):
        """
        Initialize C3VD dataset for DCP.

        Args:
            data_root: Root directory of C3VD dataset
            num_points: Number of points to sample
            split: 'train' or 'test'
            gaussian_noise: Whether to add Gaussian noise
            rot_factor: Rotation magnitude factor (rotation range = pi/rot_factor)
            trans_mag: Translation magnitude range
            train_ratio: Train/test split ratio
            random_seed: Random seed
            **kwargs: Additional arguments for C3VDDatasetBase
        """
        super().__init__(
            data_root=data_root,
            split=split,
            pair_mode="one_to_one",
            scene_split=True,
            train_ratio=train_ratio,
            random_seed=random_seed,
            val_scenes=val_scenes,
            **kwargs,
        )

        self.num_points = num_points
        self.gaussian_noise = gaussian_noise
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
        self.split = split

        print("C3VD-DCP Adapter initialized:")
        print(f"  Num points: {num_points}")
        print(f"  Gaussian noise: {gaussian_noise}")
        print(f"  Rotation factor: {rot_factor} (range: ±{180 / rot_factor:.1f}°)")
        print(f"  Translation magnitude: {trans_mag}")
        print(f"  Sampling mode: {sampling_mode}")
        print(f"  Normalize mode: {normalize_mode}")
        if self.perturbation_enabled:
            print(
                "  Unified perturbation:"
                f" rotation_deg={self.rotation_deg},"
                f" translation_m={self.translation_m},"
                f" noise_sigma={self.noise_sigma},"
                f" noise_clip={self.noise_clip},"
                f" apply_noise_to={self.apply_noise_to}"
            )

    def __getitem__(self, item: int) -> Tuple:
        """
        Get a data sample in DCP format.

        Args:
            item: Index

        Returns:
            Tuple of
            (pointcloud1, pointcloud2, R_ab, t_ab, R_ba, t_ba, euler_ab, euler_ba)
        """
        # Get base data from C3VDDatasetBase
        data = super().__getitem__(item)
        source = data["source"]  # [N, 3]
        target = data["target"]  # [M, 3]

        # Step 1: Clean point clouds (remove NaN/Inf)
        # Same as PointNetLK_c3vd Mamba3D training
        source = clean_point_cloud(source, min_points=100)
        target = clean_point_cloud(target, min_points=100)

        # Step 2: Unified benchmark sampling
        sample_seed = self.random_seed + item * 9973
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

        # Set random seed for test set (DCP does this)
        if self.split != "train":
            np.random.seed(item)

        # Generate the registration problem using the unified benchmark policy
        if self.perturbation_enabled:
            perturb_seed = None if self.split == "train" else sample_seed + 2
            source_perturbed, target_perturbed, gt_transform, _ = (
                apply_pair_perturbation(
                    source,
                    target,
                    rotation_deg=self.rotation_deg,
                    translation_m=self.translation_m,
                    noise_sigma=self.noise_sigma,
                    noise_clip=self.noise_clip,
                    apply_noise_to=self.apply_noise_to,
                    gt_transform=np.eye(4, dtype=np.float64),
                    seed=perturb_seed,
                )
            )

            source_norm, target_norm, _, source_tf, target_tf = (
                normalize_point_cloud_pair(
                    source_perturbed,
                    target_perturbed,
                    self.normalize_mode,
                )
            )
            gt_norm = target_tf @ gt_transform @ np.linalg.inv(source_tf)

            pointcloud1 = source_norm.copy()
            pointcloud2 = target_norm.copy()
            R_ab = gt_norm[:3, :3].astype(np.float32, copy=False)
            translation_ab = gt_norm[:3, 3].astype(np.float32, copy=False)
            euler_ab = Rotation.from_matrix(gt_norm[:3, :3]).as_euler("zyx")
        else:
            # Baseline-aware model-private normalization for the legacy route.
            source, target, _, _, _ = normalize_point_cloud_pair(
                source,
                target,
                self.normalize_mode,
            )
            R_ab, translation_ab, euler_ab = random_se3_transform(
                rot_factor=self.rot_factor, trans_mag=self.trans_mag
            )
            pointcloud1 = target.copy()
            pointcloud2 = apply_transform(source, R_ab, translation_ab)  # [N, 3]

            if self.gaussian_noise:
                pointcloud2 = jitter_pointcloud(pointcloud2)

        # Compute inverse transformation
        R_ba = R_ab.T
        translation_ba = -R_ba @ translation_ab

        # Compute inverse Euler angles
        # euler_ab = [angle_z, angle_y, angle_x]
        # For inverse, we need to reverse order and negate
        euler_ba = -euler_ab[::-1]

        # Convert to DCP format [3, N]
        pointcloud1 = pointcloud1.T.astype("float32")
        pointcloud2 = pointcloud2.T.astype("float32")

        return (
            pointcloud1,  # [3, N] - target (template)
            pointcloud2,  # [3, N] - transformed source
            R_ab.astype("float32"),  # [3, 3]
            translation_ab.astype("float32"),  # [3]
            R_ba.astype("float32"),  # [3, 3]
            translation_ba.astype("float32"),  # [3]
            euler_ab.astype("float32"),  # [3]
            euler_ba.astype("float32"),  # [3]
        )


def test_adapter():
    """Test the DCP adapter."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=str, required=True, help="Path to C3VD_datasets directory"
    )
    args = parser.parse_args()

    print("Testing C3VD-DCP Adapter...")
    print("=" * 60)

    # Create dataset
    dataset = C3VDForDCP(
        data_root=args.data_root,
        num_points=1024,
        split="train",
        gaussian_noise=False,
        rot_factor=4.0,
        trans_mag=0.5,
        train_ratio=0.7,
        random_seed=42,
    )

    print(f"\nDataset size: {len(dataset)}")

    if len(dataset) > 0:
        # Test loading a sample
        print("\nLoading sample 0...")
        pc1, pc2, R_ab, t_ab, R_ba, t_ba, euler_ab, euler_ba = dataset[0]

        print("\nSample 0 data:")
        print(f"  pointcloud1 shape: {pc1.shape} (expected [3, 1024])")
        print(f"  pointcloud2 shape: {pc2.shape} (expected [3, 1024])")
        print(f"  R_ab shape: {R_ab.shape} (expected [3, 3])")
        print(f"  t_ab shape: {t_ab.shape} (expected [3])")
        print(f"  R_ba shape: {R_ba.shape} (expected [3, 3])")
        print(f"  t_ba shape: {t_ba.shape} (expected [3])")
        print(f"  euler_ab shape: {euler_ab.shape} (expected [3])")
        print(f"  euler_ba shape: {euler_ba.shape} (expected [3])")

        print("\n  Euler angles (degrees):")
        print(
            f"    Forward (ab): z={np.rad2deg(euler_ab[0]):.2f}°, "
            f"y={np.rad2deg(euler_ab[1]):.2f}°, x={np.rad2deg(euler_ab[2]):.2f}°"
        )
        print(
            f"    Inverse (ba): z={np.rad2deg(euler_ba[0]):.2f}°, "
            f"y={np.rad2deg(euler_ba[1]):.2f}°, x={np.rad2deg(euler_ba[2]):.2f}°"
        )

        # Verify transformation consistency
        print("\n  Transformation consistency checks:")

        # Check 1: R_ba = R_ab^T
        R_ba_check = R_ab.T
        R_diff = np.linalg.norm(R_ba - R_ba_check)
        print(f"    |R_ba - R_ab^T|: {R_diff:.6f} (should be ~0)")

        # Check 2: t_ba = -R_ba @ t_ab
        t_ba_check = -R_ba @ t_ab
        t_diff = np.linalg.norm(t_ba - t_ba_check)
        print(f"    |t_ba - (-R_ba @ t_ab)|: {t_diff:.6f} (should be ~0)")

        # Check 3: Apply forward then inverse should give identity
        # pc2 = R_ab @ pc1 + t_ab
        # pc1_reconstructed = R_ba @ pc2 + t_ba
        pc2_expected = (R_ab @ pc1) + t_ab.reshape(3, 1)
        pc2_diff = np.linalg.norm(pc2 - pc2_expected)
        print(f"    |pc2 - (R_ab @ pc1 + t_ab)|: {pc2_diff:.6f}")

        pc1_reconstructed = (R_ba @ pc2) + t_ba.reshape(3, 1)
        recon_diff = np.linalg.norm(pc1 - pc1_reconstructed)
        print(f"    |pc1 - (R_ba @ pc2 + t_ba)|: {recon_diff:.6f} (should be ~0)")

        if R_diff < 1e-5 and t_diff < 1e-5 and recon_diff < 1e-5:
            print("\n✓ All consistency checks passed!")
        else:
            print("\n✗ Some consistency checks failed!")

        # Test multiple samples
        print("\nTesting batch loading (5 samples)...")
        for i in range(min(5, len(dataset))):
            data = dataset[i]
            assert len(data) == 8, f"Expected 8 elements, got {len(data)}"
            assert data[0].shape == (3, 1024), f"Wrong shape for pc1: {data[0].shape}"
            assert data[1].shape == (3, 1024), f"Wrong shape for pc2: {data[1].shape}"

        print("✓ Successfully loaded 5 samples")

    print("\n" + "=" * 60)
    print("DCP Adapter test completed!")


if __name__ == "__main__":
    test_adapter()
