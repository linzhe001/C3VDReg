"""
C3VD Dataset Adapter for PointNetLK_Revisited

This adapter converts C3VD dataset to PointNetLK_Revisited's expected format.
"""

import os
import sys

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from common.datasets.c3vd_base import C3VDDatasetBase
from common.utils.benchmark_preprocess import (
    apply_pair_perturbation,
    normalize_point_cloud_pair,
    sample_point_cloud,
)
from common.utils.sampling import clean_point_cloud
from common.utils.transform_utils import resample_points

# Import utils from PointNetLK_Revisited
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
pointnetlk_revisited_path = os.path.join(
    project_root, "baselines", "PointNetLK_Revisited"
)
sys.path.insert(0, pointnetlk_revisited_path)
import utils as plk_utils  # noqa: E402


class RandomTransformSE3C3VD:
    """Generate random SE(3) transformations with PointNetLK twists."""

    def __init__(self, mag=1, mag_randomly=True):
        self.mag = mag
        self.randomly = mag_randomly
        self.gt = None
        self.igt = None

    def generate_transform(self):
        """Generate random twist parameters."""
        amp = self.mag
        if self.randomly:
            amp = torch.rand(1, 1) * self.mag
        x = torch.randn(1, 6)
        x = x / x.norm(p=2, dim=1, keepdim=True) * amp
        return x

    def apply_transform(self, p0, x):
        """Apply SE(3) transformation using twist parameters."""
        # p0: [N, 3]
        # x: [1, 6], twist params
        g = plk_utils.exp(x).to(p0)  # [1, 4, 4]
        gt = plk_utils.exp(-x).to(p0)  # [1, 4, 4]
        p1 = plk_utils.transform(g, p0)
        self.gt = gt  # p1 --> p0
        self.igt = g  # p0 --> p1
        return p1

    def transform(self, tensor):
        """Transform a tensor."""
        x = self.generate_transform()
        return self.apply_transform(tensor, x)

    def __call__(self, tensor):
        return self.transform(tensor)


def add_noise(pointcloud, sigma=0.01, clip=0.05):
    """Add Gaussian noise to point cloud."""
    N, C = pointcloud.shape
    pointcloud += torch.clamp(sigma * torch.randn(N, C), -1 * clip, clip)
    return pointcloud


class C3VDForPointNetLKRevisited(torch.utils.data.Dataset):
    """
    C3VD dataset adapter for PointNetLK_Revisited.

    PointNetLK_Revisited expects data in the following format:
    Returns:
        Tuple of (p0, p1, gt) where:
        - p0: template point cloud [N, 3] (original target)
        - p1: source point cloud [N, 3] (transformed template)
        - gt: transformation matrix [4, 4] (p1 -> p0, inverse transform)
    """

    def __init__(
        self,
        data_root: str,
        num_points: int = 1024,
        split: str = "train",
        mag: float = 0.8,
        sigma: float = 0.0,
        clip: float = 0.0,
        sampling_mode: str = "voxel",
        normalize_mode: str = "none",
        perturbation_enabled: bool = False,
        rotation_deg: float = 0.0,
        translation_m: float = 0.0,
        noise_sigma: float = 0.0,
        noise_clip: float = 0.0,
        apply_noise_to: str = "source",
        train_ratio: float = 0.7,
        random_seed: int = 42,
        **kwargs,
    ):
        """
        Initialize C3VD dataset for PointNetLK_Revisited.

        Args:
            data_root: Root directory of C3VD dataset
            num_points: Number of points to sample
            split: 'train' or 'test'
            mag: Magnitude of random transformation (twist parameter)
            sigma: Gaussian noise standard deviation
            clip: Gaussian noise clipping range
            train_ratio: Train/test split ratio
            random_seed: Random seed
            **kwargs: Additional arguments for C3VDDatasetBase
        """
        # Initialize base dataset
        self.base_dataset = C3VDDatasetBase(
            data_root=data_root,
            split=split,
            pair_mode="one_to_one",
            scene_split=True,
            train_ratio=train_ratio,
            random_seed=random_seed,
            **kwargs,
        )

        self.num_points = num_points
        self.mag = mag
        self.sigma = sigma
        self.clip = clip
        self.sampling_mode = sampling_mode
        self.normalize_mode = normalize_mode
        self.perturbation_enabled = bool(perturbation_enabled)
        self.rotation_deg = float(rotation_deg)
        self.translation_m = float(translation_m)
        self.noise_sigma = float(noise_sigma)
        self.noise_clip = float(noise_clip)
        self.apply_noise_to = apply_noise_to
        self.split = split

        # Random transformation generator
        self.transf = RandomTransformSE3C3VD(mag=mag, mag_randomly=True)

        print("C3VD-PointNetLK_Revisited Adapter initialized:")
        print(f"  Num points: {num_points}")
        print(f"  Transformation magnitude: {mag}")
        print(f"  Noise (sigma, clip): ({sigma}, {clip})")
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

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        """
        Get a data sample in PointNetLK_Revisited format.

        Args:
            index: Index

        Returns:
            Tuple of (p0, p1, igt)
            - p0: template point cloud [N, 3] tensor
            - p1: source point cloud [N, 3] tensor (transformed)
            - igt: transformation matrix [4, 4] tensor (p0 -> p1)
        """
        # Get base data
        data = self.base_dataset[index]
        source = data["source"]  # [N, 3] numpy
        target = data["target"]  # [M, 3] numpy

        # Step 1: Clean point clouds (remove NaN/Inf)
        # Same as PointNetLK_c3vd Mamba3D training
        source = clean_point_cloud(source, min_points=100)
        target = clean_point_cloud(target, min_points=100)

        sample_seed = self.base_dataset.random_seed + index * 9973
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

        if self.perturbation_enabled:
            perturb_seed = None if self.split == "train" else sample_seed + 2
            (
                source_perturbed,
                target_perturbed,
                _,
                perturb_meta,
            ) = apply_pair_perturbation(
                source,
                target,
                rotation_deg=self.rotation_deg,
                translation_m=self.translation_m,
                noise_sigma=self.noise_sigma,
                noise_clip=self.noise_clip,
                apply_noise_to=self.apply_noise_to,
                seed=perturb_seed,
            )
            p0 = torch.from_numpy(target_perturbed).float()
            p1 = torch.from_numpy(source_perturbed).float()
            igt = torch.from_numpy(
                np.asarray(perturb_meta["rigid_transform"], dtype=np.float32)
            )
        else:
            # Convert to tensor
            p0 = torch.from_numpy(target).float()  # [N, 3] - template (target)
            p1_original = torch.from_numpy(source).float()  # [N, 3] - source

            # Add noise if specified
            p1 = add_noise(p1_original.clone(), sigma=self.sigma, clip=self.clip)

            # Apply random transformation to source
            p1 = self.transf(p1)
            igt = self.transf.igt.squeeze(0)  # [4, 4] - forward transformation

        # p0: template, p1: transformed source, igt: forward transform.
        # PointNetLK's loss function: ||est_g @ igt - I||, expects est_g ≈ igt^(-1)
        # Return the forward transform, as in the original PointNetLK route.
        return p0, p1, igt


class C3VDForPointNetLKRevisitedFixedPerturbation(torch.utils.data.Dataset):
    """
    C3VD dataset with fixed perturbations (for testing).

    This version uses pre-generated transformation parameters for reproducibility.
    """

    def __init__(
        self,
        data_root: str,
        num_points: int = 1024,
        split: str = "test",
        perturbation_file: str = None,
        sigma: float = 0.0,
        clip: float = 0.0,
        train_ratio: float = 0.7,
        random_seed: int = 42,
        **kwargs,
    ):
        """
        Initialize C3VD dataset with fixed perturbations.

        Args:
            data_root: Root directory of C3VD dataset
            num_points: Number of points to sample
            split: 'train' or 'test'
            perturbation_file: Path to CSV file with twist parameters [N, 6]
            sigma: Gaussian noise standard deviation
            clip: Gaussian noise clipping range
            train_ratio: Train/test split ratio
            random_seed: Random seed
            **kwargs: Additional arguments for C3VDDatasetBase
        """
        # Initialize base dataset
        self.base_dataset = C3VDDatasetBase(
            data_root=data_root,
            split=split,
            pair_mode="one_to_one",
            scene_split=True,
            train_ratio=train_ratio,
            random_seed=random_seed,
            **kwargs,
        )

        self.num_points = num_points
        self.sigma = sigma
        self.clip = clip

        # Load fixed perturbations
        if perturbation_file and os.path.exists(perturbation_file):
            import csv

            with open(perturbation_file, "r") as f:
                csvreader = csv.reader(f)
                poses = []
                for row in csvreader:
                    row = [float(i) for i in row]
                    poses.append(row)
                self.perturbations = np.array(poses)[: len(self.base_dataset)]
            print(f"Loaded fixed perturbations from {perturbation_file}")
            print(f"  Perturbation shape: {self.perturbations.shape}")
        else:
            # Generate random perturbations with fixed seed
            torch.manual_seed(713)
            self.perturbations = []
            for _ in range(len(self.base_dataset)):
                x = torch.randn(1, 6)
                x = x / x.norm(p=2, dim=1, keepdim=True) * 0.8
                self.perturbations.append(x.numpy()[0])
            self.perturbations = np.array(self.perturbations)
            print(
                f"Generated {len(self.perturbations)} random perturbations (seed=713)"
            )

        print("C3VD-PointNetLK_Revisited (Fixed Perturbation) Adapter initialized:")
        print(f"  Num points: {num_points}")
        print(f"  Noise (sigma, clip): ({sigma}, {clip})")

    def __len__(self):
        return len(self.base_dataset)

    def transform(self, p0, x):
        """Apply SE(3) transformation using twist parameters."""
        # p0: [N, 3]
        # x: [1, 6], twist-vector (rotation and translation)
        g = plk_utils.exp(x).to(p0)  # [1, 4, 4]
        p1 = plk_utils.transform(g, p0)
        igt = g.squeeze(0)
        return p1, igt

    def __getitem__(self, index):
        """
        Get a data sample with fixed perturbation.

        Args:
            index: Index

        Returns:
            Tuple of (p0, p1, igt)
        """
        # Get base data
        data = self.base_dataset[index]
        target = data["target"]  # [M, 3] numpy

        # Resample to fixed number of points
        template = resample_points(target, self.num_points)

        # Convert to tensor
        pm = torch.from_numpy(template).float()  # [N, 3]

        # Add noise if specified
        p_ = add_noise(pm.clone(), sigma=self.sigma, clip=self.clip)
        p0 = pm

        # Apply fixed transformation
        x = torch.from_numpy(self.perturbations[index][np.newaxis, ...]).to(p0)
        p1, igt = self.transform(p_, x)

        # p0: template, p1: source, igt: transform matrix from p0 to p1
        return p0, p1, igt


def test_adapter():
    """Test the PointNetLK_Revisited adapter."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=str, required=True, help="Path to C3VD_datasets directory"
    )
    args = parser.parse_args()

    print("Testing C3VD-PointNetLK_Revisited Adapter...")
    print("=" * 60)

    # Create dataset
    dataset = C3VDForPointNetLKRevisited(
        data_root=args.data_root,
        num_points=1024,
        split="train",
        mag=0.8,
        sigma=0.0,
        clip=0.0,
        train_ratio=0.7,
        random_seed=42,
    )

    print(f"\nDataset size: {len(dataset)}")

    if len(dataset) > 0:
        # Test loading a sample
        print("\nLoading sample 0...")
        p0, p1, igt = dataset[0]

        print("\nSample 0 data:")
        print(f"  p0 (template) shape: {p0.shape} (expected [1024, 3])")
        print(f"  p1 (source) shape: {p1.shape} (expected [1024, 3])")
        print(f"  igt (transform) shape: {igt.shape} (expected [4, 4])")
        print(f"  p0 dtype: {p0.dtype}")
        print(f"  p1 dtype: {p1.dtype}")
        print(f"  igt dtype: {igt.dtype}")

        # Check if transformation is correct
        print("\n  Transformation matrix (igt):")
        print(f"    {igt}")

        # Verify by applying inverse transformation
        gt = torch.inverse(igt)  # p1 -> p0
        p0_reconstructed = plk_utils.transform(gt.unsqueeze(0), p1)
        recon_error = torch.mean(torch.norm(p0 - p0_reconstructed, dim=1))
        print(f"\n  Reconstruction error: {recon_error.item():.6f} (should be ~0)")

        if recon_error < 0.01:
            print("  ✓ Transformation check passed!")
        else:
            print("  ✗ Transformation check failed!")

        # Test multiple samples
        print("\nTesting batch loading (5 samples)...")
        for i in range(min(5, len(dataset))):
            p0, p1, igt = dataset[i]
            assert p0.shape == (1024, 3), f"Wrong shape for p0: {p0.shape}"
            assert p1.shape == (1024, 3), f"Wrong shape for p1: {p1.shape}"
            assert igt.shape == (4, 4), f"Wrong shape for igt: {igt.shape}"

        print("✓ Successfully loaded 5 samples")

        # Test with DataLoader
        print("\nTesting with DataLoader (batch_size=4)...")
        from torch.utils.data import DataLoader

        loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

        for batch_idx, (p0_batch, p1_batch, igt_batch) in enumerate(loader):
            print(f"  Batch {batch_idx}:")
            print(f"    p0 shape: {p0_batch.shape} (expected [4, 1024, 3])")
            print(f"    p1 shape: {p1_batch.shape} (expected [4, 1024, 3])")
            print(f"    igt shape: {igt_batch.shape} (expected [4, 4, 4])")
            if batch_idx >= 2:  # Only test 3 batches
                break

        print("✓ DataLoader test passed!")

    print("\n" + "=" * 60)
    print("PointNetLK_Revisited Adapter test completed!")


if __name__ == "__main__":
    test_adapter()
