"""
C3VD Dataset Adapter for PointNetLK (Original)

This adapter converts C3VD dataset to PointNetLK's expected format.
PointNetLK uses the original ModelNet-style interface with CADset4tracking wrapper.
"""

import os
import sys

import numpy as np
import torch
import torch.utils.data

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

# Import PointNetLK modules
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
pointnetlk_path = os.path.join(project_root, "baselines", "PointNetLK")
sys.path.insert(0, pointnetlk_path)
import ptlk  # noqa: E402


def anatomy_label_from_scene(scene: str) -> str:
    """Map a C3VD scene name to a coarse anatomy label."""
    label = scene.split("_", 1)[0]
    aliases = {
        "sigmod": "sigmoid",
    }
    return aliases.get(label, label)


def class_names_from_scenes(scenes, label_mode: str):
    """Build deterministic classifier classes from scene names."""
    if label_mode == "scene":
        return sorted(set(scenes))
    if label_mode == "anatomy":
        return sorted({anatomy_label_from_scene(scene) for scene in scenes})
    raise ValueError(
        "Invalid classifier_label_mode: "
        f"{label_mode!r}. Expected 'scene' or 'anatomy'."
    )


class C3VDForPointNetLK(torch.utils.data.Dataset):
    """
    C3VD dataset base wrapper for PointNetLK (original).

    This class mimics the ModelNet dataset interface that PointNetLK expects.
    Returns (point_cloud, class_label) tuples like ModelNet.
    """

    def __init__(
        self,
        data_root: str,
        num_points: int = 1024,
        split: str = "train",
        train_ratio: float = 0.7,
        random_seed: int = 42,
        sampling_mode: str = "voxel",
        normalize_mode: str = "unit_cube",
        classifier_label_mode: str = "scene",
        transform=None,
        **kwargs,
    ):
        """
        Initialize C3VD dataset for PointNetLK.

        Args:
            data_root: Root directory of C3VD dataset
            num_points: Number of points to sample
            split: 'train' or 'test'
            train_ratio: Train/test split ratio
            random_seed: Random seed
            transform: Optional transform function
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
        self.sampling_mode = sampling_mode
        self.normalize_mode = normalize_mode
        self.classifier_label_mode = classifier_label_mode
        self.transform = transform
        self.split = split

        # Create a pseudo class list (PointNetLK expects this).  For C3VD we can
        # use either exact scene names or coarse anatomy labels such as sigmoid.
        label_scenes = (
            list(self.base_dataset.train_scenes)
            + list(self.base_dataset.val_scenes or [])
            + list(self.base_dataset.test_scenes or [])
        )
        self.classes = class_names_from_scenes(label_scenes, classifier_label_mode)
        self.num_classes = len(self.classes)
        self.class_to_idx = {
            class_name: idx for idx, class_name in enumerate(self.classes)
        }

        print("C3VD-PointNetLK Adapter initialized:")
        print(f"  Split: {split}")
        print(f"  Num points: {num_points}")
        print(f"  Sampling mode: {sampling_mode}")
        print(f"  Normalize mode: {normalize_mode}")
        print(f"  Classifier label mode: {classifier_label_mode}")
        print(f"  Classes: {self.classes}")
        print(f"  Num classes: {self.num_classes}")
        print(f"  Dataset size: {len(self.base_dataset)}")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        """
        Get a data sample in PointNetLK format.

        Args:
            index: Index

        Returns:
            Tuple of (point_cloud, class_label)
            - point_cloud: [N, 3] tensor (resampled target point cloud)
            - class_label: int (scene/anatomy class index)
        """
        # Get base data
        data = self.base_dataset[index]
        source = data["source"]  # [N, 3] numpy
        target = data["target"]  # [M, 3] numpy
        scene = data["scene"]

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

        # Return both source and target as a tuple
        # This allows C3VDset4tracking to use both point clouds
        point_clouds = (source, target)

        # Get classifier label.
        if self.classifier_label_mode == "scene":
            class_name = scene
        elif self.classifier_label_mode == "anatomy":
            class_name = anatomy_label_from_scene(scene)
        else:
            raise ValueError(
                "Invalid classifier_label_mode: "
                f"{self.classifier_label_mode!r}."
            )
        class_label = self.class_to_idx[class_name]

        return point_clouds, class_label


class C3VDset4tracking(torch.utils.data.Dataset):
    """
    C3VD dataset wrapper for tracking (training PointNetLK).

    This mimics the CADset4tracking interface from PointNetLK.
    Applies random rigid transformations to create training pairs.
    """

    def __init__(
        self,
        dataset,
        rigid_transform,
        source_modifier=None,
        template_modifier=None,
        perturbation_enabled: bool = False,
        rotation_deg: float = 0.0,
        translation_m: float = 0.0,
        noise_sigma: float = 0.0,
        noise_clip: float = 0.0,
        apply_noise_to: str = "source",
    ):
        """
        Initialize tracking dataset.

        Args:
            dataset: Base dataset (C3VDForPointNetLK)
            rigid_transform: Transformation function that generates random SE(3)
            source_modifier: Optional modifier for source point cloud
            template_modifier: Optional modifier for template point cloud
        """
        self.dataset = dataset
        self.rigid_transform = rigid_transform
        self.source_modifier = source_modifier
        self.template_modifier = template_modifier
        self.perturbation_enabled = bool(perturbation_enabled)
        self.rotation_deg = float(rotation_deg)
        self.translation_m = float(translation_m)
        self.noise_sigma = float(noise_sigma)
        self.noise_clip = float(noise_clip)
        self.apply_noise_to = apply_noise_to

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Get a tracking pair.

        Returns:
            Tuple of (p0, p1, gt)
            - p0: template point cloud [N, 3] (target)
            - p1: source point cloud [N, 3] (transformed source)
            - gt: transformation matrix [4, 4] (p1 -> p0, inverse transform)
        """
        point_clouds, _ = self.dataset[
            index
        ]  # point_clouds: (source, target), class_label: int
        source, target = point_clouds  # both [N, 3] numpy arrays

        # Convert to tensors
        source_tensor = torch.from_numpy(source).float()
        target_tensor = torch.from_numpy(target).float()

        # Apply source modifier if specified
        if self.source_modifier is not None:
            source_tensor = self.source_modifier(source_tensor)

        # Apply template modifier if specified
        if self.template_modifier is not None:
            p0 = self.template_modifier(target_tensor)
        else:
            p0 = target_tensor

        if self.perturbation_enabled:
            perturb_seed = (
                None
                if self.dataset.split == "train"
                else self.dataset.base_dataset.random_seed + index * 9973 + 2
            )
            source_np = source_tensor.cpu().numpy()
            target_np = p0.cpu().numpy()
            (
                source_perturbed,
                target_perturbed,
                _,
                perturb_meta,
            ) = apply_pair_perturbation(
                source_np,
                target_np,
                rotation_deg=self.rotation_deg,
                translation_m=self.translation_m,
                noise_sigma=self.noise_sigma,
                noise_clip=self.noise_clip,
                apply_noise_to=self.apply_noise_to,
                seed=perturb_seed,
            )
            p1 = torch.from_numpy(source_perturbed).float()
            p0 = torch.from_numpy(target_perturbed).float()
            igt = torch.from_numpy(
                np.asarray(perturb_meta["rigid_transform"], dtype=np.float32)
            )
        else:
            # Apply random transformation to source
            p1 = self.rigid_transform(source_tensor)

            # Get the transformation matrix
            # IMPORTANT: PointNetLK's loss function: ||est_g @ igt - I||
            # This expects est_g ≈ igt^(-1), so we should return igt (forward transform)
            # Since p1 = igt @ source, we return igt to match original PointNetLK
            igt = self.rigid_transform.igt  # igt: p0 -> p1 (forward transform)

        # p0: template, p1: transformed source, igt: forward transform.
        return p0, p1, igt


class C3VDset4tracking_fixed_perturbation(torch.utils.data.Dataset):
    """
    C3VD dataset with fixed perturbations for testing.

    This mimics the CADset4tracking_fixed_perturbation interface from PointNetLK.
    """

    @staticmethod
    def generate_perturbations(batch_size, mag, randomly=False):
        """Generate twist perturbations."""
        if randomly:
            amp = torch.rand(batch_size, 1) * mag
        else:
            amp = mag
        x = torch.randn(batch_size, 6)
        x = x / x.norm(p=2, dim=1, keepdim=True) * amp
        return x.numpy()

    @staticmethod
    def generate_rotations(batch_size, mag, randomly=False):
        """Generate rotation-only perturbations."""
        if randomly:
            amp = torch.rand(batch_size, 1) * mag
        else:
            amp = mag
        w = torch.randn(batch_size, 3)
        w = w / w.norm(p=2, dim=1, keepdim=True) * amp
        v = torch.zeros(batch_size, 3)
        x = torch.cat((w, v), dim=1)
        return x.numpy()

    def __init__(
        self,
        dataset,
        perturbation,
        source_modifier=None,
        template_modifier=None,
        fmt_trans=False,
    ):
        """
        Initialize fixed perturbation dataset.

        Args:
            dataset: Base dataset (C3VDForPointNetLK)
            perturbation: Array of twist vectors [len(dataset), 6]
            source_modifier: Optional modifier for source point cloud
            template_modifier: Optional modifier for template point cloud
            fmt_trans: If True, perturbation is (rotation, translation); else twist
        """
        self.dataset = dataset
        self.perturbation = np.array(perturbation)  # twist (len(dataset), 6)
        self.source_modifier = source_modifier
        self.template_modifier = template_modifier
        self.fmt_trans = fmt_trans

    def do_transform(self, p0, x):
        """
        Apply transformation.

        Args:
            p0: Point cloud [N, 3]
            x: Twist or (rotation, translation) [1, 6]

        Returns:
            p1: Transformed point cloud [N, 3]
            igt: Transformation matrix [4, 4]
        """
        if not self.fmt_trans:
            # x: twist-vector
            g = ptlk.se3.exp(x).to(p0)  # [1, 4, 4]
            p1 = ptlk.se3.transform(g, p0)
            igt = g.squeeze(0)  # igt: p0 -> p1
        else:
            # x: rotation and translation
            w = x[:, 0:3]
            q = x[:, 3:6]
            R = ptlk.so3.exp(w).to(p0)  # [1, 3, 3]
            g = torch.zeros(1, 4, 4)
            g[:, 3, 3] = 1
            g[:, 0:3, 0:3] = R  # rotation
            g[:, 0:3, 3] = q  # translation
            p1 = ptlk.se3.transform(g, p0)
            igt = g.squeeze(0)  # igt: p0 -> p1
        return p1, igt

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Get a sample with fixed perturbation.

        Returns:
            Tuple of (p0, p1, igt)
        """
        twist = (
            torch.from_numpy(np.array(self.perturbation[index])).contiguous().view(1, 6)
        )
        point_clouds, _ = self.dataset[index]  # point_clouds: (source, target)
        source, target = point_clouds

        # Convert to tensors
        source_tensor = torch.from_numpy(source).float()
        target_tensor = torch.from_numpy(target).float()

        x = twist.to(source_tensor)

        # Apply source modifier if specified
        if self.source_modifier is not None:
            source_tensor = self.source_modifier(source_tensor)

        # Apply transformation to source
        p1, igt = self.do_transform(source_tensor, x)

        # Apply template modifier if specified
        if self.template_modifier is not None:
            p0 = self.template_modifier(target_tensor)
        else:
            p0 = target_tensor

        # p0: template, p1: transformed source, igt: transform from p0 to p1.
        return p0, p1, igt


def test_adapter():
    """Test the PointNetLK adapter."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=str, required=True, help="Path to C3VD_datasets directory"
    )
    args = parser.parse_args()

    print("Testing C3VD-PointNetLK Adapter...")
    print("=" * 60)

    # Test 1: Base dataset
    print("\n1. Testing C3VDForPointNetLK (base dataset)...")
    dataset = C3VDForPointNetLK(
        data_root=args.data_root,
        num_points=1024,
        split="train",
        train_ratio=0.7,
        random_seed=42,
    )

    print(f"  Dataset size: {len(dataset)}")
    print(f"  Number of classes (scenes): {dataset.num_classes}")
    print(f"  Classes: {dataset.classes}")

    if len(dataset) > 0:
        # Test loading a sample
        print("\n  Loading sample 0...")
        pc, label = dataset[0]
        print(f"    Point cloud shape: {pc.shape} (expected [1024, 3])")
        print(f"    Class label: {label} (scene: {dataset.classes[label]})")
        print(f"    Point cloud dtype: {pc.dtype}")

        # Test multiple samples
        print("\n  Testing 5 samples...")
        for i in range(min(5, len(dataset))):
            pc, label = dataset[i]
            assert pc.shape == (1024, 3), f"Wrong shape: {pc.shape}"
            assert 0 <= label < dataset.num_classes, f"Invalid label: {label}"
        print("  ✓ Successfully loaded 5 samples")

    # Test 2: Tracking dataset with random transformations
    print("\n2. Testing C3VDset4tracking (with random transforms)...")

    # Create rigid transform
    from ptlk.data.transforms import RandomTransformSE3

    rigid_transform = RandomTransformSE3(mag=0.8, mag_randomly=True)

    # Create tracking dataset
    track_dataset = C3VDset4tracking(dataset=dataset, rigid_transform=rigid_transform)

    print(f"  Tracking dataset size: {len(track_dataset)}")

    if len(track_dataset) > 0:
        print("\n  Loading sample 0...")
        p0, p1, igt = track_dataset[0]

        print(f"    p0 (template) shape: {p0.shape} (expected [1024, 3])")
        print(f"    p1 (source) shape: {p1.shape} (expected [1024, 3])")
        print(f"    igt (transform) shape: {igt.shape} (expected [4, 4])")

        # Verify transformation
        print("\n    Transformation matrix (igt):")
        print(f"      {igt}")

        # Check if p1 = igt @ p0
        p0_homo = torch.cat([p0, torch.ones(p0.size(0), 1)], dim=1)  # [N, 4]
        p1_reconstructed = (igt @ p0_homo.T).T[:, :3]  # [N, 3]
        recon_error = torch.mean(torch.norm(p1 - p1_reconstructed, dim=1))
        print(f"\n    Reconstruction error: {recon_error.item():.6f} (should be ~0)")

        if recon_error < 0.01:
            print("    ✓ Transformation check passed!")
        else:
            print("    ✗ Transformation check failed!")

    # Test 3: Fixed perturbation dataset
    print("\n3. Testing C3VDset4tracking_fixed_perturbation...")

    # Generate fixed perturbations
    num_samples = min(10, len(dataset))
    perturbations = C3VDset4tracking_fixed_perturbation.generate_perturbations(
        num_samples, mag=0.8, randomly=False
    )

    # Create fixed perturbation dataset
    fixed_dataset = C3VDset4tracking_fixed_perturbation(
        dataset=dataset, perturbation=perturbations
    )

    print(f"  Fixed perturbation dataset size: {len(fixed_dataset)}")

    if len(fixed_dataset) > 0:
        print("\n  Loading sample 0...")
        p0, p1, igt = fixed_dataset[0]

        print(f"    p0 (template) shape: {p0.shape} (expected [1024, 3])")
        print(f"    p1 (source) shape: {p1.shape} (expected [1024, 3])")
        print(f"    igt (transform) shape: {igt.shape} (expected [4, 4])")

        # Test reproducibility
        print("\n  Testing reproducibility...")
        p0_2, p1_2, igt_2 = fixed_dataset[0]

        if torch.allclose(igt, igt_2):
            print("    ✓ Fixed perturbation is reproducible!")
        else:
            print("    ✗ Fixed perturbation is not reproducible!")

    # Test 4: DataLoader
    print("\n4. Testing with DataLoader...")
    from torch.utils.data import DataLoader

    loader = DataLoader(track_dataset, batch_size=4, shuffle=True, num_workers=0)

    for batch_idx, (p0_batch, p1_batch, igt_batch) in enumerate(loader):
        print(f"  Batch {batch_idx}:")
        print(f"    p0 shape: {p0_batch.shape} (expected [4, 1024, 3])")
        print(f"    p1 shape: {p1_batch.shape} (expected [4, 1024, 3])")
        print(f"    igt shape: {igt_batch.shape} (expected [4, 4, 4])")
        if batch_idx >= 2:  # Only test 3 batches
            break

    print("  ✓ DataLoader test passed!")

    print("\n" + "=" * 60)
    print("PointNetLK Adapter test completed!")


if __name__ == "__main__":
    test_adapter()
