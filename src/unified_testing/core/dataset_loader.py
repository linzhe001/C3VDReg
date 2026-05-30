"""
Dataset loader for ModelNet40 and C3VD
"""

import os
import sys
import numpy as np

# Import unified preprocessing utilities
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "common"))
)
from utils.sampling import preprocess_point_cloud, clean_point_cloud, voxel_down_sample


class DatasetLoader:
    """Unified dataset loader"""

    def __init__(
        self, dataset_type, data_root, num_points=1024, split="test", **kwargs
    ):
        """
        Args:
            dataset_type: 'modelnet' or 'c3vd'
            data_root: dataset root directory
            num_points: number of points to sample
            split: 'train' or 'test'
            **kwargs: additional dataset-specific parameters
                use_voxel_sampling: bool (C3VD only) - use VoxelGrid sampling instead of random
                                   Default: True for C3VD
        """
        self.dataset_type = dataset_type.lower()
        self.data_root = data_root
        self.num_points = num_points
        self.split = split
        self.kwargs = kwargs

        # VoxelGrid sampling control (C3VD only, default enabled)
        if self.dataset_type == "c3vd":
            self.use_voxel_sampling = kwargs.get("use_voxel_sampling", True)
        else:
            # ModelNet always uses random sampling
            self.use_voxel_sampling = False

        # load dataset
        if self.dataset_type == "modelnet":
            self.dataset = self._load_modelnet()
        elif self.dataset_type == "c3vd":
            self.dataset = self._load_c3vd()
        elif self.dataset_type == "pitvideo":
            raise ValueError(
                "dataset_type='pitvideo' has been moved to research/unified_testing_legacy "
                "and is no longer supported by the stable benchmark runtime."
            )
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        print(f"Loaded {self.dataset_type} dataset: {len(self.dataset)} samples")

    def _load_modelnet(self):
        """Load ModelNet40 dataset"""
        # try to use existing ModelNet40 dataloader
        try:
            # add PointNetLK path
            pointnetlk_path = os.path.join(
                os.path.dirname(self.data_root), "PointNetLK"
            )
            if os.path.exists(pointnetlk_path):
                sys.path.insert(0, pointnetlk_path)

            from data import ModelNet
            import ptlk.data.transforms as transforms
            import torchvision

            # get category file if specified
            category_file = self.kwargs.get("category_file", None)

            # Create transform pipeline (consistent with PointNetLK training)
            # This normalizes point clouds to unit cube [-0.5, 0.5] range
            transform = torchvision.transforms.Compose(
                [
                    transforms.Mesh2Points(),
                    transforms.OnUnitCube(),
                ]
            )

            dataset = ModelNet(
                self.data_root,
                train=(self.split == "train"),
                transform=transform,
                classinfo=None
                if category_file is None
                else self._load_category_file(category_file),
            )

            return dataset

        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not import ModelNet dataloader with transforms: {e}")
            print("Using fallback file-based loading")
            return self._load_modelnet_fallback()

    def _load_category_file(self, category_file):
        """Load category file and return classinfo tuple"""
        if not os.path.exists(category_file):
            return None

        with open(category_file, "r") as f:
            categories = [line.strip() for line in f if line.strip()]

        class_to_idx = {cat: i for i, cat in enumerate(categories)}
        return (categories, class_to_idx)

    def _load_modelnet_fallback(self):
        """Fallback ModelNet40 loader (file-based)"""
        from ..utils.file_io import get_file_list_modelnet

        category = self.kwargs.get("category", None)
        file_list = get_file_list_modelnet(
            self.data_root, category=category, split=self.split
        )

        return file_list

    def _load_c3vd(self):
        """Load C3VD dataset"""
        # try to use existing C3VD dataloader
        try:
            # add common path
            common_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "common")
            )
            if os.path.exists(common_path):
                sys.path.insert(0, common_path)

            from datasets.c3vd_base import C3VDDatasetBase

            # get parameters
            scene_split = self.kwargs.get("scene_split", True)
            train_ratio = self.kwargs.get("train_ratio", 0.7)
            random_seed = self.kwargs.get("random_seed", 42)
            pair_mode = self.kwargs.get("pair_mode", "one_to_one")
            train_scenes = self.kwargs.get("train_scenes", None)
            test_scenes = self.kwargs.get("test_scenes", None)

            dataset = C3VDDatasetBase(
                data_root=self.data_root,
                split=self.split,
                scene_split=scene_split,
                train_ratio=train_ratio,
                random_seed=random_seed,
                pair_mode=pair_mode,
                train_scenes=train_scenes,
                test_scenes=test_scenes,
            )

            return dataset

        except ImportError as e:
            print(f"Warning: Could not import C3VD dataloader: {e}")
            print("Using fallback file-based loading")
            return self._load_c3vd_fallback()

    def _load_c3vd_fallback(self):
        """Fallback C3VD loader (file-based)"""
        from ..utils.file_io import get_file_list_c3vd

        scene = self.kwargs.get("scene", None)
        file_list = get_file_list_c3vd(self.data_root, scene=scene)

        return file_list

    def _resample_points(self, points, num_points, use_voxel=False):
        """
        Resample point cloud to fixed number of points.

        Args:
            points: numpy array [N, 3]
            num_points: target number of points
            use_voxel: if True, use VoxelGrid sampling (uniform spatial distribution)
                      if False, use random sampling (faster but non-uniform)

        Returns:
            resampled points [num_points, 3]
        """
        if use_voxel:
            # Use unified VoxelGrid sampling (same as PointNetLK_c3vd training)
            return voxel_down_sample(points, num_points)
        else:
            # Random sampling (original method for ModelNet40)
            N = points.shape[0]
            if N == num_points:
                return points
            elif N > num_points:
                indices = np.random.choice(N, num_points, replace=False)
                return points[indices]
            else:
                indices = np.random.choice(N, num_points, replace=True)
                return points[indices]

    def __len__(self):
        """Get dataset length"""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get a data sample

        Returns:
            data: dict with 'source', 'target', 'metadata'
        """
        if self.dataset_type == "modelnet":
            return self._get_modelnet_item(idx)
        elif self.dataset_type == "c3vd":
            return self._get_c3vd_item(idx)
        elif self.dataset_type == "pitvideo":
            raise ValueError(
                "dataset_type='pitvideo' is not available in the stable benchmark runtime."
            )

    def _get_modelnet_item(self, idx):
        """Get ModelNet40 item"""
        try:
            # try dataset __getitem__
            points, label = self.dataset[idx]
            source = target = points  # ModelNet40: same point cloud

            metadata = {
                "idx": idx,
                "label": label if isinstance(label, int) else label.item(),
                "dataset": "modelnet",
            }

        except (AttributeError, TypeError, ValueError):
            # fallback: file-based
            from ..utils.file_io import load_point_cloud

            file_info = self.dataset[idx]
            points = load_point_cloud(file_info["full_path"], self.num_points)
            source = target = points

            metadata = {
                "idx": idx,
                "category": file_info["category"],
                "filename": file_info["filename"],
                "dataset": "modelnet",
            }

        return {"source": source, "target": target, "metadata": metadata}

    def _get_c3vd_item(self, idx):
        """Get C3VD item with unified VoxelGrid preprocessing"""
        try:
            # try dataset __getitem__
            data = self.dataset[idx]

            # C3VD dataset returns dict with 'source', 'target', etc.
            source = data["source"]  # [N, 3] numpy array
            target = data["target"]  # [M, 3] numpy array

            # Convert to numpy if torch tensor
            import torch

            if isinstance(source, torch.Tensor):
                source = source.cpu().numpy()
            if isinstance(target, torch.Tensor):
                target = target.cpu().numpy()

            # Apply unified preprocessing:
            # 1. Clean NaN/Inf (same as PointNetLK_c3vd training)
            # 2. VoxelGrid sampling (same as PointNetLK_c3vd training)
            try:
                source = clean_point_cloud(source, min_points=100)
                target = clean_point_cloud(target, min_points=100)
            except ValueError as e:
                print(
                    f"Warning: Sample {idx} has insufficient valid points after cleaning: {e}"
                )
                # Return original points without cleaning
                pass

            # Use configured sampling method (VoxelGrid or random)
            # For C3VD: default VoxelGrid (matches training), configurable in YAML
            source = self._resample_points(
                source, self.num_points, use_voxel=self.use_voxel_sampling
            )
            target = self._resample_points(
                target, self.num_points, use_voxel=self.use_voxel_sampling
            )

            metadata = {
                "idx": idx,
                "scene": data.get("scene", ""),
                "source_file": data.get("source_file", ""),
                "target_file": data.get("target_file", ""),
                "source_id": data.get("source_id", ""),
                "target_id": data.get("target_id", ""),
                "dataset": "c3vd",
            }

        except (AttributeError, TypeError, KeyError):
            # fallback: file-based
            from ..utils.file_io import load_point_cloud

            file_info = self.dataset[idx]
            source = load_point_cloud(file_info["source_path"], self.num_points)
            target = load_point_cloud(file_info["target_path"], self.num_points)

            # Apply same preprocessing for fallback
            try:
                source = clean_point_cloud(source, min_points=100)
                target = clean_point_cloud(target, min_points=100)
            except ValueError:
                pass

            source = self._resample_points(
                source, self.num_points, use_voxel=self.use_voxel_sampling
            )
            target = self._resample_points(
                target, self.num_points, use_voxel=self.use_voxel_sampling
            )

            metadata = {
                "idx": idx,
                "scene": file_info["scene"],
                "source_file": file_info["source"],
                "target_file": file_info["target"],
                "dataset": "c3vd",
            }

        return {"source": source, "target": target, "metadata": metadata}

    def get_sample_random(self):
        """
        Get a random sample from dataset

        Returns:
            data: dict with 'source', 'target', 'metadata'
        """
        idx = np.random.randint(0, len(self))
        return self[idx]

    def get_metadata(self):
        """
        Get dataset metadata

        Returns:
            metadata: dict with dataset info
        """
        return {
            "type": self.dataset_type,
            "root": self.data_root,
            "split": self.split,
            "num_points": self.num_points,
            "length": len(self),
            "kwargs": self.kwargs,
        }
