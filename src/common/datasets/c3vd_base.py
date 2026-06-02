"""
C3VD Dataset Base Class

Provides unified data loading interface for C3VD dataset across different algorithms.
Based on the implementation in PointNetLK_c3vd/ptlk/data/datasets.py
"""

import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from plyfile import PlyData
from torch.utils.data import Dataset


def plyread(file_path: str) -> np.ndarray:
    """
    Read point cloud from PLY file.

    Args:
        file_path: Path to PLY file

    Returns:
        Point cloud as numpy array [N, 3]
    """
    ply_data = PlyData.read(file_path)
    pc = np.vstack(
        [ply_data["vertex"]["x"], ply_data["vertex"]["y"], ply_data["vertex"]["z"]]
    ).T
    return pc.astype(np.float32)


class C3VDDatasetBase(Dataset):
    """
    C3VD dataset base class that provides unified data loading interface.

    This class handles:
    - Loading source and target point cloud pairs from C3VD dataset
    - Different pairing modes (one_to_one, scene_reference, etc.)
    - Scene-based train/test split

    Dataset structure:
        C3VD_datasets/
        ├── C3VD_ply_source/              # Source point clouds (depth)
        │   ├── scene_name_1/
        │   │   ├── 0000_depth_pcd.ply
        │   │   ├── 0001_depth_pcd.ply
        │   │   └── ...
        │   └── scene_name_2/
        │       └── ...
        └── visible_point_cloud_ply_depth/ # Target point clouds (visible)
            ├── scene_name_1/
            │   ├── frame_0000_visible.ply
            │   ├── frame_0001_visible.ply
            │   └── ...
            └── scene_name_2/
                └── ...
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        pair_mode: str = "one_to_one",
        scene_split: bool = True,
        train_ratio: float = 0.7,
        random_seed: int = 42,
        train_scenes: Optional[List[str]] = None,
        val_scenes: Optional[List[str]] = None,
        test_scenes: Optional[List[str]] = None,
        frame_stride: int = 1,
        max_pairs: Optional[int] = None,
    ):
        """
        Initialize C3VD dataset.

        Args:
            data_root: Root directory of C3VD dataset (containing C3VD_ply_source/)
            split: 'train' or 'test'
            pair_mode: Pairing mode - 'one_to_one' or 'scene_reference'
            scene_split: If True, split by scenes; if False, split by samples
            train_ratio: Ratio of training data
                (only used when train/test_scenes not provided)
            random_seed: Random seed for reproducible splitting
            train_scenes: Explicit list of training scenes (overrides train_ratio)
            val_scenes: Explicit list of validation scenes (overrides train_ratio)
            test_scenes: Explicit list of test scenes (overrides train_ratio)
            frame_stride: Keep only pairs whose frame index is divisible by stride
            max_pairs: Optional hard cap on the number of loaded pairs
        """
        super().__init__()

        self.data_root = data_root
        self.split = split
        self.pair_mode = pair_mode
        self.scene_split = scene_split
        self.random_seed = random_seed
        self.frame_stride = max(int(frame_stride), 1)
        self.max_pairs = max_pairs

        # Set source and target directories
        self.source_root = os.path.join(data_root, "C3VD_ply_source")
        if pair_mode == "scene_reference":
            self.target_root = os.path.join(data_root, "C3VD_ref")
        else:
            self.target_root = os.path.join(data_root, "visible_point_cloud_ply_depth")

        # Check if directories exist
        if not os.path.exists(self.source_root):
            raise FileNotFoundError(f"Source directory not found: {self.source_root}")
        if not os.path.exists(self.target_root):
            raise FileNotFoundError(f"Target directory not found: {self.target_root}")

        print("\n====== C3VD Dataset Configuration ======")
        print(f"Split: {split}")
        print(f"Pairing Mode: {pair_mode}")
        print(f"Scene Split: {scene_split}")
        print(f"Frame Stride: {self.frame_stride}")
        print(f"Max Pairs: {self.max_pairs}")
        print(f"Source Directory: {self.source_root}")
        print(f"Target Directory: {self.target_root}")

        # Get all scenes
        all_scenes = self._get_all_scenes()

        # Split scenes or load specified scenes
        if train_scenes is not None and (
            val_scenes is not None or test_scenes is not None
        ):
            self.train_scenes = train_scenes
            self.val_scenes = val_scenes if val_scenes is not None else test_scenes
            self.test_scenes = (
                test_scenes if test_scenes is not None else self.val_scenes
            )
            print("Using provided scene split:")
            print(f"  Train scenes: {len(train_scenes)}")
            print(f"  Val scenes: {len(self.val_scenes or [])}")
            print(f"  Test scenes: {len(self.test_scenes or [])}")
        else:
            self.train_scenes, self.test_scenes = self._split_scenes(
                all_scenes, train_ratio, random_seed
            )
            self.val_scenes = self.test_scenes
            print(f"Auto scene split (ratio={train_ratio}, seed={random_seed}):")
            print(f"  Train scenes: {len(self.train_scenes)}")
            print(f"  Val scenes: {len(self.val_scenes)} (fallback to test split)")
            print(f"  Test scenes: {len(self.test_scenes)}")

        # Get scenes for current split
        if split == "train":
            self.scenes = self.train_scenes
        elif split == "val":
            self.scenes = self.val_scenes
        elif split == "test":
            self.scenes = self.test_scenes
        else:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train', 'val', or 'test'"
            )

        print(f"Current split '{split}' has {len(self.scenes)} scenes")

        # Create point cloud pairs
        self.pairs = []
        self.pair_scenes = []

        if pair_mode == "one_to_one":
            self._create_one_to_one_pairs()
        elif pair_mode == "scene_reference":
            self._create_scene_reference_pairs()
        else:
            raise ValueError(f"Invalid pair_mode: {pair_mode}")

        self._apply_pair_filters()

        print(f"Total point cloud pairs loaded: {len(self.pairs)}")
        print("=========================================\n")

    def _get_all_scenes(self) -> List[str]:
        """Get all scene names from source directory."""
        scenes = []
        for scene_dir in glob.glob(os.path.join(self.source_root, "*")):
            if os.path.isdir(scene_dir):
                scenes.append(os.path.basename(scene_dir))
        scenes.sort()
        return scenes

    def _split_scenes(
        self, scenes: List[str], train_ratio: float, random_seed: int
    ) -> Tuple[List[str], List[str]]:
        """
        Split scenes into train and test sets.

        Args:
            scenes: List of all scene names
            train_ratio: Ratio of training scenes
            random_seed: Random seed for reproducibility

        Returns:
            Tuple of (train_scenes, test_scenes)
        """
        np.random.seed(random_seed)
        shuffled_scenes = scenes.copy()
        np.random.shuffle(shuffled_scenes)

        n_train = int(len(shuffled_scenes) * train_ratio)
        train_scenes = shuffled_scenes[:n_train]
        test_scenes = shuffled_scenes[n_train:]

        train_scenes.sort()
        test_scenes.sort()

        return train_scenes, test_scenes

    def _create_one_to_one_pairs(self):
        """Create one-to-one source-target pairs."""
        pair_count = 0

        for scene in self.scenes:
            # Get source files
            source_pattern = os.path.join(self.source_root, scene, "????_depth_pcd.ply")
            source_files = sorted(glob.glob(source_pattern))

            for source_file in source_files:
                # Extract frame index from source file
                basename = os.path.basename(source_file)
                frame_idx = basename[:4]  # First 4 digits
                if int(frame_idx) % self.frame_stride != 0:
                    continue

                # Construct corresponding target file
                target_file = os.path.join(
                    self.target_root, scene, f"frame_{frame_idx}_visible.ply"
                )

                # Check if target file exists
                if os.path.exists(target_file):
                    self.pairs.append((source_file, target_file))
                    self.pair_scenes.append(scene)
                    pair_count += 1
                else:
                    print(f"Warning: Target file not found: {target_file}")

        print(f"  One-to-one pairs created: {pair_count}")

    def _create_scene_reference_pairs(self):
        """Create source-to-reference pairs against the scene reference cloud."""
        pair_count = 0

        for scene in self.scenes:
            # Get source files
            source_pattern = os.path.join(self.source_root, scene, "????_depth_pcd.ply")
            source_files = sorted(glob.glob(source_pattern))

            # Get reference file (first target file in the scene)
            target_pattern = os.path.join(
                self.target_root, scene, "frame_????_visible.ply"
            )
            target_files = sorted(glob.glob(target_pattern))

            if not target_files:
                print(f"Warning: No target files found for scene {scene}")
                continue

            reference_file = target_files[0]  # Use first file as reference

            # Pair all sources with the reference
            for source_file in source_files:
                frame_idx = os.path.basename(source_file)[:4]
                if int(frame_idx) % self.frame_stride != 0:
                    continue
                self.pairs.append((source_file, reference_file))
                self.pair_scenes.append(scene)
                pair_count += 1

        print(f"  Scene-reference pairs created: {pair_count}")

    def _apply_pair_filters(self) -> None:
        if self.max_pairs is None:
            return
        if self.max_pairs < 0:
            raise ValueError("max_pairs must be non-negative.")

        pair_limit = min(len(self.pairs), int(self.max_pairs))
        self.pairs = self.pairs[:pair_limit]
        self.pair_scenes = self.pair_scenes[:pair_limit]

    def __len__(self) -> int:
        """Return number of point cloud pairs."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a point cloud pair.

        Args:
            idx: Index of the pair

        Returns:
            Dictionary containing:
                - 'source': Source point cloud [N, 3] numpy array
                - 'target': Target point cloud [M, 3] numpy array
                - 'scene': Scene name (str)
                - 'source_file': Source file path (str)
                - 'target_file': Target file path (str)
                - 'source_id': Source point cloud ID (str, e.g., '0000')
                - 'target_id': Target point cloud ID (str, e.g., '0001')
        """
        source_file, target_file = self.pairs[idx]
        scene = self.pair_scenes[idx]

        # Read point clouds
        source = plyread(source_file)
        target = plyread(target_file)

        # Check validity
        if not np.isfinite(source).all() or not np.isfinite(target).all():
            raise ValueError(f"Point cloud at index {idx} contains invalid values")

        if source.shape[0] == 0 or target.shape[0] == 0:
            raise ValueError(f"Point cloud at index {idx} is empty")

        # Extract IDs from filenames
        source_basename = os.path.basename(source_file)
        target_basename = os.path.basename(target_file)

        source_id = (
            source_basename[:4]
            if source_basename.endswith("_depth_pcd.ply")
            else "0000"
        )
        target_id = (
            target_basename[6:10] if target_basename.startswith("frame_") else "0000"
        )

        return {
            "source": source,
            "target": target,
            "scene": scene,
            "source_file": source_file,
            "target_file": target_file,
            "source_id": source_id,
            "target_id": target_id,
            "idx": idx,
        }

    def get_scene_info(self) -> Dict[str, int]:
        """
        Get statistics about scenes in the dataset.

        Returns:
            Dictionary mapping scene names to number of pairs
        """
        scene_counts = {}
        for scene in self.pair_scenes:
            scene_counts[scene] = scene_counts.get(scene, 0) + 1
        return scene_counts


if __name__ == "__main__":
    # Simple test
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=str, required=True, help="Path to C3VD_datasets directory"
    )
    args = parser.parse_args()

    print("Testing C3VDDatasetBase...")

    # Test train split
    train_dataset = C3VDDatasetBase(
        data_root=args.data_root,
        split="train",
        pair_mode="one_to_one",
        scene_split=True,
        train_ratio=0.7,
        random_seed=42,
    )

    print(f"\nTrain dataset size: {len(train_dataset)}")
    print("Scene distribution:", train_dataset.get_scene_info())

    # Test loading a sample
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print("\nSample 0:")
        print(f"  Scene: {sample['scene']}")
        print(f"  Source shape: {sample['source'].shape}")
        print(f"  Target shape: {sample['target'].shape}")
        print(f"  Source ID: {sample['source_id']}")
        print(f"  Target ID: {sample['target_id']}")

    # Test test split
    test_dataset = C3VDDatasetBase(
        data_root=args.data_root,
        split="test",
        pair_mode="one_to_one",
        scene_split=True,
        train_ratio=0.7,
        random_seed=42,
    )

    print(f"\nTest dataset size: {len(test_dataset)}")
    print("Scene distribution:", test_dataset.get_scene_info())
