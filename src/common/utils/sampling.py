"""
Unified sampling utilities for point cloud processing
Ensures consistent preprocessing across all algorithms
"""

import numpy as np
import torch


def voxel_down_sample(points, num_points):
    """
    Unified VoxelGrid-based downsampling

    This ensures spatial uniformity and preserves geometric structure.
    Same implementation as PointNetLK_c3vd for consistency.

    Args:
        points: numpy array (N, 3) or torch.Tensor (N, 3)
        num_points: target number of points

    Returns:
        sampled_points: numpy array (num_points, 3)
    """
    # Convert to numpy if torch tensor
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    # Ensure float64 for numerical stability
    points = points.astype(np.float64)

    # Compute bounding box
    min_bounds = np.min(points, axis=0)
    max_bounds = np.max(points, axis=0)

    # Compute number of voxels per axis (cubic root)
    n_side = int(np.ceil(num_points ** (1 / 3)))

    # Prevent zero range
    extents = max_bounds - min_bounds
    extents[extents == 0] = 1e-6

    # Compute voxel size
    voxel_size = extents / n_side

    # Compute voxel indices
    coords = np.floor((points - min_bounds) / voxel_size).astype(np.int64)
    coords = np.minimum(coords, n_side - 1)

    # Map 3D voxel indices to 1D keys
    keys = coords[:, 0] + coords[:, 1] * n_side + coords[:, 2] * (n_side**2)

    # Get unique voxels and mapping
    unique_keys, inverse = np.unique(keys, return_inverse=True)

    # Compute centroid of each voxel
    counts = np.bincount(inverse)
    sums = np.zeros((unique_keys.shape[0], 3), dtype=np.float64)
    for dim in range(3):
        sums[:, dim] = np.bincount(inverse, weights=points[:, dim])
    centroids = sums / counts[:, None]

    # Sample to target number
    M = centroids.shape[0]
    if M >= num_points:
        # Random selection from centroids
        indices = np.random.choice(M, num_points, replace=False)
        sampled = centroids[indices]
    else:
        # Need to fill shortage by repeating
        shortage = num_points - M
        extra_idx = np.random.choice(M, shortage, replace=True)
        sampled = np.concatenate([centroids, centroids[extra_idx]], axis=0)

    return sampled.astype(np.float32)


def clean_point_cloud(points, min_points=100):
    """
    Clean point cloud by removing NaN/Inf values

    Args:
        points: numpy array (N, 3) or torch.Tensor (N, 3)
        min_points: minimum required points after cleaning

    Returns:
        cleaned_points: numpy array (M, 3) where M >= min_points

    Raises:
        ValueError: if cleaned points < min_points
    """
    # Convert to numpy if torch tensor
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    # Filter out NaN/Inf
    mask = np.isfinite(points).all(axis=1)
    cleaned = points[mask]

    # Check minimum points requirement
    if len(cleaned) < min_points:
        raise ValueError(
            f"Point cloud has only {len(cleaned)} valid points after cleaning "
            f"(minimum required: {min_points})"
        )

    return cleaned


def preprocess_point_cloud(
    points, num_points, use_voxel_sampling=True, clean_first=True, min_points=100
):
    """
    Complete preprocessing pipeline for point clouds

    Args:
        points: input point cloud, numpy array (N, 3) or torch.Tensor (N, 3)
        num_points: target number of points
        use_voxel_sampling: if True, use VoxelGrid sampling;
                           if False, use random sampling
        clean_first: if True, remove NaN/Inf before sampling
        min_points: minimum points required after cleaning

    Returns:
        processed_points: numpy array (num_points, 3)
    """
    # Step 1: Clean if requested
    if clean_first:
        points = clean_point_cloud(points, min_points=min_points)

    # Step 2: Sample to target number
    if use_voxel_sampling:
        sampled = voxel_down_sample(points, num_points)
    else:
        # Fallback to random sampling
        sampled = random_resample(points, num_points)

    return sampled


def random_resample(points, num_points):
    """
    Simple random resampling (fallback method)

    Args:
        points: numpy array (N, 3) or torch.Tensor (N, 3)
        num_points: target number of points

    Returns:
        sampled_points: numpy array (num_points, 3)
    """
    # Convert to numpy if torch tensor
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    N = len(points)

    if N >= num_points:
        # Random selection without replacement
        idx = np.random.choice(N, num_points, replace=False)
    else:
        # Random selection with replacement
        idx = np.random.choice(N, num_points, replace=True)

    return points[idx].astype(np.float32)


def validate_point_cloud(points, name="point_cloud"):
    """
    Validate point cloud for common issues

    Args:
        points: numpy array (N, 3)
        name: name for error messages

    Raises:
        ValueError: if validation fails
    """
    if not isinstance(points, np.ndarray):
        raise ValueError(f"{name} must be numpy array")

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {points.shape}")

    if len(points) == 0:
        raise ValueError(f"{name} is empty")

    if not np.isfinite(points).all():
        n_invalid = (~np.isfinite(points)).any(axis=1).sum()
        raise ValueError(f"{name} contains {n_invalid} points with NaN/Inf values")


# Alias for backward compatibility
voxel_down_sample_numpy = voxel_down_sample
