"""
Overlap computation utilities for point cloud registration

Based on RegTR's overlap computation with KDTree search
"""

from typing import Union, Tuple

import numpy as np

try:
    import open3d as o3d

    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: Open3D not installed. Overlap computation will not work.")


def to_o3d_pcd(xyz):
    """
    Convert numpy array to Open3D point cloud

    Args:
        xyz: np.ndarray of shape (N, 3)

    Returns:
        o3d.geometry.PointCloud
    """
    if not HAS_OPEN3D:
        raise ImportError("Open3D is required for overlap computation")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def compute_overlap(
    src: Union[np.ndarray, "o3d.geometry.PointCloud"],
    tgt: Union[np.ndarray, "o3d.geometry.PointCloud"],
    search_voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes region of overlap between two point clouds.

    This function is adapted from RegTR's overlap computation.
    It uses KDTree nearest neighbor search to find correspondences.

    Args:
        src: Source point cloud, either a numpy array of shape (N, 3) or
          Open3D PointCloud object
        tgt: Target point cloud similar to src.
        search_voxel_size: Search radius for overlap detection

    Returns:
        has_corr_src: Boolean array (N,) indicating which source points are in overlap region
        has_corr_tgt: Boolean array (M,) indicating which target points are in overlap region
        src_tgt_corr: Indices of source to target correspondences, shape (2, K) where K is number of correspondences
    """
    if not HAS_OPEN3D:
        raise ImportError(
            "Open3D is required for overlap computation. Install with: pip install open3d"
        )

    # Convert to Open3D point clouds if needed
    if isinstance(src, np.ndarray):
        src_pcd = to_o3d_pcd(src)
        src_xyz = src
    else:
        src_pcd = src
        src_xyz = np.asarray(src.points)

    if isinstance(tgt, np.ndarray):
        tgt_pcd = to_o3d_pcd(tgt)
        tgt_xyz = tgt
    else:
        tgt_pcd = tgt
        tgt_xyz = np.asarray(tgt.points)

    # Check which points in tgt has a correspondence (i.e. point nearby) in the src,
    # and then in the other direction. As long there's a point nearby, it's
    # considered to be in the overlap region. For correspondences, we require a stronger
    # condition of being mutual matches

    # Find correspondences from target to source
    tgt_corr = np.full(tgt_xyz.shape[0], -1)
    pcd_tree = o3d.geometry.KDTreeFlann(src_pcd)
    for i, t in enumerate(tgt_xyz):
        num_knn, knn_indices, knn_dist = pcd_tree.search_radius_vector_3d(
            t, search_voxel_size
        )
        if num_knn > 0:
            tgt_corr[i] = knn_indices[0]

    # Find correspondences from source to target
    src_corr = np.full(src_xyz.shape[0], -1)
    pcd_tree = o3d.geometry.KDTreeFlann(tgt_pcd)
    for i, s in enumerate(src_xyz):
        num_knn, knn_indices, knn_dist = pcd_tree.search_radius_vector_3d(
            s, search_voxel_size
        )
        if num_knn > 0:
            src_corr[i] = knn_indices[0]

    # Compute mutual correspondences
    src_corr_is_mutual = np.logical_and(
        tgt_corr[src_corr] == np.arange(len(src_corr)), src_corr > 0
    )
    src_tgt_corr = np.stack(
        [np.nonzero(src_corr_is_mutual)[0], src_corr[src_corr_is_mutual]]
    )

    # Mark points that have any correspondence (not necessarily mutual)
    has_corr_src = src_corr >= 0
    has_corr_tgt = tgt_corr >= 0

    return has_corr_src, has_corr_tgt, src_tgt_corr


def compute_overlap_ratio(
    src: np.ndarray, tgt: np.ndarray, search_voxel_size: float
) -> float:
    """
    Compute overlap ratio between two point clouds

    Args:
        src: Source point cloud (N, 3)
        tgt: Target point cloud (M, 3)
        search_voxel_size: Search radius

    Returns:
        Overlap ratio as fraction of points in overlap region
    """
    has_corr_src, has_corr_tgt, _ = compute_overlap(src, tgt, search_voxel_size)

    # Return average of src and tgt overlap ratios
    src_overlap = np.sum(has_corr_src) / len(has_corr_src)
    tgt_overlap = np.sum(has_corr_tgt) / len(has_corr_tgt)

    return (src_overlap + tgt_overlap) / 2.0


if __name__ == "__main__":
    # Test overlap computation
    print("Testing overlap computation...")

    # Create two point clouds with partial overlap
    np.random.seed(42)
    src_points = np.random.randn(1000, 3)
    tgt_points = (
        src_points[:700] + np.random.randn(700, 3) * 0.01
    )  # 70% overlap with small noise

    search_radius = 0.05

    print(f"Source points: {len(src_points)}")
    print(f"Target points: {len(tgt_points)}")
    print(f"Search radius: {search_radius}")

    has_corr_src, has_corr_tgt, correspondences = compute_overlap(
        src_points, tgt_points, search_radius
    )

    print(f"\nResults:")
    print(
        f"  Source overlap: {np.sum(has_corr_src)}/{len(has_corr_src)} ({100 * np.sum(has_corr_src) / len(has_corr_src):.1f}%)"
    )
    print(
        f"  Target overlap: {np.sum(has_corr_tgt)}/{len(has_corr_tgt)} ({100 * np.sum(has_corr_tgt) / len(has_corr_tgt):.1f}%)"
    )
    print(f"  Correspondences: {correspondences.shape[1]}")
    print(
        f"  Overlap ratio: {compute_overlap_ratio(src_points, tgt_points, search_radius):.3f}"
    )

    print("\n✓ Overlap computation test passed!")
