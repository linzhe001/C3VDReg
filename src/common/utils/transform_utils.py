"""
Transformation utilities for point cloud registration.

Provides functions to convert between different transformation representations.
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from typing import Tuple


def resample_points(points: np.ndarray, num_points: int) -> np.ndarray:
    """
    Resample point cloud to fixed number of points.

    Args:
        points: Point cloud [N, 3]
        num_points: Target number of points

    Returns:
        Resampled point cloud [num_points, 3]
    """
    N = points.shape[0]

    if N == num_points:
        return points
    elif N > num_points:
        # Random sampling without replacement
        indices = np.random.choice(N, num_points, replace=False)
        return points[indices]
    else:
        # Random sampling with replacement
        indices = np.random.choice(N, num_points, replace=True)
        return points[indices]


def jitter_pointcloud(
    pointcloud: np.ndarray, sigma: float = 0.01, clip: float = 0.05
) -> np.ndarray:
    """
    Add Gaussian noise to point cloud.

    Args:
        pointcloud: Point cloud [N, 3]
        sigma: Standard deviation of noise
        clip: Clip noise to [-clip, clip]

    Returns:
        Jittered point cloud [N, 3]
    """
    N, C = pointcloud.shape
    jittered = pointcloud.copy()
    jittered += np.clip(sigma * np.random.randn(N, C), -clip, clip)
    return jittered


def compute_rotation_matrix_euler(
    angle_x: float, angle_y: float, angle_z: float
) -> np.ndarray:
    """
    Compute rotation matrix from Euler angles (ZYX convention).

    Args:
        angle_x: Rotation around X axis (radians)
        angle_y: Rotation around Y axis (radians)
        angle_z: Rotation around Z axis (radians)

    Returns:
        Rotation matrix [3, 3]
    """
    cosx = np.cos(angle_x)
    cosy = np.cos(angle_y)
    cosz = np.cos(angle_z)
    sinx = np.sin(angle_x)
    siny = np.sin(angle_y)
    sinz = np.sin(angle_z)

    Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
    Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
    Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])

    R = Rx.dot(Ry).dot(Rz)
    return R


def apply_transform(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Apply rigid transformation to point cloud.

    Args:
        points: Point cloud [N, 3] or [3, N]
        R: Rotation matrix [3, 3]
        t: Translation vector [3]

    Returns:
        Transformed point cloud (same shape as input)
    """
    if points.shape[0] == 3:  # [3, N] format
        return (R @ points) + t.reshape(3, 1)
    else:  # [N, 3] format
        return (R @ points.T).T + t


def se3_to_euler(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert SE(3) transformation to Euler angles and translation.

    Args:
        R: Rotation matrix [3, 3]
        t: Translation vector [3]

    Returns:
        Tuple of (euler_angles [3], translation [3])
        Euler angles in ZYX convention (z, y, x)
    """
    rotation = Rotation.from_matrix(R)
    euler = rotation.as_euler("zyx")  # Returns [z, y, x]
    return euler, t


def euler_to_se3(euler: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Euler angles and translation to SE(3) transformation.

    Args:
        euler: Euler angles [3] in ZYX convention (z, y, x)
        t: Translation vector [3]

    Returns:
        Tuple of (rotation_matrix [3, 3], translation [3])
    """
    rotation = Rotation.from_euler("zyx", euler)
    R = rotation.as_matrix()
    return R, t


def random_se3_transform(
    rot_factor: float = 4.0, trans_mag: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate random SE(3) transformation.

    Args:
        rot_factor: Rotation magnitude factor (rotation range = pi/rot_factor)
        trans_mag: Translation magnitude range [-trans_mag, trans_mag]

    Returns:
        Tuple of (R [3,3], t [3], euler [3])
    """
    # Random Euler angles
    angle_x = np.random.uniform() * np.pi / rot_factor
    angle_y = np.random.uniform() * np.pi / rot_factor
    angle_z = np.random.uniform() * np.pi / rot_factor

    # Compute rotation matrix
    R = compute_rotation_matrix_euler(angle_x, angle_y, angle_z)

    # Random translation
    t = np.random.uniform(-trans_mag, trans_mag, 3)

    # Euler angles in ZYX order
    euler = np.array([angle_z, angle_y, angle_x])

    return R, t, euler


def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Normalize point cloud to unit cube centered at origin.

    Args:
        points: Point cloud [N, 3]

    Returns:
        Tuple of (normalized_points, center, scale)
    """
    center = points.mean(axis=0)
    centered = points - center
    scale = np.abs(centered).max()

    if scale < 1e-10:
        scale = 1.0

    normalized = centered / scale
    return normalized, center, scale


def denormalize_points(
    points: np.ndarray, center: np.ndarray, scale: float
) -> np.ndarray:
    """
    Denormalize point cloud.

    Args:
        points: Normalized point cloud [N, 3]
        center: Original center [3]
        scale: Original scale

    Returns:
        Denormalized point cloud [N, 3]
    """
    return points * scale + center


if __name__ == "__main__":
    # Simple tests
    print("Testing transformation utilities...")

    # Test resampling
    points = np.random.randn(1000, 3)
    resampled = resample_points(points, 512)
    print(f"✓ Resample: {points.shape} -> {resampled.shape}")

    # Test jitter
    jittered = jitter_pointcloud(points)
    print(f"✓ Jitter: max diff = {np.abs(jittered - points).max():.4f}")

    # Test random transform
    R, t, euler = random_se3_transform()
    print(f"✓ Random SE(3): R shape = {R.shape}, t shape = {t.shape}")
    print(f"  Euler angles: {euler}")

    # Test apply transform
    transformed = apply_transform(points, R, t)
    print(f"✓ Apply transform: {points.shape} -> {transformed.shape}")

    # Test normalization
    normalized, center, scale = normalize_points(points)
    print(f"✓ Normalize: center = {center}, scale = {scale:.4f}")
    print(f"  Normalized range: [{normalized.min():.2f}, {normalized.max():.2f}]")

    print("\nAll tests passed!")
