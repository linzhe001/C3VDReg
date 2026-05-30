"""
SE(3) transformation utilities for point cloud registration
"""

import numpy as np
from scipy.spatial.transform import Rotation


def twist_to_se3(twist):
    """
    Convert twist vector to SE(3) matrix

    Args:
        twist: [6,] numpy array [rx, ry, rz, tx, ty, tz]

    Returns:
        g: [4, 4] SE(3) matrix
    """
    w = twist[:3]  # rotation vector
    v = twist[3:]  # translation vector

    # rotation vector -> rotation matrix
    theta = np.linalg.norm(w)
    if theta > 1e-10:
        R = Rotation.from_rotvec(w).as_matrix()
    else:
        R = np.eye(3)

    # construct SE(3) matrix
    g = np.eye(4)
    g[:3, :3] = R
    g[:3, 3] = v

    return g


def se3_to_twist(g):
    """
    Convert SE(3) matrix to twist vector

    Args:
        g: [4, 4] SE(3) matrix

    Returns:
        twist: [6,] twist vector
    """
    R = g[:3, :3]
    t = g[:3, 3]

    # rotation matrix -> rotation vector
    r = Rotation.from_matrix(R)
    w = r.as_rotvec()

    twist = np.concatenate([w, t])
    return twist


def apply_transform(points, g):
    """
    Apply SE(3) transformation to point cloud

    Args:
        points: [N, 3] or [1, N, 3] point cloud (numpy array or torch tensor)
        g: [4, 4] SE(3) matrix (numpy array)

    Returns:
        points_transformed: [N, 3] or [1, N, 3] transformed point cloud (same type and shape as input)
    """
    import torch

    # Check if input is torch tensor
    is_tensor = isinstance(points, torch.Tensor)
    has_batch_dim = False

    # Convert to numpy if needed
    if is_tensor:
        # Remove batch dimension if present
        if points.ndim == 3 and points.shape[0] == 1:
            has_batch_dim = True
            points_np = points.squeeze(0).detach().cpu().numpy()  # [1, N, 3] -> [N, 3]
        else:
            points_np = points.detach().cpu().numpy()
    else:
        if points.ndim == 3 and points.shape[0] == 1:
            has_batch_dim = True
            points_np = points.squeeze(0)  # [1, N, 3] -> [N, 3]
        else:
            points_np = points

    R = g[:3, :3]
    t = g[:3, 3]

    # points' = R @ points + t
    points_transformed = (R @ points_np.T).T + t

    # Convert back to tensor if input was tensor
    if is_tensor:
        points_transformed = (
            torch.from_numpy(points_transformed).to(points.device).type(points.dtype)
        )
        # Restore batch dimension if needed
        if has_batch_dim:
            points_transformed = points_transformed.unsqueeze(0)  # [N, 3] -> [1, N, 3]
    else:
        # Restore batch dimension if needed
        if has_batch_dim:
            points_transformed = points_transformed[
                np.newaxis, ...
            ]  # [N, 3] -> [1, N, 3]

    return points_transformed


def inverse_transform(g):
    """
    Compute SE(3) inverse transformation

    Args:
        g: [4, 4] SE(3) matrix

    Returns:
        g_inv: [4, 4] inverse matrix
    """
    R = g[:3, :3]
    t = g[:3, 3]

    R_inv = R.T
    t_inv = -R_inv @ t

    g_inv = np.eye(4)
    g_inv[:3, :3] = R_inv
    g_inv[:3, 3] = t_inv

    return g_inv


def compose_transforms(g1, g2):
    """
    Compose two SE(3) transformations: g1 * g2

    Args:
        g1: [4, 4] SE(3) matrix
        g2: [4, 4] SE(3) matrix

    Returns:
        g_composed: [4, 4] composed transformation
    """
    return g1 @ g2


def rotation_matrix_to_euler(R, degrees=True):
    """
    Convert rotation matrix to Euler angles (XYZ convention)

    Args:
        R: [3, 3] rotation matrix
        degrees: return in degrees if True, radians if False

    Returns:
        euler: [3,] array of Euler angles [rx, ry, rz]
    """
    r = Rotation.from_matrix(R)
    euler = r.as_euler("xyz", degrees=degrees)
    return euler


def euler_to_rotation_matrix(euler, degrees=True):
    """
    Convert Euler angles to rotation matrix (XYZ convention)

    Args:
        euler: [3,] array of Euler angles [rx, ry, rz]
        degrees: input in degrees if True, radians if False

    Returns:
        R: [3, 3] rotation matrix
    """
    r = Rotation.from_euler("xyz", euler, degrees=degrees)
    R = r.as_matrix()
    return R
