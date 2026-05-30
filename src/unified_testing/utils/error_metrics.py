"""
Unified error calculation functions for point cloud registration
"""

import numpy as np
from scipy.spatial.transform import Rotation
from .transform_utils import se3_to_twist


def compute_rotation_error_deg(R_pred, R_gt):
    """
    Compute rotation error using SO(3) geodesic distance

    Args:
        R_pred: [3, 3] predicted rotation matrix
        R_gt: [3, 3] ground truth rotation matrix

    Returns:
        error_deg: rotation error in degrees
    """
    # relative rotation
    R_diff = R_pred.T @ R_gt

    # use trace formula
    trace = np.trace(R_diff)

    # numerical stability
    cos_theta = np.clip((trace - 1) / 2, -1.0, 1.0)

    # angle (radians -> degrees)
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)

    return theta_deg


def compute_translation_error(t_pred, t_gt):
    """
    Compute translation error using L2 norm

    Args:
        t_pred: [3,] predicted translation vector
        t_gt: [3,] ground truth translation vector

    Returns:
        error: translation error
    """
    return np.linalg.norm(t_pred - t_gt)


def compute_se3_distance(g_pred, g_gt):
    """
    Compute SE(3) distance

    Args:
        g_pred: [4, 4] predicted SE(3) matrix
        g_gt: [4, 4] ground truth SE(3) matrix

    Returns:
        distance: SE(3) distance
    """
    # relative transformation
    g_diff = g_pred @ np.linalg.inv(g_gt)

    # convert to Lie algebra
    twist = se3_to_twist(g_diff)

    # compute norm
    distance = np.linalg.norm(twist)

    return distance


def compute_point_mse(source, target, R_pred, t_pred):
    """
    Compute point-to-point mean squared error

    Args:
        source: [N, 3] or [1, N, 3] source point cloud (numpy array or torch tensor)
        target: [N, 3] or [1, N, 3] target point cloud (numpy array or torch tensor)
        R_pred: [3, 3] predicted rotation matrix (numpy)
        t_pred: [3,] predicted translation vector (numpy)

    Returns:
        mse: mean squared error
    """
    import torch

    # Convert inputs to numpy if they are tensors
    if isinstance(source, torch.Tensor):
        source_np = (
            source.squeeze().detach().cpu().numpy()
        )  # Remove batch dim if present
    else:
        source_np = np.squeeze(source)  # Remove batch dim if present

    if isinstance(target, torch.Tensor):
        target_np = target.squeeze().detach().cpu().numpy()
    else:
        target_np = np.squeeze(target)

    # apply transformation
    source_transformed = (R_pred @ source_np.T).T + t_pred

    # compute MSE
    squared_errors = np.sum((source_transformed - target_np) ** 2, axis=1)
    mse = np.mean(squared_errors)

    return mse


def compute_all_errors(source, target, R_pred, t_pred, R_gt, t_gt):
    """
    Compute all error metrics

    Args:
        source: [N, 3] source point cloud
        target: [N, 3] target point cloud
        R_pred: [3, 3] predicted rotation matrix
        t_pred: [3,] predicted translation vector
        R_gt: [3, 3] ground truth rotation matrix
        t_gt: [3,] ground truth translation vector

    Returns:
        errors: dict with all metrics
    """
    # construct SE(3) matrices
    g_pred = np.eye(4)
    g_pred[:3, :3] = R_pred
    g_pred[:3, 3] = t_pred

    g_gt = np.eye(4)
    g_gt[:3, :3] = R_gt
    g_gt[:3, 3] = t_gt

    errors = {
        "rotation_error_deg": compute_rotation_error_deg(R_pred, R_gt),
        "translation_error": compute_translation_error(t_pred, t_gt),
        "se3_distance": compute_se3_distance(g_pred, g_gt),
        "point_mse": compute_point_mse(source, target, R_pred, t_pred),
    }

    return errors


def compute_rmse(source, target, R_pred, t_pred):
    """
    Compute root mean squared error

    Args:
        source: [N, 3] source point cloud
        target: [N, 3] target point cloud
        R_pred: [3, 3] predicted rotation matrix
        t_pred: [3,] predicted translation vector

    Returns:
        rmse: root mean squared error
    """
    mse = compute_point_mse(source, target, R_pred, t_pred)
    return np.sqrt(mse)


def is_registration_successful(
    rotation_error, translation_error, rot_threshold=5.0, trans_threshold=0.1
):
    """
    Check if registration is successful based on thresholds

    Args:
        rotation_error: rotation error in degrees
        translation_error: translation error
        rot_threshold: rotation threshold in degrees (default: 5.0)
        trans_threshold: translation threshold (default: 0.1)

    Returns:
        success: True if both errors are below thresholds
    """
    return (rotation_error < rot_threshold) and (translation_error < trans_threshold)
