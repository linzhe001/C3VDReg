"""
Utility functions for unified testing pipeline
"""

from .transform_utils import (
    twist_to_se3,
    se3_to_twist,
    apply_transform,
    inverse_transform,
    compose_transforms,
    rotation_matrix_to_euler,
    euler_to_rotation_matrix,
)

from .error_metrics import (
    compute_rotation_error_deg,
    compute_translation_error,
    compute_se3_distance,
    compute_point_mse,
    compute_all_errors,
    compute_rmse,
    is_registration_successful,
)

from .file_io import (
    load_off_file,
    load_ply_file,
    load_point_cloud,
    extract_file_paths_modelnet,
    extract_file_paths_c3vd,
    save_point_cloud_ply,
    get_file_list_modelnet,
    get_file_list_c3vd,
)

from .logger import setup_logger, get_timestamp

__all__ = [
    # transform_utils
    "twist_to_se3",
    "se3_to_twist",
    "apply_transform",
    "inverse_transform",
    "compose_transforms",
    "rotation_matrix_to_euler",
    "euler_to_rotation_matrix",
    # error_metrics
    "compute_rotation_error_deg",
    "compute_translation_error",
    "compute_se3_distance",
    "compute_point_mse",
    "compute_all_errors",
    "compute_rmse",
    "is_registration_successful",
    # file_io
    "load_off_file",
    "load_ply_file",
    "load_point_cloud",
    "extract_file_paths_modelnet",
    "extract_file_paths_c3vd",
    "save_point_cloud_ply",
    "get_file_list_modelnet",
    "get_file_list_c3vd",
    # logger
    "setup_logger",
    "get_timestamp",
]
