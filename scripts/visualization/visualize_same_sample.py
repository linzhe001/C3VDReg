#!/usr/bin/env python3
"""
Visualize point cloud registration results on the SAME sample using pert_idx=626

This script:
1. Uses the SAME point cloud sample (sigmoid_t2_a/0242)
2. Applies the SAME perturbation (pert_idx=626)
3. Shows each algorithm's prediction result on this sample

All transformations are applied in NORMALIZED space (matching the testing procedure).
"""

import os
import sys
import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R

# Configuration
DATASET_ROOT = '/mnt/f/Datasets/C3VD_sever_datasets'
OUTPUT_DIR = '/mnt/c/Users/Linzhe/Downloads/pert_626_same_sample'

# CSV files for different algorithms
CSV_FILES = {
    'mamba3d': '/home/linzhe/PCLR_compare/src/unified_testing/test_results/mamba3d_c3vd_trained_angles_40_50_60/results_angle_40.csv',
    'dcp': '/home/linzhe/PCLR_compare/src/unified_testing/test_results/dcp_c3vd_trained/results_angle_40.csv',
    'pointnetlk': '/home/linzhe/PCLR_compare/src/unified_testing/test_results/pointnetlk_c3vd/results_angle_40.csv',
    'pointnetlk_revisited': '/home/linzhe/PCLR_compare/src/unified_testing/test_results/pointnetlk_revisited_c3vd_trained/results_angle_40.csv'
}

# Target sample (same for all algorithms)
SOURCE_FILE = 'sigmoid_t2_a/0242_depth_pcd.ply'
TARGET_FILE = 'sigmoid_t2_a/frame_0242_visible.ply'

# Target perturbation index
PERT_IDX = 626


def load_ply(filepath):
    """Load PLY file and return point cloud as numpy array [N, 3]"""
    print(f"Loading: {filepath}")
    plydata = PlyData.read(filepath)
    pc = np.vstack([
        plydata['vertex']['x'],
        plydata['vertex']['y'],
        plydata['vertex']['z']
    ]).T
    return pc


def save_ply(filepath, points, colors=None):
    """
    Save point cloud to PLY file

    Args:
        filepath: output file path
        points: [N, 3] numpy array
        colors: [N, 3] numpy array (optional, RGB values 0-255)
    """
    points = points.astype(np.float32)

    if colors is not None:
        colors = colors.astype(np.uint8)
        vertex = np.array(
            [(points[i, 0], points[i, 1], points[i, 2], colors[i, 0], colors[i, 1], colors[i, 2])
             for i in range(len(points))],
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        )
    else:
        vertex = np.array(
            [(points[i, 0], points[i, 1], points[i, 2])
             for i in range(len(points))],
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        )

    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(filepath)
    print(f"Saved: {filepath}")


def joint_normalize(source, target):
    """
    Apply joint normalization (same as testing procedure)

    Args:
        source: [N, 3] source point cloud
        target: [M, 3] target point cloud

    Returns:
        source_norm: [N, 3] normalized source
        target_norm: [M, 3] normalized target
        center: [3,] center used for normalization
        scale: scalar scale factor
    """
    all_pts = np.vstack([source, target])
    min_bounds = all_pts.min(axis=0)
    max_bounds = all_pts.max(axis=0)
    center = (min_bounds + max_bounds) / 2.0
    scales = max_bounds - min_bounds
    scale = scales.max() if scales.max() > 1e-6 else 1.0

    source_norm = (source - center) / scale
    target_norm = (target - center) / scale

    return source_norm, target_norm, center, scale


def twist_to_se3(twist):
    """
    Convert twist (axis-angle + translation) to SE(3) matrix

    Args:
        twist: [6,] array [rx, ry, rz, tx, ty, tz]

    Returns:
        g: [4, 4] SE(3) transformation matrix
    """
    rx, ry, rz, tx, ty, tz = twist

    # Convert axis-angle to rotation matrix
    angle = np.sqrt(rx**2 + ry**2 + rz**2)
    if angle < 1e-10:
        rot_mat = np.eye(3)
    else:
        axis = np.array([rx, ry, rz]) / angle
        rot_mat = R.from_rotvec(axis * angle).as_matrix()

    # Construct SE(3) matrix
    g = np.eye(4)
    g[:3, :3] = rot_mat
    g[:3, 3] = [tx, ty, tz]

    return g


def apply_transform(points, g):
    """
    Apply SE(3) transformation to point cloud

    Args:
        points: [N, 3] point cloud
        g: [4, 4] SE(3) transformation matrix

    Returns:
        points_transformed: [N, 3] transformed point cloud
    """
    R = g[:3, :3]
    t = g[:3, 3]
    points_transformed = (R @ points.T).T + t
    return points_transformed


def create_colored_pc(source, target):
    """
    Merge source and target with different colors

    Args:
        source: [N, 3] source point cloud
        target: [M, 3] target point cloud

    Returns:
        points: [N+M, 3] merged point cloud
        colors: [N+M, 3] RGB colors (0-255)
    """
    points = np.vstack([source, target])

    # Source: red, Target: green
    colors_src = np.tile([255, 100, 100], (len(source), 1))  # Red for source
    colors_tgt = np.tile([100, 255, 100], (len(target), 1))  # Green for target
    colors = np.vstack([colors_src, colors_tgt])

    return points, colors


def downsample_points(points, num_points=1024):
    """Downsample point cloud to num_points"""
    if len(points) > num_points:
        indices = np.random.choice(len(points), num_points, replace=False)
        return points[indices]
    return points


def main():
    print("="*80)
    print(f"VISUALIZATION: Same Sample ({SOURCE_FILE}), Same Perturbation (pert_idx={PERT_IDX})")
    print("="*80)
    print()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ============================================================
    # 1. Load the SAME point cloud sample
    # ============================================================
    print("[1/5] Loading point cloud sample...")

    source_path = os.path.join(DATASET_ROOT, 'C3VD_ply_source', SOURCE_FILE)
    target_path = os.path.join(DATASET_ROOT, 'visible_point_cloud_ply_depth', TARGET_FILE)

    source_original = load_ply(source_path)
    target_original = load_ply(target_path)

    print(f"Source points: {len(source_original)}")
    print(f"Target points: {len(target_original)}")
    print()

    # Downsample to 1024 points (matching testing)
    np.random.seed(42)
    source_original = downsample_points(source_original, 1024)
    target_original = downsample_points(target_original, 1024)
    print(f"After downsampling: source={len(source_original)}, target={len(target_original)}")
    print()

    # ============================================================
    # 2. Apply normalization
    # ============================================================
    print("[2/5] Applying joint normalization...")

    source_norm, target_norm, center, scale = joint_normalize(source_original, target_original)

    print(f"Normalization parameters:")
    print(f"  Center: {center}")
    print(f"  Scale: {scale:.2f}")
    print()

    # Save: Step 1 - Original (normalized)
    points_original, colors_original = create_colored_pc(source_norm, target_norm)
    save_ply(
        os.path.join(OUTPUT_DIR, '01_original_normalized.ply'),
        points_original,
        colors_original
    )

    # ============================================================
    # 3. Load perturbation from pert_idx=626
    # ============================================================
    print(f"[3/5] Loading perturbation from pert_idx={PERT_IDX}...")

    # Get perturbation from mamba3d CSV (all should have same perturbation)
    df_mamba = pd.read_csv(CSV_FILES['mamba3d'])
    row_pert = df_mamba[df_mamba['pert_idx'] == PERT_IDX].iloc[0]

    twist = np.array([
        row_pert['pert_rx'], row_pert['pert_ry'], row_pert['pert_rz'],
        row_pert['pert_tx'], row_pert['pert_ty'], row_pert['pert_tz']
    ])

    print(f"Perturbation twist: {twist}")
    print(f"  Rotation: rx={twist[0]:.4f}, ry={twist[1]:.4f}, rz={twist[2]:.4f}")
    print(f"  Translation: tx={twist[3]:.6f}, ty={twist[4]:.6f}, tz={twist[5]:.6f}")
    print()

    # Apply perturbation in normalized space
    g_pert = twist_to_se3(twist)
    source_perturbed = apply_transform(source_norm, g_pert)

    # Save: Step 2 - After perturbation
    points_perturbed, colors_perturbed = create_colored_pc(source_perturbed, target_norm)
    save_ply(
        os.path.join(OUTPUT_DIR, '02_after_perturbation.ply'),
        points_perturbed,
        colors_perturbed
    )

    # ============================================================
    # 4. Load each algorithm's prediction from pert_idx=626
    # ============================================================
    print(f"[4/5] Loading predictions from each algorithm (pert_idx={PERT_IDX})...")

    algo_predictions = {}

    for algo_name, csv_file in CSV_FILES.items():
        df = pd.read_csv(csv_file)

        # Find the row with pert_idx=626
        rows = df[df['pert_idx'] == PERT_IDX]

        if len(rows) == 0:
            print(f"WARNING: {algo_name} - No row found for pert_idx={PERT_IDX}!")
            continue

        row = rows.iloc[0]

        # Extract prediction parameters
        R_pred = np.array([
            [row['pred_r11'], row['pred_r12'], row['pred_r13']],
            [row['pred_r21'], row['pred_r22'], row['pred_r23']],
            [row['pred_r31'], row['pred_r32'], row['pred_r33']]
        ])

        t_pred = np.array([row['pred_tx'], row['pred_ty'], row['pred_tz']])

        algo_predictions[algo_name] = {
            'R_pred': R_pred,
            't_pred': t_pred,
            'rotation_error': row['rotation_error_deg'],
            'translation_error': row['translation_error_m'],
            'tested_sample': row['source_file'].split('/')[-2:]
        }

        print(f"{algo_name}:")
        print(f"  Tested on: {algo_predictions[algo_name]['tested_sample']}")
        print(f"  Original errors: rot={row['rotation_error_deg']:.2f}°, trans={row['translation_error_m']:.4f}")
        print(f"  Prediction: t=[{t_pred[0]:.4f}, {t_pred[1]:.4f}, {t_pred[2]:.4f}]")

    print()

    # ============================================================
    # 5. Apply each algorithm's prediction to the SAME sample
    # ============================================================
    print(f"[5/5] Applying predictions to {SOURCE_FILE}...")

    for algo_name in ['mamba3d', 'dcp', 'pointnetlk', 'pointnetlk_revisited']:
        if algo_name not in algo_predictions:
            continue

        pred = algo_predictions[algo_name]
        R_pred = pred['R_pred']
        t_pred = pred['t_pred']

        # Construct SE(3) from prediction
        g_pred = np.eye(4)
        g_pred[:3, :3] = R_pred
        g_pred[:3, 3] = t_pred

        # Apply prediction to perturbed source
        source_registered = apply_transform(source_perturbed, g_pred)

        # Save
        points_registered, colors_registered = create_colored_pc(source_registered, target_norm)

        filename = f'03_{algo_name}_registered.ply'
        save_ply(
            os.path.join(OUTPUT_DIR, filename),
            points_registered,
            colors_registered
        )

        print(f"{algo_name}: Saved to {filename}")

    print()
    print("="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    print("Files saved:")
    print("  01_original_normalized.ply          - Original point clouds (normalized)")
    print("  02_after_perturbation.ply           - After applying perturbation")
    print("  03_mamba3d_registered.ply           - Mamba3D prediction")
    print("  03_dcp_registered.ply               - DCP prediction")
    print("  03_pointnetlk_registered.ply        - PointNetLK prediction")
    print("  03_pointnetlk_revisited_registered.ply - PointNetLK_Revisited prediction")
    print()
    print("Color scheme:")
    print("  RED   - Source point cloud")
    print("  GREEN - Target point cloud")
    print()
    print(f"Note: All visualizations use:")
    print(f"  - SAME sample: {SOURCE_FILE}")
    print(f"  - SAME perturbation: pert_idx={PERT_IDX}")
    print(f"  - Each algorithm's prediction parameters from their pert_idx={PERT_IDX} test")
    print()
    print("Algorithm comparison:")
    for algo_name in ['mamba3d', 'dcp', 'pointnetlk', 'pointnetlk_revisited']:
        if algo_name in algo_predictions:
            pred = algo_predictions[algo_name]
            print(f"  {algo_name:20s} (tested on {pred['tested_sample']})")
            print(f"    Original errors: rot={pred['rotation_error']:.2f}°, trans={pred['translation_error']:.4f}")
    print()


if __name__ == '__main__':
    main()
