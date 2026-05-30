#!/usr/bin/env python3
"""
Visualize point cloud registration results including ICP

This script visualizes all algorithms' performance (including ICP) on the SAME 
perturbation (pert_idx), showing how each algorithm performs on potentially 
different point cloud samples.

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
OUTPUT_DIR = '/mnt/c/Users/Linzhe/Downloads/pert_visualization_with_icp'
UNIFIED_TEST_DIR = '/home/linzhe/PCLR_compare/src/unified_testing'

# CSV files for different algorithms
CSV_FILES = {
    'mamba3d': '/home/linzhe/PCLR_compare/src/unified_testing/test_results/mamba3d_c3vd_trained_angles_40_50_60/results_angle_40.csv',
    'dcp': '/home/linzhe/PCLR_compare/src/unified_testing/test_results/dcp_c3vd_trained/results_angle_40.csv',
    'pointnetlk': '/home/linzhe/PCLR_compare/src/unified_testing/test_results/pointnetlk_c3vd/results_angle_40.csv',
    'pointnetlk_revisited': '/home/linzhe/PCLR_compare/src/unified_testing/test_results/pointnetlk_revisited_c3vd_trained/results_angle_40.csv',
    'icp': '/home/linzhe/PCLR_compare/src/unified_testing/test_results/icp_c3vd_angles_40_50_60/results_angle_40.csv'
}

# Target perturbation index - you can change this to visualize different samples
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
    print(f"POINT CLOUD REGISTRATION VISUALIZATION WITH ICP - pert_idx={PERT_IDX}")
    print("="*80)
    print()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ============================================================
    # 1. Load CSV data for all algorithms at specified pert_idx
    # ============================================================
    print(f"[1/4] Loading CSV data for pert_idx={PERT_IDX}...")

    algo_data = {}

    for algo_name, csv_file in CSV_FILES.items():
        if not os.path.exists(csv_file):
            print(f"WARNING: {algo_name} CSV file not found: {csv_file}")
            continue
            
        df = pd.read_csv(csv_file)

        # Find the row with specified pert_idx
        rows = df[df['pert_idx'] == PERT_IDX]

        if len(rows) == 0:
            print(f"WARNING: {algo_name} - No row found for pert_idx={PERT_IDX}!")
            continue

        row = rows.iloc[0]

        # Extract perturbation parameters
        twist = np.array([
            row['pert_rx'], row['pert_ry'], row['pert_rz'],
            row['pert_tx'], row['pert_ty'], row['pert_tz']
        ])

        # Extract prediction parameters
        R_pred = np.array([
            [row['pred_r11'], row['pred_r12'], row['pred_r13']],
            [row['pred_r21'], row['pred_r22'], row['pred_r23']],
            [row['pred_r31'], row['pred_r32'], row['pred_r33']]
        ])

        t_pred = np.array([row['pred_tx'], row['pred_ty'], row['pred_tz']])

        # Get file paths
        source_file = row['source_file']
        target_file = row['target_file']

        algo_data[algo_name] = {
            'sample_idx': int(row['sample_idx']),
            'source_file': source_file,
            'target_file': target_file,
            'twist': twist,
            'R_pred': R_pred,
            't_pred': t_pred,
            'rotation_error': row['rotation_error_deg'],
            'translation_error': row['translation_error_m']
        }

        print(f"{algo_name.upper()}:")
        print(f"  sample_idx: {algo_data[algo_name]['sample_idx']}")
        print(f"  source: {source_file.split('/')[-2:]}")
        print(f"  Rotation error: {algo_data[algo_name]['rotation_error']:.2f}°")
        print(f"  Translation error: {algo_data[algo_name]['translation_error']:.4f}")

    print()

    # Verify all algorithms use the SAME perturbation
    print("Verifying perturbation consistency across algorithms...")
    if len(algo_data) > 0:
        reference_algo = list(algo_data.keys())[0]
        reference_twist = algo_data[reference_algo]['twist']
        for algo_name in algo_data.keys():
            if not np.allclose(algo_data[algo_name]['twist'], reference_twist):
                print(f"WARNING: {algo_name} has different perturbation!")
            else:
                print(f"  ✓ {algo_name} uses same perturbation")
    print()

    # ============================================================
    # 2. Process each algorithm separately
    # ============================================================
    print(f"[2/4] Processing each algorithm's point cloud...")

    for algo_name in ['mamba3d', 'dcp', 'pointnetlk', 'pointnetlk_revisited', 'icp']:
        if algo_name not in algo_data:
            continue

        print(f"\n{algo_name.upper()}:")
        print("-" * 60)

        data = algo_data[algo_name]

        # Load point clouds
        source_path = data['source_file']
        target_path = data['target_file']

        source_original = load_ply(source_path)
        target_original = load_ply(target_path)

        # Downsample
        np.random.seed(42)
        source_original = downsample_points(source_original, 1024)
        target_original = downsample_points(target_original, 1024)

        # Normalize
        source_norm, target_norm, center, scale = joint_normalize(source_original, target_original)
        print(f"  Normalization: center={center}, scale={scale:.2f}")

        # Apply perturbation
        twist = data['twist']
        g_pert = twist_to_se3(twist)
        source_perturbed = apply_transform(source_norm, g_pert)

        # Apply prediction
        R_pred = data['R_pred']
        t_pred = data['t_pred']
        g_pred = np.eye(4)
        g_pred[:3, :3] = R_pred
        g_pred[:3, 3] = t_pred
        source_registered = apply_transform(source_perturbed, g_pred)

        # Save visualizations
        # 1. Original
        points_original, colors_original = create_colored_pc(source_norm, target_norm)
        save_ply(
            os.path.join(OUTPUT_DIR, f'{algo_name}_01_original.ply'),
            points_original,
            colors_original
        )

        # 2. After perturbation
        points_perturbed, colors_perturbed = create_colored_pc(source_perturbed, target_norm)
        save_ply(
            os.path.join(OUTPUT_DIR, f'{algo_name}_02_perturbed.ply'),
            points_perturbed,
            colors_perturbed
        )

        # 3. After registration
        points_registered, colors_registered = create_colored_pc(source_registered, target_norm)
        save_ply(
            os.path.join(OUTPUT_DIR, f'{algo_name}_03_registered.ply'),
            points_registered,
            colors_registered
        )

    print()
    print("="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    print(f"Generated files for pert_idx={PERT_IDX}:")
    print()
    
    # Create performance summary table
    print("Performance Summary:")
    print("-" * 80)
    print(f"{'Algorithm':<25} {'Rotation Error':>15} {'Translation Error':>20} {'Sample'}")
    print("-" * 80)
    
    for algo_name in ['mamba3d', 'dcp', 'pointnetlk', 'pointnetlk_revisited', 'icp']:
        if algo_name in algo_data:
            data = algo_data[algo_name]
            sample_str = f"{data['source_file'].split('/')[-2]}/{data['source_file'].split('/')[-1][:8]}"
            print(f"{algo_name.upper():<25} {data['rotation_error']:>12.2f}° {data['translation_error']:>19.4f} {sample_str}")
            
    print("-" * 80)
    print()
    print("Files generated for each algorithm:")
    for algo_name in ['mamba3d', 'dcp', 'pointnetlk', 'pointnetlk_revisited', 'icp']:
        if algo_name in algo_data:
            print(f"  {algo_name}:")
            print(f"    - {algo_name}_01_original.ply")
            print(f"    - {algo_name}_02_perturbed.ply")
            print(f"    - {algo_name}_03_registered.ply")

    print()
    print("Color scheme:")
    print("  RED   - Source point cloud")
    print("  GREEN - Target point cloud")
    print()
    print(f"Note: All algorithms use the SAME perturbation (pert_idx={PERT_IDX}),")
    print("      but may be tested on DIFFERENT point cloud samples.")
    print()


if __name__ == '__main__':
    main()
