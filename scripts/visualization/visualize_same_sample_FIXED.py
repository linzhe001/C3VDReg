import os
import sys
import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
import open3d as o3d

# --- Configuration ---
DATASET_ROOT = '/mnt/f/Datasets/C3VD_sever_datasets'
# New folder for the adjusted result
OUTPUT_DIR = '/mnt/c/Users/Linzhe/Downloads/pert_315_angle_50_FINAL_DCP_ADJUSTED'
BASE_RESULTS_PATH = "/home/linzhe/PCLR_compare/src/unified_testing/test_results"

# *** CHOSEN ANGLE AND PERT_IDX FROM ANALYSIS ***
ANGLE = 50
PERT_IDX = 315

# Algorithms and their result folders, as defined in analyze_results.py for C3VD
ALGORITHMS = {
    "ICP": "icp_c3vd",
    "DCP": "dcp_c3vd_trained",
    "PointNetLK": "pointnetlk_c3vd",
    "PointNetLK_revisited": "pointnetlk_revisited_c3vd_from_scratch_all_angles",
    "Mamba3D": "pointnetlk_mamba3d_c3vd_need"
}

# Target sample (same for all algorithms)
SOURCE_FILE = 'sigmoid_t2_a/0242_depth_pcd.ply'
TARGET_FILE = 'sigmoid_t2_a/frame_0242_visible.ply'

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
    """
    rx, ry, rz, tx, ty, tz = twist
    angle = np.sqrt(rx**2 + ry**2 + rz**2)
    if angle < 1e-10:
        rot_mat = np.eye(3)
    else:
        axis = np.array([rx, ry, rz]) / angle
        rot_mat = R.from_rotvec(axis * angle).as_matrix()
    g = np.eye(4)
    g[:3, :3] = rot_mat
    g[:3, 3] = [tx, ty, tz]
    return g

def apply_transform(points, g):
    """
    Apply SE(3) transformation to point cloud
    """
    R_mat = g[:3, :3]
    t_vec = g[:3, 3]
    points_transformed = (R_mat @ points.T).T + t_vec
    return points_transformed

def create_colored_pc(source, target):
    """
    Merge source and target with different colors
    """
    points = np.vstack([source, target])
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

def to_o3d_pcd(points):
    """Convert numpy array to Open3D PointCloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def main():
    print("="*80)
    print(f"VISUALIZATION: Best Sample Showcase (Angle={ANGLE}, Pert_Idx={PERT_IDX})")
    print("  *** WITH MANUAL ADJUSTMENT FOR DCP TRANSLATION ***")
    print("="*80)
    print()

    # Moved from global scope
    CSV_FILES = {}
    for name, folder in ALGORITHMS.items():
        path = os.path.join(BASE_RESULTS_PATH, folder, f"results_angle_{ANGLE}_corrected.csv")
        if not os.path.exists(path):
            path = os.path.join(BASE_RESULTS_PATH, folder, f"results_angle_{ANGLE}.csv")
        
        if os.path.exists(path):
            CSV_FILES[name] = path
        else:
            print(f"FATAL: No result file found for {name} at angle {ANGLE}. Exiting.")
            sys.exit(1)

    ALL_ALGORITHMS = list(ALGORITHMS.keys())

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[1/5] Loading point cloud sample...")
    source_path = os.path.join(DATASET_ROOT, 'C3VD_ply_source', SOURCE_FILE)
    target_path = os.path.join(DATASET_ROOT, 'visible_point_cloud_ply_depth', TARGET_FILE)
    source_original = load_ply(source_path)
    target_original = load_ply(target_path)
    print(f"Source points: {len(source_original)}, Target points: {len(target_original)}\n")

    np.random.seed(42)
    source_original = downsample_points(source_original, 4096)
    target_original = downsample_points(target_original, 4096)
    print(f"After downsampling: source={len(source_original)}, target={len(target_original)}\n")

    print("[2/5] Applying joint normalization...")
    source_norm, target_norm, center, scale = joint_normalize(source_original, target_original)
    print(f"Normalization parameters: Center: {center}, Scale: {scale:.2f}\n")

    points_original, colors_original = create_colored_pc(source_norm, target_norm)
    save_ply(os.path.join(OUTPUT_DIR, '01_original_normalized.ply'), points_original, colors_original)

    print(f"[3/5] Loading perturbation from pert_idx={PERT_IDX} at angle={ANGLE}...")
    df_ref = pd.read_csv(CSV_FILES['Mamba3D'])
    row_pert_series = df_ref[df_ref['pert_idx'] == PERT_IDX]
    if row_pert_series.empty:
        print(f"FATAL: Could not find pert_idx {PERT_IDX} in reference CSV. Exiting.")
        sys.exit(1)
    row_pert = row_pert_series.iloc[0]
    
    twist = row_pert[['pert_rx', 'pert_ry', 'pert_rz', 'pert_tx', 'pert_ty', 'pert_tz']].to_numpy()
    print(f"Perturbation twist: {twist}\n")

    g_pert = twist_to_se3(twist)
    source_perturbed = apply_transform(source_norm, g_pert)
    points_perturbed, colors_perturbed = create_colored_pc(source_perturbed, target_norm)
    save_ply(os.path.join(OUTPUT_DIR, '02_after_perturbation.ply'), points_perturbed, colors_perturbed)

    # Calculate and print Chamfer Distance for the perturbed cloud
    print("\n[3.5/5] Calculating initial Chamfer Distance after perturbation...")
    source_perturbed_o3d = to_o3d_pcd(source_perturbed)
    target_norm_o3d = to_o3d_pcd(target_norm)
    dist_s_t = np.asarray(source_perturbed_o3d.compute_point_cloud_distance(target_norm_o3d))
    dist_t_s = np.asarray(target_norm_o3d.compute_point_cloud_distance(source_perturbed_o3d))
    chamfer_dist_perturbed = np.mean(dist_s_t) + np.mean(dist_t_s)
    print(f"Chamfer Distance after perturbation (before registration): {chamfer_dist_perturbed:.6f}\n")

    print(f"[4/5] Loading predictions...")
    print(f"  Note: Translation values from CSV will be divided by scale={scale:.2f} to normalize.")
    print()

    algo_predictions = {}
    for algo_name in ALL_ALGORITHMS:
        csv_file = CSV_FILES.get(algo_name)
        df = pd.read_csv(csv_file)
        rows = df[df['pert_idx'] == PERT_IDX]
        if len(rows) == 0:
            print(f"WARNING: {algo_name} - No row found for pert_idx={PERT_IDX}!")
            continue
        row = rows.iloc[0]

        R_pred = row[['pred_r11', 'pred_r12', 'pred_r13', 'pred_r21', 'pred_r22', 'pred_r23', 'pred_r31', 'pred_r32', 'pred_r33']].to_numpy().reshape(3, 3)
        
        t_pred_csv = row[['pred_tx', 'pred_ty', 'pred_tz']].to_numpy()
        t_pred_normalized = t_pred_csv / scale

        # User request: Manually scale down DCP translation for visualization
        if algo_name == 'DCP':
            print("  *** Applying manual scaling for DCP translation (dividing by 10)... ***")
            t_pred_normalized = t_pred_normalized / 10

        algo_predictions[algo_name] = {
            'R_pred': R_pred,
            't_pred_normalized': t_pred_normalized,
            'rotation_error': row['rotation_error_deg'],
            'translation_error_m_from_csv': row['translation_error_m'],
        }

        R_gt = g_pert[:3, :3].T
        t_gt = -R_gt @ g_pert[:3, 3]
        trans_error_norm_recalculated = np.linalg.norm(t_pred_normalized - t_gt)
        algo_predictions[algo_name]['translation_error_norm_recalculated'] = trans_error_norm_recalculated

        print(f"{algo_name} (from {os.path.basename(csv_file)}):")
        print(f"  CSV errors: rot={row['rotation_error_deg']:.2f}°, trans={row['translation_error_m']:.4f}m")
        print(f"  Recalculated trans error (norm): {trans_error_norm_recalculated:.6f} -> {trans_error_norm_recalculated * scale:.2f} mm")
        print()

    print(f"[5/5] Applying predictions and calculating Chamfer Distance...")
    target_norm_o3d = to_o3d_pcd(target_norm)
    for algo_name, pred in algo_predictions.items():
        g_pred = np.eye(4)
        g_pred[:3, :3] = pred['R_pred']
        g_pred[:3, 3] = pred['t_pred_normalized']
        source_registered = apply_transform(source_perturbed, g_pred)
        
        # Calculate Chamfer Distance
        source_registered_o3d = to_o3d_pcd(source_registered)
        dist_s_t = np.asarray(source_registered_o3d.compute_point_cloud_distance(target_norm_o3d))
        dist_t_s = np.asarray(target_norm_o3d.compute_point_cloud_distance(source_registered_o3d))
        chamfer_dist = np.mean(dist_s_t) + np.mean(dist_t_s)
        pred['chamfer_distance'] = chamfer_dist

        points_registered, colors_registered = create_colored_pc(source_registered, target_norm)
        filename = f'03_{algo_name}_registered.ply'
        save_ply(os.path.join(OUTPUT_DIR, filename), points_registered, colors_registered)
        print(f"{algo_name}: Saved to {filename}. Chamfer Distance: {chamfer_dist:.6f}")

    print()
    print("="*80)
    print("VISUALIZATION COMPLETE")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80)
    print("\nAlgorithm comparison (recalculated):")
    print("-" * 90)
    print(f"{'Algorithm':<25} {'Rot Error':>12} {'Trans Error (mm)':>18} {'Chamfer Distance':>20}")
    print("-" * 90)
    # Sort by rotation error for final display
    sorted_algos = sorted(algo_predictions.items(), key=lambda item: item[1]['rotation_error'])
    for algo_name, pred in sorted_algos:
        trans_err_mm = pred['translation_error_norm_recalculated'] * scale
        note = "(Manually Adjusted)" if algo_name == "DCP" else ""
        print(f"  {algo_name:<23} {pred['rotation_error']:>10.2f}° {trans_err_mm:>16.2f} mm {pred['chamfer_distance']:>19.4f} {note}")
    print("-" * 90)
    print()

if __name__ == '__main__':
    main()
