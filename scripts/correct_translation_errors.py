
import os
import sys
import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
import glob
import argparse

def load_ply(filepath):
    """Load PLY file and return point cloud as numpy array [N, 3]"""
    try:
        plydata = PlyData.read(filepath)
        pc = np.vstack([
            plydata['vertex']['x'],
            plydata['vertex']['y'],
            plydata['vertex']['z']
        ]).T
        return pc
    except Exception as e:
        print(f"ERROR: Failed to load PLY file {filepath}: {e}", file=sys.stderr)
        return None

def joint_normalize(source, target):
    """
    Apply joint normalization (same as testing procedure)
    """
    if source is None or target is None:
        return None, None, None, None
        
    all_pts = np.vstack([source, target])
    min_bounds = all_pts.min(axis=0)
    max_bounds = all_pts.max(axis=0)
    center = (min_bounds + max_bounds) / 2.0
    scales = max_bounds - min_bounds
    scale = scales.max() if scales.max() > 1e-6 else 1.0

    # source_norm = (source - center) / scale
    # target_norm = (target - center) / scale

    return None, None, center, scale

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

def correct_csv(csv_path, c3vd_root, modelnet_root):
    """
    Corrects the translation error in a given CSV file.
    """
    print(f"Processing file: {csv_path}")
    df = pd.read_csv(csv_path)

    if 'translation_error_m' not in df.columns:
        print(f"WARNING: 'translation_error_m' column not found in {csv_path}. Skipping.")
        return

    corrected_errors = []
    
    # Check if the file is modelnet or c3vd
    is_modelnet = 'modelnet' in csv_path.lower()
    is_c3vd = 'c3vd' in csv_path.lower()

    if not is_c3vd and not is_modelnet:
        print(f"WARNING: Cannot determine dataset (C3VD or ModelNet) for {csv_path}. Skipping.")
        return

    for index, row in df.iterrows():
        source_file_val = row['source_file']
        target_file_val = row['target_file']

        # Handle cases where file paths are missing (e.g., 'N/A' or NaN)
        if not isinstance(source_file_val, str) or source_file_val == 'N/A':
            # print(f"INFO: Skipping row {index} due to missing source file path.", file=sys.stderr)
            corrected_errors.append(row['translation_error_m']) # Keep original if fails
            continue

        # Determine dataset and construct paths
        if is_c3vd:
            dataset_root = c3vd_root
            source_path = os.path.join(dataset_root, 'C3VD_ply_source', source_file_val)
            target_path = os.path.join(dataset_root, 'visible_point_cloud_ply_depth', target_file_val)
        else: # is_modelnet
            dataset_root = modelnet_root
            # ModelNet paths are absolute in the CSV
            source_path = source_file_val
            target_path = target_file_val


        # Load point clouds
        source_original = load_ply(source_path)
        target_original = load_ply(target_path)

        if source_original is None or target_original is None:
            print(f"WARNING: Skipping row {index} due to missing point cloud data.", file=sys.stderr)
            corrected_errors.append(row['translation_error_m']) # Keep original if fails
            continue

        # Calculate scale
        _, _, _, scale = joint_normalize(source_original, target_original)
        
        if scale is None:
            print(f"WARNING: Skipping row {index} due to normalization failure.", file=sys.stderr)
            corrected_errors.append(row['translation_error_m']) # Keep original if fails
            continue

        # Get predicted translation
        t_pred_csv = np.array([row['pred_tx'], row['pred_ty'], row['pred_tz']])
        t_pred_normalized = t_pred_csv / scale

        # Get ground truth transformation
        twist_gt = np.array([
            row['pert_rx'], row['pert_ry'], row['pert_rz'],
            row['pert_tx'], row['pert_ty'], row['pert_tz']
        ])
        g_pert = twist_to_se3(twist_gt)
        R_gt = g_pert[:3, :3].T
        t_gt = -R_gt @ g_pert[:3, 3]

        # Calculate corrected translation error
        trans_error_corrected = np.linalg.norm(t_pred_normalized - t_gt)
        corrected_errors.append(trans_error_corrected)

    # Update the dataframe
    df['translation_error_m'] = corrected_errors
    
    # Save to a new file
    output_path = csv_path.replace('.csv', '_corrected.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved corrected file to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Correct translation errors in PCLR result CSVs.")
    parser.add_argument("directories", nargs='+', help="List of directories to process.")
    parser.add_argument("--c3vd_root", default="/mnt/f/Datasets/C3VD_sever_datasets", help="Root directory for the C3VD dataset.")
    parser.add_argument("--modelnet_root", default="/mnt/f/Datasets/ModelNet40_ply_hdf5_2048", help="Root directory for the ModelNet40 dataset.")
    args = parser.parse_args()

    for directory in args.directories:
        if not os.path.isdir(directory):
            print(f"WARNING: Directory not found: {directory}", file=sys.stderr)
            continue
        
        print(f"--- Processing directory: {directory} ---")
        csv_files = glob.glob(os.path.join(directory, "results_angle_*.csv"))
        
        # Exclude files that are already corrected
        csv_files = [f for f in csv_files if '_corrected' not in f]

        if not csv_files:
            print("No 'results_angle_*.csv' files found to correct.")
            continue

        for csv_file in csv_files:
            correct_csv(csv_file, args.c3vd_root, args.modelnet_root)
    
    print("--- Correction script finished. ---")

if __name__ == "__main__":
    main()
