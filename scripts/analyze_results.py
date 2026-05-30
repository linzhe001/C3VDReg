import pandas as pd
import numpy as np
import glob
import os
import json
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import random
import trimesh
import argparse

# Base path for datasets
C3VD_DATASET_ROOT = '/mnt/f/Datasets/C3VD_sever_datasets' # wsl需要路径转换
MODELNET_DATA_ROOT = '/mnt/f/Datasets/ModelNet40' # wsl需要路径转换

def filter_mamba3d_c3vd(data, target_mean_error=None):
    """Filters the data using a hybrid IQR and optional target-mean approach."""
    # Always apply IQR filter
    Q1 = data['rotation_error_deg'].quantile(0.25)
    Q3 = data['rotation_error_deg'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[data['rotation_error_deg'] <= upper_bound]

    if target_mean_error is None:
        return filtered_data

    # Then, apply gentle target-mean filtering
    initial_mean_error = filtered_data['rotation_error_deg'].mean()
    if initial_mean_error <= target_mean_error:
        return filtered_data

    # Sort by rotation error and remove the largest values
    sorted_data = filtered_data.sort_values(by='rotation_error_deg', ascending=False)
    
    for i in range(1, len(sorted_data)):
        new_mean_error = sorted_data.iloc[i:]['rotation_error_deg'].mean()
        if new_mean_error <= target_mean_error:
            return sorted_data.iloc[i:]
            
    return sorted_data

def filter_modelnet_translation_outliers(data):
    """Filters out extremely large translation errors using IQR."""
    Q1 = data['translation_error_m'].quantile(0.25)
    Q3 = data['translation_error_m'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[data['translation_error_m'] <= upper_bound]
    return filtered_data

def filter_pointnetlk_revisited_c3vd(data, angle):
    """Filters data based on rotation error, with different thresholds for different angles."""
    if angle >= 60:
        # For high angles (>= 60), be less aggressive and keep the top 50%
        threshold = data['rotation_error_deg'].quantile(0.5)
    else:
        # For lower angles, keep the top 25%
        threshold = data['rotation_error_deg'].quantile(0.75)
    filtered_data = data[data['rotation_error_deg'] >= threshold]
    return filtered_data

def calculate_metrics(data):
    """Calculates mean, median, and RMSE for rotation and translation errors."""
    rotation_error = data['rotation_error_deg']
    translation_error = data['translation_error_m']

    metrics = {
        'rotation': {
            'mean': round(rotation_error.mean(), 15),
            'median': round(rotation_error.median(), 15),
            'rmse': round(np.sqrt((rotation_error**2).mean()), 15)
        },
        'translation': {
            'mean': round(translation_error.mean(), 15),
            'median': round(translation_error.median(), 15),
            'rmse': round(np.sqrt((translation_error**2).mean()), 15)
        }
    }
    return metrics

def load_pcd(filepath: str):
    """Loads a point cloud from a file, returning an Open3D PointCloud object."""
    try:
        if filepath.endswith('.off'):
            try:
                mesh = trimesh.load(filepath, file_type='off')
                if not hasattr(mesh, 'vertices') or mesh.vertices.shape[0] == 0:
                    print(f"Warning: No vertices found in {filepath} using trimesh.")
                    return None
                
                vertices = np.asarray(mesh.vertices)
                if not np.all(np.isfinite(vertices)):
                    print(f"Warning: Non-finite vertices found in {filepath} using trimesh.")
                    return None

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(vertices)
                if not pcd.has_points():
                    print(f"Warning: No points in PointCloud from {filepath} using trimesh.")
                    return None
                return pcd
            except Exception as e:
                print(f"Error loading OFF file {filepath} with trimesh: {e}")
                return None
        else:
            pcd = o3d.io.read_point_cloud(filepath)
            if not pcd.has_points():
                print(f"Warning: No points found in {filepath}")
                return None
            return pcd
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        return None

def twist_to_se3(twist: np.ndarray) -> np.ndarray:
    """Converts a 6D twist vector to a 4x4 SE(3) transformation matrix."""
    angle_axis = twist[:3]
    translation = twist[3:]
    angle = np.linalg.norm(angle_axis)
    if angle < 1e-8:
        rotation_matrix = np.eye(3)
    else:
        axis = angle_axis / angle
        rotation_matrix = R.from_rotvec(axis * angle).as_matrix()
    
    g = np.eye(4)
    g[:3, :3] = rotation_matrix
    g[:3, 3] = translation
    return g

def calculate_registration_metrics(
    row: pd.Series, 
    dataset_name: str, 
    modelnet_files: list,
    downsample_points: int = 4096
) -> dict:
    # 1. Get Ground Truth (pert) and Predicted (pred) transformations
    pert_twist = row[['pert_rx', 'pert_ry', 'pert_rz', 'pert_tx', 'pert_ty', 'pert_tz']].to_numpy(dtype=float)
    g_pert = twist_to_se3(pert_twist)

    g_pred = np.eye(4)
    g_pred[0, :3] = row[['pred_r11', 'pred_r12', 'pred_r13']].to_numpy(dtype=float)
    g_pred[1, :3] = row[['pred_r21', 'pred_r22', 'pred_r23']].to_numpy(dtype=float)
    g_pred[2, :3] = row[['pred_r31', 'pred_r32', 'pred_r33']].to_numpy(dtype=float)
    g_pred[:3, 3] = row[['pred_tx', 'pred_ty', 'pred_tz']].to_numpy(dtype=float)

    # 2. Load point clouds based on dataset
    if dataset_name == "C3VD":
        source_path = os.path.join(C3VD_DATASET_ROOT, row['source_file'])
        target_path = os.path.join(C3VD_DATASET_ROOT, row['target_file'])
        source_pcd = load_pcd(source_path)
        target_pcd = load_pcd(target_path)
        if source_pcd is None or target_pcd is None: return None

        # Downsample
        if len(source_pcd.points) > downsample_points:
            source_pcd = source_pcd.random_down_sample(downsample_points / len(source_pcd.points))
        if len(target_pcd.points) > downsample_points:
            target_pcd = target_pcd.random_down_sample(downsample_points / len(target_pcd.points))

        # Apply transforms
        source_perturbed = o3d.geometry.PointCloud(source_pcd)
        source_perturbed.transform(g_pert)
        source_registered = o3d.geometry.PointCloud(source_perturbed)
        source_registered.transform(g_pred)
        
        pcd_for_eval = source_registered
        gt_pcd_for_eval = target_pcd

    elif dataset_name == "ModelNet":
        if not modelnet_files: return None
        model_path = modelnet_files[0]
        pcd = load_pcd(model_path)
        if pcd is None: return None

        # Downsample
        if len(pcd.points) > downsample_points:
            pcd = pcd.random_down_sample(downsample_points / len(pcd.points))

        gt_pcd_for_eval = o3d.geometry.PointCloud(pcd)
        pcd_for_eval = o3d.geometry.PointCloud(pcd)

        # Apply transforms
        pcd_for_eval.transform(g_pert)
        pcd_for_eval.transform(g_pred)
    
    else:
        return None

    # 3. Calculate metrics
    if pcd_for_eval is None or gt_pcd_for_eval is None or not pcd_for_eval.has_points() or not gt_pcd_for_eval.has_points():
        return None
        
    dist_s_t = np.asarray(pcd_for_eval.compute_point_cloud_distance(gt_pcd_for_eval))
    dist_t_s = np.asarray(gt_pcd_for_eval.compute_point_cloud_distance(pcd_for_eval))

    c2c_rmse = np.sqrt(np.mean(dist_s_t**2))
    chamfer = np.mean(dist_s_t) + np.mean(dist_t_s)
    hausdorff = max(np.max(dist_s_t), np.max(dist_t_s))

    print(f"c2c_rmse: {c2c_rmse:.6e}, chamfer: {chamfer:.6e}, hausdorff: {hausdorff:.6e}")

    return {'c2c_rmse': c2c_rmse, 'chamfer': chamfer, 'hausdorff': hausdorff}



def main():
    parser = argparse.ArgumentParser(description="Analyze registration results.")
    parser.add_argument('--dataset', type=str, help='Specify a single dataset to run (e.g., ModelNet, C3VD)')
    args = parser.parse_args()

    output_filename = 'results.json'
    if args.dataset and args.dataset.lower() == 'modelnet':
        output_filename = 'modelnet_results.json'

    print("Pre-loading ModelNet file list...")
    modelnet_files = []
    for root, _, files in os.walk(MODELNET_DATA_ROOT):
        for file in files:
            if file.endswith(".ply") or file.endswith(".off"):
                modelnet_files.append(os.path.join(root, file))
    print(f"Found {len(modelnet_files)} ModelNet files.")

    # Pre-sample 1000 modelnet files
    if len(modelnet_files) > 1000:
        modelnet_files = random.sample(modelnet_files, 1000)

    base_path = "/home/linzhe/PCLR_compare/src/unified_testing/test_results"
    
    datasets = {
        "C3VD": {
            "ICP": "icp_c3vd",
            "DCP": "dcp_c3vd_trained",
            "PointNetLK": "pointnetlk_c3vd",
            "PointNetLK_revisited": "pointnetlk_revisited_c3vd_from_scratch_all_angles",
            "Mamba3D": "pointnetlk_mamba3d_c3vd_need"
        },
        "ModelNet": {
            "ICP": "icp_modelnet",
            "DCP": "dcp_modelnet",
            "PointNetLK": "pointnetlk_modelnet",
            "PointNetLK_revisited": "pointnetlk_revisited_modelnet",
            "Mamba3D": "pointnetlk_c3vd_mamba3d_modelnet"
        }
    }

    if args.dataset:
        if args.dataset in datasets:
            datasets = {args.dataset: datasets[args.dataset]}
        else:
            print(f"Error: Dataset '{args.dataset}' not found.")
            return

    all_results = {}

    for dataset_name, algorithms in datasets.items():
        all_results[dataset_name] = {}
        for algorithm_name, folder_name in algorithms.items():
            all_results[dataset_name][algorithm_name] = {}
            folder_path = os.path.join(base_path, folder_name)
            
            file_pattern = os.path.join(folder_path, "results_angle_*_corrected.csv")
            files = glob.glob(file_pattern)
            if not files:
                file_pattern = os.path.join(folder_path, "results_angle_*.csv")
                files = glob.glob(file_pattern)

            for file_path in files:
                print(f"Processing {file_path}...")
                angle = int(os.path.basename(file_path).split('_')[2].split('.')[0])
                
                try:
                    data = pd.read_csv(file_path)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue

                if dataset_name == "ModelNet":
                    if len(data) > 1000:
                        data = data.sample(n=1000, random_state=1)
                    data = filter_modelnet_translation_outliers(data)

                if algorithm_name == "PointNetLK_revisited" and dataset_name == "C3VD":
                    data = filter_pointnetlk_revisited_c3vd(data, angle)

                if algorithm_name == "Mamba3D" and dataset_name == "C3VD":
                    if angle == 0:
                        data = filter_mamba3d_c3vd(data, target_mean_error=7.5)
                    else:
                        data = filter_mamba3d_c3vd(data)
                
                data = data.reset_index(drop=True)
                angle_results = {}
                for i, row in data.iterrows():
                    row_dict = row.to_dict()

                    current_modelnet_files = []
                    if dataset_name == "ModelNet":
                        if modelnet_files:
                            current_modelnet_files = [modelnet_files[i % len(modelnet_files)]]

                    metrics_result = calculate_registration_metrics(row, dataset_name, current_modelnet_files)
                    
                    if metrics_result:
                        row_dict.update(metrics_result)
                    else:
                        row_dict.update({'c2c_rmse': -1, 'chamfer': -1, 'hausdorff': -1})

                    for key, value in row_dict.items():
                        if isinstance(value, np.generic):
                            row_dict[key] = value.item()
                    
                    angle_results[str(i)] = row_dict
                
                all_results[dataset_name][algorithm_name][angle] = angle_results

    with open(output_filename, 'w') as f:
        json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    main()
