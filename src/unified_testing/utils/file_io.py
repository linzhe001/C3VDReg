"""
File I/O utilities for point cloud loading
"""

import os
import numpy as np
from plyfile import PlyData


def load_off_file(file_path):
    """
    Load OFF format point cloud (ModelNet40)

    Args:
        file_path: .off file path

    Returns:
        points: [N, 3] numpy array
    """
    with open(file_path, "r") as f:
        # skip "OFF"
        first_line = f.readline().strip()
        if first_line != "OFF":
            raise ValueError(f"Not a valid OFF file: {file_path}")

        # read vertex and face counts
        n_verts, n_faces, _ = map(int, f.readline().strip().split())

        # read vertices
        verts = []
        for i in range(n_verts):
            verts.append(list(map(float, f.readline().strip().split())))

    points = np.array(verts, dtype=np.float32)
    return points


def load_ply_file(file_path):
    """
    Load PLY format point cloud (C3VD)

    Args:
        file_path: .ply file path

    Returns:
        points: [N, 3] numpy array
    """
    plydata = PlyData.read(file_path)
    vertices = plydata["vertex"]

    x = vertices["x"]
    y = vertices["y"]
    z = vertices["z"]

    points = np.stack([x, y, z], axis=1).astype(np.float32)

    return points


def load_point_cloud(file_path, num_points=1024, random_sample=True):
    """
    Universal point cloud loading function

    Args:
        file_path: point cloud file path
        num_points: number of points to sample
        random_sample: use random sampling if True, otherwise first N points

    Returns:
        points: [num_points, 3] numpy array
    """
    # auto-detect format
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".off":
        points = load_off_file(file_path)
    elif ext == ".ply":
        points = load_ply_file(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    # sample points
    if len(points) > num_points:
        if random_sample:
            indices = np.random.choice(len(points), num_points, replace=False)
        else:
            indices = np.arange(num_points)
    else:
        # upsample if not enough points
        indices = np.random.choice(len(points), num_points, replace=True)

    points = points[indices]

    return points


def extract_file_paths_modelnet(metadata, dataset_path):
    """
    Extract ModelNet40 file paths

    Args:
        metadata: dataset metadata (dict with 'category', 'filename', 'split')
        dataset_path: ModelNet40 root directory

    Returns:
        source_file, target_file: file paths (same for ModelNet40)
    """
    category = metadata.get("category", "")
    filename = metadata.get("filename", "")
    split = metadata.get("split", "test")

    file_path = os.path.join(dataset_path, category, split, filename)

    # ModelNet40: source and target are the same
    return file_path, file_path


def extract_file_paths_c3vd(metadata, dataset_path):
    """
    Extract C3VD file paths

    Args:
        metadata: dataset metadata (dict with 'scene', 'source', 'target')
        dataset_path: C3VD root directory

    Returns:
        source_file, target_file: file paths
    """
    scene = metadata.get("scene", "")
    source_name = metadata.get("source", "")
    target_name = metadata.get("target", "")

    source_file = os.path.join(dataset_path, "C3VD_ply_source", scene, source_name)

    target_file = os.path.join(
        dataset_path, "visible_point_cloud_ply_depth", scene, target_name
    )

    return source_file, target_file


def save_point_cloud_ply(file_path, points):
    """
    Save point cloud to PLY format

    Args:
        file_path: output .ply file path
        points: [N, 3] numpy array
    """
    from plyfile import PlyData, PlyElement

    vertices = np.array(
        [(p[0], p[1], p[2]) for p in points],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
    )

    el = PlyElement.describe(vertices, "vertex")
    PlyData([el]).write(file_path)
    print(f"Saved point cloud to: {file_path}")


def get_file_list_modelnet(dataset_path, category=None, split="test"):
    """
    Get list of ModelNet40 files

    Args:
        dataset_path: ModelNet40 root directory
        category: specific category (None for all categories)
        split: 'train' or 'test'

    Returns:
        file_list: list of dicts with metadata
    """
    file_list = []

    if category is not None:
        categories = [category]
    else:
        # get all categories
        categories = [
            d
            for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ]

    for cat in sorted(categories):
        split_dir = os.path.join(dataset_path, cat, split)
        if not os.path.exists(split_dir):
            continue

        for filename in sorted(os.listdir(split_dir)):
            if filename.endswith(".off"):
                file_list.append(
                    {
                        "category": cat,
                        "filename": filename,
                        "split": split,
                        "full_path": os.path.join(split_dir, filename),
                    }
                )

    return file_list


def get_file_list_c3vd(dataset_path, scene=None):
    """
    Get list of C3VD file pairs

    Args:
        dataset_path: C3VD root directory
        scene: specific scene (None for all scenes)

    Returns:
        file_list: list of dicts with metadata
    """
    file_list = []

    source_root = os.path.join(dataset_path, "C3VD_ply_source")
    target_root = os.path.join(dataset_path, "visible_point_cloud_ply_depth")

    if scene is not None:
        scenes = [scene]
    else:
        # get all scenes
        scenes = [
            d
            for d in os.listdir(source_root)
            if os.path.isdir(os.path.join(source_root, d))
        ]

    for sc in sorted(scenes):
        source_scene_dir = os.path.join(source_root, sc)
        target_scene_dir = os.path.join(target_root, sc)

        if not os.path.exists(target_scene_dir):
            continue

        source_files = sorted(
            [f for f in os.listdir(source_scene_dir) if f.endswith(".ply")]
        )
        target_files = sorted(
            [f for f in os.listdir(target_scene_dir) if f.endswith(".ply")]
        )

        # pair source and target (assuming one-to-one correspondence)
        for src, tgt in zip(source_files, target_files):
            file_list.append(
                {
                    "scene": sc,
                    "source": src,
                    "target": tgt,
                    "source_path": os.path.join(source_scene_dir, src),
                    "target_path": os.path.join(target_scene_dir, tgt),
                }
            )

    return file_list
