#!/usr/bin/env python3
"""
Convert DA2 depth map (PNG) to point cloud
Author: Generated for depth to point cloud conversion
"""

import numpy as np
import cv2
import open3d as o3d
import argparse
from pathlib import Path


def depth_to_pointcloud(depth_image_path, output_path=None, focal_length=None,
                        max_depth=10.0, depth_scale=1.0):
    """
    Convert depth image to 3D point cloud

    Args:
        depth_image_path: Path to depth PNG image
        output_path: Output path for point cloud (.ply or .pcd)
        focal_length: Camera focal length (default: image width)
        max_depth: Maximum depth value for filtering outliers
        depth_scale: Scale factor for depth values (adjust based on DA2 output)

    Returns:
        open3d.geometry.PointCloud object
    """

    # Read depth image
    depth_img = cv2.imread(str(depth_image_path), cv2.IMREAD_UNCHANGED)

    if depth_img is None:
        raise ValueError(f"Failed to load depth image from {depth_image_path}")

    print(f"Loaded depth image: {depth_img.shape}, dtype: {depth_img.dtype}")
    print(f"Depth range: [{depth_img.min()}, {depth_img.max()}]")

    # Convert to float and normalize if needed
    if depth_img.dtype == np.uint8:
        # 8-bit depth map (0-255)
        depth = depth_img.astype(np.float32) / 255.0 * depth_scale
    elif depth_img.dtype == np.uint16:
        # 16-bit depth map (0-65535)
        depth = depth_img.astype(np.float32) / 65535.0 * depth_scale
    else:
        # Already float
        depth = depth_img.astype(np.float32) * depth_scale

    height, width = depth.shape[:2]

    # Set default focal length (can adjust based on your camera)
    if focal_length is None:
        focal_length = width  # Simple assumption

    # Camera intrinsics
    cx = width / 2.0
    cy = height / 2.0
    fx = fy = focal_length

    # Create mesh grid
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)

    # Back-project to 3D
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Stack coordinates
    points = np.stack([x, y, z], axis=-1)
    points = points.reshape(-1, 3)

    # Filter invalid points (depth = 0 or too large)
    valid_mask = (points[:, 2] > 0) & (points[:, 2] < max_depth)
    points = points[valid_mask]

    print(f"Generated {len(points)} valid points")

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Optional: add colors from original image if available
    # For grayscale depth map, use depth as color
    colors = np.zeros_like(points)
    colors[:, 0] = np.clip(points[:, 2] / max_depth, 0, 1)  # Red channel based on depth
    colors[:, 1] = 0.5  # Green
    colors[:, 2] = 1.0 - colors[:, 0]  # Blue (inverse of depth)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(output_path), pcd)
        print(f"Saved point cloud to {output_path}")

    return pcd


def visualize_pointcloud(pcd):
    """Visualize point cloud using Open3D"""
    print("Visualizing point cloud... (Press Q to close)")
    o3d.visualization.draw_geometries([pcd],
                                      window_name="Point Cloud from Depth",
                                      width=1024,
                                      height=768,
                                      point_show_normal=False)


def main():
    parser = argparse.ArgumentParser(description="Convert DA2 depth PNG to point cloud")
    parser.add_argument('--input', '-i', required=True,
                        help='Input depth PNG image path')
    parser.add_argument('--output', '-o', default=None,
                        help='Output point cloud path (.ply or .pcd)')
    parser.add_argument('--focal-length', '-f', type=float, default=None,
                        help='Camera focal length (default: image width)')
    parser.add_argument('--max-depth', type=float, default=10.0,
                        help='Maximum depth for filtering (default: 10.0)')
    parser.add_argument('--depth-scale', type=float, default=10.0,
                        help='Depth scale factor (default: 10.0 for DA2)')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Visualize the point cloud')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save point cloud (only visualize)')

    args = parser.parse_args()

    # Set default output path if not provided
    if args.output is None and not args.no_save:
        input_path = Path(args.input)
        args.output = input_path.with_suffix('.ply')

    # Convert depth to point cloud
    pcd = depth_to_pointcloud(
        depth_image_path=args.input,
        output_path=args.output if not args.no_save else None,
        focal_length=args.focal_length,
        max_depth=args.max_depth,
        depth_scale=args.depth_scale
    )

    # Visualize if requested
    if args.visualize:
        visualize_pointcloud(pcd)

    print("Done!")


if __name__ == "__main__":
    main()
