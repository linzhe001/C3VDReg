#!/usr/bin/env python3
"""
Convert DA2 depth map (PNG) to point cloud with proper camera calibration
Supports camera intrinsics and distortion correction
"""

import numpy as np
import cv2
import open3d as o3d
import argparse
import json
from pathlib import Path


class CameraIntrinsics:
    """Camera intrinsic parameters"""

    def __init__(self, fx, fy, cx, cy, width, height,
                 distortion_coeffs=None, depth_scale=1.0):
        """
        Args:
            fx, fy: Focal lengths in pixels
            cx, cy: Principal point (optical center)
            width, height: Image dimensions
            distortion_coeffs: [k1, k2, p1, p2, k3] for OpenCV distortion model
            depth_scale: Scale factor for depth values
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.distortion_coeffs = distortion_coeffs
        self.depth_scale = depth_scale

    @classmethod
    def from_dict(cls, params):
        """Load from dictionary"""
        return cls(
            fx=params['fx'],
            fy=params['fy'],
            cx=params['cx'],
            cy=params['cy'],
            width=params['width'],
            height=params['height'],
            distortion_coeffs=params.get('distortion_coeffs'),
            depth_scale=params.get('depth_scale', 1.0)
        )

    @classmethod
    def from_json(cls, json_path):
        """Load from JSON file"""
        with open(json_path, 'r') as f:
            params = json.load(f)
        return cls.from_dict(params)

    @classmethod
    def from_fov(cls, width, height, hfov_deg, depth_scale=1.0):
        """
        Create intrinsics from horizontal field of view

        Args:
            width, height: Image dimensions
            hfov_deg: Horizontal field of view in degrees
            depth_scale: Depth scale factor
        """
        hfov_rad = np.deg2rad(hfov_deg)
        fx = width / (2 * np.tan(hfov_rad / 2))
        fy = fx  # Assume square pixels
        cx = width / 2.0
        cy = height / 2.0
        return cls(fx, fy, cx, cy, width, height, depth_scale=depth_scale)

    def get_camera_matrix(self):
        """Get OpenCV camera matrix (K)"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)

    def to_dict(self):
        """Export to dictionary"""
        return {
            'fx': self.fx,
            'fy': self.fy,
            'cx': self.cx,
            'cy': self.cy,
            'width': self.width,
            'height': self.height,
            'distortion_coeffs': self.distortion_coeffs,
            'depth_scale': self.depth_scale
        }

    def save_json(self, json_path):
        """Save to JSON file"""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Saved camera intrinsics to {json_path}")


def depth_to_pointcloud(depth_image_path, camera_intrinsics, output_path=None,
                        max_depth=10.0, undistort=True):
    """
    Convert depth image to 3D point cloud with proper camera calibration

    Args:
        depth_image_path: Path to depth PNG image
        camera_intrinsics: CameraIntrinsics object
        output_path: Output path for point cloud (.ply or .pcd)
        max_depth: Maximum depth value for filtering outliers
        undistort: Whether to undistort image coordinates

    Returns:
        open3d.geometry.PointCloud object
    """

    # Read depth image
    depth_img = cv2.imread(str(depth_image_path), cv2.IMREAD_UNCHANGED)

    if depth_img is None:
        raise ValueError(f"Failed to load depth image from {depth_image_path}")

    print(f"Loaded depth image: {depth_img.shape}, dtype: {depth_img.dtype}")
    print(f"Depth range: [{depth_img.min()}, {depth_img.max()}]")

    # Convert to float and normalize
    if depth_img.dtype == np.uint8:
        depth = depth_img.astype(np.float32) / 255.0 * camera_intrinsics.depth_scale
    elif depth_img.dtype == np.uint16:
        depth = depth_img.astype(np.float32) / 65535.0 * camera_intrinsics.depth_scale
    else:
        depth = depth_img.astype(np.float32) * camera_intrinsics.depth_scale

    height, width = depth.shape[:2]

    # Verify image dimensions match camera intrinsics
    if width != camera_intrinsics.width or height != camera_intrinsics.height:
        print(f"WARNING: Image size ({width}x{height}) doesn't match "
              f"camera intrinsics ({camera_intrinsics.width}x{camera_intrinsics.height})")

    # Create pixel coordinate grid
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)

    # Handle distortion if specified
    if undistort and camera_intrinsics.distortion_coeffs is not None:
        print("Applying distortion correction...")

        # Flatten coordinates
        uv = np.stack([u.ravel(), v.ravel()], axis=1).astype(np.float32)
        uv = uv.reshape(-1, 1, 2)

        # Undistort using OpenCV
        K = camera_intrinsics.get_camera_matrix()
        D = np.array(camera_intrinsics.distortion_coeffs, dtype=np.float32)
        uv_undistorted = cv2.undistortPoints(uv, K, D, P=K)

        # Reshape back
        u = uv_undistorted[:, 0, 0].reshape(height, width)
        v = uv_undistorted[:, 0, 1].reshape(height, width)

    # Back-project to 3D using camera intrinsics
    z = depth
    x = (u - camera_intrinsics.cx) * z / camera_intrinsics.fx
    y = (v - camera_intrinsics.cy) * z / camera_intrinsics.fy

    # Stack coordinates
    points = np.stack([x, y, z], axis=-1)
    points = points.reshape(-1, 3)

    # Filter invalid points
    valid_mask = (points[:, 2] > 0) & (points[:, 2] < max_depth)
    points = points[valid_mask]

    print(f"Generated {len(points)} valid points")

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Add depth-based coloring
    colors = np.zeros_like(points)
    depth_norm = np.clip(points[:, 2] / max_depth, 0, 1)
    colors[:, 0] = depth_norm  # Red
    colors[:, 1] = 0.5
    colors[:, 2] = 1.0 - depth_norm  # Blue
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(output_path), pcd)
        print(f"Saved point cloud to {output_path}")

    return pcd


def create_default_intrinsics_template(output_path="camera_intrinsics.json"):
    """Create a template JSON file for camera intrinsics"""
    template = {
        "_comment": "Camera intrinsic parameters",
        "width": 542,
        "height": 542,
        "fx": 542.0,
        "fy": 542.0,
        "cx": 271.0,
        "cy": 271.0,
        "distortion_coeffs": None,
        "depth_scale": 10.0,
        "_notes": {
            "fx_fy": "Focal lengths in pixels",
            "cx_cy": "Principal point (optical center)",
            "distortion_coeffs": "[k1, k2, p1, p2, k3] for radial-tangential model, or null",
            "depth_scale": "Multiply normalized depth [0,1] by this to get metric depth"
        }
    }

    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)

    print(f"Created template camera intrinsics file: {output_path}")
    print("Please edit this file with your actual camera parameters!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert DA2 depth PNG to point cloud with camera calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use camera intrinsics from JSON file
  python %(prog)s -i depth.png -c camera.json -o output.ply

  # Use horizontal field of view
  python %(prog)s -i depth.png --hfov 60 -o output.ply

  # Create template JSON file for editing
  python %(prog)s --create-template

  # Use default simple intrinsics (fx=fy=width)
  python %(prog)s -i depth.png -o output.ply
        """
    )

    parser.add_argument('--input', '-i',
                        help='Input depth PNG image path')
    parser.add_argument('--output', '-o',
                        help='Output point cloud path (.ply or .pcd)')
    parser.add_argument('--camera-json', '-c',
                        help='Camera intrinsics JSON file')
    parser.add_argument('--hfov', type=float,
                        help='Horizontal field of view in degrees (alternative to JSON)')
    parser.add_argument('--max-depth', type=float, default=10.0,
                        help='Maximum depth for filtering (default: 10.0)')
    parser.add_argument('--depth-scale', type=float, default=10.0,
                        help='Depth scale factor (default: 10.0)')
    parser.add_argument('--no-undistort', action='store_true',
                        help='Skip distortion correction')
    parser.add_argument('--create-template', action='store_true',
                        help='Create template camera intrinsics JSON file')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Visualize the point cloud')

    args = parser.parse_args()

    # Handle template creation
    if args.create_template:
        create_default_intrinsics_template()
        return

    # Validate required arguments
    if not args.input:
        parser.error("--input is required (unless using --create-template)")

    # Load or create camera intrinsics
    if args.camera_json:
        print(f"Loading camera intrinsics from {args.camera_json}")
        camera_intrinsics = CameraIntrinsics.from_json(args.camera_json)
    elif args.hfov:
        print(f"Creating intrinsics from horizontal FOV: {args.hfov}°")
        depth_img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
        height, width = depth_img.shape[:2]
        camera_intrinsics = CameraIntrinsics.from_fov(
            width, height, args.hfov, args.depth_scale
        )
    else:
        print("Using default intrinsics (fx=fy=width, no distortion)")
        depth_img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
        height, width = depth_img.shape[:2]
        camera_intrinsics = CameraIntrinsics(
            fx=width, fy=width,
            cx=width/2.0, cy=height/2.0,
            width=width, height=height,
            depth_scale=args.depth_scale
        )

    print("\nCamera Intrinsics:")
    print(f"  fx={camera_intrinsics.fx:.2f}, fy={camera_intrinsics.fy:.2f}")
    print(f"  cx={camera_intrinsics.cx:.2f}, cy={camera_intrinsics.cy:.2f}")
    print(f"  Image size: {camera_intrinsics.width}x{camera_intrinsics.height}")
    print(f"  Distortion: {camera_intrinsics.distortion_coeffs}")
    print(f"  Depth scale: {camera_intrinsics.depth_scale}")

    # Set default output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = input_path.with_suffix('.ply')

    # Convert depth to point cloud
    pcd = depth_to_pointcloud(
        depth_image_path=args.input,
        camera_intrinsics=camera_intrinsics,
        output_path=args.output,
        max_depth=args.max_depth,
        undistort=not args.no_undistort
    )

    # Visualize if requested
    if args.visualize:
        print("Visualizing point cloud... (Press Q to close)")
        o3d.visualization.draw_geometries([pcd],
                                          window_name="Point Cloud from Depth",
                                          width=1024, height=768)

    print("Done!")


if __name__ == "__main__":
    main()
