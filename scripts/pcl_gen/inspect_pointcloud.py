#!/usr/bin/env python3
"""Quick script to inspect point cloud statistics"""

import open3d as o3d
import numpy as np

# Load point cloud
pcd = o3d.io.read_point_cloud("tmp3q86tvc3.ply")

# Get points as numpy array
points = np.asarray(pcd.points)

print("=== Point Cloud Statistics ===")
print(f"Total points: {len(points)}")
print(f"\nCoordinate ranges:")
print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
print(f"\nMean: ({points[:, 0].mean():.3f}, {points[:, 1].mean():.3f}, {points[:, 2].mean():.3f})")
print(f"Std:  ({points[:, 0].std():.3f}, {points[:, 1].std():.3f}, {points[:, 2].std():.3f})")

# Bounding box
bbox = pcd.get_axis_aligned_bounding_box()
print(f"\nBounding box extent: {bbox.get_extent()}")
