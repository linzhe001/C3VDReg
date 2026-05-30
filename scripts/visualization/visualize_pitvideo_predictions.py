#!/usr/bin/env python3
"""
Visualize PITvideo test predictions by applying predicted transforms.

This script:
1. Reads the predictions JSON file
2. Loads source (dense) and target (sparse) point clouds
3. Applies the predicted transform to source
4. Saves the transformed point clouds with different colors for visualization
"""

import json
import numpy as np
import argparse
from pathlib import Path
from plyfile import PlyData, PlyElement
import sys

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def load_ply(filepath):
    """Load PLY file and return point cloud as numpy array [N, 3]"""
    print(f"  Loading: {filepath}")
    plydata = PlyData.read(filepath)
    pc = np.vstack([
        plydata['vertex']['x'],
        plydata['vertex']['y'],
        plydata['vertex']['z']
    ]).T
    return pc


def save_ply(filepath, points, colors=None):
    """
    Save point cloud to PLY file with optional colors.
    
    Args:
        filepath: output file path
        points: [N, 3] numpy array
        colors: [N, 3] numpy array (optional, RGB values 0-255)
    """
    points = points.astype(np.float32)
    
    if colors is not None:
        colors = colors.astype(np.uint8)
        vertex = np.array(
            [(points[i, 0], points[i, 1], points[i, 2], 
              colors[i, 0], colors[i, 1], colors[i, 2])
             for i in range(len(points))],
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                   ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        )
    else:
        vertex = np.array(
            [(points[i, 0], points[i, 1], points[i, 2])
             for i in range(len(points))],
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        )
    
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(filepath)
    print(f"  Saved: {filepath}")


def apply_transform(points, transform):
    """
    Apply 4x4 homogeneous transform to points.
    
    Args:
        points: [N, 3] numpy array
        transform: [4, 4] numpy array
    
    Returns:
        transformed_points: [N, 3] numpy array
    """
    # Convert to homogeneous coordinates
    points_homo = np.hstack([points, np.ones((len(points), 1))])  # [N, 4]
    
    # Apply transform
    transformed_homo = (transform @ points_homo.T).T  # [N, 4]
    
    # Convert back to 3D
    transformed_points = transformed_homo[:, :3]
    
    return transformed_points


def visualize_pair(pair_data, output_dir, data_root):
    """
    Visualize a single pair by applying predicted transform.
    
    Creates 4 output files:
    1. target_original.ply - target point cloud (green)
    2. source_original.ply - source point cloud before transform (red)
    3. source_transformed.ply - source after applying predicted transform (blue)
    4. combined.ply - all three point clouds together for comparison
    """
    pair_id = pair_data['pair_id']
    source_file = pair_data['source_file']
    target_file = pair_data['target_file']
    transform = np.array(pair_data['transform'])
    
    print(f"\nProcessing pair {pair_id}...")
    
    # Convert paths if they start with /mnt/c/ (WSL) to use data_root
    if source_file.startswith('/mnt/c/'):
        # Extract relative path from full path
        # e.g., /mnt/c/.../PITvideo_paired/test/143/143_034367 - Cloud.ply
        # -> test/143/143_034367 - Cloud.ply
        parts = source_file.split('PITvideo_paired/')
        if len(parts) == 2:
            source_file = data_root / parts[1]
        else:
            source_file = Path(source_file)
    else:
        source_file = Path(source_file)
    
    if target_file.startswith('/mnt/c/'):
        parts = target_file.split('PITvideo_paired/')
        if len(parts) == 2:
            target_file = data_root / parts[1]
        else:
            target_file = Path(target_file)
    else:
        target_file = Path(target_file)
    
    # Load point clouds
    source_pc = load_ply(source_file)
    target_pc = load_ply(target_file)
    
    print(f"  Source shape: {source_pc.shape}")
    print(f"  Target shape: {target_pc.shape}")
    
    # Apply predicted transform to source
    source_transformed = apply_transform(source_pc, transform)
    
    # Create output directory for this pair
    pair_output_dir = output_dir / f"pair_{pair_id}"
    pair_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define colors
    color_target = np.array([0, 255, 0], dtype=np.uint8)      # Green
    color_source = np.array([255, 0, 0], dtype=np.uint8)      # Red
    color_transformed = np.array([0, 0, 255], dtype=np.uint8) # Blue
    
    # Save individual point clouds
    save_ply(
        pair_output_dir / "target_original.ply",
        target_pc,
        np.tile(color_target, (len(target_pc), 1))
    )
    
    save_ply(
        pair_output_dir / "source_original.ply",
        source_pc,
        np.tile(color_source, (len(source_pc), 1))
    )
    
    save_ply(
        pair_output_dir / "source_transformed.ply",
        source_transformed,
        np.tile(color_transformed, (len(source_transformed), 1))
    )
    
    # Save combined point cloud
    combined_points = np.vstack([target_pc, source_pc, source_transformed])
    combined_colors = np.vstack([
        np.tile(color_target, (len(target_pc), 1)),
        np.tile(color_source, (len(source_pc), 1)),
        np.tile(color_transformed, (len(source_transformed), 1))
    ])
    
    save_ply(
        pair_output_dir / "combined.ply",
        combined_points,
        combined_colors
    )
    
    print(f"  ✓ Visualization saved to: {pair_output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize PITvideo test predictions"
    )
    parser.add_argument(
        '--predictions',
        type=Path,
        default=Path('results/pitvideo_test_predictions_finetuned.json'),
        help='Path to predictions JSON file'
    )
    parser.add_argument(
        '--data-root',
        type=Path,
        required=True,
        help='Path to PITvideo_paired root directory'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('visualization/pitvideo_results'),
        help='Output directory for visualization files'
    )
    parser.add_argument(
        '--pairs',
        type=str,
        nargs='*',
        default=None,
        help='Specific pair IDs to visualize (e.g., 143 241). If not specified, visualize all.'
    )
    args = parser.parse_args()
    
    # Load predictions
    print(f"Loading predictions from: {args.predictions}")
    with open(args.predictions) as f:
        data = json.load(f)
    
    test_pairs = data['test_pairs']
    print(f"Found {len(test_pairs)} test pairs")
    
    # Filter pairs if specified
    if args.pairs:
        test_pairs = [p for p in test_pairs if p['pair_id'] in args.pairs]
        print(f"Filtering to {len(test_pairs)} specified pairs: {args.pairs}")
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Process each pair
    print("\n" + "=" * 60)
    print("Starting visualization...")
    print("=" * 60)
    
    for pair in test_pairs:
        visualize_pair(pair, args.output, args.data_root)
    
    print("\n" + "=" * 60)
    print(f"✓ All visualizations completed!")
    print(f"Output directory: {args.output}")
    print("=" * 60)
    
    print("\nVisualization files per pair:")
    print("  - target_original.ply: Target point cloud (GREEN)")
    print("  - source_original.ply: Source before registration (RED)")
    print("  - source_transformed.ply: Source after registration (BLUE)")
    print("  - combined.ply: All three together for comparison")
    print("\nYou can open these files in CloudCompare or MeshLab for visualization.")


if __name__ == '__main__':
    main()
