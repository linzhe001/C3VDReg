#!/usr/bin/env python3
"""
Build a deterministic inventory and optional manifest for the C3VD-derived
paired point cloud registration dataset used by this repository.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SCHEMA_VERSION = "0.1.0"
DEFAULT_SEED = 42
DEFAULT_FRAME_STRIDE = 10
DEFAULT_DATASET_NAME = "C3VD-derived paired point cloud corpus"

DEFAULT_SPLITS = {
    "train": [
        "cecum_t2_a",
        "cecum_t2_b",
        "cecum_t2_c",
        "cecum_t4_a",
        "cecum_t4_b",
        "desc_t4_a",
        "sigmoid_t1_a",
        "sigmoid_t2_a",
        "trans_t1_b",
        "trans_t2_a",
        "trans_t2_c",
        "trans_t3_a",
        "trans_t3_b",
    ],
    "val": [
        "cecum_t1_b",
        "sigmoid_t3_a",
        "trans_t4_b",
    ],
    "test": [
        "cecum_t1_a",
        "cecum_t3_a",
        "sigmoid_t3_b",
        "trans_t1_a",
        "trans_t2_b",
        "trans_t4_a",
    ],
}


@dataclass(frozen=True)
class DatasetLayout:
    data_root: Path
    source_root: Path
    visible_root: Path
    ref_root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=(
            "Dataset root containing C3VD_ply_source/, "
            "visible_point_cloud_ply_depth/, and C3VD_ref/."
        ),
    )
    parser.add_argument(
        "--subset-config-out",
        type=Path,
        default=None,
        help="Write the deterministic subset/split config JSON to this path.",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Optionally write a JSONL manifest with one entry per one-to-one pair.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=DEFAULT_FRAME_STRIDE,
        help="Frame stride used to define the MVP subset. Default: 10.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Recorded random seed for reproducibility metadata. Default: 42.",
    )
    return parser.parse_args()


def resolve_layout(data_root: Path) -> DatasetLayout:
    layout = DatasetLayout(
        data_root=data_root,
        source_root=data_root / "C3VD_ply_source",
        visible_root=data_root / "visible_point_cloud_ply_depth",
        ref_root=data_root / "C3VD_ref",
    )
    missing = [
        path
        for path in (layout.source_root, layout.visible_root, layout.ref_root)
        if not path.exists()
    ]
    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Dataset layout is incomplete. Missing: {missing_str}")
    return layout


def list_scenes(root: Path) -> list[str]:
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def count_pose_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for _ in handle)


def parse_ply_vertex_count(path: Path) -> int:
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise ValueError(f"Could not read vertex count from {path}")


def build_scene_inventory(layout: DatasetLayout) -> dict[str, dict[str, int]]:
    inventory: dict[str, dict[str, int]] = {}
    scenes = list_scenes(layout.source_root)
    for scene in scenes:
        source_files = sorted((layout.source_root / scene).glob("????_depth_pcd.ply"))
        visible_files = sorted(
            (layout.visible_root / scene).glob("frame_????_visible.ply")
        )
        ref_pose = layout.ref_root / scene / "pose.txt"
        visible_pose = layout.visible_root / scene / "pose.txt"
        reference_path = layout.ref_root / scene / "coverage_mesh.ply"
        if not ref_pose.exists():
            raise FileNotFoundError(f"Missing reference pose file: {ref_pose}")
        if not visible_pose.exists():
            raise FileNotFoundError(f"Missing visible pose file: {visible_pose}")
        if not reference_path.exists():
            raise FileNotFoundError(f"Missing reference asset: {reference_path}")

        ref_pose_lines = count_pose_lines(ref_pose)
        visible_pose_lines = count_pose_lines(visible_pose)
        source_count = len(source_files)
        visible_count = len(visible_files)
        if not (source_count == visible_count == ref_pose_lines == visible_pose_lines):
            raise ValueError(
                "Count mismatch for scene "
                f"{scene}: source={source_count}, visible={visible_count}, "
                f"ref_pose={ref_pose_lines}, visible_pose={visible_pose_lines}"
            )

        inventory[scene] = {
            "pair_count": source_count,
            "ref_pose_lines": ref_pose_lines,
            "visible_pose_lines": visible_pose_lines,
            "reference_vertices": parse_ply_vertex_count(reference_path),
        }
    return inventory


def validate_splits(all_scenes: Iterable[str], splits: dict[str, list[str]]) -> None:
    known = set(all_scenes)
    assigned = set()
    for split_name, scenes in splits.items():
        for scene in scenes:
            if scene not in known:
                raise ValueError(
                    f"Scene '{scene}' from split '{split_name}' "
                    "does not exist in the dataset root."
                )
            if scene in assigned:
                raise ValueError(f"Scene '{scene}' is assigned to multiple splits.")
            assigned.add(scene)
    missing = sorted(known - assigned)
    if missing:
        raise ValueError(f"Unassigned scenes in split config: {missing}")


def scene_subset_count(pair_count: int, frame_stride: int) -> int:
    return (pair_count + frame_stride - 1) // frame_stride


def build_subset_config(
    layout: DatasetLayout,
    inventory: dict[str, dict[str, int]],
    frame_stride: int,
    seed: int,
) -> dict[str, object]:
    validate_splits(inventory.keys(), DEFAULT_SPLITS)

    per_scene_subset = {
        scene: scene_subset_count(stats["pair_count"], frame_stride)
        for scene, stats in inventory.items()
    }
    split_pair_counts = {
        split: sum(inventory[scene]["pair_count"] for scene in scenes)
        for split, scenes in DEFAULT_SPLITS.items()
    }
    split_subset_counts = {
        split: sum(per_scene_subset[scene] for scene in scenes)
        for split, scenes in DEFAULT_SPLITS.items()
    }
    total_pairs = sum(stats["pair_count"] for stats in inventory.values())
    total_subset_pairs = sum(per_scene_subset.values())

    prefix_pair_counts: dict[str, int] = {}
    prefix_subset_counts: dict[str, int] = {}
    for scene, stats in inventory.items():
        prefix = scene.split("_")[0]
        prefix_pair_counts[prefix] = (
            prefix_pair_counts.get(prefix, 0) + stats["pair_count"]
        )
        prefix_subset_counts[prefix] = (
            prefix_subset_counts.get(prefix, 0) + per_scene_subset[scene]
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": DEFAULT_DATASET_NAME,
        "task_family": "point_cloud_registration",
        "pair_modes": {
            "primary": "one_to_one",
            "secondary": "scene_reference",
        },
        "data_root": str(layout.data_root),
        "directories": {
            "source_root": str(layout.source_root),
            "visible_root": str(layout.visible_root),
            "reference_root": str(layout.ref_root),
        },
        "seed": seed,
        "scene_splits": DEFAULT_SPLITS,
        "scene_pair_counts": {
            scene: inventory[scene]["pair_count"] for scene in sorted(inventory)
        },
        "scene_subset_counts": {
            scene: per_scene_subset[scene] for scene in sorted(inventory)
        },
        "split_pair_counts": split_pair_counts,
        "split_subset_counts": split_subset_counts,
        "prefix_pair_counts": dict(sorted(prefix_pair_counts.items())),
        "prefix_subset_counts": dict(sorted(prefix_subset_counts.items())),
        "reference_vertices": {
            "unique_values": sorted(
                {stats["reference_vertices"] for stats in inventory.values()}
            ),
            "count_per_scene": {
                scene: inventory[scene]["reference_vertices"]
                for scene in sorted(inventory)
            },
        },
        "subset_strategy": {
            "name": "mvp_10pct_stride10",
            "type": "scene_safe_temporal_stride",
            "frame_stride": frame_stride,
            "selection_rule": (
                "Apply the frozen scene split first, then keep only pairs "
                "whose zero-based frame_idx % frame_stride == 0 within each "
                "scene. Do not randomly drop views."
            ),
            "total_pairs": total_pairs,
            "total_subset_pairs": total_subset_pairs,
            "effective_ratio": round(total_subset_pairs / total_pairs, 6),
            "expected_speedup_vs_full_pairs": round(
                total_pairs / total_subset_pairs, 4
            ),
        },
        "notes": [
            (
                "The current dataset root is a preprocessed C3VD-derived "
                "paired point cloud corpus, not the raw official RGB-D release."
            ),
            (
                "All 22 reference assets are named coverage_mesh.ply but "
                "currently store fixed-size 50000-vertex reference point clouds."
            ),
            (
                "desc_t4_a is kept in train because it is the only "
                "desc-prefixed scene in the current dataset root."
            ),
        ],
    }


def iter_manifest_entries(
    layout: DatasetLayout,
    inventory: dict[str, dict[str, int]],
    frame_stride: int,
) -> Iterable[dict[str, object]]:
    scene_to_split = {
        scene: split for split, scenes in DEFAULT_SPLITS.items() for scene in scenes
    }
    for scene in sorted(inventory):
        split = scene_to_split[scene]
        region = scene.split("_")[0]
        source_files = sorted((layout.source_root / scene).glob("????_depth_pcd.ply"))
        target_root = layout.visible_root / scene
        reference_path = layout.ref_root / scene / "coverage_mesh.ply"
        ref_pose_path = layout.ref_root / scene / "pose.txt"
        target_pose_path = layout.visible_root / scene / "pose.txt"
        for source_path in source_files:
            frame_idx = int(source_path.name[:4])
            target_path = target_root / f"frame_{frame_idx:04d}_visible.ply"
            if not target_path.exists():
                raise FileNotFoundError(
                    f"Missing target file for {source_path}: {target_path}"
                )
            yield {
                "schema_version": SCHEMA_VERSION,
                "sample_id": f"{scene}:{frame_idx:04d}",
                "scene": scene,
                "region": region,
                "split": split,
                "frame_idx": frame_idx,
                "pair_mode": "one_to_one",
                "subset_selected": frame_idx % frame_stride == 0,
                "source_path": str(source_path),
                "target_path": str(target_path),
                "reference_path": str(reference_path),
                "reference_pose_path": str(ref_pose_path),
                "target_pose_path": str(target_pose_path),
                "reference_vertices": inventory[scene]["reference_vertices"],
            }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_manifest(path: Path, entries: Iterable[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")
            count += 1
    return count


def main() -> int:
    args = parse_args()
    layout = resolve_layout(args.data_root.resolve())
    inventory = build_scene_inventory(layout)
    subset_config = build_subset_config(
        layout=layout,
        inventory=inventory,
        frame_stride=args.frame_stride,
        seed=args.seed,
    )

    if args.subset_config_out is not None:
        write_json(args.subset_config_out, subset_config)

    manifest_count = None
    if args.manifest_out is not None:
        manifest_count = write_manifest(
            args.manifest_out,
            iter_manifest_entries(
                layout=layout,
                inventory=inventory,
                frame_stride=args.frame_stride,
            ),
        )

    summary = {
        "dataset_root": str(layout.data_root),
        "scene_count": len(inventory),
        "total_pairs": sum(stats["pair_count"] for stats in inventory.values()),
        "split_pair_counts": subset_config["split_pair_counts"],
        "split_subset_counts": subset_config["split_subset_counts"],
        "frame_stride": args.frame_stride,
        "subset_config_out": str(args.subset_config_out)
        if args.subset_config_out
        else None,
        "manifest_out": str(args.manifest_out) if args.manifest_out else None,
        "manifest_entries_written": manifest_count,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
