"""Manifest schema and digest helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.benchmarking.metrics.units import PointUnit, canonicalize_point_unit

Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    scene_id: str
    trajectory_id: str
    frame_id: int
    split: Split
    source_path: str
    target_path: str
    gt_transform: list[list[float]]
    point_unit: PointUnit
    overlap_ratio: float | None
    metadata: dict[str, Any]


def _identity_transform() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _parse_frame_id(record: dict[str, Any]) -> int:
    if "frame_id" in record:
        return int(record["frame_id"])
    if "frame_idx" in record:
        return int(record["frame_idx"])
    sample_id = str(record.get("sample_id", ""))
    if ":" in sample_id:
        return int(sample_id.rsplit(":", 1)[-1])
    raise KeyError("Manifest record requires 'frame_id' or 'frame_idx'.")


def validate_manifest_record(record: dict[str, Any]) -> ManifestRecord:
    """Validate a manifest record and coerce legacy aliases into the canonical form."""

    sample_id = str(record["sample_id"])
    scene_id = str(record.get("scene_id") or record.get("scene"))
    if not scene_id:
        raise KeyError("Manifest record requires 'scene_id' or legacy alias 'scene'.")

    split = str(record["split"])
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split '{split}'.")

    trajectory_id = str(record.get("trajectory_id") or scene_id)
    frame_id = _parse_frame_id(record)
    source_path = str(record["source_path"])
    target_path = str(record["target_path"])
    gt_transform = record.get("gt_transform")
    if gt_transform is None:
        pair_mode = str(record.get("pair_mode", "one_to_one"))
        if pair_mode != "one_to_one":
            raise KeyError(
                "Manifest record requires 'gt_transform' for non one-to-one pairs."
            )
        gt_transform = _identity_transform()

    if not isinstance(gt_transform, list) or len(gt_transform) != 4:
        raise ValueError("gt_transform must be a 4x4 nested list.")
    for row in gt_transform:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError("gt_transform must be a 4x4 nested list.")
        for value in row:
            if not isinstance(value, (int, float)):
                raise ValueError("gt_transform values must be numeric.")

    point_unit = canonicalize_point_unit(str(record.get("point_unit", "m")))

    overlap_ratio = record.get("overlap_ratio")
    if overlap_ratio is not None:
        overlap_ratio = float(overlap_ratio)
        if not 0.0 <= overlap_ratio <= 1.0:
            raise ValueError("overlap_ratio must be within [0, 1].")

    metadata = dict(record.get("metadata", {}))
    for key, value in record.items():
        if key not in {
            "sample_id",
            "scene_id",
            "scene",
            "trajectory_id",
            "frame_id",
            "frame_idx",
            "split",
            "source_path",
            "target_path",
            "gt_transform",
            "point_unit",
            "overlap_ratio",
            "metadata",
        }:
            metadata.setdefault(key, value)

    return ManifestRecord(
        sample_id=sample_id,
        scene_id=scene_id,
        trajectory_id=trajectory_id,
        frame_id=frame_id,
        split=split,  # type: ignore[arg-type]
        source_path=source_path,
        target_path=target_path,
        gt_transform=[[float(value) for value in row] for row in gt_transform],
        point_unit=point_unit,
        overlap_ratio=overlap_ratio,
        metadata=metadata,
    )


def compute_manifest_digest(manifest_path: str | Path) -> str:
    """Compute a sha256 digest for a JSON/JSONL manifest payload."""

    manifest_path = Path(manifest_path)
    payload = manifest_path.read_bytes()
    return hashlib.sha256(payload).hexdigest()
