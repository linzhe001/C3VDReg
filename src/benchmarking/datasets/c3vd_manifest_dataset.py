"""Manifest-driven C3VD dataset loader."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from plyfile import PlyData

from src.benchmarking.manifest_schema import ManifestRecord, validate_manifest_record
from src.benchmarking.preprocess.pipeline import PreprocessPipeline


def _read_ply_xyz(path: Path) -> np.ndarray:
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    return np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)


class C3VDManifestDataset:
    """Load point-pair samples from a canonical benchmark manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        preprocess_pipeline: PreprocessPipeline | None = None,
        preprocess_profile_id: str | None = None,
        seed: int = 42,
        preprocess_overrides: Mapping[str, Any] | None = None,
        perturbation_config: Mapping[str, Any] | None = None,
        subset_config_path: str | Path | None = None,
        subset_name: str | None = None,
        dataset_root: str | Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.preprocess_pipeline = preprocess_pipeline
        self.preprocess_profile_id = preprocess_profile_id
        self.seed = seed
        self.preprocess_overrides = dict(preprocess_overrides or {})
        self.perturbation_config = dict(perturbation_config or {})
        self.subset_name = subset_name
        self.dataset_root = Path(dataset_root).resolve() if dataset_root else None
        self._subset_frame_stride = self._load_subset_stride(subset_config_path)
        self.records = self._load_records(split)

    def _load_subset_stride(self, subset_config_path: str | Path | None) -> int | None:
        if subset_config_path is None or self.subset_name is None:
            return None
        payload = json.loads(Path(subset_config_path).read_text(encoding="utf-8"))
        subset_strategy = payload.get("subset_strategy", {})
        if "stride" in self.subset_name and "frame_stride" in subset_strategy:
            return int(subset_strategy["frame_stride"])
        return None

    def _load_records(self, split: str) -> list[ManifestRecord]:
        scene_to_split: dict[str, str] = {}
        records: list[ManifestRecord] = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = validate_manifest_record(json.loads(line))
                if (
                    record.scene_id in scene_to_split
                    and scene_to_split[record.scene_id] != record.split
                ):
                    raise ValueError(
                        f"Scene '{record.scene_id}' appears in multiple splits: "
                        f"{scene_to_split[record.scene_id]} and {record.split}."
                    )
                scene_to_split[record.scene_id] = record.split
                if record.split != split:
                    continue
                if (
                    self._subset_frame_stride is not None
                    and record.frame_id % self._subset_frame_stride != 0
                ):
                    continue
                records.append(record)
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        if self.dataset_root is None:
            return path.resolve()
        return (self.dataset_root / path).resolve()

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        source_points = _read_ply_xyz(self._resolve_path(record.source_path))
        target_points = _read_ply_xyz(self._resolve_path(record.target_path))
        sample = {
            "record": record,
            "sample_id": record.sample_id,
            "scene_id": record.scene_id,
            "trajectory_id": record.trajectory_id,
            "frame_id": record.frame_id,
            "split": record.split,
            "gt_transform": np.asarray(record.gt_transform, dtype=np.float64),
            "point_unit": record.point_unit,
            "overlap_ratio": record.overlap_ratio,
            "metadata": dict(record.metadata),
            "source_points": source_points,
            "target_points": target_points,
            "source_points_raw": source_points.copy(),
            "target_points_raw": target_points.copy(),
        }
        if (
            self.preprocess_pipeline is not None
            and self.preprocess_profile_id is not None
        ):
            sample = self.preprocess_pipeline.run(
                sample=sample,
                profile_id=self.preprocess_profile_id,
                seed=self.seed + index,
                sampling_override=self.preprocess_overrides.get("sampling_override"),
                num_points_override=self.preprocess_overrides.get(
                    "num_points_override"
                ),
                perturbation_config=self.perturbation_config,
            )
        return sample

    def to_manifest_rows(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.records]
