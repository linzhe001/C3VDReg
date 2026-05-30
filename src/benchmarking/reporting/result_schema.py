"""Per-sample result schema for analysis-ready benchmark outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ResultRecord:
    sample_id: str
    scene_id: str
    trajectory_id: str
    frame_id: int
    split: str
    model_id: str
    preprocess_profile_id: str
    adapter_private_transform_id: str
    refinement_track: str
    train_regime: str
    seed: int | None
    manifest_digest: str
    config_digest: str
    git_snapshot: str | None
    checkpoint_id: str | None
    rre_deg: float
    rte_mm: float
    rmse_mm: float | None
    success_1deg_1mm: int
    success_3deg_3mm: int
    success_5deg_5mm: int
    success_10deg_10mm: int
    visible_nn_mean_mm: float | None
    visible_nn_median_mm: float | None
    visible_nn_p90_mm: float | None
    trimmed_chamfer_mm: float | None
    overlap_only_distance_mm: float | None
    preprocess_time_ms: float | None
    inference_time_ms: float | None
    refinement_time_ms: float | None
    latency_ms: float | None
    peak_memory_mb: float | None
    overlap_bin: str | None
    rotation_bin: str | None
    translation_bin: str | None
    artifact_bin: str | None
    train_time_ms: float | None = None
    train_peak_memory_mb: float | None = None
    train_peak_allocated_mb: float | None = None
    failure_tags: list[str] = field(default_factory=list)


def serialize_result_record(record: ResultRecord) -> dict[str, object]:
    return asdict(record)


def validate_result_record(record: dict[str, Any]) -> ResultRecord:
    required_fields = {
        "sample_id",
        "scene_id",
        "trajectory_id",
        "frame_id",
        "split",
        "model_id",
        "preprocess_profile_id",
        "adapter_private_transform_id",
        "refinement_track",
        "train_regime",
        "manifest_digest",
        "config_digest",
        "rre_deg",
        "rte_mm",
        "success_1deg_1mm",
        "success_3deg_3mm",
        "success_5deg_5mm",
        "success_10deg_10mm",
    }
    missing = sorted(required_fields - set(record))
    if missing:
        raise KeyError(f"ResultRecord is missing required fields: {missing}")
    if str(record["split"]) not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split '{record['split']}'.")

    result = ResultRecord(
        sample_id=str(record["sample_id"]),
        scene_id=str(record["scene_id"]),
        trajectory_id=str(record["trajectory_id"]),
        frame_id=int(record["frame_id"]),
        split=str(record["split"]),
        model_id=str(record["model_id"]),
        preprocess_profile_id=str(record["preprocess_profile_id"]),
        adapter_private_transform_id=str(record["adapter_private_transform_id"]),
        refinement_track=str(record["refinement_track"]),
        train_regime=str(record["train_regime"]),
        seed=int(record["seed"]) if record.get("seed") is not None else None,
        manifest_digest=str(record["manifest_digest"]),
        config_digest=str(record["config_digest"]),
        git_snapshot=(
            str(record["git_snapshot"])
            if record.get("git_snapshot") is not None
            else None
        ),
        checkpoint_id=(
            str(record["checkpoint_id"])
            if record.get("checkpoint_id") is not None
            else None
        ),
        rre_deg=float(record["rre_deg"]),
        rte_mm=float(record["rte_mm"]),
        rmse_mm=float(record["rmse_mm"]) if record.get("rmse_mm") is not None else None,
        success_1deg_1mm=int(record["success_1deg_1mm"]),
        success_3deg_3mm=int(record["success_3deg_3mm"]),
        success_5deg_5mm=int(record["success_5deg_5mm"]),
        success_10deg_10mm=int(record["success_10deg_10mm"]),
        visible_nn_mean_mm=(
            float(record["visible_nn_mean_mm"])
            if record.get("visible_nn_mean_mm") is not None
            else None
        ),
        visible_nn_median_mm=(
            float(record["visible_nn_median_mm"])
            if record.get("visible_nn_median_mm") is not None
            else None
        ),
        visible_nn_p90_mm=(
            float(record["visible_nn_p90_mm"])
            if record.get("visible_nn_p90_mm") is not None
            else None
        ),
        trimmed_chamfer_mm=(
            float(record["trimmed_chamfer_mm"])
            if record.get("trimmed_chamfer_mm") is not None
            else None
        ),
        overlap_only_distance_mm=(
            float(record["overlap_only_distance_mm"])
            if record.get("overlap_only_distance_mm") is not None
            else None
        ),
        preprocess_time_ms=(
            float(record["preprocess_time_ms"])
            if record.get("preprocess_time_ms") is not None
            else None
        ),
        inference_time_ms=(
            float(record["inference_time_ms"])
            if record.get("inference_time_ms") is not None
            else None
        ),
        refinement_time_ms=(
            float(record["refinement_time_ms"])
            if record.get("refinement_time_ms") is not None
            else None
        ),
        latency_ms=(
            float(record["latency_ms"])
            if record.get("latency_ms") is not None
            else None
        ),
        peak_memory_mb=(
            float(record["peak_memory_mb"])
            if record.get("peak_memory_mb") is not None
            else None
        ),
        overlap_bin=str(record["overlap_bin"])
        if record.get("overlap_bin") is not None
        else None,
        rotation_bin=(
            str(record["rotation_bin"])
            if record.get("rotation_bin") is not None
            else None
        ),
        translation_bin=(
            str(record["translation_bin"])
            if record.get("translation_bin") is not None
            else None
        ),
        artifact_bin=str(record["artifact_bin"])
        if record.get("artifact_bin") is not None
        else None,
        train_time_ms=(
            float(record["train_time_ms"])
            if record.get("train_time_ms") is not None
            else None
        ),
        train_peak_memory_mb=(
            float(record["train_peak_memory_mb"])
            if record.get("train_peak_memory_mb") is not None
            else None
        ),
        train_peak_allocated_mb=(
            float(record["train_peak_allocated_mb"])
            if record.get("train_peak_allocated_mb") is not None
            else None
        ),
        failure_tags=[str(tag) for tag in record.get("failure_tags", [])],
    )
    return result
