"""Measured dataset profile export for DPG-HPT."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.benchmarking.config.loader import _load_structured_file
from src.benchmarking.datasets.c3vd_manifest_dataset import _read_ply_xyz
from src.benchmarking.manifest_schema import (
    compute_manifest_digest,
    validate_manifest_record,
)

DEFAULT_SAMPLE_FRAMES = ("first", "middle", "last")
DEFAULT_MODALITIES = {
    "source_modality": "depth_reprojected_point_cloud",
    "target_modality": "ct_mesh_raycasted_visible_point_cloud",
}
DEFAULT_DOMAIN = "medical_endoscopic_cross_modal"
DEFAULT_PAIR_TYPE = "partial_to_partial"
DEFAULT_SPLIT_POLICY = "scene_safe"
DEFAULT_POSE_DIRECTION = "source_to_target"
DEFAULT_POSE_DIRECTION_CONFIDENCE = "benchmark_manifest_contract"
POSE_SE3_TOLERANCE = 1e-3


def _empty_pose_profile(translation_unit: str | None = None) -> dict[str, Any]:
    return {
        "storage_field": None,
        "storage_field_present_fraction": None,
        "pose_source": "unknown",
        "transform_format": "unknown",
        "transform_shapes": [],
        "shape_counts": {},
        "direction": "unknown",
        "direction_confidence": "missing_profile_evidence",
        "translation_unit": translation_unit,
        "translation_norm": {"p10": None, "p50": None, "p90": None},
        "rotation_determinant": {"p10": None, "p50": None, "p90": None},
        "rotation_orthonormal_error": {"p10": None, "p50": None, "p90": None},
        "valid_se3_fraction": None,
        "bottom_row_valid_fraction": None,
        "checked_record_count": None,
        "explicit_transform_count": None,
        "default_identity_count": None,
        "evidence": [],
    }


def _percentile_stats(values: Sequence[float | int]) -> dict[str, float | None]:
    if not values:
        return {"p10": None, "p50": None, "p90": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def _transform_shape_label(transform: np.ndarray) -> str:
    if transform.ndim != 2:
        return "x".join(str(dim) for dim in transform.shape)
    if transform.shape == (4, 4):
        return "4x4_homogeneous"
    if transform.shape == (3, 4):
        return "3x4_compact"
    return "x".join(str(dim) for dim in transform.shape)


def _se3_diagnostics(transform_values: Sequence[Sequence[float]]) -> dict[str, Any]:
    transform = np.asarray(transform_values, dtype=np.float64)
    shape = _transform_shape_label(transform)
    finite = bool(np.isfinite(transform).all())
    bottom_row_valid = None
    rotation_determinant = None
    rotation_orthonormal_error = None
    translation_norm = None
    valid_se3 = False

    if transform.shape in {(3, 4), (4, 4)} and finite:
        if transform.shape == (4, 4):
            bottom_row_valid = bool(
                np.allclose(
                    transform[3],
                    np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
                    atol=POSE_SE3_TOLERANCE,
                )
            )
            transform_3x4 = transform[:3, :4]
        else:
            bottom_row_valid = True
            transform_3x4 = transform

        rotation = transform_3x4[:3, :3]
        translation = transform_3x4[:3, 3]
        rotation_determinant = float(np.linalg.det(rotation))
        rotation_orthonormal_error = float(
            np.linalg.norm(rotation.T @ rotation - np.eye(3, dtype=np.float64))
        )
        translation_norm = float(np.linalg.norm(translation))
        valid_se3 = (
            bottom_row_valid
            and abs(rotation_determinant - 1.0) <= POSE_SE3_TOLERANCE
            and rotation_orthonormal_error <= POSE_SE3_TOLERANCE
        )

    return {
        "shape": shape,
        "translation_norm": translation_norm,
        "rotation_determinant": rotation_determinant,
        "rotation_orthonormal_error": rotation_orthonormal_error,
        "bottom_row_valid": bottom_row_valid,
        "valid_se3": valid_se3,
    }


def _fraction(values: Sequence[bool]) -> float | None:
    if not values:
        return None
    return float(sum(1 for value in values if value) / len(values))


def _pose_summary(
    records: Sequence[dict[str, Any]],
    translation_unit: str,
) -> dict[str, Any]:
    diagnostics = [
        _se3_diagnostics(record["gt_transform"])
        for record in records
        if "gt_transform" in record
    ]
    if not diagnostics:
        return _empty_pose_profile(translation_unit=translation_unit)

    shape_counts: dict[str, int] = defaultdict(int)
    translation_norms: list[float] = []
    rotation_determinants: list[float] = []
    rotation_orthonormal_errors: list[float] = []
    bottom_row_flags: list[bool] = []
    valid_se3_flags: list[bool] = []
    explicit_transform_flags = [
        bool(record.get("gt_transform_declared"))
        for record in records
        if "gt_transform" in record
    ]
    for item in diagnostics:
        shape_counts[str(item["shape"])] += 1
        if item["translation_norm"] is not None:
            translation_norms.append(float(item["translation_norm"]))
        if item["rotation_determinant"] is not None:
            rotation_determinants.append(float(item["rotation_determinant"]))
        if item["rotation_orthonormal_error"] is not None:
            rotation_orthonormal_errors.append(
                float(item["rotation_orthonormal_error"])
            )
        if item["bottom_row_valid"] is not None:
            bottom_row_flags.append(bool(item["bottom_row_valid"]))
        valid_se3_flags.append(bool(item["valid_se3"]))

    sorted_shape_counts = dict(sorted(shape_counts.items()))
    explicit_transform_count = sum(
        1 for value in explicit_transform_flags if value
    )
    default_identity_count = len(explicit_transform_flags) - explicit_transform_count
    if explicit_transform_count == len(explicit_transform_flags):
        pose_source = "manifest_explicit_gt_transform"
    elif default_identity_count == len(explicit_transform_flags):
        pose_source = "manifest_default_identity_for_one_to_one"
    else:
        pose_source = "mixed_manifest_explicit_and_default_identity"
    return {
        "storage_field": "gt_transform",
        "storage_field_present_fraction": _fraction(explicit_transform_flags),
        "pose_source": pose_source,
        "transform_format": "homogeneous_SE3",
        "transform_shapes": list(sorted_shape_counts),
        "shape_counts": sorted_shape_counts,
        "direction": DEFAULT_POSE_DIRECTION,
        "direction_confidence": DEFAULT_POSE_DIRECTION_CONFIDENCE,
        "translation_unit": translation_unit,
        "translation_norm": _percentile_stats(translation_norms),
        "rotation_determinant": _percentile_stats(rotation_determinants),
        "rotation_orthonormal_error": _percentile_stats(rotation_orthonormal_errors),
        "valid_se3_fraction": _fraction(valid_se3_flags),
        "bottom_row_valid_fraction": _fraction(bottom_row_flags),
        "checked_record_count": len(diagnostics),
        "explicit_transform_count": explicit_transform_count,
        "default_identity_count": default_identity_count,
        "evidence": [
            {
                "source": "src/benchmarking/manifest_schema.py",
                "supports": [
                    "validate_manifest_record requires 4x4 gt_transform",
                    "missing one_to_one gt_transform defaults to identity",
                    "ManifestRecord preserves gt_transform for dataset loaders",
                ],
            },
            {
                "source": "src/benchmarking/metrics/pose_metrics.py",
                "supports": [
                    "benchmark metrics apply gt_transform to source points",
                    "metrics compare transformed source points in target space",
                    "the benchmark pose contract is source_to_target",
                ],
            },
        ],
    }


def _bbox_diag(points: np.ndarray) -> float:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return float(np.linalg.norm(maxs - mins))


def _estimate_spacing(points: np.ndarray, max_points: int = 2048) -> float:
    if len(points) < 2:
        return 0.0
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, num=max_points, dtype=int)
        sampled = points[indices]
    else:
        sampled = points
    diff = sampled[:, None, :] - sampled[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    nearest = dists.min(axis=1)
    finite_nearest = nearest[np.isfinite(nearest)]
    return float(np.median(finite_nearest)) if len(finite_nearest) else 0.0


def _select_scene_records(
    records_by_scene: dict[str, list[dict[str, Any]]],
    sample_frames: Sequence[str],
) -> list[dict[str, Any]]:
    selectors = tuple(sample_frames) if sample_frames else DEFAULT_SAMPLE_FRAMES
    selected: list[dict[str, Any]] = []
    for scene_id in sorted(records_by_scene):
        scene_records = sorted(
            records_by_scene[scene_id],
            key=lambda item: item["frame_id"],
        )
        if not scene_records:
            continue
        chosen_indices: set[int] = set()
        for selector in selectors:
            if selector == "first":
                chosen_indices.add(0)
            elif selector == "middle":
                chosen_indices.add(len(scene_records) // 2)
            elif selector == "last":
                chosen_indices.add(len(scene_records) - 1)
            else:
                raise ValueError(
                    f"Unsupported sample frame selector '{selector}'. "
                    "Expected one of: first, middle, last."
                )
        selected.extend(scene_records[index] for index in sorted(chosen_indices))
    return selected


def _resolve_subset_path(
    subset_config_path: str | Path | None,
    data_config: dict[str, Any],
) -> Path | None:
    if subset_config_path is not None:
        return Path(subset_config_path).resolve()
    declared_path = data_config.get("subset_config_path")
    if declared_path is None:
        return None
    return Path(declared_path).resolve()


def _load_selection_context(
    dataset_config_path: str | Path,
    subset_config_path: str | Path | None,
) -> dict[str, Any]:
    config = _load_structured_file(Path(dataset_config_path))
    benchmark = config.get("benchmark", {})
    data = config.get("data", {})
    subset_path = _resolve_subset_path(subset_config_path, data)
    subset_payload = (
        json.loads(subset_path.read_text(encoding="utf-8"))
        if subset_path is not None
        else {}
    )
    return {
        "config": config,
        "benchmark": benchmark,
        "data": data,
        "manifest_path": Path(data["manifest_path"]).resolve(),
        "subset_config_path": subset_path,
        "subset_payload": subset_payload,
    }


def _load_selected_records(context: dict[str, Any]) -> list[dict[str, Any]]:
    benchmark = context["benchmark"]
    subset_payload = context["subset_payload"]
    split = str(benchmark.get("split", "test"))
    subset_name = benchmark.get("subset_name")
    frame_stride = None
    if subset_name and "stride" in str(subset_name):
        subset_strategy = subset_payload.get("subset_strategy", {})
        frame_stride_value = subset_strategy.get("frame_stride", 0)
        frame_stride = int(frame_stride_value) or None

    selected: list[dict[str, Any]] = []
    with context["manifest_path"].open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw_record = json.loads(line)
            record = validate_manifest_record(raw_record)
            if record.split != split:
                continue
            if frame_stride is not None and record.frame_id % frame_stride != 0:
                continue
            selected.append(
                {
                    "sample_id": record.sample_id,
                    "scene_id": record.scene_id,
                    "trajectory_id": record.trajectory_id,
                    "frame_id": record.frame_id,
                    "split": record.split,
                    "source_path": record.source_path,
                    "target_path": record.target_path,
                    "gt_transform": record.gt_transform,
                    "gt_transform_declared": (
                        raw_record.get("gt_transform") is not None
                    ),
                    "pair_mode": str(raw_record.get("pair_mode", "one_to_one")),
                    "point_unit": record.point_unit,
                    "metadata": dict(record.metadata),
                }
            )
    return selected


def _resolve_point_path(dataset_root: Path | None, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if dataset_root is None:
        return path.resolve()
    return (dataset_root / path).resolve()


def _infer_unit(
    declared_unit: str,
    target_bbox_diag_p50: float | None,
    target_spacing_p50: float | None,
) -> tuple[str, list[str]]:
    notes = [f"declared_point_unit={declared_unit}"]
    if target_bbox_diag_p50 is None or target_spacing_p50 is None:
        notes.append("insufficient_geometry_for_unit_inference")
        return declared_unit, notes
    small_extent = target_bbox_diag_p50 < 0.2
    dense_spacing = target_spacing_p50 < 0.01
    if declared_unit == "m" and small_extent and dense_spacing:
        notes.append("small_metric_extent_detected; flagging_mm_like_for_review")
        return "mm_like", notes
    notes.append("geometry_consistent_with_declared_unit")
    return declared_unit, notes


def _profile_digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_profile_stub(
    registry_path: str | Path,
    dataset_id: str,
) -> dict[str, Any]:
    """Load one dataset profile stub from the durable registry."""

    payload = _load_structured_file(Path(registry_path))
    datasets = payload.get("datasets", {})
    profile_payload = datasets.get(dataset_id)
    if profile_payload is None:
        requested_id = str(dataset_id).lower()
        for registry_key, candidate_profile in datasets.items():
            candidate_ids = [
                registry_key,
                candidate_profile.get("dataset_id"),
                *candidate_profile.get("aliases", []),
            ]
            if requested_id in {
                str(candidate_id).lower()
                for candidate_id in candidate_ids
                if candidate_id is not None
            }:
                profile_payload = candidate_profile
                break
    if profile_payload is None:
        raise KeyError(f"Dataset profile stub not found: {dataset_id}")
    profile = json.loads(json.dumps(profile_payload))
    profile.setdefault("dataset_id", dataset_id)
    profile.setdefault("profile_version", 1)
    profile.setdefault("profile_source", "registry_stub")
    profile.setdefault("coverage", {})
    profile["coverage"].setdefault("split", "unknown")
    profile["coverage"].setdefault("scene_count", None)
    profile["coverage"].setdefault("pair_count_selected", None)
    profile["coverage"].setdefault("pair_count_measured", None)
    profile.setdefault("route_hints", {})
    profile.setdefault("notes", ["loaded_from_dataset_profile_registry_stub"])
    pose_defaults = _empty_pose_profile(
        translation_unit=profile.get("data", {}).get("coordinate_unit")
    )
    profile.setdefault("pose", {})
    for key, value in pose_defaults.items():
        profile["pose"].setdefault(key, json.loads(json.dumps(value)))
    profile.setdefault("geometry", {})
    profile["geometry"].setdefault(
        "source_point_count",
        {"p10": None, "p50": None, "p90": None},
    )
    profile["geometry"].setdefault(
        "target_point_count",
        {"p10": None, "p50": None, "p90": None},
    )
    profile["geometry"].setdefault(
        "bbox_diag",
        {
            "source_p10": None,
            "source_p50": None,
            "source_p90": None,
            "target_p10": None,
            "target_p50": None,
            "target_p90": None,
        },
    )
    for key in (
        "source_p10",
        "source_p50",
        "source_p90",
        "target_p10",
        "target_p50",
        "target_p90",
    ):
        profile["geometry"]["bbox_diag"].setdefault(key, None)
    profile["geometry"].setdefault(
        "nearest_neighbor_spacing",
        {
            "source_p10": None,
            "source_p50": None,
            "source_p90": None,
            "target_p10": None,
            "target_p50": None,
            "target_p90": None,
        },
    )
    for key in (
        "source_p10",
        "source_p50",
        "source_p90",
        "target_p10",
        "target_p50",
        "target_p90",
    ):
        profile["geometry"]["nearest_neighbor_spacing"].setdefault(key, None)
    profile.setdefault("digests", {})
    profile["digests"].setdefault("manifest_digest", None)
    if not profile["digests"].get("profile_digest"):
        digest_payload = {
            key: value
            for key, value in profile.items()
            if key != "digests"
        }
        profile["digests"]["profile_digest"] = _profile_digest(digest_payload)
    return profile


def _density_ratio(
    source_counts: Sequence[int],
    target_counts: Sequence[int],
) -> float | None:
    source_p50 = _percentile_stats(source_counts)["p50"]
    target_p50 = _percentile_stats(target_counts)["p50"]
    if source_p50 is None or target_p50 in {None, 0}:
        return None
    return float(source_p50) / float(target_p50)


def _geometry_summary(values: Sequence[float]) -> dict[str, float | None]:
    stats = _percentile_stats(values)
    return {
        "p10": stats["p10"],
        "p50": stats["p50"],
        "p90": stats["p90"],
    }


def _geometry_pair_summary(
    source_values: Sequence[float],
    target_values: Sequence[float],
) -> dict[str, float | None]:
    source_stats = _percentile_stats(source_values)
    target_stats = _percentile_stats(target_values)
    return {
        "source_p10": source_stats["p10"],
        "source_p50": source_stats["p50"],
        "source_p90": source_stats["p90"],
        "target_p10": target_stats["p10"],
        "target_p50": target_stats["p50"],
        "target_p90": target_stats["p90"],
    }


def measure_dataset_profile(
    dataset_config_path: str | Path,
    subset_config_path: str | Path | None,
    dataset_id: str,
    sample_frames: Sequence[str] = DEFAULT_SAMPLE_FRAMES,
) -> dict[str, Any]:
    """Measure a target dataset profile from a manifest-driven dataset config."""

    context = _load_selection_context(
        dataset_config_path=dataset_config_path,
        subset_config_path=subset_config_path,
    )
    selected_records = _load_selected_records(context)
    if not selected_records:
        raise ValueError(
            "No manifest records matched the requested split/subset selection."
        )

    records_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in selected_records:
        records_by_scene[record["scene_id"]].append(record)

    measured_records = _select_scene_records(records_by_scene, sample_frames)
    dataset_root_value = context["data"].get("dataset_root")
    dataset_root = (
        Path(dataset_root_value).resolve()
        if dataset_root_value is not None
        else context["manifest_path"].parent
    )

    source_counts: list[int] = []
    target_counts: list[int] = []
    source_diags: list[float] = []
    target_diags: list[float] = []
    source_spacings: list[float] = []
    target_spacings: list[float] = []

    for record in measured_records:
        source_path = _resolve_point_path(dataset_root, record["source_path"])
        target_path = _resolve_point_path(dataset_root, record["target_path"])
        source_points = _read_ply_xyz(source_path)
        target_points = _read_ply_xyz(target_path)
        source_counts.append(int(source_points.shape[0]))
        target_counts.append(int(target_points.shape[0]))
        source_diags.append(_bbox_diag(source_points))
        target_diags.append(_bbox_diag(target_points))
        source_spacings.append(_estimate_spacing(source_points))
        target_spacings.append(_estimate_spacing(target_points))

    benchmark = context["benchmark"]
    subset_payload = context["subset_payload"]
    split = str(benchmark.get("split", "test"))
    declared_point_unit = str(benchmark.get("point_unit", "m"))
    bbox_summary = _geometry_pair_summary(source_diags, target_diags)
    spacing_summary = _geometry_pair_summary(source_spacings, target_spacings)
    inferred_unit, unit_notes = _infer_unit(
        declared_unit=declared_point_unit,
        target_bbox_diag_p50=bbox_summary["target_p50"],
        target_spacing_p50=spacing_summary["target_p50"],
    )
    pose_profile = _pose_summary(
        records=measured_records,
        translation_unit=declared_point_unit,
    )
    pose_shapes = ",".join(pose_profile["transform_shapes"]) or "unknown"
    unit_notes = [
        *unit_notes,
        f"pose_storage_field={pose_profile['storage_field']}",
        f"pose_source={pose_profile['pose_source']}",
        f"pose_transform_shapes={pose_shapes}",
        f"pose_direction={pose_profile['direction']}",
    ]

    split_pair_counts = subset_payload.get("split_pair_counts", {})
    split_subset_counts = subset_payload.get("split_subset_counts", {})
    payload: dict[str, Any] = {
        "dataset_id": dataset_id,
        "profile_version": 1,
        "profile_source": "measured",
        "created_at": None,
        "data": {
            "domain": DEFAULT_DOMAIN,
            "coordinate_unit": declared_point_unit,
            "inferred_unit": inferred_unit,
            "pair_type": DEFAULT_PAIR_TYPE,
            "source_modality": DEFAULT_MODALITIES["source_modality"],
            "target_modality": DEFAULT_MODALITIES["target_modality"],
            "split_policy": DEFAULT_SPLIT_POLICY,
        },
        "coverage": {
            "split": split,
            "scene_count": len(records_by_scene),
            "pair_count_selected": len(selected_records),
            "pair_count_measured": len(measured_records),
            "split_pair_count_full": split_pair_counts.get(split),
            "split_pair_count_subset": split_subset_counts.get(split),
            "sample_frames": list(sample_frames or DEFAULT_SAMPLE_FRAMES),
        },
        "geometry": {
            "source_point_count": _geometry_summary(source_counts),
            "target_point_count": _geometry_summary(target_counts),
            "bbox_diag": bbox_summary,
            "nearest_neighbor_spacing": spacing_summary,
            "density_ratio": {
                "source_to_target_count_p50": _density_ratio(
                    source_counts,
                    target_counts,
                ),
            },
        },
        "pose": pose_profile,
        "digests": {
            "manifest_digest": compute_manifest_digest(context["manifest_path"]),
            "profile_digest": None,
        },
        "route_hints": {
            "preferred_reference_domains": [
                "indoor_metric_scene",
                "local_scene_fragment",
            ],
            "rejected_reference_domains": [
                "outdoor_lidar",
                "normalized_object",
            ],
        },
        "notes": unit_notes,
    }
    payload["digests"]["profile_digest"] = _profile_digest(payload)
    return payload


def _format_geometry_row(
    label: str,
    stats: dict[str, float | None],
) -> str:
    return (
        f"| {label} | {stats['p10']} | {stats['p50']} | {stats['p90']} |"
    )


def render_dataset_profile_markdown(profile: dict[str, Any]) -> str:
    """Render a compact markdown summary for a measured dataset profile."""

    data = profile["data"]
    coverage = profile["coverage"]
    geometry = profile["geometry"]
    pose = profile.get(
        "pose",
        _empty_pose_profile(translation_unit=data.get("coordinate_unit")),
    )
    digests = profile["digests"]
    pose_shapes = ", ".join(pose.get("transform_shapes") or ["unknown"])
    lines = [
        f"# Dataset Profile: {profile['dataset_id']}",
        "",
        "## Summary",
        "",
        f"- profile_version: `{profile['profile_version']}`",
        f"- profile_source: `{profile['profile_source']}`",
        f"- domain: `{data['domain']}`",
        f"- pair_type: `{data['pair_type']}`",
        f"- coordinate_unit: `{data['coordinate_unit']}`",
        f"- inferred_unit: `{data['inferred_unit']}`",
        f"- split_policy: `{data['split_policy']}`",
        f"- split: `{coverage['split']}`",
        f"- scene_count: `{coverage['scene_count']}`",
        f"- pair_count_selected: `{coverage['pair_count_selected']}`",
        f"- pair_count_measured: `{coverage['pair_count_measured']}`",
        "",
        "## Geometry",
        "",
        "| Metric | p10 | p50 | p90 |",
        "| --- | ---: | ---: | ---: |",
        _format_geometry_row(
            "source_point_count",
            geometry["source_point_count"],
        ),
        _format_geometry_row(
            "target_point_count",
            geometry["target_point_count"],
        ),
        _format_geometry_row(
            "source_bbox_diag",
            {
                "p10": geometry["bbox_diag"]["source_p10"],
                "p50": geometry["bbox_diag"]["source_p50"],
                "p90": geometry["bbox_diag"]["source_p90"],
            },
        ),
        _format_geometry_row(
            "target_bbox_diag",
            {
                "p10": geometry["bbox_diag"]["target_p10"],
                "p50": geometry["bbox_diag"]["target_p50"],
                "p90": geometry["bbox_diag"]["target_p90"],
            },
        ),
        _format_geometry_row(
            "source_nn_spacing",
            {
                "p10": geometry["nearest_neighbor_spacing"]["source_p10"],
                "p50": geometry["nearest_neighbor_spacing"]["source_p50"],
                "p90": geometry["nearest_neighbor_spacing"]["source_p90"],
            },
        ),
        _format_geometry_row(
            "target_nn_spacing",
            {
                "p10": geometry["nearest_neighbor_spacing"]["target_p10"],
                "p50": geometry["nearest_neighbor_spacing"]["target_p50"],
                "p90": geometry["nearest_neighbor_spacing"]["target_p90"],
            },
        ),
        "",
        "## Pose",
        "",
        f"- storage_field: `{pose.get('storage_field')}`",
        (
            "- storage_field_present_fraction: "
            f"`{pose.get('storage_field_present_fraction')}`"
        ),
        f"- pose_source: `{pose.get('pose_source')}`",
        f"- transform_format: `{pose.get('transform_format')}`",
        f"- transform_shapes: `{pose_shapes}`",
        f"- direction: `{pose.get('direction')}`",
        f"- direction_confidence: `{pose.get('direction_confidence')}`",
        f"- translation_unit: `{pose.get('translation_unit')}`",
        f"- valid_se3_fraction: `{pose.get('valid_se3_fraction')}`",
        f"- checked_record_count: `{pose.get('checked_record_count')}`",
        f"- explicit_transform_count: `{pose.get('explicit_transform_count')}`",
        f"- default_identity_count: `{pose.get('default_identity_count')}`",
        "",
        "| Pose Metric | p10 | p50 | p90 |",
        "| --- | ---: | ---: | ---: |",
        _format_geometry_row(
            "translation_norm",
            pose.get("translation_norm", {"p10": None, "p50": None, "p90": None}),
        ),
        _format_geometry_row(
            "rotation_determinant",
            pose.get(
                "rotation_determinant",
                {"p10": None, "p50": None, "p90": None},
            ),
        ),
        _format_geometry_row(
            "rotation_orthonormal_error",
            pose.get(
                "rotation_orthonormal_error",
                {"p10": None, "p50": None, "p90": None},
            ),
        ),
        "",
        "## Digests",
        "",
        f"- manifest_digest: `{digests['manifest_digest']}`",
        f"- profile_digest: `{digests['profile_digest']}`",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in profile.get("notes", []))
    return "\n".join(lines) + "\n"


def write_dataset_profile_outputs(
    profile: dict[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown outputs for a measured dataset profile."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{profile['dataset_id']}.json"
    md_path = output_dir / f"{profile['dataset_id']}.md"
    json_payload = json.dumps(profile, indent=2, ensure_ascii=True) + "\n"
    json_path.write_text(json_payload, encoding="utf-8")
    md_path.write_text(
        render_dataset_profile_markdown(profile),
        encoding="utf-8",
    )
    return json_path, md_path
