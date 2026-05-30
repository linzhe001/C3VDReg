"""End-to-end benchmark eval runner."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.benchmarking.analysis.failure_analysis import (
    assign_failure_tags,
    mine_failure_cases,
)
from src.benchmarking.analysis.geometry_diagnostics import (
    build_distance_heatmap_manifest,
    compute_visible_distance_stats,
)
from src.benchmarking.analysis.run_card import build_run_card
from src.benchmarking.datasets.c3vd_manifest_dataset import C3VDManifestDataset
from src.benchmarking.diagnostics.compatibility import (
    assert_baseline_repo_clean,
    assert_model_preprocess_compatibility,
    assert_runtime_policy_compatible,
)
from src.benchmarking.manifest_schema import compute_manifest_digest
from src.benchmarking.metrics.pose_metrics import compute_pose_metrics
from src.benchmarking.metrics.units import distance_to_millimeters
from src.benchmarking.preprocess.pipeline import PreprocessPipeline
from src.benchmarking.preprocess.registry import PreprocessRegistry
from src.benchmarking.registry.model_registry import ModelRegistry
from src.benchmarking.reporting.aggregate_results import aggregate_results
from src.benchmarking.reporting.build_html_report import build_html_report
from src.benchmarking.reporting.export_figures import (
    export_curve_figures,
    export_geometry_figures,
)
from src.benchmarking.reporting.export_tables import (
    export_bucket_tables,
    export_leaderboard_tables,
)
from src.benchmarking.reporting.result_schema import (
    ResultRecord,
    serialize_result_record,
)
from src.utils.git_snapshot import git_snapshot


def _sha256_json(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _load_symbol(path: str) -> Any:
    module_name, symbol_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def _instantiate_adapter(spec: Any, config: dict[str, Any]) -> Any:
    adapter_class = _load_symbol(spec.adapter_path)
    adapter = adapter_class(SimpleNamespace(**config))
    if hasattr(adapter, "load_model"):
        adapter.load_model(config.get("checkpoint_path"))
    return adapter


def _prediction_to_transform(prediction: Any) -> np.ndarray:
    if isinstance(prediction, tuple) and len(prediction) == 2:
        rotation, translation = prediction
        rotation = np.asarray(rotation, dtype=np.float64)
        translation = np.asarray(translation, dtype=np.float64).reshape(3)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        return transform

    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape == (4, 4):
        return prediction
    raise ValueError(f"Unsupported prediction format with shape {prediction.shape}.")


def _find_train_metadata_from_checkpoint(
    checkpoint_path: str | None,
) -> Path | None:
    if not checkpoint_path:
        return None
    path = Path(checkpoint_path).expanduser().resolve()
    for parent in [path.parent, *path.parents]:
        candidate = parent / "run_metadata.json"
        if candidate.exists():
            return candidate
    return None


def _load_train_metadata(config: dict[str, Any]) -> dict[str, Any]:
    runtime = dict(config.get("runtime", {}))
    metadata_path = runtime.get("train_metadata_path")
    path = (
        Path(str(metadata_path)).expanduser().resolve()
        if metadata_path
        else _find_train_metadata_from_checkpoint(
            config["model"].get("checkpoint_path")
        )
    )
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["train_metadata_path"] = str(path)
    return payload


def _rotation_bin(gt_transform: np.ndarray) -> str:
    cos_theta = np.clip((np.trace(gt_transform[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.degrees(np.arccos(cos_theta)))
    if angle < 1.0:
        return "0-1deg"
    if angle < 5.0:
        return "1-5deg"
    if angle < 10.0:
        return "5-10deg"
    return "10+deg"


def _translation_bin(gt_transform: np.ndarray, point_unit: str) -> str:
    distance_mm = distance_to_millimeters(
        np.linalg.norm(gt_transform[:3, 3]), point_unit
    )
    if distance_mm < 1.0:
        return "0-1mm"
    if distance_mm < 5.0:
        return "1-5mm"
    if distance_mm < 10.0:
        return "5-10mm"
    return "10+mm"


def _overlap_bin(overlap_ratio: float | None) -> str:
    if overlap_ratio is None:
        return "missing"
    if overlap_ratio < 0.25:
        return "very_low"
    if overlap_ratio < 0.5:
        return "low"
    if overlap_ratio < 0.75:
        return "medium"
    return "high"


def run_eval(
    config: dict[str, Any],
    model_registry: ModelRegistry | None = None,
) -> dict[str, Any]:
    """Run the benchmark eval loop and export analysis-ready artifacts."""

    repo_root = Path(__file__).resolve().parents[3]
    output_dir = Path(config["runtime"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_config_path = output_dir / "normalized_eval_config.json"
    normalized_config_path.write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    config_digest = _sha256_json(config)
    manifest_digest = compute_manifest_digest(config["data"]["manifest_path"])
    snapshot = git_snapshot(
        repo_root=repo_root,
        config_path=normalized_config_path,
        manifest_path=config["data"]["manifest_path"],
        subset_config_path=config["data"].get("subset_config_path"),
        preprocess_profile_id=config["preprocess"]["profile"],
        model_id=config["model"]["id"],
        checkpoint_id=config["model"].get("checkpoint_path"),
        output_path=output_dir / "git_snapshot.json",
    )

    model_registry = model_registry or ModelRegistry()
    preprocess_registry = PreprocessRegistry()
    spec = assert_model_preprocess_compatibility(
        registry=model_registry,
        model_id=config["model"]["id"],
        preprocess_profile_id=config["preprocess"]["profile"],
    )
    assert_runtime_policy_compatible(spec, str(config["runtime"]["device"]))
    assert_baseline_repo_clean(repo_root=repo_root, spec=spec)
    if not spec.capabilities.supports_eval:
        raise ValueError(
            f"Model '{spec.model_id}' does not currently support benchmark eval."
        )
    train_metadata = _load_train_metadata(config)
    adapter_config = dict(config["model"].get("overrides", {}))
    adapter_config.setdefault("device", config["runtime"]["device"])
    adapter_config["checkpoint_path"] = config["model"].get("checkpoint_path")
    adapter = _instantiate_adapter(spec, adapter_config)

    dataset = C3VDManifestDataset(
        manifest_path=config["data"]["manifest_path"],
        split=config["benchmark"]["split"],
        preprocess_pipeline=PreprocessPipeline(preprocess_registry),
        preprocess_profile_id=config["preprocess"]["profile"],
        seed=int(config["preprocess"]["seed"]),
        preprocess_overrides={
            "sampling_override": config["preprocess"].get("sampling_override"),
            "num_points_override": config["preprocess"].get("num_points_override"),
        },
        perturbation_config=config.get("perturbation", {}),
        subset_config_path=config["data"].get("subset_config_path"),
        subset_name=config["benchmark"].get("subset_name"),
        dataset_root=config["data"].get("dataset_root"),
    )

    results: list[dict[str, object]] = []
    pose_transform_rows: list[dict[str, object]] = []
    worst_render_case: dict[str, object] | None = None
    point_unit = str(config["benchmark"].get("point_unit", "m"))
    export_transforms = bool(config.get("analysis", {}).get("export_transforms", False))
    for sample in dataset:
        source_points = np.asarray(sample["source_points"], dtype=np.float64)
        target_points = np.asarray(sample["target_points"], dtype=np.float64)
        infer_start = perf_counter()
        prediction = adapter.predict(source_points, target_points)
        inference_time_ms = (perf_counter() - infer_start) * 1000.0
        pred_transform = _prediction_to_transform(prediction)
        gt_transform = np.asarray(sample["gt_transform"], dtype=np.float64)
        pose_metrics = compute_pose_metrics(
            pred_transform=pred_transform,
            gt_transform=gt_transform,
            source_points=source_points,
            target_points=target_points,
            point_unit=point_unit,
        )
        geometry = compute_visible_distance_stats(
            source_points=source_points,
            target_points=target_points,
            pred_transform=pred_transform,
            gt_transform=gt_transform,
            sample_count=int(config["analysis"]["geometry"]["sample_count"]),
            distance_mode=str(config["analysis"]["geometry"]["distance_mode"]),
            point_unit=point_unit,
        )

        record = ResultRecord(
            sample_id=str(sample["sample_id"]),
            scene_id=str(sample["scene_id"]),
            trajectory_id=str(sample["trajectory_id"]),
            frame_id=int(sample["frame_id"]),
            split=str(sample["split"]),
            model_id=spec.model_id,
            preprocess_profile_id=str(sample["preprocess_profile_id"]),
            adapter_private_transform_id=spec.capabilities.private_input_transform_id,
            refinement_track=spec.capabilities.refinement_mode,
            train_regime=(
                str(train_metadata.get("train_bridge"))
                if train_metadata
                else "eval_only"
            ),
            seed=int(config["preprocess"]["seed"]),
            manifest_digest=manifest_digest,
            config_digest=config_digest,
            git_snapshot=str(snapshot["git_commit"]),
            checkpoint_id=config["model"].get("checkpoint_path"),
            rre_deg=float(pose_metrics["rre_deg"]),
            rte_mm=float(pose_metrics["rte_mm"]),
            rmse_mm=(
                float(pose_metrics["rmse_mm"])
                if pose_metrics["rmse_mm"] is not None
                else None
            ),
            success_1deg_1mm=int(pose_metrics["success_1deg_1mm"]),
            success_3deg_3mm=int(pose_metrics["success_3deg_3mm"]),
            success_5deg_5mm=int(pose_metrics["success_5deg_5mm"]),
            success_10deg_10mm=int(pose_metrics["success_10deg_10mm"]),
            visible_nn_mean_mm=float(geometry["visible_nn_mean_mm"]),
            visible_nn_median_mm=float(geometry["visible_nn_median_mm"]),
            visible_nn_p90_mm=float(geometry["visible_nn_p90_mm"]),
            trimmed_chamfer_mm=float(geometry["trimmed_chamfer_mm"]),
            overlap_only_distance_mm=float(geometry["overlap_only_distance_mm"]),
            preprocess_time_ms=float(sample.get("preprocess_time_ms", 0.0)),
            inference_time_ms=inference_time_ms,
            refinement_time_ms=0.0,
            latency_ms=float(sample.get("preprocess_time_ms", 0.0)) + inference_time_ms,
            peak_memory_mb=0.0,
            overlap_bin=_overlap_bin(sample.get("overlap_ratio")),
            rotation_bin=_rotation_bin(gt_transform),
            translation_bin=_translation_bin(gt_transform, point_unit),
            artifact_bin=sample["metadata"].get("artifact_bin"),
            train_time_ms=(
                float(train_metadata["train_time_ms"])
                if train_metadata.get("train_time_ms") is not None
                else None
            ),
            train_peak_memory_mb=(
                float(train_metadata["train_peak_memory_mb"])
                if train_metadata.get("train_peak_memory_mb") is not None
                else None
            ),
            train_peak_allocated_mb=(
                float(train_metadata["train_peak_allocated_mb"])
                if train_metadata.get("train_peak_allocated_mb") is not None
                else None
            ),
            failure_tags=[],
        )
        record.failure_tags = assign_failure_tags(serialize_result_record(record))
        results.append(serialize_result_record(record))
        if export_transforms:
            pose_transform_rows.append(
                {
                    "sample_id": record.sample_id,
                    "scene_id": record.scene_id,
                    "trajectory_id": record.trajectory_id,
                    "frame_id": record.frame_id,
                    "split": record.split,
                    "model_id": spec.model_id,
                    "point_unit": point_unit,
                    "rre_deg": float(pose_metrics["rre_deg"]),
                    "rte_mm": float(pose_metrics["rte_mm"]),
                    "pred_transform": pred_transform.tolist(),
                    "gt_transform": gt_transform.tolist(),
                }
            )
        if (
            worst_render_case is None
            or float(geometry["visible_nn_mean_mm"])
            > float(worst_render_case["visible_nn_mean_mm"])
        ):
            worst_render_case = {
                "sample_id": record.sample_id,
                "scene_id": record.scene_id,
                "visible_nn_mean_mm": float(geometry["visible_nn_mean_mm"]),
                "rre_deg": float(pose_metrics["rre_deg"]),
                "rte_mm": float(pose_metrics["rte_mm"]),
                "point_unit": point_unit,
                "source_points": source_points,
                "target_points": target_points,
                "pred_transform": pred_transform,
            }

    results_path = output_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row) + "\n")

    if export_transforms:
        transforms_path = output_dir / "pose_transforms.jsonl"
        with transforms_path.open("w", encoding="utf-8") as handle:
            for row in pose_transform_rows:
                handle.write(json.dumps(row) + "\n")

    aggregate_summary = aggregate_results(
        results,
        bucket_keys=list(config["analysis"]["bucket_keys"]),
    )
    table_paths = export_leaderboard_tables(
        aggregate_summary=aggregate_summary,
        output_dir=output_dir,
        markdown_tables=bool(config["analysis"]["export"]["markdown_tables"]),
    )
    table_paths.update(
        export_bucket_tables(aggregate_summary=aggregate_summary, output_dir=output_dir)
    )

    figure_paths = {}
    if config["analysis"]["export"]["png"]:
        figure_paths.update(export_curve_figures(results, output_dir))
        if (
            config["analysis"]["geometry"]["export_histogram"]
            or config["analysis"]["geometry"]["export_cdf"]
        ):
            figure_paths.update(
                export_geometry_figures(
                    results,
                    output_dir,
                    render_case=worst_render_case,
                )
            )

    qualitative_dir = output_dir / "qualitative" / "failure_gallery"
    qualitative_dir.mkdir(parents=True, exist_ok=True)
    failure_cases = mine_failure_cases(
        results,
        topk=int(config["analysis"]["qualitative"]["topk_failures"]),
    )
    (qualitative_dir / "failure_gallery_manifest.json").write_text(
        json.dumps(failure_cases, indent=2) + "\n",
        encoding="utf-8",
    )
    (qualitative_dir / "heatmap_manifest.json").write_text(
        json.dumps(
            build_distance_heatmap_manifest(
                results,
                topk=int(config["analysis"]["qualitative"]["topk_failures"]),
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    run_card = build_run_card(config, snapshot, aggregate_summary["overall"])
    run_card_path = output_dir / "run_card.json"
    run_card_path.write_text(json.dumps(run_card, indent=2) + "\n", encoding="utf-8")

    report_path: str | None = None
    if config["runtime"]["export_html"] and config["analysis"]["export"]["html"]:
        report_path = build_html_report(
            output_dir=output_dir,
            aggregate_summary=aggregate_summary,
            table_paths=table_paths,
            figure_paths=figure_paths,
            run_card=run_card,
        )

    return {
        "output_dir": str(output_dir),
        "results_path": str(results_path),
        "run_card_path": str(run_card_path),
        "report_path": report_path,
        "sample_count": len(results),
        "overall": aggregate_summary["overall"],
    }
