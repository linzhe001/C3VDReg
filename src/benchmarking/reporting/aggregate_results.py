"""Aggregation helpers built on top of ResultRecord JSONL rows."""

from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any

from src.benchmarking.analysis.bucketizer import build_bucket_views
from src.benchmarking.analysis.curve_metrics import (
    build_success_latency_pareto,
    compute_multithreshold_recall,
)
from src.benchmarking.analysis.metric_catalog import get_official_metric_catalog
from src.benchmarking.reporting.result_schema import (
    serialize_result_record,
    validate_result_record,
)


def _safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _safe_median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _safe_p90(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * 0.9))
    return ordered[index]


def _threshold_rate(values: list[float], threshold: float) -> float | None:
    if not values:
        return None
    return sum(value <= threshold for value in values) / len(values)


def _single_value(records: list[dict[str, Any]], key: str) -> str:
    values = {str(record.get(key)) for record in records}
    return values.pop() if len(values) == 1 else "mixed"


def _safe_single_or_mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    if not values:
        return None
    unique_values = {round(value, 9) for value in values}
    return values[0] if len(unique_values) == 1 else _safe_mean(values)


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    catalog = get_official_metric_catalog()
    summary: dict[str, Any] = {
        "model_id": _single_value(records, "model_id"),
        "preprocess_profile_id": _single_value(records, "preprocess_profile_id"),
        "refinement_track": _single_value(records, "refinement_track"),
        "sample_count": len(records),
        "manifest_digest": _single_value(records, "manifest_digest"),
        "config_digest": _single_value(records, "config_digest"),
        "git_snapshot": _single_value(records, "git_snapshot"),
    }
    for spec in catalog:
        numeric_values = [
            float(record[spec.name])
            for record in records
            if record.get(spec.name) is not None
        ]
        if "success_rate" in spec.aggregate_modes:
            summary[f"{spec.name}_rate"] = _safe_mean(numeric_values)
        if "mean" in spec.aggregate_modes:
            summary[f"{spec.name}_mean"] = _safe_mean(numeric_values)
        if "median" in spec.aggregate_modes:
            summary[f"{spec.name}_median"] = _safe_median(numeric_values)
        if "p90" in spec.aggregate_modes:
            summary[f"{spec.name}_p90"] = _safe_p90(numeric_values)

    summary["registration_recall@rre_1deg_rte_1mm"] = summary.get(
        "success_1deg_1mm_rate"
    )
    summary["registration_recall@rre_3deg_rte_3mm"] = summary.get(
        "success_3deg_3mm_rate"
    )
    summary["registration_recall@rre_5deg_rte_5mm"] = summary.get(
        "success_5deg_5mm_rate"
    )
    summary["registration_recall@rre_10deg_rte_10mm"] = summary.get(
        "success_10deg_10mm_rate"
    )
    rre_values = [
        float(record["rre_deg"])
        for record in records
        if record.get("rre_deg") is not None
    ]
    rte_values = [
        float(record["rte_mm"])
        for record in records
        if record.get("rte_mm") is not None
    ]
    summary["rot_hit_5deg_rate"] = _threshold_rate(rre_values, 5.0)
    summary["trans_hit_5mm_rate"] = _threshold_rate(rte_values, 5.0)
    summary["rot_hit_10deg_rate"] = _threshold_rate(rre_values, 10.0)
    summary["trans_hit_10mm_rate"] = _threshold_rate(rte_values, 10.0)
    summary["train_time_ms"] = _safe_single_or_mean(records, "train_time_ms")
    summary["train_peak_memory_mb"] = _safe_single_or_mean(
        records,
        "train_peak_memory_mb",
    )
    summary["train_peak_allocated_mb"] = _safe_single_or_mean(
        records,
        "train_peak_allocated_mb",
    )
    return summary


def export_markdown_tables(
    aggregate_summary: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = output_dir / "leaderboard_main.md"

    rows = aggregate_summary["leaderboard_rows"]
    if not rows:
        leaderboard_path.write_text("# Leaderboard\n\n_No records_\n", encoding="utf-8")
        return {"leaderboard_main": str(leaderboard_path)}

    columns = [
        "model_id",
        "preprocess_profile_id",
        "registration_recall@rre_5deg_rte_5mm",
        "rot_hit_5deg_rate",
        "trans_hit_5mm_rate",
        "rre_deg_mean",
        "rre_deg_median",
        "rre_deg_p90",
        "rte_mm_mean",
        "rte_mm_median",
        "rte_mm_p90",
        "visible_nn_mean_mm_mean",
        "trimmed_chamfer_mm_mean",
        "train_time_ms",
        "train_peak_memory_mb",
        "latency_ms_mean",
        "sample_count",
    ]
    header = "| " + " | ".join(columns) + " |\n"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |\n"
    body = "".join(
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |\n"
        for row in rows
    )
    leaderboard_path.write_text(
        "# Leaderboard\n\n" + header + separator + body, encoding="utf-8"
    )
    return {"leaderboard_main": str(leaderboard_path)}


def aggregate_results(
    records: list[dict[str, object]],
    bucket_keys: list[str] | None = None,
) -> dict[str, Any]:
    validated = [
        serialize_result_record(validate_result_record(record)) for record in records
    ]
    bucket_keys = bucket_keys or [
        "overlap_bin",
        "rotation_bin",
        "translation_bin",
        "scene_id",
        "preprocess_profile_id",
        "refinement_track",
    ]

    overall = _summarize_records(validated)
    bucket_views = build_bucket_views(validated, bucket_keys)

    bucket_summaries: dict[str, list[dict[str, Any]]] = {}
    for key, views in bucket_views.items():
        rows: list[dict[str, Any]] = []
        for bucket in views:
            bucket_records = bucket["records"]
            summary = _summarize_records(bucket_records)
            rows.append(
                {
                    "bucket_key": key,
                    "bucket_value": bucket["bucket_value"],
                    "count": bucket["count"],
                    "registration_recall@rre_5deg_rte_5mm": summary.get(
                        "registration_recall@rre_5deg_rte_5mm"
                    ),
                    "rot_hit_5deg_rate": summary.get("rot_hit_5deg_rate"),
                    "trans_hit_5mm_rate": summary.get("trans_hit_5mm_rate"),
                    "rre_deg_mean": summary.get("rre_deg_mean"),
                    "rte_mm_mean": summary.get("rte_mm_mean"),
                    "latency_ms_mean": summary.get("latency_ms_mean"),
                    "visible_nn_mean_mm_mean": summary.get("visible_nn_mean_mm_mean"),
                    "trimmed_chamfer_mm_mean": summary.get(
                        "trimmed_chamfer_mm_mean"
                    ),
                }
            )
        bucket_summaries[key] = rows

    grouped_rows = {
        "scene": _group_by(validated, "scene_id"),
        "profile": _group_by(validated, "preprocess_profile_id"),
        "track": _group_by(validated, "refinement_track"),
    }

    return {
        "records": validated,
        "overall": overall,
        "leaderboard_rows": [overall],
        "multithreshold_rows": compute_multithreshold_recall(validated),
        "pareto_rows": build_success_latency_pareto(validated),
        "efficiency_rows": [
            {
                "model_id": overall.get("model_id"),
                "preprocess_profile_id": overall.get("preprocess_profile_id"),
                "train_time_ms": overall.get("train_time_ms"),
                "train_peak_memory_mb": overall.get("train_peak_memory_mb"),
                "train_peak_allocated_mb": overall.get("train_peak_allocated_mb"),
                "preprocess_time_ms_mean": overall.get("preprocess_time_ms_mean"),
                "latency_ms_mean": overall.get("latency_ms_mean"),
                "latency_ms_p90": overall.get("latency_ms_p90"),
                "inference_time_ms_mean": overall.get("inference_time_ms_mean"),
                "refinement_time_ms_mean": overall.get("refinement_time_ms_mean"),
                "peak_memory_mb_mean": overall.get("peak_memory_mb_mean"),
                "sample_count": overall.get("sample_count"),
            }
        ],
        "geometry_rows": [
            {
                "model_id": overall.get("model_id"),
                "preprocess_profile_id": overall.get("preprocess_profile_id"),
                "visible_nn_mean_mm_mean": overall.get("visible_nn_mean_mm_mean"),
                "visible_nn_p90_mm_p90": overall.get("visible_nn_p90_mm_p90"),
                "trimmed_chamfer_mm_mean": overall.get("trimmed_chamfer_mm_mean"),
            }
        ],
        "bucket_summaries": bucket_summaries,
        "grouped_rows": grouped_rows,
    }


def _group_by(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        label = str(record.get(key, "missing"))
        grouped.setdefault(label, []).append(record)

    rows: list[dict[str, Any]] = []
    for label, items in sorted(grouped.items()):
        summary = _summarize_records(items)
        summary[key] = label
        rows.append(summary)
    return rows
