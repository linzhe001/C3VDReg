"""CSV and Markdown exporters for benchmark tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.benchmarking.reporting.aggregate_results import export_markdown_tables


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _build_overview_payload(aggregate_summary: dict[str, Any]) -> dict[str, Any]:
    overall = aggregate_summary["overall"]
    return {
        "recommended_reading_order": [
            {
                "path": "summary_overview.md",
                "purpose": "Start here for the key conclusions of this run.",
            },
            {
                "path": "leaderboard/leaderboard_main.csv",
                "purpose": (
                    "Main result table. Prioritize RR@5deg/5mm, marginal hit "
                    "rates, RRE, RTE, trimmed Chamfer, training time, and peak "
                    "training memory."
                ),
            },
            {
                "path": "leaderboard/efficiency_summary.csv",
                "purpose": (
                    "Speed and resource table. Prioritize training time, peak "
                    "training memory, preprocess time, inference time, and "
                    "latency."
                ),
            },
            {
                "path": "geometry/geometry_summary.csv",
                "purpose": (
                    "Geometry error summary. Prioritize visible_nn_mean and "
                    "Chamfer."
                ),
            },
            {
                "path": "report.html",
                "purpose": "HTML overview of plots and linked artifacts.",
            },
        ],
        "headline_metrics": {
            "model_id": overall.get("model_id"),
            "preprocess_profile_id": overall.get("preprocess_profile_id"),
            "sample_count": overall.get("sample_count"),
            "registration_recall@rre_5deg_rte_5mm": overall.get(
                "registration_recall@rre_5deg_rte_5mm"
            ),
            "rot_hit_5deg_rate": overall.get("rot_hit_5deg_rate"),
            "trans_hit_5mm_rate": overall.get("trans_hit_5mm_rate"),
            "rre_deg_mean": overall.get("rre_deg_mean"),
            "rre_deg_median": overall.get("rre_deg_median"),
            "rre_deg_p90": overall.get("rre_deg_p90"),
            "rte_mm_mean": overall.get("rte_mm_mean"),
            "rte_mm_median": overall.get("rte_mm_median"),
            "rte_mm_p90": overall.get("rte_mm_p90"),
            "visible_nn_mean_mm_mean": overall.get("visible_nn_mean_mm_mean"),
            "trimmed_chamfer_mm_mean": overall.get("trimmed_chamfer_mm_mean"),
            "train_time_ms": overall.get("train_time_ms"),
            "train_peak_memory_mb": overall.get("train_peak_memory_mb"),
        },
        "efficiency_metrics": {
            "train_time_ms": overall.get("train_time_ms"),
            "train_peak_memory_mb": overall.get("train_peak_memory_mb"),
            "train_peak_allocated_mb": overall.get("train_peak_allocated_mb"),
            "preprocess_time_ms_mean": overall.get("preprocess_time_ms_mean"),
            "inference_time_ms_mean": overall.get("inference_time_ms_mean"),
            "refinement_time_ms_mean": overall.get("refinement_time_ms_mean"),
            "latency_ms_mean": overall.get("latency_ms_mean"),
            "latency_ms_p90": overall.get("latency_ms_p90"),
            "peak_memory_mb_mean": overall.get("peak_memory_mb_mean"),
        },
        "geometry_metrics": {
            "visible_nn_mean_mm_mean": overall.get("visible_nn_mean_mm_mean"),
            "trimmed_chamfer_mm_mean": overall.get("trimmed_chamfer_mm_mean"),
        },
    }


def _write_overview_files(
    output_dir: Path,
    aggregate_summary: dict[str, Any],
) -> dict[str, str]:
    overview = _build_overview_payload(aggregate_summary)
    overview_json = output_dir / "summary_overview.json"
    overview_md = output_dir / "summary_overview.md"

    overview_json.write_text(
        json.dumps(overview, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Quick Result Overview",
        "",
        "## Recommended Reading Order",
    ]
    for index, item in enumerate(overview["recommended_reading_order"], start=1):
        lines.append(
            f"{index}. `{item['path']}`: {item['purpose']}"
        )

    lines.extend(
        [
            "",
            "## Headline Metrics",
        ]
    )
    for key, value in overview["headline_metrics"].items():
        lines.append(f"- {key}: {_format_value(value)}")

    lines.extend(
        [
            "",
            "## Speed And Resources",
        ]
    )
    for key, value in overview["efficiency_metrics"].items():
        lines.append(f"- {key}: {_format_value(value)}")

    lines.extend(
        [
            "",
            "## Geometry Summary",
        ]
    )
    for key, value in overview["geometry_metrics"].items():
        lines.append(f"- {key}: {_format_value(value)}")

    overview_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "summary_overview_json": str(overview_json),
        "summary_overview_md": str(overview_md),
    }


def export_leaderboard_tables(
    aggregate_summary: dict[str, Any],
    output_dir: str | Path,
    markdown_tables: bool = True,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    leaderboard_dir = output_dir / "leaderboard"
    geometry_dir = output_dir / "geometry"

    main_csv = leaderboard_dir / "leaderboard_main.csv"
    multithreshold_csv = leaderboard_dir / "leaderboard_multithreshold.csv"
    efficiency_csv = leaderboard_dir / "efficiency_summary.csv"
    geometry_csv = geometry_dir / "geometry_summary.csv"

    _write_csv(main_csv, aggregate_summary["leaderboard_rows"])
    _write_csv(multithreshold_csv, aggregate_summary["multithreshold_rows"])
    _write_csv(efficiency_csv, aggregate_summary["efficiency_rows"])
    _write_csv(geometry_csv, aggregate_summary["geometry_rows"])

    outputs = {
        "leaderboard_main": str(main_csv),
        "leaderboard_multithreshold": str(multithreshold_csv),
        "efficiency_summary": str(efficiency_csv),
        "geometry_summary": str(geometry_csv),
    }
    outputs.update(_write_overview_files(output_dir, aggregate_summary))
    if markdown_tables:
        outputs.update(export_markdown_tables(aggregate_summary, leaderboard_dir))
    return outputs


def export_bucket_tables(
    aggregate_summary: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    bucket_dir = output_dir / "buckets"
    outputs: dict[str, str] = {}
    for bucket_key, rows in aggregate_summary["bucket_summaries"].items():
        normalized = bucket_key.replace("_bin", "").replace("_id", "")
        path = bucket_dir / f"bucket_{normalized}.csv"
        _write_csv(path, rows)
        outputs[bucket_key] = str(path)
    return outputs
