"""Curve metrics for benchmark reporting."""

from __future__ import annotations

from collections import defaultdict


def compute_multithreshold_recall(
    records: list[dict[str, object]],
) -> list[dict[str, float]]:
    thresholds = (
        (1.0, 1.0, "success_1deg_1mm"),
        (3.0, 3.0, "success_3deg_3mm"),
        (5.0, 5.0, "success_5deg_5mm"),
        (10.0, 10.0, "success_10deg_10mm"),
    )
    count = max(len(records), 1)
    return [
        {
            "rotation_deg": rotation_deg,
            "translation_mm": translation_mm,
            "recall": sum(int(record.get(field, 0)) for record in records) / count,
        }
        for rotation_deg, translation_mm, field in thresholds
    ]


def build_success_latency_pareto(
    records: list[dict[str, object]],
) -> list[dict[str, float]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        label = (
            f"{record.get('model_id', 'unknown')}|"
            f"{record.get('preprocess_profile_id', 'unknown')}|"
            f"{record.get('refinement_track', 'unknown')}"
        )
        grouped[label].append(record)

    pareto_rows: list[dict[str, float]] = []
    for label, items in sorted(grouped.items()):
        finite_latencies = [
            float(item["latency_ms"])
            for item in items
            if item.get("latency_ms") is not None
        ]
        pareto_rows.append(
            {
                "label": label,
                "success_rate": sum(
                    int(item.get("success_5deg_5mm", 0)) for item in items
                )
                / max(len(items), 1),
                "latency_ms": sum(finite_latencies) / max(len(finite_latencies), 1),
            }
        )
    return pareto_rows
