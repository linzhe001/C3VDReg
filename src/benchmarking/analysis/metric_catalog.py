"""Official analysis metric catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Goal = Literal["min", "max"]


@dataclass(frozen=True)
class AnalysisMetricSpec:
    name: str
    group: str
    goal: Goal
    required_for_main_table: bool
    aggregate_modes: tuple[str, ...]


def get_official_metric_catalog() -> list[AnalysisMetricSpec]:
    return [
        AnalysisMetricSpec("rre_deg", "pose", "min", True, ("mean", "median", "p90")),
        AnalysisMetricSpec("rte_mm", "pose", "min", True, ("mean", "median", "p90")),
        AnalysisMetricSpec("rmse_mm", "pose", "min", True, ("mean", "median")),
        AnalysisMetricSpec(
            "success_1deg_1mm", "robustness", "max", False, ("success_rate",)
        ),
        AnalysisMetricSpec(
            "success_3deg_3mm", "robustness", "max", False, ("success_rate",)
        ),
        AnalysisMetricSpec(
            "success_5deg_5mm", "robustness", "max", True, ("success_rate",)
        ),
        AnalysisMetricSpec(
            "success_10deg_10mm", "robustness", "max", False, ("success_rate",)
        ),
        AnalysisMetricSpec(
            "visible_nn_mean_mm", "geometry", "min", False, ("mean", "median", "p90")
        ),
        AnalysisMetricSpec(
            "trimmed_chamfer_mm", "geometry", "min", False, ("mean", "median")
        ),
        AnalysisMetricSpec(
            "preprocess_time_ms", "efficiency", "min", False, ("mean", "median", "p90")
        ),
        AnalysisMetricSpec(
            "inference_time_ms", "efficiency", "min", False, ("mean", "median", "p90")
        ),
        AnalysisMetricSpec(
            "refinement_time_ms", "efficiency", "min", False, ("mean", "median", "p90")
        ),
        AnalysisMetricSpec(
            "latency_ms", "efficiency", "min", True, ("mean", "median", "p90")
        ),
        AnalysisMetricSpec(
            "peak_memory_mb", "efficiency", "min", False, ("mean", "p90")
        ),
    ]
