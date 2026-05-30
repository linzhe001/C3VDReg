#!/usr/bin/env python3
"""Rerender hard-protocol worst-distance qualitative cases without rerunning eval."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark.render_bmvc_qualitative_panels import (  # noqa: E402
    _instantiate_adapter,
    _load_json,
    _load_sample,
    _prediction_to_transform,
)
from src.benchmarking.reporting.export_figures import (  # noqa: E402
    export_pointcloud_alignment_transition_animation,
    export_pointcloud_distance_render,
)

PROTOCOL_ROOT = REPO_ROOT / "outputs" / "benchmark" / "r25_90_t100_500mm_protocol"


def _metadata_paths() -> list[Path]:
    return sorted(
        PROTOCOL_ROOT.glob(
            "*/eval_test/qualitative/distance_render/*_worst_visible_distance.json"
        )
    )


def _eval_dir_from_metadata(path: Path) -> Path:
    return path.parents[2]


def _build_render_case(
    metadata_path: Path,
    adapter_cache: dict[Path, Any],
) -> tuple[Path, dict[str, object]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    eval_dir = _eval_dir_from_metadata(metadata_path)
    config = _load_json(eval_dir / "normalized_eval_config.json")
    sample = _load_sample(config, str(metadata["sample_id"]))
    if eval_dir not in adapter_cache:
        adapter_cache[eval_dir] = _instantiate_adapter(config)
    adapter = adapter_cache[eval_dir]

    source = np.asarray(sample["source_points"], dtype=np.float64)
    target = np.asarray(sample["target_points"], dtype=np.float64)
    pred_transform = _prediction_to_transform(adapter.predict(source, target))
    return eval_dir, {
        "sample_id": metadata["sample_id"],
        "scene_id": metadata["scene_id"],
        "visible_nn_mean_mm": metadata["visible_nn_mean_mm"],
        "rre_deg": metadata["rre_deg"],
        "rte_mm": metadata["rte_mm"],
        "point_unit": config["benchmark"].get("point_unit", "m"),
        "source_points": source,
        "target_points": target,
        "pred_transform": pred_transform,
    }


def main() -> int:
    adapter_cache: dict[Path, Any] = {}
    outputs: list[dict[str, str]] = []
    for metadata_path in _metadata_paths():
        eval_dir, render_case = _build_render_case(metadata_path, adapter_cache)
        distance_outputs = export_pointcloud_distance_render(render_case, eval_dir)
        animation_outputs = export_pointcloud_alignment_transition_animation(
            render_case,
            eval_dir,
        )
        outputs.append(
            {
                "eval_dir": str(eval_dir.relative_to(REPO_ROOT)),
                **distance_outputs,
                **animation_outputs,
            }
        )
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
