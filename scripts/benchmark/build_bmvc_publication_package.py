#!/usr/bin/env python3
"""Build BMVC publication protocol, analysis, and figure artifacts.

The script is intentionally read-only with respect to benchmark results: it joins
existing R90/T500mm result bundles and exports publication-facing summaries.
"""

# ruff: noqa: E501,I001

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmarking.datasets.c3vd_manifest_dataset import C3VDManifestDataset  # noqa: E402
from src.benchmarking.preprocess.pipeline import PreprocessPipeline  # noqa: E402
from src.benchmarking.preprocess.registry import PreprocessRegistry  # noqa: E402


PROTOCOL_ROOT = REPO_ROOT / "outputs" / "benchmark" / "r90_t500mm_protocol"
ERROR_ROOT = PROTOCOL_ROOT / "error_analysis"
FIGURE_ROOT = REPO_ROOT / "outputs" / "benchmark" / "figures" / "paper"
FAILURE_FIGURE_ROOT = REPO_ROOT / "outputs" / "benchmark" / "figures" / "failure_cases"
PACKAGE_ROOT = REPO_ROOT / "outputs" / "benchmark" / "bmvc_publication_package"
PROTOCOL_DOC = REPO_ROOT / "docs" / "C3VD_Raycasting_PCR_Benchmark_Protocol.md"
DATASET_ROOT_PLACEHOLDER = "${C3VD_RAYCASTING_ROOT}"
GEOMETRY_DIFFICULTY_PATH = ERROR_ROOT / "independent_geometry_difficulty.csv"


@dataclass(frozen=True)
class RunSpec:
    model_key: str
    display_name: str
    result_dir: Path
    retained: bool = True


RUN_SPECS = (
    RunSpec(
        "geotransformer",
        "GeoTransformer",
        PROTOCOL_ROOT / "geotransformer" / "eval_test",
    ),
    RunSpec(
        "regtr",
        "RegTR",
        REPO_ROOT
        / "outputs"
        / "benchmark"
        / "r90_t500mm_from_scratch_fixed_regtr_dcp"
        / "regtr"
        / "eval_test_latency_rerun2",
    ),
    RunSpec(
        "mamba2_direct",
        "Mamba2 direct",
        REPO_ROOT
        / "outputs"
        / "benchmark"
        / "mamba2_followup_point_order_pair_initializer"
        / "direct_sort_xyz_e5"
        / "eval_test_maxiter10",
    ),
    RunSpec(
        "pointnetlk_revisited",
        "PointNetLK Revisited",
        PROTOCOL_ROOT / "pointnetlk_revisited" / "eval_test",
    ),
    RunSpec(
        "pointnetlk",
        "PointNetLK",
        PROTOCOL_ROOT / "pointnetlk" / "eval_test",
    ),
    RunSpec(
        "dcp",
        "DCP",
        REPO_ROOT
        / "outputs"
        / "benchmark"
        / "r90_t500mm_from_scratch_fixed_regtr_dcp"
        / "dcp"
        / "eval_test",
    ),
    RunSpec("icp", "ICP", PROTOCOL_ROOT / "icp" / "eval_test"),
)


MODEL_COLORS = {
    "geotransformer": "#2f6fbb",
    "regtr": "#bf6f2a",
    "mamba2_direct": "#3f8c54",
    "pointnetlk_revisited": "#8e5fb8",
    "pointnetlk": "#5f6c7b",
    "dcp": "#b44c54",
    "icp": "#111827",
}


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_csv_single_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else {}


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _safe_float(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return float(mean(clean)) if clean else math.nan


def _safe_median(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return float(median(clean)) if clean else math.nan


def _safe_p90(values: list[float]) -> float:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return math.nan
    index = int(round((len(clean) - 1) * 0.9))
    return float(clean[index])


def _apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (transform[:3, :3] @ points.T).T + transform[:3, 3]


def _sample_evenly(points: np.ndarray, sample_count: int) -> np.ndarray:
    if len(points) <= sample_count:
        return points.astype(np.float64, copy=False)
    indices = np.linspace(0, len(points) - 1, num=sample_count, dtype=int)
    return points[indices].astype(np.float64, copy=False)


def _distance_to_mm(distance: float, point_unit: str) -> float:
    if point_unit == "m":
        return distance * 1000.0
    return distance


def _format_float(value: Any, digits: int = 3) -> str:
    numeric = _safe_float(value)
    if not math.isfinite(numeric):
        return ""
    return f"{numeric:.{digits}f}"


def _format_pct(value: Any) -> str:
    numeric = _safe_float(value)
    if not math.isfinite(numeric):
        return ""
    return f"{numeric * 100.0:.2f}"


def _relative_to_dataset(path_value: str, dataset_root: Path | None) -> str:
    path = Path(path_value)
    if dataset_root is not None and path.is_absolute():
        try:
            return str(path.relative_to(dataset_root))
        except ValueError:
            return str(path)
    return str(path)


def _identity_transform() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _load_available_runs() -> list[RunSpec]:
    runs: list[RunSpec] = []
    for spec in RUN_SPECS:
        if (spec.result_dir / "results.jsonl").exists():
            runs.append(spec)
    return runs


def _load_run_records(runs: list[RunSpec]) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {}
    for spec in runs:
        rows = _read_jsonl(spec.result_dir / "results.jsonl")
        for row in rows:
            row["_model_key"] = spec.model_key
            row["_display_name"] = spec.display_name
            row["_result_dir"] = _rel(spec.result_dir)
        records[spec.model_key] = rows
    return records


def _first_config(runs: list[RunSpec]) -> dict[str, Any]:
    for spec in runs:
        path = spec.result_dir / "normalized_eval_config.json"
        if path.exists():
            return _read_json(path)
    raise FileNotFoundError("No normalized_eval_config.json found in retained runs.")


def build_pair_manifest(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(config["data"]["manifest_path"])
    dataset_root_value = config["data"].get("dataset_root")
    dataset_root = Path(dataset_root_value) if dataset_root_value else None
    rows = _read_jsonl(manifest_path)

    full_manifest_path = PROTOCOL_ROOT / "pair_manifest.jsonl"
    test_manifest_path = PROTOCOL_ROOT / "pair_manifest_test.jsonl"
    split_indices: dict[str, int] = defaultdict(int)
    split_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    test_scene_counts: Counter[str] = Counter()
    perturbation = dict(config.get("perturbation", {}))
    base_seed = int(config["preprocess"]["seed"])
    input_points = int(config["preprocess"].get("num_points_override") or 8192)
    exported_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    for raw in rows:
        split = str(raw["split"])
        split_index = split_indices[split]
        split_indices[split] += 1
        scene_id = str(raw.get("scene_id") or raw.get("scene"))
        frame_id = int(raw.get("frame_id", raw.get("frame_idx", 0)))
        gt_transform = raw.get("gt_transform") or _identity_transform()
        output = {
            "pair_id": str(raw["sample_id"]),
            "split": split,
            "split_index": split_index,
            "scene_id": scene_id,
            "trajectory_id": str(raw.get("trajectory_id") or scene_id),
            "frame_id": frame_id,
            "source_path": _relative_to_dataset(str(raw["source_path"]), dataset_root),
            "target_path": _relative_to_dataset(str(raw["target_path"]), dataset_root),
            "path_root": DATASET_ROOT_PLACEHOLDER,
            "gt_transform_source_to_target": gt_transform,
            "gt_transform_note": (
                "Base raycasting pair transform before benchmark source-only "
                "perturbation; one_to_one rows use identity."
            ),
            "point_unit": config["benchmark"].get("point_unit", "mm_like"),
            "input_points": input_points,
            "preprocess_profile_id": config["preprocess"]["profile"],
            "preprocess_seed": base_seed + split_index,
            "source_perturbation_seed": base_seed + split_index + 2,
            "perturbation": {
                "enabled": bool(perturbation.get("enabled", False)),
                "rotation_deg": float(perturbation.get("rotation_deg", 0.0)),
                "translation_m": float(perturbation.get("translation_m", 0.0)),
                "noise_sigma": float(perturbation.get("noise_sigma", 0.0)),
                "noise_clip": float(perturbation.get("noise_clip", 0.0)),
                "apply_noise_to": str(perturbation.get("apply_noise_to", "source")),
            },
            "metadata": {
                "schema_version": raw.get("schema_version"),
                "region": raw.get("region"),
                "pair_mode": raw.get("pair_mode", "one_to_one"),
                "reference_path": _relative_to_dataset(
                    str(raw.get("reference_path", "")), dataset_root
                )
                if raw.get("reference_path")
                else None,
                "source_manifest_name": manifest_path.name,
            },
        }
        exported_rows.append(output)
        split_counts[split] += 1
        scene_counts[scene_id] += 1
        if split == "test":
            test_rows.append(output)
            test_scene_counts[scene_id] += 1

    for path, payload in (
        (full_manifest_path, exported_rows),
        (test_manifest_path, test_rows),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in payload:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "source_manifest_path": str(manifest_path),
        "dataset_root_placeholder": DATASET_ROOT_PLACEHOLDER,
        "full_manifest_path": _rel(full_manifest_path),
        "test_manifest_path": _rel(test_manifest_path),
        "split_counts": dict(sorted(split_counts.items())),
        "test_scene_counts": dict(sorted(test_scene_counts.items())),
        "input_points": input_points,
        "preprocess_profile_id": config["preprocess"]["profile"],
        "perturbation": {
            "rotation_deg": float(perturbation.get("rotation_deg", 0.0)),
            "translation_m": float(perturbation.get("translation_m", 0.0)),
            "noise_sigma": float(perturbation.get("noise_sigma", 0.0)),
            "apply_noise_to": str(perturbation.get("apply_noise_to", "source")),
        },
    }
    write_pair_manifest_summary(summary)
    write_protocol_doc(summary)
    return summary


def write_pair_manifest_summary(summary: dict[str, Any]) -> None:
    lines = [
        "# C3VD Raycasting R90/T500mm Pair Manifest Summary",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Full manifest | `{summary['full_manifest_path']}` |",
        f"| Test manifest | `{summary['test_manifest_path']}` |",
        f"| Input points | `{summary['input_points']}` source and target points |",
        f"| Preprocess profile | `{summary['preprocess_profile_id']}` |",
        "| Perturbation | "
        f"source-only R{summary['perturbation']['rotation_deg']:.0f}/"
        f"T{summary['perturbation']['translation_m']:.0f}mm, "
        f"noise_sigma={summary['perturbation']['noise_sigma']:.1f} |",
        f"| Dataset root placeholder | `{summary['dataset_root_placeholder']}` |",
        "",
        "## Split Counts",
        "",
        "| Split | Pairs |",
        "| --- | ---: |",
    ]
    for split, count in summary["split_counts"].items():
        lines.append(f"| `{split}` | {count} |")
    lines.extend(
        [
            "",
            "## Test Scene Counts",
            "",
            "| Scene | Test pairs |",
            "| --- | ---: |",
        ]
    )
    for scene, count in summary["test_scene_counts"].items():
        lines.append(f"| `{scene}` | {count} |")
    lines.extend(
        [
            "",
            "说明：manifest 中的 `gt_transform_source_to_target` 是 raycasting pair "
            "在 source-only perturbation 之前的 source-to-target transform。当前 "
            "`one_to_one` pair 使用 identity；R90/T500mm 的 perturbed GT 由 "
            "`preprocess_seed` 和 `source_perturbation_seed` 确定性生成。",
            "",
        ]
    )
    path = PROTOCOL_ROOT / "pair_manifest_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")


def write_protocol_doc(summary: dict[str, Any]) -> None:
    text = f"""# C3VD Raycasting PCR Benchmark Protocol

本文档固化 BMVC C3VD raycasting point-cloud registration benchmark 的当前主协议。它服务于论文、supplement 和可复现包，不替代底层训练/评测代码。

## 任务边界

- 数据来源：MambaNetLK_IPCAI workflow 中生成的 C3VD raycasting point clouds。
- 任务：endoscopic rigid pairwise partial-to-partial point-cloud registration。
- 不覆盖：C3VDv2 sequence、deformation tracking、clean/debris robustness、screening sequence、online hidden leaderboard。

## Pair 构造

- `source_path` 指向由 endoscopic depth map reprojection 得到的 source point cloud。
- `target_path` 指向同一虚拟相机位姿下从 CT mesh raycasting 得到的 visible target point cloud。
- 当前主 pair mode 是 `one_to_one`，base `gt_transform_source_to_target` 为 identity；评测时对 source 施加 deterministic source-only perturbation，并同步更新 source-to-target GT。
- 坐标单位按 benchmark config 记录为 `mm_like`；RRE 用 degree，RTE 和 visible nearest-neighbour distance 用 mm 报告。

## Frozen Main Protocol

| Field | Value |
| --- | --- |
| Split | `test` |
| Test pairs | `{summary['split_counts'].get('test', 0)}` |
| Input points | `{summary['input_points']}` source + `{summary['input_points']}` target |
| Preprocess profile | `{summary['preprocess_profile_id']}` |
| Perturbation | source-only `R{summary['perturbation']['rotation_deg']:.0f}/T{summary['perturbation']['translation_m']:.0f}mm` |
| Noise | `noise_sigma={summary['perturbation']['noise_sigma']:.1f}` |
| Primary recall | `RR@5deg/5mm`, `RR@10deg/10mm` |
| Resource metrics | inference ms, total latency ms, p90 latency, standardized eval memory |

## Manifest Artifacts

- Full split manifest: `{summary['full_manifest_path']}`
- Test-only manifest: `{summary['test_manifest_path']}`
- Manifest summary: `outputs/benchmark/r90_t500mm_protocol/pair_manifest_summary.md`

Manifest paths are stored relative to `{summary['dataset_root_placeholder']}` so the package is not tied to a local absolute dataset root.

## Metrics

Let `T_gt` be the source-to-target ground-truth transform after source-only perturbation, and `T_pred` be the model-predicted source-to-target transform. The relative transform is:

```text
T_err = inv(T_gt) @ T_pred
```

RRE is the geodesic angle of the rotational part of `T_err`:

```text
RRE = arccos((trace(R_err) - 1) / 2)
```

RTE is the Euclidean norm of the translation part of `T_err`, converted to mm:

```text
RTE = ||t_err||_2
```

`RR@5deg/5mm` counts a pair as successful when `RRE <= 5deg` and `RTE <= 5mm`; `RR@10deg/10mm` uses `10deg` and `10mm`.

Visible NN is a diagnostic distance from predicted-aligned source points to target points, restricted by the current visible/overlap-aware evaluation mask. Because overlap labels are currently missing in the manifest, difficulty buckets use a prediction-independent GT-aligned Visible NN proxy and should be described as proxy difficulty, not true overlap.

## Evaluation Command Template

```bash
python scripts/runners/eval_benchmark.py \\
  --config outputs/benchmark/r90_t500mm_protocol/configs/eval_<model>.yaml
```

The retained leaderboard rows all use `split=test`, `subset_name=null`, `num_points_override=8192`, `rotation_deg=90`, `translation_m=500`, `noise_sigma=0`, and `apply_noise_to=source`.

## License And Ethics Notes

This package describes a derived raycasting benchmark built from the C3VD/MambaNetLK_IPCAI preprocessing workflow. The upstream C3VD project page states that C3VD is licensed under CC BY-NC-SA 4.0 and describes the data as acquired from high-fidelity colon phantom models. Treat derived raycasting point clouds as license-constrained artifacts: release them only under compatible terms after institutional review, or release the manifest, generation/preprocessing scripts, metric implementation, and commands instead. No new human-subject data collection is introduced by this protocol.
"""
    PROTOCOL_DOC.write_text(text, encoding="utf-8")


def build_result_source_audit(runs: list[RunSpec]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in runs:
        config_path = spec.result_dir / "normalized_eval_config.json"
        leaderboard_path = spec.result_dir / "leaderboard" / "leaderboard_main.csv"
        config = _read_json(config_path) if config_path.exists() else {}
        leaderboard = (
            _read_csv_single_row(leaderboard_path) if leaderboard_path.exists() else {}
        )
        rows.append(
            {
                "model_key": spec.model_key,
                "display_name": spec.display_name,
                "result_dir": _rel(spec.result_dir),
                "sample_count": leaderboard.get("sample_count"),
                "split": config.get("benchmark", {}).get("split"),
                "subset_name": config.get("benchmark", {}).get("subset_name"),
                "preprocess_profile": config.get("preprocess", {}).get("profile"),
                "num_points_override": config.get("preprocess", {}).get(
                    "num_points_override"
                ),
                "rotation_deg": config.get("perturbation", {}).get("rotation_deg"),
                "translation_m": config.get("perturbation", {}).get("translation_m"),
                "noise_sigma": config.get("perturbation", {}).get("noise_sigma"),
                "apply_noise_to": config.get("perturbation", {}).get(
                    "apply_noise_to"
                ),
                "manifest_digest": leaderboard.get("manifest_digest"),
                "config_digest": leaderboard.get("config_digest"),
                "rr5": leaderboard.get("registration_recall@rre_5deg_rte_5mm"),
                "rr10": leaderboard.get("registration_recall@rre_10deg_rte_10mm"),
                "rre_deg_mean": leaderboard.get("rre_deg_mean"),
                "rte_mm_mean": leaderboard.get("rte_mm_mean"),
                "latency_ms_mean": leaderboard.get("latency_ms_mean"),
                "retained": spec.retained,
            }
        )
    fields = [
        "model_key",
        "display_name",
        "result_dir",
        "sample_count",
        "split",
        "subset_name",
        "preprocess_profile",
        "num_points_override",
        "rotation_deg",
        "translation_m",
        "noise_sigma",
        "apply_noise_to",
        "manifest_digest",
        "config_digest",
        "rr5",
        "rr10",
        "rre_deg_mean",
        "rte_mm_mean",
        "latency_ms_mean",
        "retained",
    ]
    _write_csv(PROTOCOL_ROOT / "result_source_audit.csv", rows, fields)
    write_result_source_audit_md(rows)
    return rows


def write_result_source_audit_md(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# R90/T500mm Result Source Audit",
        "",
        "| Model | Samples | RR@5 (%) | RR@10 (%) | RRE mean | RTE mean | Latency ms | Result dir |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['display_name']} | {row.get('sample_count', '')} | "
            f"{_format_pct(row.get('rr5'))} | {_format_pct(row.get('rr10'))} | "
            f"{_format_float(row.get('rre_deg_mean'), 2)} | "
            f"{_format_float(row.get('rte_mm_mean'), 2)} | "
            f"{_format_float(row.get('latency_ms_mean'), 1)} | "
            f"`{row['result_dir']}` |"
        )
    lines.append("")
    (PROTOCOL_ROOT / "result_source_audit.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def build_error_distribution_summary(
    runs: list[RunSpec],
    records_by_model: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in runs:
        records = records_by_model[spec.model_key]
        sample_count = len(records)
        rre_values = [_safe_float(row.get("rre_deg")) for row in records]
        rte_values = [_safe_float(row.get("rte_mm")) for row in records]
        rot5 = sum(value <= 5.0 for value in rre_values if math.isfinite(value))
        trans5 = sum(value <= 5.0 for value in rte_values if math.isfinite(value))
        rot10 = sum(value <= 10.0 for value in rre_values if math.isfinite(value))
        trans10 = sum(value <= 10.0 for value in rte_values if math.isfinite(value))
        rr5 = sum(
            _safe_float(row.get("rre_deg")) <= 5.0
            and _safe_float(row.get("rte_mm")) <= 5.0
            for row in records
        )
        rr10 = sum(
            _safe_float(row.get("rre_deg")) <= 10.0
            and _safe_float(row.get("rte_mm")) <= 10.0
            for row in records
        )
        rows.append(
            {
                "model_key": spec.model_key,
                "display_name": spec.display_name,
                "sample_count": sample_count,
                "rot_hit_5deg_rate": rot5 / sample_count,
                "trans_hit_5mm_rate": trans5 / sample_count,
                "joint_rr5_rate": rr5 / sample_count,
                "rot_hit_10deg_rate": rot10 / sample_count,
                "trans_hit_10mm_rate": trans10 / sample_count,
                "joint_rr10_rate": rr10 / sample_count,
                "rre_deg_mean": _safe_mean(rre_values),
                "rre_deg_median": _safe_median(rre_values),
                "rre_deg_p90": _safe_p90(rre_values),
                "rte_mm_mean": _safe_mean(rte_values),
                "rte_mm_median": _safe_median(rte_values),
                "rte_mm_p90": _safe_p90(rte_values),
            }
        )

    fields = [
        "model_key",
        "display_name",
        "sample_count",
        "rot_hit_5deg_rate",
        "trans_hit_5mm_rate",
        "joint_rr5_rate",
        "rot_hit_10deg_rate",
        "trans_hit_10mm_rate",
        "joint_rr10_rate",
        "rre_deg_mean",
        "rre_deg_median",
        "rre_deg_p90",
        "rte_mm_mean",
        "rte_mm_median",
        "rte_mm_p90",
    ]
    _write_csv(PROTOCOL_ROOT / "error_distribution_summary.csv", rows, fields)
    write_error_distribution_summary_md(rows)
    return rows


def write_error_distribution_summary_md(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Error Distribution Diagnostic Summary",
        "",
        "This table separates marginal rotation/translation hit rates from joint recall and adds median/p90 errors. It explains cases where a method has a nonzero strict recall but poor mean RRE/RTE because most failures have a heavy-tailed error distribution.",
        "",
        "| Model | R<=5 (%) | T<=5 (%) | RR@5 (%) | RTE median | RTE p90 | RRE median | RRE p90 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['display_name']} | "
            f"{_format_pct(row['rot_hit_5deg_rate'])} | "
            f"{_format_pct(row['trans_hit_5mm_rate'])} | "
            f"{_format_pct(row['joint_rr5_rate'])} | "
            f"{_format_float(row['rte_mm_median'], 2)} | "
            f"{_format_float(row['rte_mm_p90'], 2)} | "
            f"{_format_float(row['rre_deg_median'], 2)} | "
            f"{_format_float(row['rre_deg_p90'], 2)} |"
        )
    lines.append("")
    (PROTOCOL_ROOT / "error_distribution_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _model_record_map(
    records_by_model: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        model_key: {str(row["sample_id"]): row for row in rows}
        for model_key, rows in records_by_model.items()
    }


def build_independent_geometry_difficulty(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Compute prediction-independent geometry difficulty from GT-aligned pairs."""

    if GEOMETRY_DIFFICULTY_PATH.exists():
        rows = _read_csv_rows(GEOMETRY_DIFFICULTY_PATH)
        return {
            row["sample_id"]: {
                "difficulty_bucket": row["difficulty_bucket"],
                "visible_nn_proxy_mm": _safe_float(row["gt_visible_nn_mean_mm"]),
                "gt_visible_nn_mean_mm": _safe_float(row["gt_visible_nn_mean_mm"]),
                "gt_visible_nn_p90_mm": _safe_float(row["gt_visible_nn_p90_mm"]),
                "gt_trimmed_chamfer_mm": _safe_float(row["gt_trimmed_chamfer_mm"]),
                "scene_id": row["scene_id"],
                "frame_id": int(row["frame_id"]),
            }
            for row in rows
        }

    ERROR_ROOT.mkdir(parents=True, exist_ok=True)
    dataset = C3VDManifestDataset(
        manifest_path=config["data"]["manifest_path"],
        split=config["benchmark"]["split"],
        preprocess_pipeline=PreprocessPipeline(PreprocessRegistry()),
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
    point_unit = str(config["benchmark"].get("point_unit", "mm_like"))
    raw_rows: list[dict[str, Any]] = []
    for sample in dataset:
        source = _sample_evenly(np.asarray(sample["source_points"]), 2048)
        target = _sample_evenly(np.asarray(sample["target_points"]), 2048)
        gt_transform = np.asarray(sample["gt_transform"], dtype=np.float64)
        gt_source = _apply_transform(source, gt_transform)
        source_to_target = cKDTree(target).query(gt_source)[0]
        target_to_source = cKDTree(gt_source).query(target)[0]
        chamfer = np.concatenate([source_to_target, target_to_source])
        cutoff = np.percentile(chamfer, 90)
        trimmed = chamfer[chamfer <= cutoff]
        raw_rows.append(
            {
                "sample_id": str(sample["sample_id"]),
                "scene_id": str(sample["scene_id"]),
                "frame_id": int(sample["frame_id"]),
                "gt_visible_nn_mean_mm": _distance_to_mm(
                    float(np.mean(source_to_target)),
                    point_unit,
                ),
                "gt_visible_nn_p90_mm": _distance_to_mm(
                    float(np.percentile(source_to_target, 90)),
                    point_unit,
                ),
                "gt_trimmed_chamfer_mm": _distance_to_mm(
                    float(np.mean(trimmed)),
                    point_unit,
                ),
            }
        )

    q33, q66 = np.quantile(
        [row["gt_visible_nn_mean_mm"] for row in raw_rows],
        [1.0 / 3.0, 2.0 / 3.0],
    )
    for row in raw_rows:
        proxy = float(row["gt_visible_nn_mean_mm"])
        if proxy <= q33:
            row["difficulty_bucket"] = "easy"
        elif proxy <= q66:
            row["difficulty_bucket"] = "medium"
        else:
            row["difficulty_bucket"] = "hard"
    _write_csv(
        GEOMETRY_DIFFICULTY_PATH,
        raw_rows,
        [
            "sample_id",
            "scene_id",
            "frame_id",
            "difficulty_bucket",
            "gt_visible_nn_mean_mm",
            "gt_visible_nn_p90_mm",
            "gt_trimmed_chamfer_mm",
        ],
    )
    return {
        row["sample_id"]: {
            "difficulty_bucket": row["difficulty_bucket"],
            "visible_nn_proxy_mm": row["gt_visible_nn_mean_mm"],
            "gt_visible_nn_mean_mm": row["gt_visible_nn_mean_mm"],
            "gt_visible_nn_p90_mm": row["gt_visible_nn_p90_mm"],
            "gt_trimmed_chamfer_mm": row["gt_trimmed_chamfer_mm"],
            "scene_id": row["scene_id"],
            "frame_id": row["frame_id"],
        }
        for row in raw_rows
    }


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _difficulty_buckets(
    records_by_model: dict[str, list[dict[str, Any]]],
    independent_difficulty: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if independent_difficulty:
        return independent_difficulty

    visible_by_sample: dict[str, list[float]] = defaultdict(list)
    scene_by_sample: dict[str, str] = {}
    frame_by_sample: dict[str, int] = {}
    for rows in records_by_model.values():
        for row in rows:
            sample_id = str(row["sample_id"])
            visible = _safe_float(row.get("visible_nn_mean_mm"))
            if math.isfinite(visible):
                visible_by_sample[sample_id].append(visible)
            scene_by_sample[sample_id] = str(row.get("scene_id", ""))
            frame_by_sample[sample_id] = int(row.get("frame_id", 0))

    proxies = {
        sample_id: float(median(values))
        for sample_id, values in visible_by_sample.items()
        if values
    }
    q33, q66 = np.quantile(list(proxies.values()), [1.0 / 3.0, 2.0 / 3.0])
    buckets: dict[str, dict[str, Any]] = {}
    for sample_id, proxy in proxies.items():
        if proxy <= q33:
            bucket = "easy"
        elif proxy <= q66:
            bucket = "medium"
        else:
            bucket = "hard"
        buckets[sample_id] = {
            "difficulty_bucket": bucket,
            "visible_nn_proxy_mm": proxy,
            "scene_id": scene_by_sample.get(sample_id, ""),
            "frame_id": frame_by_sample.get(sample_id, 0),
        }
    return buckets


def build_join_and_summaries(
    runs: list[RunSpec],
    records_by_model: dict[str, list[dict[str, Any]]],
    independent_difficulty: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ERROR_ROOT.mkdir(parents=True, exist_ok=True)
    model_maps = _model_record_map(records_by_model)
    difficulty = _difficulty_buckets(records_by_model, independent_difficulty)
    sample_ids = sorted(difficulty)
    run_keys = [spec.model_key for spec in runs]

    joined_rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        base = difficulty[sample_id]
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "scene_id": base["scene_id"],
            "frame_id": base["frame_id"],
            "difficulty_bucket": base["difficulty_bucket"],
            "visible_nn_proxy_mm": base["visible_nn_proxy_mm"],
            "gt_visible_nn_mean_mm": base.get("gt_visible_nn_mean_mm", ""),
            "gt_visible_nn_p90_mm": base.get("gt_visible_nn_p90_mm", ""),
            "gt_trimmed_chamfer_mm": base.get("gt_trimmed_chamfer_mm", ""),
        }
        for model_key in run_keys:
            record = model_maps.get(model_key, {}).get(sample_id)
            if not record:
                continue
            prefix = f"{model_key}."
            row[prefix + "rre_deg"] = record.get("rre_deg")
            row[prefix + "rte_mm"] = record.get("rte_mm")
            row[prefix + "success_5deg_5mm"] = record.get("success_5deg_5mm")
            row[prefix + "success_10deg_10mm"] = record.get("success_10deg_10mm")
            row[prefix + "visible_nn_mean_mm"] = record.get("visible_nn_mean_mm")
            row[prefix + "latency_ms"] = record.get("latency_ms")
            row[prefix + "failure_tags"] = ";".join(record.get("failure_tags", []))
        joined_rows.append(row)

    join_fields = [
        "sample_id",
        "scene_id",
        "frame_id",
        "difficulty_bucket",
        "visible_nn_proxy_mm",
        "gt_visible_nn_mean_mm",
        "gt_visible_nn_p90_mm",
        "gt_trimmed_chamfer_mm",
    ]
    for model_key in run_keys:
        join_fields.extend(
            [
                f"{model_key}.rre_deg",
                f"{model_key}.rte_mm",
                f"{model_key}.success_5deg_5mm",
                f"{model_key}.success_10deg_10mm",
                f"{model_key}.visible_nn_mean_mm",
                f"{model_key}.latency_ms",
                f"{model_key}.failure_tags",
            ]
        )
    _write_csv(ERROR_ROOT / "per_pair_joined_results.csv", joined_rows, join_fields)

    scene_summary = summarize_by(records_by_model, ["scene_id"], difficulty)
    difficulty_summary = summarize_by(
        records_by_model,
        ["difficulty_bucket"],
        difficulty,
    )
    failure_summary = summarize_failure_tags(records_by_model)
    representatives = select_representative_cases(records_by_model, difficulty)
    (ERROR_ROOT / "representative_cases.json").write_text(
        json.dumps(representatives, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_error_markdown_summaries(scene_summary, difficulty_summary, representatives)
    return {
        "joined_rows": joined_rows,
        "difficulty": difficulty,
        "scene_summary": scene_summary,
        "difficulty_summary": difficulty_summary,
        "failure_summary": failure_summary,
        "representatives": representatives,
    }


def _summarize_records(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rre = [_safe_float(row.get("rre_deg")) for row in rows]
    rte = [_safe_float(row.get("rte_mm")) for row in rows]
    visible = [_safe_float(row.get("visible_nn_mean_mm")) for row in rows]
    latency = [_safe_float(row.get("latency_ms")) for row in rows]
    return {
        "count": len(rows),
        "rr5": _safe_mean([_safe_float(row.get("success_5deg_5mm")) for row in rows]),
        "rr10": _safe_mean(
            [_safe_float(row.get("success_10deg_10mm")) for row in rows]
        ),
        "rre_deg_mean": _safe_mean(rre),
        "rre_deg_median": _safe_median(rre),
        "rte_mm_mean": _safe_mean(rte),
        "rte_mm_median": _safe_median(rte),
        "visible_nn_mean_mm_mean": _safe_mean(visible),
        "latency_ms_mean": _safe_mean(latency),
        "latency_ms_p90": _safe_p90(latency),
    }


def summarize_by(
    records_by_model: dict[str, list[dict[str, Any]]],
    keys: list[str],
    difficulty: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for model_key, rows in records_by_model.items():
        for row in rows:
            enriched = dict(row)
            sample_difficulty = difficulty.get(str(row.get("sample_id")), {})
            enriched["difficulty_bucket"] = sample_difficulty.get("difficulty_bucket")
            group_key = tuple(
                [model_key]
                + [str(enriched.get(key, "missing")) for key in keys]
            )
            grouped[group_key].append(enriched)

    out_rows: list[dict[str, Any]] = []
    for group_key, rows in sorted(grouped.items()):
        model_key = group_key[0]
        summary = _summarize_records(rows)
        out = {"model_key": model_key}
        for index, key in enumerate(keys, start=1):
            out[key] = group_key[index]
        out.update(summary)
        out_rows.append(out)

    fields = ["model_key", *keys, *list(_summarize_records([]).keys())]
    filename = (
        "per_scene_error_summary.csv"
        if keys == ["scene_id"]
        else "geometry_difficulty_summary.csv"
    )
    _write_csv(ERROR_ROOT / filename, out_rows, fields)
    return out_rows


def summarize_failure_tags(
    records_by_model: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_key, records in records_by_model.items():
        counts: Counter[str] = Counter()
        for record in records:
            tags = record.get("failure_tags", [])
            if not tags:
                counts["none"] += 1
            for tag in tags:
                counts[str(tag)] += 1
        total = len(records)
        for tag, count in sorted(counts.items()):
            rows.append(
                {
                    "model_key": model_key,
                    "failure_tag": tag,
                    "count": count,
                    "rate": count / total if total else math.nan,
                    "sample_count": total,
                }
            )
    _write_csv(
        ERROR_ROOT / "failure_tag_summary.csv",
        rows,
        ["model_key", "failure_tag", "count", "rate", "sample_count"],
    )
    return rows


def _public_case(record: dict[str, Any], label: str, difficulty: dict[str, Any]) -> dict[str, Any]:
    sample_id = str(record["sample_id"])
    return {
        "label": label,
        "model_key": record.get("_model_key"),
        "display_name": record.get("_display_name"),
        "sample_id": sample_id,
        "scene_id": record.get("scene_id"),
        "frame_id": record.get("frame_id"),
        "difficulty_bucket": difficulty.get(sample_id, {}).get("difficulty_bucket"),
        "visible_nn_proxy_mm": difficulty.get(sample_id, {}).get(
            "visible_nn_proxy_mm"
        ),
        "rre_deg": record.get("rre_deg"),
        "rte_mm": record.get("rte_mm"),
        "visible_nn_mean_mm": record.get("visible_nn_mean_mm"),
        "success_5deg_5mm": record.get("success_5deg_5mm"),
        "success_10deg_10mm": record.get("success_10deg_10mm"),
        "failure_tags": record.get("failure_tags", []),
        "result_dir": record.get("_result_dir"),
    }


def select_representative_cases(
    records_by_model: dict[str, list[dict[str, Any]]],
    difficulty: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    geot = records_by_model.get("geotransformer", [])
    successes = [
        row
        for row in geot
        if int(row.get("success_10deg_10mm", 0)) == 1
    ]
    if successes:
        row = min(
            successes,
            key=lambda item: _safe_float(item.get("rre_deg"))
            + _safe_float(item.get("rte_mm")) / 10.0,
        )
        cases["easy_success"] = _public_case(row, "easy_success", difficulty)

    regtr = records_by_model.get("regtr", [])
    translation = [
        row
        for row in regtr
        if "large_translation" in set(row.get("failure_tags", []))
    ]
    if translation:
        row = max(translation, key=lambda item: _safe_float(item.get("rte_mm")))
        cases["translation_heavy_failure"] = _public_case(
            row,
            "translation_heavy_failure",
            difficulty,
        )

    severe_pool = records_by_model.get("dcp", []) + records_by_model.get(
        "pointnetlk",
        [],
    )
    if severe_pool:
        row = max(
            severe_pool,
            key=lambda item: _safe_float(item.get("rre_deg"))
            + _safe_float(item.get("rte_mm")) / 10.0,
        )
        cases["severe_failure"] = _public_case(row, "severe_failure", difficulty)

    model_maps = _model_record_map(records_by_model)
    all_fail_samples = []
    for sample_id, diff in difficulty.items():
        rows = [
            model_rows[sample_id]
            for model_rows in model_maps.values()
            if sample_id in model_rows
        ]
        if rows and all(int(row.get("success_10deg_10mm", 0)) == 0 for row in rows):
            all_fail_samples.append((sample_id, diff["visible_nn_proxy_mm"], rows))
    if all_fail_samples:
        sample_id, _, rows = max(all_fail_samples, key=lambda item: item[1])
        row = max(rows, key=lambda item: _safe_float(item.get("visible_nn_mean_mm")))
        cases["hard_geometry_failure"] = _public_case(
            row,
            "hard_geometry_failure",
            difficulty,
        )

    plausible = [
        row
        for row in geot
        if int(row.get("success_10deg_10mm", 0)) == 0
        and _safe_float(row.get("visible_nn_mean_mm")) <= 1.0
        and (
            _safe_float(row.get("rre_deg")) > 10.0
            or _safe_float(row.get("rte_mm")) > 10.0
        )
    ]
    if plausible:
        row = min(plausible, key=lambda item: _safe_float(item.get("visible_nn_mean_mm")))
        cases["plausible_but_wrong"] = _public_case(
            row,
            "plausible_but_wrong",
            difficulty,
        )
    return cases


def write_error_markdown_summaries(
    scene_summary: list[dict[str, Any]],
    difficulty_summary: list[dict[str, Any]],
    representatives: dict[str, Any],
) -> None:
    lines = [
        "# Difficulty Bucket Summary",
        "",
        "Difficulty uses prediction-independent GT-aligned Visible NN tertiles. It is a geometry proxy, not a true overlap label.",
        "",
        "| Model | Bucket | Count | RR@5 (%) | RR@10 (%) | RRE mean | RTE mean | RTE median |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    bucket_order = {"easy": 0, "medium": 1, "hard": 2}
    ordered_difficulty = sorted(
        difficulty_summary,
        key=lambda item: (
            str(item.get("model_key", "")),
            bucket_order.get(str(item.get("difficulty_bucket")), 99),
        ),
    )
    for row in ordered_difficulty:
        lines.append(
            f"| `{row['model_key']}` | `{row['difficulty_bucket']}` | {row['count']} | "
            f"{_format_pct(row['rr5'])} | {_format_pct(row['rr10'])} | "
            f"{_format_float(row['rre_deg_mean'], 2)} | "
            f"{_format_float(row['rte_mm_mean'], 2)} | "
            f"{_format_float(row['rte_mm_median'], 2)} |"
        )
    (PROTOCOL_ROOT / "difficulty_bucket_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    scene_lines = [
        "# Per-Scene Summary",
        "",
        "| Model | Scene | Count | RR@5 (%) | RR@10 (%) | RRE mean | RTE mean | Visible NN mean |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in scene_summary:
        scene_lines.append(
            f"| `{row['model_key']}` | `{row['scene_id']}` | {row['count']} | "
            f"{_format_pct(row['rr5'])} | {_format_pct(row['rr10'])} | "
            f"{_format_float(row['rre_deg_mean'], 2)} | "
            f"{_format_float(row['rte_mm_mean'], 2)} | "
            f"{_format_float(row['visible_nn_mean_mm_mean'], 2)} |"
        )
    (PROTOCOL_ROOT / "per_scene_summary.md").write_text(
        "\n".join(scene_lines) + "\n",
        encoding="utf-8",
    )

    rep_lines = [
        "# Representative Error Cases",
        "",
        "| Label | Model | Sample | Scene | RRE | RTE | Visible NN | Tags |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for label, row in representatives.items():
        rep_lines.append(
            f"| `{label}` | `{row.get('model_key')}` | `{row.get('sample_id')}` | "
            f"`{row.get('scene_id')}` | {_format_float(row.get('rre_deg'), 2)} | "
            f"{_format_float(row.get('rte_mm'), 2)} | "
            f"{_format_float(row.get('visible_nn_mean_mm'), 2)} | "
            f"`{';'.join(row.get('failure_tags', []))}` |"
        )
    (ERROR_ROOT / "representative_cases.md").write_text(
        "\n".join(rep_lines) + "\n",
        encoding="utf-8",
    )


def _overall_rows(runs: list[RunSpec]) -> list[dict[str, Any]]:
    rows = []
    for spec in runs:
        path = spec.result_dir / "leaderboard" / "leaderboard_main.csv"
        if path.exists():
            row = _read_csv_single_row(path)
            row["model_key"] = spec.model_key
            row["display_name"] = spec.display_name
            rows.append(row)
    return rows


def build_figures(
    runs: list[RunSpec],
    records_by_model: dict[str, list[dict[str, Any]]],
    analysis: dict[str, Any],
) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    overall = _overall_rows(runs)
    plot_rr_bar(overall)
    plot_cdf(records_by_model, "rre_deg", "RRE (deg)", "rre_cdf.png")
    plot_cdf(records_by_model, "rte_mm", "RTE (mm)", "rte_cdf.png")
    plot_visible_proxy_cdf(analysis["difficulty"])
    plot_scene_rr_heatmap(runs, analysis["scene_summary"])
    plot_difficulty_rr_rte(runs, analysis["difficulty_summary"])
    plot_failure_tags(runs, analysis["failure_summary"])
    write_qualitative_selection_manifest(analysis["representatives"])


def plot_rr_bar(overall: list[dict[str, Any]]) -> None:
    labels = [row["display_name"] for row in overall]
    x = np.arange(len(labels))
    width = 0.36
    rr5 = [
        _safe_float(row.get("registration_recall@rre_5deg_rte_5mm")) * 100.0
        for row in overall
    ]
    rr10 = [
        _safe_float(row.get("registration_recall@rre_10deg_rte_10mm")) * 100.0
        for row in overall
    ]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.bar(x - width / 2, rr5, width, label="RR@5deg/5mm", color="#2f6fbb")
    ax.bar(x + width / 2, rr10, width, label="RR@10deg/10mm", color="#3f8c54")
    ax.set_ylabel("Registration recall (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0.0, max(rr10 + rr5 + [1.0]) * 1.2)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "main_rr_bar.png", dpi=220)
    plt.close(fig)


def plot_cdf(
    records_by_model: dict[str, list[dict[str, Any]]],
    metric: str,
    xlabel: str,
    filename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    for model_key, rows in records_by_model.items():
        values = sorted(
            _safe_float(row.get(metric))
            for row in rows
            if math.isfinite(_safe_float(row.get(metric)))
        )
        if not values:
            continue
        y = np.arange(1, len(values) + 1) / len(values)
        ax.plot(
            values,
            y,
            label=model_key,
            color=MODEL_COLORS.get(model_key),
            linewidth=1.8,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / filename, dpi=220)
    plt.close(fig)


def plot_visible_proxy_cdf(difficulty: dict[str, dict[str, Any]]) -> None:
    values = sorted(float(row["visible_nn_proxy_mm"]) for row in difficulty.values())
    y = np.arange(1, len(values) + 1) / len(values)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(values, y, color="#2f6fbb", linewidth=2.0)
    ax.set_xlabel("Median Visible NN proxy (mm)")
    ax.set_ylabel("CDF")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "visible_nn_cdf.png", dpi=220)
    plt.close(fig)


def plot_scene_rr_heatmap(runs: list[RunSpec], scene_summary: list[dict[str, Any]]) -> None:
    scenes = sorted({str(row["scene_id"]) for row in scene_summary})
    model_keys = [spec.model_key for spec in runs]
    matrix = np.full((len(model_keys), len(scenes)), np.nan)
    for row in scene_summary:
        if row["model_key"] in model_keys:
            i = model_keys.index(row["model_key"])
            j = scenes.index(str(row["scene_id"]))
            matrix[i, j] = _safe_float(row["rr5"]) * 100.0
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0)
    ax.set_yticks(np.arange(len(model_keys)))
    ax.set_yticklabels(model_keys)
    ax.set_xticks(np.arange(len(scenes)))
    ax.set_xticklabels(scenes, rotation=35, ha="right")
    ax.set_title("Per-scene RR@5deg/5mm (%)")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("RR@5 (%)")
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "scene_bucket_rr.png", dpi=220)
    plt.close(fig)


def plot_difficulty_rr_rte(
    runs: list[RunSpec],
    difficulty_summary: list[dict[str, Any]],
) -> None:
    buckets = ["easy", "medium", "hard"]
    model_keys = [spec.model_key for spec in runs]
    lookup = {
        (row["model_key"], row["difficulty_bucket"]): row
        for row in difficulty_summary
    }
    x = np.arange(len(buckets))
    width = min(0.12, 0.78 / max(len(model_keys), 1))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharex=True)
    for index, model_key in enumerate(model_keys):
        offset = (index - (len(model_keys) - 1) / 2.0) * width
        rr = [
            _safe_float(lookup.get((model_key, bucket), {}).get("rr5")) * 100.0
            for bucket in buckets
        ]
        rte = [
            _safe_float(lookup.get((model_key, bucket), {}).get("rte_mm_median"))
            for bucket in buckets
        ]
        axes[0].bar(
            x + offset,
            rr,
            width,
            label=model_key,
            color=MODEL_COLORS.get(model_key),
        )
        axes[1].bar(
            x + offset,
            rte,
            width,
            label=model_key,
            color=MODEL_COLORS.get(model_key),
        )
    axes[0].set_ylabel("RR@5 (%)")
    axes[1].set_ylabel("Median RTE (mm)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(buckets)
        ax.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "difficulty_bucket_rr_rte.png", dpi=220)
    plt.close(fig)


def plot_failure_tags(runs: list[RunSpec], failure_summary: list[dict[str, Any]]) -> None:
    tags = ["large_rotation", "large_translation", "density_mismatch"]
    model_keys = [spec.model_key for spec in runs]
    lookup = {
        (row["model_key"], row["failure_tag"]): _safe_float(row["rate"]) * 100.0
        for row in failure_summary
    }
    x = np.arange(len(model_keys))
    bottom = np.zeros(len(model_keys))
    colors = ["#2f6fbb", "#bf6f2a", "#8e5fb8"]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for tag, color in zip(tags, colors, strict=True):
        values = [lookup.get((model_key, tag), 0.0) for model_key in model_keys]
        ax.bar(x, values, bottom=bottom, label=tag, color=color)
        bottom += np.asarray(values)
    ax.set_ylabel("Tag rate, stacked (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(model_keys, rotation=25, ha="right")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "failure_tag_distribution.png", dpi=220)
    plt.close(fig)


def write_qualitative_selection_manifest(representatives: dict[str, Any]) -> None:
    manifest_path = FIGURE_ROOT / "qualitative_selection_manifest.json"
    manifest_path.write_text(
        json.dumps(representatives, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    existing = [
        PROTOCOL_ROOT
        / "geotransformer"
        / "eval_test"
        / "qualitative"
        / "distance_render"
        / "cecum_t3_a_0329_worst_visible_distance.png",
        PROTOCOL_ROOT
        / "pointnetlk_revisited"
        / "eval_test"
        / "qualitative"
        / "distance_render"
        / "trans_t4_a_0265_worst_visible_distance.png",
    ]
    existing = [path for path in existing if path.exists()]
    if not existing:
        return
    fig, axes = plt.subplots(1, len(existing), figsize=(6.5 * len(existing), 4.2))
    if len(existing) == 1:
        axes = [axes]
    for ax, image_path in zip(axes, existing, strict=True):
        ax.imshow(plt.imread(image_path))
        ax.set_title(image_path.parent.parent.parent.parent.name)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "qualitative_available_renders.png", dpi=180)
    plt.close(fig)


def _copy_artifact(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _write_text_artifact(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _package_rel(path: Path) -> str:
    return str(path.relative_to(PACKAGE_ROOT))


def write_package_readme(
    summary: dict[str, Any],
    result_rows: list[dict[str, Any]],
) -> None:
    retained_models = ", ".join(row["display_name"] for row in result_rows)
    text = f"""# BMVC C3VD Raycasting Publication Package

This bundle contains the anonymized, publication-facing artifacts for the
C3VD raycasting point-cloud registration benchmark paper.

It intentionally excludes raw point clouds, model checkpoints, raw eval
configuration dumps, and raw run cards because those files contain local
machine paths or license-constrained data. The package keeps only manifests,
summary tables, joined analysis tables, figures, paper source, schema notes,
and command templates.

## Contents

- `paper/`: BMVC LaTeX source, bibliography, class/style files, and paper images.
- `docs/`: benchmark protocol and publication status notes.
- `manifests/`: full and test pair manifests with dataset-root placeholders.
- `results/`: retained leaderboard source audit.
- `analysis/`: joined per-pair table, difficulty/scene summaries, and representative cases.
- `figures/`: paper figures and deterministic qualitative failure panels.
- `reproducibility/`: result schema, command template, environment notes, and release audit.

## Frozen Protocol

- Test split: `{summary['split_counts'].get('test', 0)}` pairs.
- Full manifest: `{summary['split_counts'].get('train', 0)}` train, `{summary['split_counts'].get('val', 0)}` val, `{summary['split_counts'].get('test', 0)}` test pairs.
- Input points: `{summary['input_points']}` source and `{summary['input_points']}` target points.
- Preprocess profile: `{summary['preprocess_profile_id']}`.
- Perturbation: source-only R{summary['perturbation']['rotation_deg']:.0f}/T{summary['perturbation']['translation_m']:.0f}mm with noise sigma {summary['perturbation']['noise_sigma']:.1f}.
- Retained models: {retained_models}.

## Main Result Pointers

- Main summary: `results/result_source_audit.md`.
- Error distribution diagnostic: `results/error_distribution_summary.md`.
- Joined per-pair table: `analysis/per_pair_joined_results.csv`.
- Difficulty summary: `analysis/difficulty_bucket_summary.md`.
- Qualitative contact sheet: `figures/paper/qualitative_alignment_panels.png`.

Difficulty buckets use prediction-independent GT-aligned Visible NN tertiles.
They are geometry-proxy buckets, not true overlap labels.
"""
    _write_text_artifact(PACKAGE_ROOT / "README.md", text)


def write_reproducibility_notes() -> None:
    text = """# Reproducibility Notes

## Environment

The local experiments were run with Python 3.10, PyTorch 2.8, Open3D 0.19,
NumPy 2.2, SciPy 1.15, h5py 3.14, matplotlib 3.10, and PyYAML 6.0.
The retained learning baselines also depend on their public vendor code and
compiled CUDA/sparse extensions where applicable.

Install the project dependencies in an isolated environment, then make the
C3VD raycasting data root available through an environment variable:

```bash
export C3VD_RAYCASTING_ROOT=/path/to/C3VD_raycasting_root
```

## Rebuilding Publication Artifacts

From the repository root, after retained eval bundles are present:

```bash
python scripts/benchmark/build_bmvc_publication_package.py
python scripts/benchmark/render_bmvc_qualitative_panels.py
```

The first command joins retained result bundles, regenerates manifests,
analysis tables, figures, and this release package. The second command
rerenders deterministic qualitative panels from the selected cases.

## Running An Evaluation

Use `reproducibility/eval_config_template.yaml` as a placeholder template.
It is not a model-specific frozen config; it avoids local checkpoint and
dataset paths by design. A concrete run still needs a model-specific checkpoint
and adapter settings.

```bash
python scripts/runners/eval_benchmark.py \\
  --config reproducibility/eval_config_template.yaml
```

Official leaderboard rows must use `split=test`, `subset_name=null`,
`num_points_override=8192`, `rotation_deg=90`, `translation_m=500`,
`noise_sigma=0`, and `apply_noise_to=source`.

## Data Release Boundary

The manifests use `${C3VD_RAYCASTING_ROOT}` placeholders. Direct release of
derived raycasting point clouds should be handled under the upstream C3VD
license boundary and any institutional review requirements. If direct point
cloud redistribution is not approved, release manifests, scripts, metrics, and
commands rather than repackaged point clouds.
"""
    _write_text_artifact(PACKAGE_ROOT / "reproducibility" / "REPRODUCIBILITY.md", text)


def write_result_schema_notes() -> None:
    text = """# Result Schema

Per-sample benchmark outputs follow `src/benchmarking/reporting/result_schema.py`.
The publication package does not include raw `results.jsonl` files because raw
run bundles may contain local paths. It includes a joined CSV table with the
same metric columns needed for paper analysis.

## Required Identity Fields

- `sample_id`: stable pair id, formatted as `scene_id:frame_id`.
- `scene_id`: test scene identifier.
- `trajectory_id`: trajectory or scene grouping.
- `frame_id`: integer frame id.
- `split`: `train`, `val`, or `test`.
- `model_id`: evaluated baseline id.
- `preprocess_profile_id`: benchmark preprocess route.
- `adapter_private_transform_id`: adapter transform convention id.
- `refinement_track`: declared refinement path.
- `train_regime`: training/eval regime label.
- `manifest_digest`: source manifest digest.
- `config_digest`: normalized config digest.

## Required Metric Fields

- `rre_deg`: relative rotation error in degrees.
- `rte_mm`: relative translation error in millimetres.
- `success_1deg_1mm`, `success_3deg_3mm`, `success_5deg_5mm`, `success_10deg_10mm`: thresholded recall flags.

## Optional Diagnostic Fields

- `rmse_mm`
- `visible_nn_mean_mm`, `visible_nn_median_mm`, `visible_nn_p90_mm`
- `trimmed_chamfer_mm`
- `overlap_only_distance_mm`
- `preprocess_time_ms`, `inference_time_ms`, `refinement_time_ms`, `latency_ms`
- `peak_memory_mb`
- `overlap_bin`, `rotation_bin`, `translation_bin`, `artifact_bin`
- `failure_tags`

## Publication Joined Table

`analysis/per_pair_joined_results.csv` stores one row per test pair and prefixes
model-specific metrics as `<model_key>.<metric>`, for example
`geotransformer.rre_deg` or `icp.success_5deg_5mm`.
"""
    _write_text_artifact(PACKAGE_ROOT / "reproducibility" / "RESULT_SCHEMA.md", text)


def write_eval_config_template() -> None:
    text = """benchmark:
  name: c3vd_raycasting_v1
  split: test
  subset_name: null
  point_unit: mm_like
  official_track: main

data:
  manifest_path: ${C3VD_RAYCASTING_ROOT}/c3vd_raycasting_manifest.jsonl
  subset_config_path: configs/subset_config.json
  dataset_root: ${C3VD_RAYCASTING_ROOT}

preprocess:
  profile: canonical_v1
  seed: 42
  sampling_override: null
  num_points_override: 8192

perturbation:
  enabled: true
  rotation_deg: 90.0
  translation_m: 500.0
  noise_sigma: 0.0
  noise_clip: 0.0
  apply_noise_to: source

model:
  id: geotransformer
  checkpoint_path: ${MODEL_CHECKPOINT}
  overrides: {}

runtime:
  device: cuda:0
  batch_size: 1
  num_workers: 0
  export_html: false
  output_dir: outputs/benchmark/reproduced_eval
  train_metadata_path: null

analysis:
  qualitative:
    export_failure_gallery: false
  export:
    html: false
    png: true
    markdown_tables: true
"""
    _write_text_artifact(
        PACKAGE_ROOT / "reproducibility" / "eval_config_template.yaml",
        text,
    )


def audit_package_anonymization() -> None:
    patterns = ("linzhe", "/mnt/", "F:\\", "C:\\")
    suffixes = {
        ".aux",
        ".bbl",
        ".bib",
        ".bst",
        ".cls",
        ".csv",
        ".json",
        ".jsonl",
        ".md",
        ".sty",
        ".tex",
        ".txt",
        ".yaml",
        ".yml",
    }
    matches: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(pattern in line for pattern in patterns):
                matches.append(f"{_package_rel(path)}:{line_number}: {line}")

    if matches:
        status = "FAILED"
        detail = "\n".join(f"- `{match}`" for match in matches)
    else:
        status = "PASSED"
        detail = "- No local user, local mount, Windows drive, or absolute machine path markers found in text artifacts."

    text = f"""# Release Anonymization Audit

Status: {status}

Scope: all text artifacts under this publication package.

{detail}
"""
    _write_text_artifact(PACKAGE_ROOT / "reproducibility" / "ANONYMIZATION_AUDIT.md", text)


def build_reproducibility_package(
    summary: dict[str, Any],
    result_rows: list[dict[str, Any]],
) -> None:
    if PACKAGE_ROOT.exists():
        shutil.rmtree(PACKAGE_ROOT)
    PACKAGE_ROOT.mkdir(parents=True, exist_ok=True)

    copy_pairs = [
        (PROTOCOL_DOC, PACKAGE_ROOT / "docs" / "C3VD_Raycasting_PCR_Benchmark_Protocol.md"),
        (
            PROTOCOL_ROOT / "pair_manifest.jsonl",
            PACKAGE_ROOT / "manifests" / "pair_manifest.jsonl",
        ),
        (
            PROTOCOL_ROOT / "pair_manifest_test.jsonl",
            PACKAGE_ROOT / "manifests" / "pair_manifest_test.jsonl",
        ),
        (
            PROTOCOL_ROOT / "pair_manifest_summary.md",
            PACKAGE_ROOT / "manifests" / "pair_manifest_summary.md",
        ),
        (
            PROTOCOL_ROOT / "result_source_audit.csv",
            PACKAGE_ROOT / "results" / "result_source_audit.csv",
        ),
        (
            PROTOCOL_ROOT / "result_source_audit.md",
            PACKAGE_ROOT / "results" / "result_source_audit.md",
        ),
        (
            PROTOCOL_ROOT / "error_distribution_summary.csv",
            PACKAGE_ROOT / "results" / "error_distribution_summary.csv",
        ),
        (
            PROTOCOL_ROOT / "error_distribution_summary.md",
            PACKAGE_ROOT / "results" / "error_distribution_summary.md",
        ),
        (
            PROTOCOL_ROOT / "difficulty_bucket_summary.md",
            PACKAGE_ROOT / "analysis" / "difficulty_bucket_summary.md",
        ),
        (
            PROTOCOL_ROOT / "per_scene_summary.md",
            PACKAGE_ROOT / "analysis" / "per_scene_summary.md",
        ),
        (
            ERROR_ROOT / "per_pair_joined_results.csv",
            PACKAGE_ROOT / "analysis" / "per_pair_joined_results.csv",
        ),
        (
            ERROR_ROOT / "per_scene_error_summary.csv",
            PACKAGE_ROOT / "analysis" / "per_scene_error_summary.csv",
        ),
        (
            ERROR_ROOT / "geometry_difficulty_summary.csv",
            PACKAGE_ROOT / "analysis" / "geometry_difficulty_summary.csv",
        ),
        (
            ERROR_ROOT / "independent_geometry_difficulty.csv",
            PACKAGE_ROOT / "analysis" / "independent_geometry_difficulty.csv",
        ),
        (
            ERROR_ROOT / "failure_tag_summary.csv",
            PACKAGE_ROOT / "analysis" / "failure_tag_summary.csv",
        ),
        (
            ERROR_ROOT / "representative_cases.json",
            PACKAGE_ROOT / "analysis" / "representative_cases.json",
        ),
        (
            ERROR_ROOT / "representative_cases.md",
            PACKAGE_ROOT / "analysis" / "representative_cases.md",
        ),
        (
            REPO_ROOT / "BMVC2026_Linzhe" / "bmvc_c3vd_benchmark_draft.tex",
            PACKAGE_ROOT / "paper" / "bmvc_c3vd_benchmark_draft.tex",
        ),
        (
            REPO_ROOT / "BMVC2026_Linzhe" / "bmvc_review.tex",
            PACKAGE_ROOT / "paper" / "bmvc_review.tex",
        ),
        (
            REPO_ROOT / "BMVC2026_Linzhe" / "bmvc_final.tex",
            PACKAGE_ROOT / "paper" / "bmvc_final.tex",
        ),
        (
            REPO_ROOT / "BMVC2026_Linzhe" / "bmvc_c3vd_benchmark_draft.pdf",
            PACKAGE_ROOT / "paper" / "bmvc_c3vd_benchmark_draft.pdf",
        ),
        (
            REPO_ROOT / "BMVC2026_Linzhe" / "bmvc2k.cls",
            PACKAGE_ROOT / "paper" / "bmvc2k.cls",
        ),
        (
            REPO_ROOT / "BMVC2026_Linzhe" / "bmvc2k.bst",
            PACKAGE_ROOT / "paper" / "bmvc2k.bst",
        ),
        (
            REPO_ROOT / "BMVC2026_Linzhe" / "bmvc2k_natbib.sty",
            PACKAGE_ROOT / "paper" / "bmvc2k_natbib.sty",
        ),
        (
            REPO_ROOT / "BMVC2026_Linzhe" / "c3vd_benchmark.bib",
            PACKAGE_ROOT / "paper" / "c3vd_benchmark.bib",
        ),
    ]
    for source, destination in copy_pairs:
        _copy_artifact(source, destination)

    for image_path in sorted((REPO_ROOT / "BMVC2026_Linzhe" / "images").glob("*")):
        if image_path.name.startswith("eg1_") or image_path.name.startswith("root3"):
            continue
        _copy_artifact(image_path, PACKAGE_ROOT / "paper" / "images" / image_path.name)

    for figure_path in sorted(FIGURE_ROOT.glob("*")):
        _copy_artifact(figure_path, PACKAGE_ROOT / "figures" / "paper" / figure_path.name)

    for figure_path in sorted(FAILURE_FIGURE_ROOT.glob("*")):
        _copy_artifact(
            figure_path,
            PACKAGE_ROOT / "figures" / "failure_cases" / figure_path.name,
        )

    write_package_readme(summary, result_rows)
    write_reproducibility_notes()
    write_result_schema_notes()
    write_eval_config_template()
    audit_package_anonymization()

    manifest_rows = []
    for path in sorted(PACKAGE_ROOT.rglob("*")):
        if path.is_file():
            manifest_rows.append(_package_rel(path))
    _write_text_artifact(
        PACKAGE_ROOT / "ARTIFACT_MANIFEST.txt",
        "\n".join(manifest_rows),
    )


def update_results_summary() -> None:
    path = REPO_ROOT / "outputs" / "benchmark" / "test_results_summary.md"
    marker = "\n## BMVC Publication Package\n"
    section = """## BMVC Publication Package

Generated publication-facing artifacts:

- Protocol doc: `docs/C3VD_Raycasting_PCR_Benchmark_Protocol.md`
- Pair manifest: `outputs/benchmark/r90_t500mm_protocol/pair_manifest.jsonl`
- Test pair manifest: `outputs/benchmark/r90_t500mm_protocol/pair_manifest_test.jsonl`
- Result source audit: `outputs/benchmark/r90_t500mm_protocol/result_source_audit.md`
- Error distribution diagnostic: `outputs/benchmark/r90_t500mm_protocol/error_distribution_summary.md`
- Joined per-pair table: `outputs/benchmark/r90_t500mm_protocol/error_analysis/per_pair_joined_results.csv`
- Difficulty summary: `outputs/benchmark/r90_t500mm_protocol/difficulty_bucket_summary.md`
- Per-scene summary: `outputs/benchmark/r90_t500mm_protocol/per_scene_summary.md`
- Representative cases: `outputs/benchmark/r90_t500mm_protocol/error_analysis/representative_cases.json`
- Paper figures: `outputs/benchmark/figures/paper/`
- Anonymous reproducibility package: `outputs/benchmark/bmvc_publication_package/`

Difficulty buckets currently use prediction-independent GT-aligned Visible NN tertiles because manifest overlap labels are missing. Do not describe these buckets as true overlap.
"""
    original = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in original:
        original = original.split(marker, 1)[0].rstrip() + "\n"
    path.write_text(original.rstrip() + "\n\n" + section, encoding="utf-8")


def main() -> int:
    runs = _load_available_runs()
    if not runs:
        raise RuntimeError("No retained result bundles found.")
    config = _first_config(runs)
    summary = build_pair_manifest(config)
    result_rows = build_result_source_audit(runs)
    records_by_model = _load_run_records(runs)
    build_error_distribution_summary(runs, records_by_model)
    independent_difficulty = build_independent_geometry_difficulty(config)
    analysis = build_join_and_summaries(runs, records_by_model, independent_difficulty)
    build_figures(runs, records_by_model, analysis)
    update_results_summary()
    build_reproducibility_package(summary, result_rows)
    print(
        json.dumps(
            {
                "runs": [spec.model_key for spec in runs],
                "protocol_doc": _rel(PROTOCOL_DOC),
                "error_analysis": _rel(ERROR_ROOT),
                "figures": _rel(FIGURE_ROOT),
                "package": _rel(PACKAGE_ROOT),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
