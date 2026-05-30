#!/usr/bin/env python3
# ruff: noqa: E402
"""Build paper figures for the frame-wise C3VDReg benchmark revision."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")
for _font_path in (
    "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Bold.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Italic.otf",
):
    if Path(_font_path).exists():
        font_manager.fontManager.addfont(_font_path)
matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Nimbus Roman"],
        "mathtext.fontset": "stix",
    }
)

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark.render_bmvc_qualitative_panels import (  # noqa: E402
    _apply_transform,
    _camera_basis_from_direction,
    _camera_direction_to_mpl_view,
    _choose_error_view,
    _fibonacci_sphere_directions,
    _load_json,
    _load_sample,
    _sample_evenly,
    _style_axis,
)
from src.benchmarking.datasets.c3vd_manifest_dataset import _read_ply_xyz  # noqa: E402
from src.benchmarking.metrics.pose_metrics import compute_pose_metrics  # noqa: E402
from src.benchmarking.metrics.units import (  # noqa: E402
    canonicalize_point_unit,
    point_unit_to_mm_scale,
)
from src.unified_testing.adapters.geotransformer_adapter import (  # noqa: E402
    GeoTransformerAdapter,
)

BENCH_ROOT = REPO_ROOT / "outputs" / "benchmark"
HARD_ROOT = BENCH_ROOT / "r25_90_t100_500mm_protocol"
PAPER_FIGURE_DIR = BENCH_ROOT / "figures" / "paper"
BMVC_IMAGE_DIR = REPO_ROOT / "BMVC2026_Linzhe" / "images"
LOCAL_GLOBAL_ROOT = BENCH_ROOT / "local_to_global_diagnostic"
FULL_CT_MANIFEST = LOCAL_GLOBAL_ROOT / "full_ct_subset_manifest.jsonl"
SELECTION_JSON = LOCAL_GLOBAL_ROOT / "selected_samples.json"
FULL_MODEL_OBJ = Path("/mnt/f/Datasets/C3VD_sever_datasets/C3VD_ref/full_model.obj")
FULL_MODEL_PLY = Path("/mnt/f/Datasets/C3VD_sever_datasets/C3VD_ref/full_model.ply")
FULL_MODEL_SCENE_TRANSFORMS = (
    BENCH_ROOT
    / "full_model_scene_alignment"
    / "full_model_to_scene_coverage_transforms.json"
)
REVISED_ROW_DIR = PAPER_FIGURE_DIR / "revised_row_panels"
HELD_OUT_SCENES = (
    "cecum_t1_a",
    "cecum_t3_a",
    "sigmoid_t3_b",
    "trans_t1_a",
    "trans_t2_b",
    "trans_t4_a",
)
MODEL_CONFIGS = {
    "geotransformer": HARD_ROOT
    / "geotransformer"
    / "eval_test"
    / "normalized_eval_config.json",
    "icp": HARD_ROOT / "icp" / "eval_test" / "normalized_eval_config.json",
}
GEOT_FAILURE_CASES = (
    ("qualitative_case_a.png", "(a) GeoTransformer success"),
    ("qualitative_case_b.png", "(b) GeoTransformer plausible but wrong"),
)
CONSTRUCTION_SAMPLE_ID = "sigmoid_t3_b:0196"
GEOT_FAILURE_SAMPLE_IDS = (
    "sigmoid_t3_b:0261",
    "sigmoid_t3_b:0036",
    "cecum_t3_a:0612",
    "cecum_t3_a:0220",
    "trans_t4_a:0298",
    "trans_t4_a:0078",
    "trans_t2_b:0019",
    "trans_t1_a:0018",
)
LOCAL_GLOBAL_COMPARISON_SAMPLE_IDS = (
    "sigmoid_t3_b:0235",
    "trans_t4_a:0222",
)
LOCAL_GLOBAL_ALIGNMENT_VIEW_OVERRIDES = {
    "cecum_t1_a:0030": (0.0, 90.0),
}
LOCAL_GLOBAL_ALIGNMENT_ZOOM_OVERRIDES = {
    "cecum_t1_a:0030": 1.12,
}
CT_TARGET_SAMPLE_COUNT = 32768
PANEL_SOURCE_SAMPLE_COUNT = 2048
PANEL_TARGET_SAMPLE_COUNT = 4096
PANEL_FULL_MODEL_INITIAL_SAMPLE_COUNT = 12000
ALIGNMENT_ROW_WIDTH_IN = 13.8
ALIGNMENT_ROW_HEIGHT_IN = 3.18
LOCAL_GLOBAL_ROW_WIDTH_IN = 8.4
LOCAL_GLOBAL_ROW_HEIGHT_IN = 2.85


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _balanced_sample(
    rows: list[dict[str, Any]], per_scene: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for scene in HELD_OUT_SCENES:
        scene_rows = [
            row
            for row in rows
            if row.get("split") == "test"
            and (row.get("scene") or row.get("scene_id")) == scene
        ]
        scene_rows = sorted(
            scene_rows,
            key=lambda row: int(row.get("frame_idx", row.get("frame_id", 0))),
        )
        if not scene_rows:
            raise RuntimeError(f"No test rows found for scene {scene!r}.")
        count = min(per_scene, len(scene_rows))
        indices = np.linspace(0, len(scene_rows) - 1, count, dtype=int)
        selected.extend(scene_rows[int(index)] for index in indices)
    return selected


def prepare_local_to_global(args: argparse.Namespace) -> int:
    base_config = _load_json(MODEL_CONFIGS["geotransformer"])
    base_manifest = Path(base_config["data"]["manifest_path"])
    base_rows = _read_jsonl(base_manifest)
    selected = _balanced_sample(base_rows, int(args.per_scene))
    full_rows: list[dict[str, Any]] = []
    for row in selected:
        full_row = dict(row)
        full_row["target_path"] = row["reference_path"]
        full_row["target_setting"] = "full_scene_mesh_sampled_to_8192"
        full_row["visible_target_path"] = row["target_path"]
        full_row["benchmark_view"] = "direct_framewise_local_to_global_diagnostic"
        full_rows.append(full_row)
    _write_jsonl(FULL_CT_MANIFEST, full_rows)

    sample_ids = [str(row["sample_id"]) for row in full_rows]
    _write_json(
        SELECTION_JSON,
        {
            "per_scene": int(args.per_scene),
            "sample_count": len(sample_ids),
            "sample_ids": sample_ids,
            "manifest": str(FULL_CT_MANIFEST.relative_to(REPO_ROOT)),
        },
    )

    config_paths: dict[str, str] = {}
    for model_key, config_path in MODEL_CONFIGS.items():
        config = _load_json(config_path)
        config["data"]["manifest_path"] = str(FULL_CT_MANIFEST)
        config["runtime"]["output_dir"] = str(
            LOCAL_GLOBAL_ROOT / model_key / "eval_test"
        )
        config["runtime"]["export_html"] = False
        config["analysis"]["geometry"]["export_histogram"] = False
        config["analysis"]["geometry"]["export_cdf"] = False
        config["analysis"]["export"]["html"] = False
        config["analysis"]["export"]["png"] = False
        config["analysis"]["export"]["markdown_tables"] = True
        out_config = LOCAL_GLOBAL_ROOT / "configs" / f"eval_{model_key}_full_ct.json"
        _write_json(out_config, config)
        config_paths[model_key] = str(out_config.relative_to(REPO_ROOT))

    _write_json(
        LOCAL_GLOBAL_ROOT / "run_manifest.json",
        {
            "full_ct_manifest": str(FULL_CT_MANIFEST.relative_to(REPO_ROOT)),
            "config_paths": config_paths,
            "description": (
                "Balanced diagnostic: direct single-frame source to the aligned full "
                "scene mesh "
                "target, sampled to the same 8192-point input budget."
            ),
        },
    )
    print(
        json.dumps(
            {"manifest": str(FULL_CT_MANIFEST), "configs": config_paths}, indent=2
        )
    )
    return 0


def _load_result_rows(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): row for row in _read_jsonl(path)}


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    rre = np.asarray([float(row["rre_deg"]) for row in rows], dtype=np.float64)
    rte = np.asarray([float(row["rte_mm"]) for row in rows], dtype=np.float64)
    rr5 = np.asarray([int(row["success_5deg_5mm"]) for row in rows], dtype=np.float64)
    rr10 = np.asarray(
        [int(row["success_10deg_10mm"]) for row in rows], dtype=np.float64
    )
    return {
        "n": float(len(rows)),
        "rr5": float(np.mean(rr5) * 100.0),
        "rr10": float(np.mean(rr10) * 100.0),
        "rre_median": float(np.median(rre)),
        "rre_p90": float(np.percentile(rre, 90.0)),
        "rte_median": float(np.median(rte)),
        "rte_p90": float(np.percentile(rte, 90.0)),
    }


def summarize_local_to_global(_args: argparse.Namespace) -> int:
    selection = _load_json(SELECTION_JSON)
    sample_ids = list(map(str, selection["sample_ids"]))
    rows_out: list[dict[str, Any]] = []
    for model_key in ("geotransformer", "icp"):
        full_path = LOCAL_GLOBAL_ROOT / model_key / "eval_test" / "results.jsonl"
        ray_path = HARD_ROOT / model_key / "eval_test" / "results.jsonl"
        if not full_path.exists():
            raise FileNotFoundError(f"Missing full-CT result file: {full_path}")
        full_by_id = _load_result_rows(full_path)
        ray_by_id = _load_result_rows(ray_path)
        for setting, by_id in (
            ("full_scene_local_to_global", full_by_id),
            ("raycast_partial_to_partial", ray_by_id),
        ):
            matched = [
                by_id[sample_id] for sample_id in sample_ids if sample_id in by_id
            ]
            summary = _summarize_rows(matched)
            rows_out.append({"model": model_key, "target_setting": setting, **summary})

    LOCAL_GLOBAL_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = LOCAL_GLOBAL_ROOT / "local_to_global_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "target_setting",
                "n",
                "rr5",
                "rr10",
                "rre_median",
                "rre_p90",
                "rte_median",
                "rte_p90",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    md_path = LOCAL_GLOBAL_ROOT / "local_to_global_summary.md"
    lines = [
        (
            "| Model | Target setting | n | RR@5 | RR@10 | RRE med | "
            "RRE p90 | RTE med | RTE p90 |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows_out:
        template = (
            "| {model} | {target_setting} | {n:.0f} | {rr5:.2f} | "
            "{rr10:.2f} | {rre_median:.2f} | {rre_p90:.2f} | "
            "{rte_median:.2f} | {rte_p90:.2f} |"
        )
        lines.append(
            template.format(**row)
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def render_geotransformer_figure4(_args: argparse.Namespace) -> int:
    images: list[np.ndarray] = []
    titles: list[str] = []
    for filename, title in GEOT_FAILURE_CASES:
        path = PAPER_FIGURE_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing qualitative row image: {path}")
        images.append(mpimg.imread(path))
        titles.append(title)

    width = 12.2
    row_height = 2.9
    fig, axes = plt.subplots(
        len(images), 1, figsize=(width, row_height * len(images)), dpi=220
    )
    if len(images) == 1:
        axes = [axes]
    for ax, image, title in zip(axes, images, titles, strict=True):
        ax.imshow(image)
        ax.set_axis_off()
        ax.text(
            0.005,
            0.985,
            title,
            transform=ax.transAxes,
            fontsize=14.0,
            fontweight="bold",
            va="top",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
        )
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005, hspace=0.02)
    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    BMVC_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        out_path = PAPER_FIGURE_DIR / f"geotransformer_failure_cases.{suffix}"
        fig.savefig(out_path)
        shutil.copy2(out_path, BMVC_IMAGE_DIR / out_path.name)
    plt.close(fig)
    print(
        str(
            (PAPER_FIGURE_DIR / "geotransformer_failure_cases.pdf").relative_to(
                REPO_ROOT
            )
        )
    )
    return 0


def _read_transform(model_key: str, sample_id: str) -> np.ndarray:
    path = (
        HARD_ROOT
        / "pose_transform_exports"
        / model_key
        / "eval_test"
        / "pose_transforms.jsonl"
    )
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("sample_id")) == sample_id:
                return np.asarray(row["pred_transform"], dtype=np.float64)
    raise KeyError(f"{sample_id!r} not found in {path}")


def _read_result_row(model_key: str, sample_id: str) -> dict[str, Any]:
    rows = _load_result_rows(HARD_ROOT / model_key / "eval_test" / "results.jsonl")
    return rows[sample_id]


def _stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _sanitize_filename(text: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in text)


def _prediction_to_transform(prediction: Any) -> np.ndarray:
    if isinstance(prediction, tuple) and len(prediction) == 2:
        rotation, translation = prediction
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
        transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
        return transform
    transform = np.asarray(prediction, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Unsupported prediction shape {transform.shape}.")
    return transform


def _sample_random(points: np.ndarray, sample_count: int, seed_text: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= sample_count:
        return points.astype(np.float64, copy=False)
    rng = np.random.default_rng(_stable_seed(seed_text))
    indices = rng.choice(len(points), size=sample_count, replace=False)
    return points[indices].astype(np.float64, copy=False)


def _ensure_full_model_ply() -> Path:
    if FULL_MODEL_PLY.exists():
        return FULL_MODEL_PLY
    if not FULL_MODEL_OBJ.exists():
        raise FileNotFoundError(f"Missing full CT mesh: {FULL_MODEL_OBJ}")
    vertices: list[tuple[float, float, float]] = []
    with FULL_MODEL_OBJ.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) >= 4:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not vertices:
        raise RuntimeError(f"No vertices found in {FULL_MODEL_OBJ}")
    vertex_array = np.asarray(
        vertices,
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
    )
    FULL_MODEL_PLY.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex_array, "vertex")], text=False).write(
        FULL_MODEL_PLY
    )
    return FULL_MODEL_PLY


def _load_pose_matrix(pose_path: str | Path, frame_id: int) -> np.ndarray:
    lines = Path(pose_path).read_text(encoding="utf-8").splitlines()
    if frame_id >= len(lines):
        raise IndexError(f"frame {frame_id} is out of range for {pose_path}")
    values = np.asarray(
        [float(value) for value in lines[frame_id].split(",")],
        dtype=np.float64,
    )
    if values.size != 16:
        raise ValueError(f"Expected 16 pose values in {pose_path}:{frame_id}")
    # C3VD pose files are stored in row-vector layout; transpose to the
    # column-vector convention used by _apply_transform.
    return values.reshape(4, 4).T


def _load_frame_full_model_target(
    sample: dict[str, Any],
    *,
    sample_count: int,
) -> np.ndarray:
    full_points = _read_ply_xyz(_ensure_full_model_ply())
    sample_id = str(sample["sample_id"])
    sampled = _sample_random(
        full_points,
        sample_count,
        f"full-model-target:{sample_id}:{sample_count}",
    )
    metadata = dict(sample["metadata"])
    pose_path = metadata.get("target_pose_path") or metadata.get("reference_pose_path")
    frame_pose = _load_pose_matrix(str(pose_path), int(sample["frame_id"]))
    return _apply_transform(sampled, frame_pose)


def _load_raw_full_model_target(
    sample: dict[str, Any],
    *,
    sample_count: int,
) -> np.ndarray:
    full_points = _read_ply_xyz(_ensure_full_model_ply())
    return _sample_random(
        full_points,
        sample_count,
        f"raw-full-model-target:{sample['sample_id']}:{sample_count}",
    )


def _load_full_scene_target(
    sample: dict[str, Any],
    *,
    sample_count: int,
) -> np.ndarray:
    metadata = dict(sample["metadata"])
    reference_path = metadata.get("reference_path")
    if reference_path is None:
        raise KeyError(f"Sample {sample['sample_id']} has no reference_path.")
    full_scene_points = _read_ply_xyz(Path(str(reference_path)))
    return _sample_random(
        full_scene_points,
        sample_count,
        f"full-scene-target:{sample['sample_id']}:{sample_count}",
    )


def _load_full_model_to_scene_transform(sample: dict[str, Any]) -> np.ndarray:
    if not FULL_MODEL_SCENE_TRANSFORMS.exists():
        raise FileNotFoundError(
            "Missing recovered full-model scene transforms: "
            f"{FULL_MODEL_SCENE_TRANSFORMS}"
        )
    scene_id = str(sample["scene_id"])
    transform_data = _load_json(FULL_MODEL_SCENE_TRANSFORMS)
    for row in transform_data.get("scenes", []):
        if str(row.get("scene")) == scene_id:
            return np.asarray(row["full_model_to_coverage_transform"], dtype=np.float64)
    raise KeyError(f"Missing full-model transform for scene {scene_id!r}.")


def _load_scene_aligned_full_model_target(
    sample: dict[str, Any],
    *,
    sample_count: int,
) -> np.ndarray:
    full_points = _read_ply_xyz(_ensure_full_model_ply())
    sampled = _sample_random(
        full_points,
        sample_count,
        f"scene-aligned-full-model-target:{sample['sample_id']}:{sample_count}",
    )
    transform = _load_full_model_to_scene_transform(sample)
    return _apply_transform(sampled, transform)


def _full_model_gt_transform(sample: dict[str, Any]) -> np.ndarray:
    metadata = dict(sample["metadata"])
    pose_path = metadata.get("target_pose_path") or metadata.get("reference_pose_path")
    frame_pose = _load_pose_matrix(str(pose_path), int(sample["frame_id"]))
    frame_gt = np.asarray(sample["gt_transform"], dtype=np.float64)
    return np.linalg.inv(frame_pose) @ frame_gt


def _instantiate_geotransformer(config: dict[str, Any]) -> GeoTransformerAdapter:
    adapter_args = dict(config["model"].get("overrides", {}))
    adapter_args.setdefault("device", config["runtime"]["device"])
    adapter_args["checkpoint_path"] = config["model"].get("checkpoint_path")
    adapter = GeoTransformerAdapter(SimpleNamespace(**adapter_args))
    adapter.load_model(config["model"].get("checkpoint_path"))
    return adapter


def _target_context_points(
    target: np.ndarray,
    focus_points: np.ndarray,
    *,
    sample_count: int,
) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64)
    focus_points = np.asarray(focus_points, dtype=np.float64)
    if len(target) <= sample_count:
        return target
    center = np.median(focus_points, axis=0)
    distances = np.linalg.norm(target - center, axis=1)
    indices = np.argpartition(distances, sample_count - 1)[:sample_count]
    return target[indices]


def _projected_pair_score(
    source_points: np.ndarray,
    target_points: np.ndarray,
    camera_direction: np.ndarray,
) -> float:
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    if len(source_points) == 0 or len(target_points) == 0:
        return 0.0
    right, up = _camera_basis_from_direction(camera_direction)
    source_2d = np.stack([source_points @ right, source_points @ up], axis=1)
    target_2d = np.stack([target_points @ right, target_points @ up], axis=1)
    source_center = np.median(source_2d, axis=0)
    target_center = np.median(target_2d, axis=0)
    center_gap = float(np.linalg.norm(source_center - target_center))
    source_span = np.ptp(source_2d, axis=0)
    target_span = np.ptp(target_2d, axis=0)
    source_area = float(np.sqrt(max(source_span[0] * source_span[1], 0.0)))
    target_area = float(np.sqrt(max(target_span[0] * target_span[1], 0.0)))
    return center_gap + 0.12 * source_area + 0.08 * target_area


def _choose_pair_separation_view(
    pairs: list[tuple[np.ndarray, np.ndarray, float]],
) -> tuple[float, float]:
    best_score = -np.inf
    best_direction = None
    for candidate in _fibonacci_sphere_directions(96):
        score = 0.0
        for source_points, target_points, weight in pairs:
            score += float(weight) * _projected_pair_score(
                source_points,
                target_points,
                candidate,
            )
        if score > best_score:
            best_score = score
            best_direction = candidate
    if best_direction is None:
        return _camera_direction_to_mpl_view(np.array([1.55, 1.18, 0.92]))
    return _camera_direction_to_mpl_view(best_direction)


def _render_alignment_row(
    *,
    source: np.ndarray,
    target: np.ndarray,
    pred_transform: np.ndarray,
    gt_transform: np.ndarray,
    point_unit: str,
    row_title: str,
    output_base: Path,
    target_kind: str,
    source_seed_key: str | None = None,
    standardize_source_panels: bool = False,
    initial_view: tuple[float, float] | None = None,
    alignment_view: tuple[float, float] | None = None,
    alignment_zoom: float | None = None,
    show_row_title: bool = True,
    show_residual_cdf: bool = True,
    show_error_colorbar: bool = True,
    figure_size_in: tuple[float, float] | None = None,
) -> dict[str, Any]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    pred_transform = np.asarray(pred_transform, dtype=np.float64)
    gt_transform = np.asarray(gt_transform, dtype=np.float64)
    mm_scale = point_unit_to_mm_scale(canonicalize_point_unit(point_unit))

    source_small = _sample_random(
        source,
        PANEL_SOURCE_SAMPLE_COUNT,
        f"{source_seed_key or output_base.name}:source:{len(source)}",
    )
    target_overview_count = (
        PANEL_FULL_MODEL_INITIAL_SAMPLE_COUNT
        if target_kind == "ct_frame"
        else PANEL_TARGET_SAMPLE_COUNT
    )
    target_overview = _sample_random(
        target,
        target_overview_count,
        f"{output_base.name}:target-plot:{len(target)}",
    )
    target_metric = target
    pred_source = _apply_transform(source_small, pred_transform)
    gt_source = _apply_transform(source_small, gt_transform)
    target_pred_context = _target_context_points(
        target_metric,
        pred_source,
        sample_count=PANEL_TARGET_SAMPLE_COUNT,
    )
    target_gt_context = _target_context_points(
        target_metric,
        gt_source,
        sample_count=PANEL_TARGET_SAMPLE_COUNT,
    )
    distances_mm = cKDTree(target_metric).query(pred_source, workers=-1)[0] * mm_scale
    ordered = np.sort(distances_mm[np.isfinite(distances_mm)])
    pose = compute_pose_metrics(
        pred_transform=pred_transform,
        gt_transform=gt_transform,
        source_points=source_small,
        point_unit=point_unit,
    )

    if ordered.size:
        color_max = max(float(np.percentile(ordered, 95.0)), 3.0)
        x_max = max(float(np.percentile(ordered, 98.0)), 10.0)
    else:
        color_max = 10.0
        x_max = 10.0
    color_max = min(color_max, 80.0)
    camera_direction = _choose_error_view(pred_source, distances_mm)
    elev, azim = _camera_direction_to_mpl_view(camera_direction)
    if standardize_source_panels:
        initial_elev, initial_azim = initial_view or _choose_pair_separation_view(
            [(source_small, target_overview, 1.0)]
        )
        align_elev, align_azim = alignment_view or _choose_pair_separation_view(
            [
                (pred_source, target_pred_context, 1.0),
                (gt_source, target_gt_context, 0.75),
            ]
        )
    else:
        initial_elev, initial_azim = elev, azim
        align_elev, align_azim = elev, azim

    fig = plt.figure(
        figsize=figure_size_in or (ALIGNMENT_ROW_WIDTH_IN, ALIGNMENT_ROW_HEIGHT_IN),
        dpi=190,
    )
    if show_residual_cdf:
        axes = [
            fig.add_axes([0.000, 0.075, 0.255, 0.785], projection="3d"),
            fig.add_axes([0.312, 0.075, 0.245, 0.785], projection="3d"),
            fig.add_axes([0.590, 0.085, 0.205, 0.775], projection="3d"),
        ]
        hist_ax = fig.add_axes([0.833, 0.185, 0.160, 0.675])
    else:
        axes = [
            fig.add_axes([0.008, 0.075, 0.310, 0.805], projection="3d"),
            fig.add_axes([0.345, 0.075, 0.310, 0.805], projection="3d"),
            fig.add_axes([0.682, 0.075, 0.310, 0.805], projection="3d"),
        ]
        hist_ax = None
    colorbar_ax = (
        fig.add_axes([0.280, 0.275, 0.010, 0.475])
        if show_error_colorbar
        else None
    )

    initial_target = target_overview
    axes[0].scatter(
        initial_target[:, 0],
        initial_target[:, 1],
        initial_target[:, 2],
        s=1.8 if target_kind == "ct_frame" else 5.8,
        c="#b8c0cc",
        alpha=0.18 if target_kind == "ct_frame" else 0.42,
    )
    axes[0].scatter(
        source_small[:, 0],
        source_small[:, 1],
        source_small[:, 2],
        s=7.0,
        c="#2563a9",
        alpha=0.82,
    )

    axes[1].scatter(
        target_pred_context[:, 0],
        target_pred_context[:, 1],
        target_pred_context[:, 2],
        s=4.2,
        c="#c9ced6",
        alpha=0.22 if target_kind == "ct_frame" else 0.28,
    )
    rendered = axes[1].scatter(
        pred_source[:, 0],
        pred_source[:, 1],
        pred_source[:, 2],
        s=7.4,
        c=distances_mm,
        cmap="inferno",
        vmin=0.0,
        vmax=color_max,
        alpha=0.9,
    )

    axes[2].scatter(
        target_gt_context[:, 0],
        target_gt_context[:, 1],
        target_gt_context[:, 2],
        s=4.2,
        c="#b8c0cc",
        alpha=0.24 if target_kind == "ct_frame" else 0.35,
    )
    axes[2].scatter(
        gt_source[:, 0],
        gt_source[:, 1],
        gt_source[:, 2],
        s=7.0,
        c="#1f8a5b",
        alpha=0.82,
    )

    titles = ("Initial", "Prediction error", "Ground truth")
    for index, (ax, title) in enumerate(zip(axes, titles, strict=True)):
        if index == 0:
            if standardize_source_panels:
                view_points = np.vstack([source_small, initial_target])
                zoom = 1.02
                clip_percentiles = (
                    (0.5, 99.5) if target_kind == "ct_frame" else (1.0, 99.0)
                )
                equal_axes = True
                view_elev, view_azim = initial_elev, initial_azim
            else:
                view_points = np.vstack([source_small, target_overview])
                zoom = 1.0
                clip_percentiles = (0.0, 100.0)
                equal_axes = False
                view_elev, view_azim = elev, azim
        elif index == 1:
            view_points = (
                np.vstack([pred_source, target_pred_context])
                if standardize_source_panels
                else pred_source
            )
            zoom = (
                alignment_zoom
                if standardize_source_panels and alignment_zoom is not None
                else 0.92
                if standardize_source_panels
                else 1.25
                if target_kind == "ct_frame"
                else 0.78
            )
            clip_percentiles = (2.0, 98.0)
            equal_axes = True
            view_elev, view_azim = (
                (align_elev, align_azim) if standardize_source_panels else (elev, azim)
            )
        else:
            view_points = (
                np.vstack([gt_source, target_gt_context])
                if standardize_source_panels
                else gt_source
            )
            zoom = (
                alignment_zoom
                if standardize_source_panels and alignment_zoom is not None
                else 0.78
                if standardize_source_panels
                else 1.25
                if target_kind == "ct_frame"
                else 0.78
            )
            clip_percentiles = (2.0, 98.0)
            equal_axes = True
            view_elev, view_azim = (
                (align_elev, align_azim) if standardize_source_panels else (elev, azim)
            )
        _style_axis(
            ax,
            view_points,
            view_elev,
            view_azim,
            zoom=zoom,
            clip_percentiles=clip_percentiles,
            equal_axes=equal_axes,
        )
        ax.set_title(title, fontsize=24.0, pad=1.0)

    if colorbar_ax is not None:
        colorbar = fig.colorbar(rendered, cax=colorbar_ax)
        colorbar.ax.set_ylabel("")
        colorbar.ax.yaxis.set_ticks_position("left")
        colorbar.ax.tick_params(labelsize=16.0, length=2, pad=1)

    if hist_ax is not None:
        if ordered.size:
            cdf = np.linspace(0.0, 1.0, ordered.size)
            hist_ax.plot(ordered, cdf, color="#222222", linewidth=1.8)
            hist_ax.axvline(5.0, color="#2f6fbb", linewidth=1.0, linestyle="--")
            hist_ax.axvline(10.0, color="#bf6f2a", linewidth=1.0, linestyle="--")
            hist_ax.set_xlim(0.0, x_max)
            hist_ax.set_ylim(0.0, 1.0)
        hist_ax.grid(alpha=0.22, linewidth=0.8)
        hist_ax.set_title("Residual CDF", fontsize=24.0, pad=2)
        hist_ax.set_xlabel("NN distance (mm)", fontsize=20.0, labelpad=0)
        hist_ax.set_ylabel("")
        hist_ax.tick_params(axis="both", labelsize=18.0, pad=1)
        nn_p90_text = (
            float(np.percentile(ordered, 90.0)) if ordered.size else float("nan")
        )
        hist_ax.text(
            0.02,
            0.05,
            f"RRE {float(pose['rre_deg']):.1f} deg\n"
            f"RTE {float(pose['rte_mm']):.1f} mm\n"
            f"NN p90 {nn_p90_text:.2f} mm",
            transform=hist_ax.transAxes,
            fontsize=15.8,
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.86},
        )
    if show_row_title:
        fig.text(0.010, 0.020, row_title, fontsize=11.5, ha="left", va="bottom")

    output_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_base.with_suffix(".pdf")
    png_path = output_base.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=190)
    plt.close(fig)
    return {
        "row_title": row_title,
        "target_kind": target_kind,
        "panel_pdf": str(pdf_path.relative_to(REPO_ROOT)),
        "panel_png": str(png_path.relative_to(REPO_ROOT)),
        "rre_deg": float(pose["rre_deg"]),
        "rte_mm": float(pose["rte_mm"]),
        "nn_mean_mm": float(np.mean(ordered)) if ordered.size else float("nan"),
        "nn_p90_mm": float(np.percentile(ordered, 90.0))
        if ordered.size
        else float("nan"),
        "source_points": int(len(source)),
        "target_points": int(len(target)),
        "source_panels_standardized": bool(standardize_source_panels),
    }


def _clear_bmvc_row_pdfs(prefix: str) -> None:
    BMVC_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for path in BMVC_IMAGE_DIR.glob(f"{prefix}_row_*.pdf"):
        path.unlink()


def _copy_bmvc_row_pdf(row_pdf: Path, output_name: str) -> str:
    BMVC_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = BMVC_IMAGE_DIR / output_name
    shutil.copy2(row_pdf, output_path)
    return str(output_path.relative_to(REPO_ROOT))


def _stack_row_pdfs(
    *,
    pdf_paths: list[Path],
    output_pdf: Path,
    row_width_in: float = ALIGNMENT_ROW_WIDTH_IN,
    row_height_in: float = ALIGNMENT_ROW_HEIGHT_IN,
) -> None:
    if not pdf_paths:
        raise ValueError("No row PDFs to stack.")
    if shutil.which("pdflatex") is None:
        raise RuntimeError("pdflatex is required to stack vector row PDFs.")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    page_height = row_height_in * len(pdf_paths)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        include_lines: list[str] = []
        for index, pdf_path in enumerate(pdf_paths, start=1):
            row_name = f"row_{index:02d}.pdf"
            shutil.copy2(pdf_path, tmp_path / row_name)
            include_lines.extend(
                [
                    rf"\noindent\includegraphics[width=\paperwidth]{{{row_name}}}",
                    r"\par\nointerlineskip",
                ]
            )
        tex_path = tmp_path / "stacked_rows.tex"
        tex_path.write_text(
            "\n".join(
                [
                    r"\documentclass{article}",
                    rf"\usepackage[paperwidth={row_width_in}in,"
                    rf"paperheight={page_height:.4f}in,margin=0in]{{geometry}}",
                    r"\usepackage{graphicx}",
                    r"\pagestyle{empty}",
                    r"\setlength{\parindent}{0pt}",
                    r"\setlength{\parskip}{0pt}",
                    r"\begin{document}",
                    *include_lines,
                    r"\end{document}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_path.name,
            ],
            cwd=tmp_path,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        shutil.copy2(tmp_path / "stacked_rows.pdf", output_pdf)


def _stack_alignment_rows(
    *,
    pdf_paths: list[Path],
    png_paths: list[Path],
    output_base: Path,
    max_rows: int | None = None,
    row_width_in: float = ALIGNMENT_ROW_WIDTH_IN,
    row_height_in: float = ALIGNMENT_ROW_HEIGHT_IN,
    png_row_height_in: float = 2.38,
) -> None:
    if max_rows is not None:
        pdf_paths = pdf_paths[:max_rows]
        png_paths = png_paths[:max_rows]
    _stack_row_pdfs(
        pdf_paths=pdf_paths,
        output_pdf=output_base.with_suffix(".pdf"),
        row_width_in=row_width_in,
        row_height_in=row_height_in,
    )

    images = [mpimg.imread(path) for path in png_paths]
    if not images:
        raise ValueError("No row PNGs to stack.")
    fig, axes = plt.subplots(
        len(images),
        1,
        figsize=(row_width_in, png_row_height_in * len(images)),
        dpi=180,
    )
    if len(images) == 1:
        axes = [axes]
    for ax, image in zip(axes, images, strict=True):
        ax.imshow(image)
        ax.set_axis_off()
    fig.subplots_adjust(left=0.002, right=0.998, top=0.998, bottom=0.002, hspace=0.01)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    fig.savefig(png_path)
    BMVC_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        output_base.with_suffix(".pdf"),
        BMVC_IMAGE_DIR / f"{output_base.name}.pdf",
    )
    shutil.copy2(png_path, BMVC_IMAGE_DIR / png_path.name)
    plt.close(fig)


def render_geotransformer_failure_rows(_args: argparse.Namespace) -> int:
    config = _load_json(MODEL_CONFIGS["geotransformer"])
    point_unit = str(config["benchmark"].get("point_unit", "mm_like"))
    row_dir = REVISED_ROW_DIR / "geotransformer_failures"
    rows: list[dict[str, Any]] = []
    pdf_paths: list[Path] = []
    png_paths: list[Path] = []
    _clear_bmvc_row_pdfs("geotransformer_failure_cases")
    for index, sample_id in enumerate(GEOT_FAILURE_SAMPLE_IDS[:5], start=1):
        sample = _load_sample(config, sample_id)
        result = _read_result_row("geotransformer", sample_id)
        pred_transform = _read_transform("geotransformer", sample_id)
        output_base = row_dir / (
            f"{index:02d}_{_sanitize_filename(sample_id)}_geotransformer_failure"
        )
        row = _render_alignment_row(
            source=np.asarray(sample["source_points"], dtype=np.float64),
            target=np.asarray(sample["target_points"], dtype=np.float64),
            pred_transform=pred_transform,
            gt_transform=np.asarray(sample["gt_transform"], dtype=np.float64),
            point_unit=point_unit,
            row_title=(
                f"GeoTransformer failure {index}: {sample_id} "
                f"(RRE={float(result['rre_deg']):.1f} deg, "
                f"RTE={float(result['rte_mm']):.1f} mm)"
            ),
            output_base=output_base,
            target_kind="frame_wise",
            show_row_title=False,
        )
        row["sample_id"] = sample_id
        row["scene_id"] = str(sample["scene_id"])
        row_pdf = output_base.with_suffix(".pdf")
        row["bmvc_panel_pdf"] = _copy_bmvc_row_pdf(
            row_pdf,
            f"geotransformer_failure_cases_row_{index:02d}.pdf",
        )
        rows.append(row)
        pdf_paths.append(row_pdf)
        png_paths.append(output_base.with_suffix(".png"))

    _stack_alignment_rows(
        pdf_paths=pdf_paths,
        png_paths=png_paths,
        output_base=PAPER_FIGURE_DIR / "geotransformer_failure_cases",
    )
    _write_json(
        PAPER_FIGURE_DIR / "geotransformer_failure_cases_manifest.json",
        {"rows": rows},
    )
    print(
        str(
            (PAPER_FIGURE_DIR / "geotransformer_failure_cases.pdf").relative_to(
                REPO_ROOT
            )
        )
    )
    return 0


def render_local_to_global_row_comparison(_args: argparse.Namespace) -> int:
    config = _load_json(MODEL_CONFIGS["geotransformer"])
    point_unit = str(config["benchmark"].get("point_unit", "mm_like"))
    adapter = _instantiate_geotransformer(config)
    row_dir = REVISED_ROW_DIR / "local_to_global_comparison"
    rows: list[dict[str, Any]] = []
    pdf_paths: list[Path] = []
    png_paths: list[Path] = []
    transform_rows: list[dict[str, Any]] = []
    _clear_bmvc_row_pdfs("local_to_global_vs_raycast_target")

    for sample_index, sample_id in enumerate(
        LOCAL_GLOBAL_COMPARISON_SAMPLE_IDS, start=1
    ):
        sample = _load_sample(config, sample_id)
        source = np.asarray(sample["source_points"], dtype=np.float64)
        gt_transform = np.asarray(sample["gt_transform"], dtype=np.float64)
        source_seed_key = f"{sample_id}:local-to-global-comparison"

        ct_target = _load_scene_aligned_full_model_target(
            sample,
            sample_count=CT_TARGET_SAMPLE_COUNT,
        )
        ct_gt_transform = gt_transform
        frame_target = np.asarray(sample["target_points"], dtype=np.float64)
        ct_pred = _prediction_to_transform(adapter.predict(source, ct_target))
        frame_pred = _read_transform("geotransformer", sample_id)
        view_source = _sample_random(
            source,
            PANEL_SOURCE_SAMPLE_COUNT,
            f"{source_seed_key}:source:{len(source)}",
        )
        view_gt_source = _apply_transform(view_source, gt_transform)
        view_ct_pred_source = _apply_transform(view_source, ct_pred)
        view_frame_pred_source = _apply_transform(view_source, frame_pred)
        view_ct_gt_target = _target_context_points(
            ct_target,
            view_gt_source,
            sample_count=PANEL_TARGET_SAMPLE_COUNT,
        )
        view_frame_gt_target = _target_context_points(
            frame_target,
            view_gt_source,
            sample_count=PANEL_TARGET_SAMPLE_COUNT,
        )
        view_ct_initial_target = _sample_random(
            ct_target,
            PANEL_FULL_MODEL_INITIAL_SAMPLE_COUNT,
            f"{source_seed_key}:ct-initial-target:{len(ct_target)}",
        )
        view_frame_initial_target = _sample_random(
            frame_target,
            PANEL_TARGET_SAMPLE_COUNT,
            f"{source_seed_key}:frame-initial-target:{len(frame_target)}",
        )
        view_ct_pred_target = _target_context_points(
            ct_target,
            view_ct_pred_source,
            sample_count=PANEL_TARGET_SAMPLE_COUNT,
        )
        view_frame_pred_target = _target_context_points(
            frame_target,
            view_frame_pred_source,
            sample_count=PANEL_TARGET_SAMPLE_COUNT,
        )
        initial_view = _choose_pair_separation_view(
            [
                (view_source, view_ct_initial_target, 1.0),
                (view_source, view_frame_initial_target, 1.0),
            ]
        )
        alignment_view = _choose_pair_separation_view(
            [
                (view_ct_pred_source, view_ct_pred_target, 1.0),
                (view_frame_pred_source, view_frame_pred_target, 1.0),
                (view_gt_source, view_ct_gt_target, 0.65),
                (view_gt_source, view_frame_gt_target, 0.65),
            ]
        )
        alignment_view = LOCAL_GLOBAL_ALIGNMENT_VIEW_OVERRIDES.get(
            sample_id,
            alignment_view,
        )
        alignment_zoom = LOCAL_GLOBAL_ALIGNMENT_ZOOM_OVERRIDES.get(sample_id)
        ct_output = row_dir / (
            f"{sample_index:02d}_{_sanitize_filename(sample_id)}_ct_frame"
        )
        ct_row = _render_alignment_row(
            source=source,
            target=ct_target,
            pred_transform=ct_pred,
            gt_transform=ct_gt_transform,
            point_unit=point_unit,
            row_title=(
                f"Scene-aligned full model: {sample_id} "
                f"(source 8192 -> full model {CT_TARGET_SAMPLE_COUNT})"
            ),
            output_base=ct_output,
            target_kind="ct_frame",
            source_seed_key=source_seed_key,
            standardize_source_panels=True,
            initial_view=initial_view,
            alignment_view=alignment_view,
            alignment_zoom=alignment_zoom,
            show_row_title=False,
            show_residual_cdf=False,
            show_error_colorbar=False,
            figure_size_in=(LOCAL_GLOBAL_ROW_WIDTH_IN, LOCAL_GLOBAL_ROW_HEIGHT_IN),
        )
        ct_row["sample_id"] = sample_id
        ct_row["setting"] = "scene_aligned_full_model"
        ct_pdf = ct_output.with_suffix(".pdf")
        ct_row["bmvc_panel_pdf"] = _copy_bmvc_row_pdf(
            ct_pdf,
            (
                f"local_to_global_vs_raycast_target_row_{len(pdf_paths) + 1:02d}"
                "_scene_aligned_full_model.pdf"
            ),
        )
        rows.append(ct_row)
        pdf_paths.append(ct_pdf)
        png_paths.append(ct_output.with_suffix(".png"))
        transform_rows.append(
            {
                "sample_id": sample_id,
                "setting": "scene_aligned_full_model",
                "target_points": CT_TARGET_SAMPLE_COUNT,
                "pred_transform": ct_pred.tolist(),
                "gt_transform": ct_gt_transform.tolist(),
                "rre_deg": ct_row["rre_deg"],
                "rte_mm": ct_row["rte_mm"],
                "nn_p90_mm": ct_row["nn_p90_mm"],
            }
        )

        frame_output = row_dir / (
            f"{sample_index:02d}_{_sanitize_filename(sample_id)}_frame_wise"
        )
        frame_row = _render_alignment_row(
            source=source,
            target=frame_target,
            pred_transform=frame_pred,
            gt_transform=gt_transform,
            point_unit=point_unit,
            row_title=(
                f"Frame-wise/raycast target: {sample_id} "
                "(source 8192 -> target 8192)"
            ),
            output_base=frame_output,
            target_kind="frame_wise",
            source_seed_key=source_seed_key,
            standardize_source_panels=True,
            initial_view=initial_view,
            alignment_view=alignment_view,
            alignment_zoom=alignment_zoom,
            show_row_title=False,
            show_residual_cdf=False,
            show_error_colorbar=False,
            figure_size_in=(LOCAL_GLOBAL_ROW_WIDTH_IN, LOCAL_GLOBAL_ROW_HEIGHT_IN),
        )
        frame_row["sample_id"] = sample_id
        frame_row["setting"] = "frame_wise_raycast"
        frame_pdf = frame_output.with_suffix(".pdf")
        frame_row["bmvc_panel_pdf"] = _copy_bmvc_row_pdf(
            frame_pdf,
            (
                f"local_to_global_vs_raycast_target_row_{len(pdf_paths) + 1:02d}"
                "_frame_wise_raycast.pdf"
            ),
        )
        rows.append(frame_row)
        pdf_paths.append(frame_pdf)
        png_paths.append(frame_output.with_suffix(".png"))
        transform_rows.append(
            {
                "sample_id": sample_id,
                "setting": "frame_wise_raycast",
                "target_points": int(len(frame_target)),
                "pred_transform": frame_pred.tolist(),
                "gt_transform": gt_transform.tolist(),
                "rre_deg": frame_row["rre_deg"],
                "rte_mm": frame_row["rte_mm"],
                "nn_p90_mm": frame_row["nn_p90_mm"],
            }
        )

    _stack_alignment_rows(
        pdf_paths=pdf_paths,
        png_paths=png_paths,
        output_base=PAPER_FIGURE_DIR / "local_to_global_vs_raycast_target",
        row_width_in=LOCAL_GLOBAL_ROW_WIDTH_IN,
        row_height_in=LOCAL_GLOBAL_ROW_HEIGHT_IN,
        png_row_height_in=LOCAL_GLOBAL_ROW_HEIGHT_IN,
    )
    _write_json(
        PAPER_FIGURE_DIR / "local_to_global_vs_raycast_target.json",
        {
            "scene_aligned_full_model_target_points": CT_TARGET_SAMPLE_COUNT,
            "full_model_ply": str(FULL_MODEL_PLY),
            "full_model_scene_transforms": str(FULL_MODEL_SCENE_TRANSFORMS),
            "rows": rows,
            "transform_rows": transform_rows,
        },
    )
    print(
        str(
            (PAPER_FIGURE_DIR / "local_to_global_vs_raycast_target.pdf").relative_to(
                REPO_ROOT
            )
        )
    )
    return 0


def render_local_to_global_figure(_args: argparse.Namespace) -> int:
    config = _load_json(MODEL_CONFIGS["geotransformer"])
    sample = _load_sample(config, CONSTRUCTION_SAMPLE_ID)
    source = np.asarray(sample["source_points"], dtype=np.float64)
    raycast_target = np.asarray(sample["target_points"], dtype=np.float64)
    metadata = dict(sample["metadata"])
    reference_path = Path(str(metadata["reference_path"]))
    full_target = _read_ply_xyz(reference_path)
    full_target = _sample_evenly(full_target, 12000)
    source_small = _sample_evenly(source, 2048)
    raycast_small = _sample_evenly(raycast_target, 2048)
    pred_transform = _read_transform("geotransformer", CONSTRUCTION_SAMPLE_ID)
    pred_source = _apply_transform(source_small, pred_transform)
    distances_mm = cKDTree(raycast_small).query(pred_source)[
        0
    ] * point_unit_to_mm_scale(
        canonicalize_point_unit(str(config["benchmark"].get("point_unit", "mm_like")))
    )
    color_max = max(5.0, min(25.0, float(np.percentile(distances_mm, 96.0))))
    local_direction = _choose_error_view(pred_source, distances_mm)
    local_elev, local_azim = _camera_direction_to_mpl_view(local_direction)
    full_elev, full_azim = 21.0, -42.0
    result = _read_result_row("geotransformer", CONSTRUCTION_SAMPLE_ID)

    fig = plt.figure(figsize=(13.4, 3.95), dpi=210)
    grid = fig.add_gridspec(1, 4, wspace=0.03)
    axes = [fig.add_subplot(grid[0, idx], projection="3d") for idx in range(4)]

    axes[0].scatter(
        source_small[:, 0],
        source_small[:, 1],
        source_small[:, 2],
        s=7.5,
        c="#2563a9",
        alpha=0.86,
    )
    _style_axis(
        axes[0],
        source_small,
        local_elev,
        local_azim,
        zoom=0.75,
        clip_percentiles=(2.0, 98.0),
    )
    axes[0].set_title("(a) Single-frame source", fontsize=15.5, pad=2)

    axes[1].scatter(
        full_target[:, 0],
        full_target[:, 1],
        full_target[:, 2],
        s=1.0,
        c="#c5cbd3",
        alpha=0.18,
    )
    axes[1].scatter(
        source_small[:, 0],
        source_small[:, 1],
        source_small[:, 2],
        s=5.0,
        c="#2563a9",
        alpha=0.82,
    )
    _style_axis(
        axes[1],
        np.vstack([full_target, source_small]),
        full_elev,
        full_azim,
        zoom=1.0,
        clip_percentiles=(0.0, 100.0),
    )
    axes[1].set_title("(b) Full CT mesh target", fontsize=15.5, pad=2)

    axes[2].scatter(
        raycast_small[:, 0],
        raycast_small[:, 1],
        raycast_small[:, 2],
        s=5.0,
        c="#b8c0cc",
        alpha=0.42,
    )
    axes[2].scatter(
        source_small[:, 0],
        source_small[:, 1],
        source_small[:, 2],
        s=7.0,
        c="#2563a9",
        alpha=0.82,
    )
    _style_axis(
        axes[2],
        np.vstack([raycast_small, source_small]),
        local_elev,
        local_azim,
        zoom=0.85,
        clip_percentiles=(2.0, 98.0),
    )
    axes[2].set_title("(c) Raycast visible target", fontsize=15.5, pad=2)

    axes[3].scatter(
        raycast_small[:, 0],
        raycast_small[:, 1],
        raycast_small[:, 2],
        s=4.2,
        c="#c5cbd3",
        alpha=0.35,
    )
    rendered = axes[3].scatter(
        pred_source[:, 0],
        pred_source[:, 1],
        pred_source[:, 2],
        s=6.5,
        c=distances_mm,
        cmap="inferno",
        vmin=0.0,
        vmax=color_max,
        alpha=0.9,
    )
    _style_axis(
        axes[3],
        np.vstack([raycast_small, pred_source]),
        local_elev,
        local_azim,
        zoom=0.78,
        clip_percentiles=(3.0, 97.0),
    )
    axes[3].set_title(
        "(d) GeoT on raycast target\n"
        f"{float(result['rre_deg']):.1f}deg / {float(result['rte_mm']):.1f}mm",
        fontsize=15.0,
        pad=2,
    )

    cax = fig.add_axes([0.925, 0.22, 0.012, 0.54])
    colorbar = fig.colorbar(rendered, cax=cax)
    colorbar.set_label("NN distance (mm)", fontsize=11.5, labelpad=3)
    colorbar.ax.tick_params(labelsize=10.5, length=2, pad=1)
    fig.subplots_adjust(left=0.004, right=0.910, top=0.95, bottom=0.02)

    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    BMVC_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        out_path = PAPER_FIGURE_DIR / f"local_to_global_vs_raycast_target.{suffix}"
        fig.savefig(out_path)
        shutil.copy2(out_path, BMVC_IMAGE_DIR / out_path.name)
    plt.close(fig)
    _write_json(
        PAPER_FIGURE_DIR / "local_to_global_vs_raycast_target.json",
        {
            "sample_id": CONSTRUCTION_SAMPLE_ID,
            "reference_path": str(reference_path),
            "geotransformer_rre_deg": float(result["rre_deg"]),
            "geotransformer_rte_mm": float(result["rte_mm"]),
            "output_pdf": (
                "outputs/benchmark/figures/paper/"
                "local_to_global_vs_raycast_target.pdf"
            ),
        },
    )
    print(
        str(
            (PAPER_FIGURE_DIR / "local_to_global_vs_raycast_target.pdf").relative_to(
                REPO_ROOT
            )
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-local-to-global")
    prepare.add_argument("--per-scene", type=int, default=30)
    subparsers.add_parser("summarize-local-to-global")
    subparsers.add_parser("render-geotransformer-figure4")
    subparsers.add_parser("render-geotransformer-failure-rows")
    subparsers.add_parser("render-local-to-global-figure")
    subparsers.add_parser("render-local-to-global-row-comparison")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare-local-to-global":
        return prepare_local_to_global(args)
    if args.command == "summarize-local-to-global":
        return summarize_local_to_global(args)
    if args.command == "render-geotransformer-figure4":
        return render_geotransformer_failure_rows(args)
    if args.command == "render-geotransformer-failure-rows":
        return render_geotransformer_failure_rows(args)
    if args.command == "render-local-to-global-figure":
        return render_local_to_global_row_comparison(args)
    if args.command == "render-local-to-global-row-comparison":
        return render_local_to_global_row_comparison(args)
    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
