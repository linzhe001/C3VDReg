#!/usr/bin/env python3
# ruff: noqa: E402
"""Build combined tables and figures for the R25-90/T100-500mm protocol."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ROOT = REPO_ROOT / "outputs" / "benchmark" / "r25_90_t100_500mm_protocol"
FIGURE_ROOT = PROTOCOL_ROOT / "figures"
PAPER_IMAGE_ROOT = REPO_ROOT / "BMVC2026_Linzhe" / "images"
DIFFICULTY_SOURCE = (
    REPO_ROOT
    / "outputs"
    / "benchmark"
    / "r90_t500mm_protocol"
    / "error_analysis"
    / "independent_geometry_difficulty.csv"
)


@dataclass(frozen=True)
class RunSpec:
    model_key: str
    display_name: str
    eval_dir: Path


RUN_SPECS = (
    RunSpec(
        "geotransformer",
        "GeoTransformer",
        PROTOCOL_ROOT / "geotransformer" / "eval_test",
    ),
    RunSpec("regtr", "RegTR", PROTOCOL_ROOT / "regtr" / "eval_test"),
    RunSpec(
        "mamba2_direct",
        "PointNetLK-Mamba",
        PROTOCOL_ROOT / "mamba2_direct" / "eval_test",
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
    RunSpec("dcp", "DCP", PROTOCOL_ROOT / "dcp" / "eval_test"),
    RunSpec("icp", "ICP", PROTOCOL_ROOT / "icp" / "eval_test"),
)

MODEL_COLORS = {
    "geotransformer": "#2563eb",
    "regtr": "#c2410c",
    "mamba2_direct": "#047857",
    "pointnetlk_revisited": "#7c3aed",
    "pointnetlk": "#64748b",
    "dcp": "#be123c",
    "icp": "#111827",
}

BUCKET_LABELS = ["Easy", "Medium", "Hard"]
MODEL_LABELS = {
    "geotransformer": "GeoT",
    "regtr": "RegTR",
    "mamba2_direct": "PNLK-M",
    "pointnetlk_revisited": "PNLK-R",
    "pointnetlk": "PNLK",
    "dcp": "DCP",
    "icp": "ICP",
}

RR_CMAP = LinearSegmentedColormap.from_list(
    "rr_cmap",
    ["#f8fafc", "#dff7ed", "#69c39b", "#047857"],
)
ERROR_CMAP = LinearSegmentedColormap.from_list(
    "error_cmap",
    ["#f8fafc", "#fee8c8", "#f59e0b", "#b45309"],
)
TAG_CMAP = LinearSegmentedColormap.from_list(
    "tag_cmap",
    ["#f8fafc", "#e0e7ff", "#818cf8", "#4338ca"],
)


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _short_scene_label(scene_id: str) -> str:
    scene_labels = {
        "cecum_t1_a": "Cecum 1",
        "cecum_t3_a": "Cecum 2",
        "sigmoid_t3_b": "Sigmoid 1",
        "trans_t1_a": "Trans. 1",
        "trans_t2_b": "Trans. 2",
        "trans_t4_a": "Trans. 3",
    }
    return scene_labels.get(scene_id, scene_id.replace("_", "\n"))


def _save_figure(
    fig: plt.Figure,
    filename: str,
    *,
    dpi: int = 220,
    bbox_inches: str | None = None,
    pad_inches: float = 0.1,
) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    PAPER_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, Any] = {"dpi": dpi}
    if bbox_inches is not None:
        save_kwargs.update({"bbox_inches": bbox_inches, "pad_inches": pad_inches})
    output_name = Path(filename)
    pdf_name = output_name.with_suffix(".pdf")
    fig.savefig(FIGURE_ROOT / pdf_name, **save_kwargs)
    fig.savefig(PAPER_IMAGE_ROOT / pdf_name, **save_kwargs)
    if output_name.suffix.lower() == ".png":
        fig.savefig(FIGURE_ROOT / output_name, **save_kwargs)


def _trim_plot_image(
    image: np.ndarray,
    *,
    threshold: float = 0.985,
    margin_fraction: float = 0.012,
) -> np.ndarray:
    if image.ndim != 3:
        return image
    rgb = image[..., :3]
    alpha = image[..., 3] if image.shape[2] == 4 else np.ones(image.shape[:2])
    content = (np.min(rgb, axis=2) < threshold) | (alpha < 0.99)
    rows, cols = np.where(content)
    if rows.size == 0 or cols.size == 0:
        return image
    margin_y = max(6, int(round(image.shape[0] * margin_fraction)))
    margin_x = max(6, int(round(image.shape[1] * margin_fraction)))
    top = max(0, int(rows.min()) - margin_y)
    bottom = min(image.shape[0], int(rows.max()) + margin_y + 1)
    left = max(0, int(cols.min()) - margin_x)
    right = min(image.shape[1], int(cols.max()) + margin_x + 1)
    return image[top:bottom, left:right, :]


def _add_canvas_panel_labels(
    fig: plt.Figure,
    axes: list[plt.Axes] | np.ndarray,
    labels: str,
    *,
    y_offset: float = 0.018,
) -> None:
    for ax, label in zip(np.ravel(axes), labels, strict=True):
        bbox = ax.get_position()
        fig.text(
            bbox.x0,
            bbox.y1 + y_offset,
            f"({label})",
            ha="left",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="#0f172a",
        )


def _add_canvas_panel_headers(
    fig: plt.Figure,
    axes: list[plt.Axes] | np.ndarray,
    labels: str,
    titles: list[str],
    *,
    y_offset: float = 0.010,
    title_x_offset: float = 0.036,
) -> None:
    for ax, label, title in zip(np.ravel(axes), labels, titles, strict=True):
        bbox = ax.get_position()
        y = bbox.y1 + y_offset
        fig.text(
            bbox.x0,
            y,
            f"({label})",
            ha="left",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="#0f172a",
        )
        fig.text(
            bbox.x0 + title_x_offset,
            y,
            title,
            ha="left",
            va="bottom",
            fontsize=10.2,
            fontweight="bold",
            color="#0f172a",
        )


def _add_centered_canvas_panel_headers(
    fig: plt.Figure,
    axes: list[plt.Axes] | np.ndarray,
    labels: str,
    titles: list[str],
    *,
    y_offset: float = 0.010,
    fontsize: float = 10.2,
) -> None:
    for ax, label, title in zip(np.ravel(axes), labels, titles, strict=True):
        bbox = ax.get_position()
        fig.text(
            (bbox.x0 + bbox.x1) / 2.0,
            bbox.y1 + y_offset,
            f"({label})  {title}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
            color="#0f172a",
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_csv_single(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows[0]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def _safe_mean(values: list[float]) -> float:
    clean = _finite(values)
    return float(mean(clean)) if clean else math.nan


def _safe_median(values: list[float]) -> float:
    clean = _finite(values)
    return float(median(clean)) if clean else math.nan


def _safe_p90(values: list[float]) -> float:
    clean = sorted(_finite(values))
    if not clean:
        return math.nan
    return float(clean[int(round((len(clean) - 1) * 0.9))])


def _pct(value: Any) -> float:
    return _safe_float(value) * 100.0


def _fmt(value: Any, digits: int = 2) -> str:
    numeric = _safe_float(value)
    if not math.isfinite(numeric):
        return ""
    return f"{numeric:.{digits}f}"


def _annotate_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    fmt: str,
    high_is_good: bool,
) -> None:
    finite = matrix[np.isfinite(matrix)]
    threshold = float(np.nanmax(finite)) * 0.58 if finite.size else math.inf
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            if not math.isfinite(float(value)):
                label = "-"
                color = "#64748b"
            else:
                label = fmt.format(value)
                dark_cell = value >= threshold if high_is_good else value >= threshold
                color = "white" if dark_cell else "#0f172a"
            ax.text(
                col_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=7,
                color=color,
                fontweight="bold" if label != "-" else "normal",
            )


def _load_records() -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {}
    missing: list[str] = []
    for spec in RUN_SPECS:
        path = spec.eval_dir / "results.jsonl"
        if not path.exists():
            missing.append(_rel(path))
            continue
        rows = _read_jsonl(path)
        for row in rows:
            row["_model_key"] = spec.model_key
            row["_display_name"] = spec.display_name
            row["_result_dir"] = _rel(spec.eval_dir)
        records[spec.model_key] = rows
    if missing:
        raise FileNotFoundError("Missing hard protocol results: " + ", ".join(missing))
    return records


def build_combined_rows(
    records_by_model: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in RUN_SPECS:
        leaderboard = _read_csv_single(
            spec.eval_dir / "leaderboard" / "leaderboard_main.csv"
        )
        records = records_by_model[spec.model_key]
        rre = [_safe_float(row.get("rre_deg")) for row in records]
        rte = [_safe_float(row.get("rte_mm")) for row in records]
        rows.append(
            {
                "model_key": spec.model_key,
                "display_name": spec.display_name,
                "sample_count": int(_safe_float(leaderboard.get("sample_count"))),
                "rr5_pct": _pct(
                    leaderboard.get("registration_recall@rre_5deg_rte_5mm")
                ),
                "rr10_pct": _pct(
                    leaderboard.get("registration_recall@rre_10deg_rte_10mm")
                ),
                "rot_hit_5deg_pct": _pct(leaderboard.get("rot_hit_5deg_rate")),
                "trans_hit_5mm_pct": _pct(leaderboard.get("trans_hit_5mm_rate")),
                "rot_hit_10deg_pct": _pct(leaderboard.get("rot_hit_10deg_rate")),
                "trans_hit_10mm_pct": _pct(leaderboard.get("trans_hit_10mm_rate")),
                "rre_deg_mean": _safe_float(leaderboard.get("rre_deg_mean")),
                "rre_deg_median": _safe_float(leaderboard.get("rre_deg_median")),
                "rre_deg_p90": _safe_float(leaderboard.get("rre_deg_p90")),
                "rte_mm_mean": _safe_float(leaderboard.get("rte_mm_mean")),
                "rte_mm_median": _safe_float(leaderboard.get("rte_mm_median")),
                "rte_mm_p90": _safe_float(leaderboard.get("rte_mm_p90")),
                "rre_deg_mean_recomputed": _safe_mean(rre),
                "rte_mm_mean_recomputed": _safe_mean(rte),
                "visible_nn_mean_mm": _safe_float(
                    leaderboard.get("visible_nn_mean_mm_mean")
                ),
                "trimmed_chamfer_mm_mean": _safe_float(
                    leaderboard.get("trimmed_chamfer_mm_mean")
                ),
                "trimmed_chamfer_mm_median": _safe_float(
                    leaderboard.get("trimmed_chamfer_mm_median")
                ),
                "latency_ms_mean": _safe_float(leaderboard.get("latency_ms_mean")),
                "latency_ms_p90": _safe_float(leaderboard.get("latency_ms_p90")),
                "train_time_ms": _safe_float(leaderboard.get("train_time_ms")),
                "train_peak_memory_mb": _safe_float(
                    leaderboard.get("train_peak_memory_mb")
                ),
                "result_dir": _rel(spec.eval_dir),
            }
        )
    return sorted(
        rows,
        key=lambda row: (-_safe_float(row["rr5_pct"]), row["display_name"]),
    )


def write_markdown(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# R25-90/T100-500mm Hard Protocol Summary",
        "",
        (
            "All rows use 2088 C3VD raycasting test pairs, 8192 input points, "
            "source-only rotation sampled from 25-90 degrees, translation sampled "
            "from 100-500 mm, and no added noise."
        ),
        "",
        (
            "| Model | RR@5 | RR@10 | R<=5 | T<=5 | RRE mean | RTE mean | "
            "RTE median | RTE p90 | Trim CD | Latency |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['display_name']} | {_fmt(row['rr5_pct'])} | "
            f"{_fmt(row['rr10_pct'])} | {_fmt(row['rot_hit_5deg_pct'])} | "
            f"{_fmt(row['trans_hit_5mm_pct'])} | {_fmt(row['rre_deg_mean'])} | "
            f"{_fmt(row['rte_mm_mean'])} | {_fmt(row['rte_mm_median'])} | "
            f"{_fmt(row['rte_mm_p90'])} | "
            f"{_fmt(row['trimmed_chamfer_mm_mean'])} | "
            f"{_fmt(row['latency_ms_mean'], 1)} |"
        )
    lines.extend(
        [
            "",
            (
                "Interpretation: `R<=5` and `T<=5` are marginal hit rates. "
                "`RR@5` is their joint success and is therefore much lower "
                "whenever one error family dominates."
            ),
            "",
        ]
    )
    (PROTOCOL_ROOT / "combined_leaderboard.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_latex_tables(rows: list[dict[str, Any]]) -> None:
    main_lines = [
        "% Auto-generated by scripts/benchmark/build_hard_protocol_summary.py",
        "\\begin{tabular}{lrrrrrrr}",
        "\\hline",
        (
            "Model & RR@5 & RR@10 & RRE mean & RTE mean & Visible NN & "
            "Trim CD & Latency \\\\"
        ),
        " & (\\%) & (\\%) & (deg) & (mm) & (mm) & (mm) & (ms) \\\\",
        "\\hline",
    ]
    diag_lines = [
        "% Auto-generated by scripts/benchmark/build_hard_protocol_summary.py",
        "\\begin{tabular}{lrrrrrrr}",
        "\\hline",
        (
            "Model & R$\\leq$5 & T$\\leq$5 & RR@5 & RTE med. & "
            "RTE p90 & RRE med. & RRE p90 \\\\"
        ),
        " & (\\%) & (\\%) & (\\%) & (mm) & (mm) & (deg) & (deg) \\\\",
        "\\hline",
    ]
    for row in rows:
        main_lines.append(
            f"{row['display_name']} & {_fmt(row['rr5_pct'])} & "
            f"{_fmt(row['rr10_pct'])} & {_fmt(row['rre_deg_mean'])} & "
            f"{_fmt(row['rte_mm_mean'])} & {_fmt(row['visible_nn_mean_mm'])} & "
            f"{_fmt(row['trimmed_chamfer_mm_mean'])} & "
            f"{_fmt(row['latency_ms_mean'], 1)} \\\\"
        )
        diag_lines.append(
            f"{row['display_name']} & {_fmt(row['rot_hit_5deg_pct'])} & "
            f"{_fmt(row['trans_hit_5mm_pct'])} & {_fmt(row['rr5_pct'])} & "
            f"{_fmt(row['rte_mm_median'])} & {_fmt(row['rte_mm_p90'])} & "
            f"{_fmt(row['rre_deg_median'])} & {_fmt(row['rre_deg_p90'])} \\\\"
        )
    main_lines.extend(["\\hline", "\\end{tabular}", ""])
    diag_lines.extend(["\\hline", "\\end{tabular}", ""])
    (PROTOCOL_ROOT / "latex_main_table.tex").write_text(
        "\n".join(main_lines),
        encoding="utf-8",
    )
    (PROTOCOL_ROOT / "latex_error_distribution_table.tex").write_text(
        "\n".join(diag_lines),
        encoding="utf-8",
    )


def plot_rr_bar(rows: list[dict[str, Any]]) -> None:
    labels = [
        MODEL_LABELS.get(str(row["model_key"]), str(row["display_name"]))
        for row in rows
    ]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(4.35, 3.05))
    ax.bar(
        x - width / 2,
        [row["rr5_pct"] for row in rows],
        width,
        label="RR@5deg/5mm",
        color="#2f6fbb",
    )
    ax.bar(
        x + width / 2,
        [row["rr10_pct"] for row in rows],
        width,
        label="RR@10deg/10mm",
        color="#3f8c54",
    )
    ax.set_ylabel("Registration recall (%)", fontsize=16.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=14.2)
    ax.tick_params(axis="y", labelsize=14.2)
    ax.set_ylim(0.0, max([row["rr10_pct"] for row in rows] + [1.0]) * 1.25)
    ax.legend(frameon=False, fontsize=13.4)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(pad=0.25)
    _save_figure(fig, "hard_main_rr_bar.png")
    plt.close(fig)


def _smooth_quantile_curve(
    values: np.ndarray,
    quantile_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.linspace(0.0, 1.0, quantile_count)
    x = np.quantile(values, y, method="linear")
    if quantile_count < 17:
        return x, y

    window = min(45, max(11, (quantile_count // 16) | 1))
    pad = window // 2
    kernel = np.hanning(window)
    kernel /= np.sum(kernel)
    smooth = np.convolve(np.pad(x, pad, mode="edge"), kernel, mode="valid")
    smooth[0] = x[0]
    smooth[-1] = x[-1]
    smooth = np.maximum.accumulate(smooth)
    smooth = np.minimum(smooth, x[-1])
    smooth[-1] = x[-1]
    return smooth, y


def plot_cdf(
    records_by_model: dict[str, list[dict[str, Any]]],
    metric: str,
    xlabel: str,
    filename: str,
    title: str,
    x_limit: float,
    thresholds: tuple[float, ...],
) -> None:
    fig, ax = plt.subplots(figsize=(4.15, 3.05), dpi=150)
    for spec in RUN_SPECS:
        values = sorted(
            _safe_float(row.get(metric))
            for row in records_by_model[spec.model_key]
            if math.isfinite(_safe_float(row.get(metric)))
        )
        if not values:
            continue
        values_array = np.asarray(values, dtype=float)
        quantile_count = min(420, max(140, len(values_array) // 3))
        smoothed_values, y = _smooth_quantile_curve(values_array, quantile_count)
        ax.plot(
            smoothed_values,
            y,
            label=MODEL_LABELS.get(spec.model_key, spec.display_name),
            color=MODEL_COLORS.get(spec.model_key),
            linewidth=2.15,
            alpha=0.94,
        )
    for threshold in thresholds:
        ax.axvline(threshold, color="#94a3b8", linestyle=":", linewidth=0.9)
    ax.set_xlabel(xlabel, fontsize=16.2)
    ax.set_ylabel("Empirical CDF", fontsize=16.2)
    ax.tick_params(axis="both", labelsize=14.0)
    ax.set_xlim(0.0, x_limit)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.20, linewidth=0.8)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#e2e8f0",
        fontsize=13.2,
        ncol=2,
        loc="lower right",
    )
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(0.8)
    fig.tight_layout(pad=0.25)
    _save_figure(fig, filename)
    plt.close(fig)


def plot_scene_rr_heatmap(records_by_model: dict[str, list[dict[str, Any]]]) -> None:
    scenes = sorted(
        {
            str(row.get("scene_id"))
            for records in records_by_model.values()
            for row in records
        }
    )
    matrix = np.full((len(RUN_SPECS), len(scenes)), np.nan)
    for row_index, spec in enumerate(RUN_SPECS):
        for col_index, scene in enumerate(scenes):
            records = [
                row
                for row in records_by_model[spec.model_key]
                if str(row.get("scene_id")) == scene
            ]
            matrix[row_index, col_index] = _safe_mean(
                [_safe_float(row.get("success_5deg_5mm")) for row in records]
            ) * 100.0
    fig, ax = plt.subplots(figsize=(4.80, 3.12))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0)
    ax.set_yticks(np.arange(len(RUN_SPECS)))
    ax.set_yticklabels(
        [MODEL_LABELS[spec.model_key] for spec in RUN_SPECS],
        fontsize=14.0,
    )
    ax.set_xticks(np.arange(len(scenes)))
    ax.set_xticklabels(
        [_short_scene_label(scene) for scene in scenes],
        rotation=25,
        ha="right",
        fontsize=13.6,
    )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("RR@5 (%)", fontsize=14.0)
    cbar.ax.tick_params(labelsize=13.0)
    fig.tight_layout(pad=0.25)
    _save_figure(fig, "hard_scene_bucket_rr.png")
    plt.close(fig)


def plot_main_benchmark_views() -> None:
    panels = [
        FIGURE_ROOT / "hard_main_rr_bar.png",
        FIGURE_ROOT / "hard_scene_bucket_rr.png",
        FIGURE_ROOT / "hard_rre_cdf.png",
        FIGURE_ROOT / "hard_rte_cdf.png",
    ]
    images = [_trim_plot_image(plt.imread(path)) for path in panels]
    fig, axes = plt.subplots(2, 2, figsize=(10.55, 7.05), dpi=180)
    for ax, image in zip(np.ravel(axes), images, strict=True):
        ax.imshow(image)
        ax.axis("off")
    fig.tight_layout(pad=0.15, w_pad=0.28, h_pad=0.54)
    _save_figure(
        fig,
        "hard_main_benchmark_views.png",
        dpi=240,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def _load_difficulty_lookup() -> dict[str, str]:
    if not DIFFICULTY_SOURCE.exists():
        return {}
    return {
        row["sample_id"]: row["difficulty_bucket"]
        for row in _read_csv_rows(DIFFICULTY_SOURCE)
    }


def _load_difficulty_rows() -> dict[str, dict[str, Any]]:
    if not DIFFICULTY_SOURCE.exists():
        return {}
    return {
        row["sample_id"]: {
            "difficulty_bucket": row.get("difficulty_bucket"),
            "visible_nn_proxy_mm": _safe_float(row.get("gt_visible_nn_mean_mm")),
            "trimmed_chamfer_proxy_mm": _safe_float(row.get("gt_trimmed_chamfer_mm")),
        }
        for row in _read_csv_rows(DIFFICULTY_SOURCE)
    }


def _difficulty_matrices(
    records_by_model: dict[str, list[dict[str, Any]]],
) -> tuple[list[str], np.ndarray, np.ndarray] | None:
    difficulty_lookup = _load_difficulty_lookup()
    if not difficulty_lookup:
        return None
    buckets = ["easy", "medium", "hard"]
    rr_matrix = np.full((len(RUN_SPECS), len(buckets)), np.nan)
    rte_matrix = np.full((len(RUN_SPECS), len(buckets)), np.nan)
    for row_index, spec in enumerate(RUN_SPECS):
        for col_index, bucket in enumerate(buckets):
            records = [
                row
                for row in records_by_model[spec.model_key]
                if difficulty_lookup.get(str(row.get("sample_id"))) == bucket
            ]
            strict_success = [
                _safe_float(row.get("success_5deg_5mm")) for row in records
            ]
            rr_matrix[row_index, col_index] = _safe_mean(strict_success) * 100.0
            rte_matrix[row_index, col_index] = _safe_median(
                [_safe_float(row.get("rte_mm")) for row in records]
            )
    return buckets, rr_matrix, rte_matrix


def _failure_tag_matrix(
    records_by_model: dict[str, list[dict[str, Any]]],
) -> tuple[list[str], np.ndarray]:
    tags = ["large_rotation", "large_translation", "density_mismatch"]
    matrix = np.zeros((len(RUN_SPECS), len(tags)), dtype=float)
    for row_index, spec in enumerate(RUN_SPECS):
        for col_index, tag in enumerate(tags):
            records = records_by_model[spec.model_key]
            count = sum(tag in set(row.get("failure_tags", [])) for row in records)
            matrix[row_index, col_index] = (count / len(records)) * 100.0
    tag_labels = ["Large\nrotation", "Large\ntranslation", "Density\nmismatch"]
    return tag_labels, matrix


def plot_difficulty_rr_rte(
    records_by_model: dict[str, list[dict[str, Any]]],
) -> None:
    difficulty_data = _difficulty_matrices(records_by_model)
    if difficulty_data is None:
        return
    buckets, rr_matrix, rte_matrix = difficulty_data

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), dpi=150)
    rr_image = axes[0].imshow(rr_matrix, cmap=RR_CMAP, vmin=0.0, vmax=20.0)
    rte_image = axes[1].imshow(rte_matrix, cmap=ERROR_CMAP, vmin=0.0, vmax=350.0)

    for ax, title in zip(
        axes,
        ["Strict success by difficulty", "Translation tail by difficulty"],
        strict=True,
    ):
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_xticks(np.arange(len(buckets)))
        ax.set_xticklabels(BUCKET_LABELS)
        ax.set_yticks(np.arange(len(RUN_SPECS)))
        ax.set_yticklabels([MODEL_LABELS[spec.model_key] for spec in RUN_SPECS])
        ax.tick_params(axis="both", labelsize=8)
        ax.set_xticks(np.arange(-0.5, len(buckets), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(RUN_SPECS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    _annotate_heatmap(axes[0], rr_matrix, "{:.1f}%", high_is_good=True)
    _annotate_heatmap(axes[1], rte_matrix, "{:.0f}", high_is_good=False)
    axes[0].set_xlabel("GT-aligned Visible NN tertile")
    axes[1].set_xlabel("GT-aligned Visible NN tertile")
    axes[0].set_ylabel("Model")
    fig.colorbar(rr_image, ax=axes[0], fraction=0.046, pad=0.02).set_label(
        "RR@5 (%)"
    )
    fig.colorbar(rte_image, ax=axes[1], fraction=0.046, pad=0.02).set_label(
        "Median RTE (mm)"
    )
    fig.suptitle(
        "Difficulty buckets expose success concentration and translation failures",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    _save_figure(fig, "hard_difficulty_bucket_rr_rte.png")
    plt.close(fig)


def plot_failure_tags(records_by_model: dict[str, list[dict[str, Any]]]) -> None:
    tag_labels, matrix = _failure_tag_matrix(records_by_model)
    fig, ax = plt.subplots(figsize=(7.8, 4.4), dpi=150)
    image = ax.imshow(matrix, cmap=TAG_CMAP, vmin=0.0, vmax=100.0, aspect="auto")
    ax.set_title(
        "Failure taxonomy: tags are not mutually exclusive",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xticks(np.arange(len(tag_labels)))
    ax.set_xticklabels(tag_labels)
    ax.set_yticks(np.arange(len(RUN_SPECS)))
    ax.set_yticklabels([MODEL_LABELS[spec.model_key] for spec in RUN_SPECS])
    ax.tick_params(axis="both", labelsize=8)
    ax.set_xticks(np.arange(-0.5, len(tag_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(RUN_SPECS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _annotate_heatmap(ax, matrix, "{:.0f}%", high_is_good=False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Tagged pairs (%)")
    fig.text(
        0.5,
        0.025,
        "A pair may receive multiple tags; each cell reports an independent rate.",
        fontsize=8,
        color="#475569",
        ha="center",
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.98))
    _save_figure(fig, "hard_failure_tag_distribution.png")
    plt.close(fig)


def _style_matrix_axis(
    ax: plt.Axes,
    x_labels: list[str],
    *,
    show_y_labels: bool,
) -> None:
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(len(RUN_SPECS)))
    if show_y_labels:
        ax.set_yticklabels([MODEL_LABELS[spec.model_key] for spec in RUN_SPECS])
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", left=False)
    ax.tick_params(axis="both", labelsize=7.8)
    ax.set_xticks(np.arange(-0.5, len(x_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(RUN_SPECS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.35)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_difficulty_failure_panels(
    records_by_model: dict[str, list[dict[str, Any]]],
) -> None:
    difficulty_data = _difficulty_matrices(records_by_model)
    if difficulty_data is None:
        return
    buckets, rr_matrix, rte_matrix = difficulty_data
    tag_labels, tag_matrix = _failure_tag_matrix(records_by_model)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.7, 5.10),
        dpi=180,
        gridspec_kw={"width_ratios": [1.08, 1.08, 1.0]},
    )
    images = [
        axes[0].imshow(rr_matrix, cmap=RR_CMAP, vmin=0.0, vmax=25.0),
        axes[1].imshow(rte_matrix, cmap=ERROR_CMAP, vmin=0.0, vmax=350.0),
        axes[2].imshow(tag_matrix, cmap=TAG_CMAP, vmin=0.0, vmax=100.0),
    ]
    titles = [
        "Strict success by geometry tertile",
        "Translation tail by geometry tertile",
        "Rule-based failure tags",
    ]
    _style_matrix_axis(axes[0], BUCKET_LABELS, show_y_labels=True)
    _style_matrix_axis(axes[1], BUCKET_LABELS, show_y_labels=False)
    compact_tag_labels = ["Large\nrot.", "Large\ntrans.", "Density\nmismatch"]
    _style_matrix_axis(axes[2], compact_tag_labels, show_y_labels=False)
    axes[0].set_xlabel("GT-aligned Visible NN tertile")
    axes[1].set_xlabel("GT-aligned Visible NN tertile")
    axes[2].set_xlabel("Independent tag rate")

    _annotate_heatmap(axes[0], rr_matrix, "{:.1f}%", high_is_good=True)
    _annotate_heatmap(axes[1], rte_matrix, "{:.0f}", high_is_good=False)
    _annotate_heatmap(axes[2], tag_matrix, "{:.0f}%", high_is_good=False)

    labels = ["RR@5 (%)", "Median RTE (mm)", "Tagged pairs (%)"]
    for ax, image, label in zip(axes, images, labels, strict=True):
        cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.018)
        cbar.set_label(label, fontsize=7.8)
        cbar.ax.tick_params(labelsize=7.2)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90), pad=0.48, w_pad=0.64)
    _add_centered_canvas_panel_headers(
        fig,
        axes,
        "abc",
        titles,
        y_offset=0.040,
        fontsize=10.2,
    )
    _save_figure(
        fig,
        "hard_difficulty_failure_analysis.png",
        dpi=240,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def _model_record_map(
    records_by_model: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        model_key: {str(row["sample_id"]): row for row in records}
        for model_key, records in records_by_model.items()
    }


def _public_case(
    record: dict[str, Any],
    label: str,
    difficulty_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    sample_id = str(record["sample_id"])
    difficulty = difficulty_rows.get(sample_id, {})
    return {
        "label": label,
        "model_key": record.get("_model_key"),
        "display_name": record.get("_display_name"),
        "sample_id": sample_id,
        "scene_id": record.get("scene_id"),
        "frame_id": record.get("frame_id"),
        "difficulty_bucket": difficulty.get("difficulty_bucket"),
        "visible_nn_proxy_mm": difficulty.get("visible_nn_proxy_mm"),
        "trimmed_chamfer_proxy_mm": difficulty.get("trimmed_chamfer_proxy_mm"),
        "rre_deg": record.get("rre_deg"),
        "rte_mm": record.get("rte_mm"),
        "visible_nn_mean_mm": record.get("visible_nn_mean_mm"),
        "trimmed_chamfer_mm": record.get("trimmed_chamfer_mm"),
        "success_5deg_5mm": record.get("success_5deg_5mm"),
        "success_10deg_10mm": record.get("success_10deg_10mm"),
        "failure_tags": record.get("failure_tags", []),
        "result_dir": record.get("_result_dir"),
    }


def select_qualitative_cases(
    records_by_model: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    difficulty_rows = _load_difficulty_rows()
    cases: dict[str, Any] = {}

    geot = records_by_model.get("geotransformer", [])
    successes = [row for row in geot if int(row.get("success_5deg_5mm", 0)) == 1]
    if successes:
        row = min(
            successes,
            key=lambda item: _safe_float(item.get("rre_deg"))
            + _safe_float(item.get("rte_mm")) / 10.0,
        )
        cases["easy_success"] = _public_case(row, "easy_success", difficulty_rows)

    geot_failures = [
        row for row in geot if int(row.get("success_10deg_10mm", 0)) == 0
    ]
    if geot_failures:
        visible_values = sorted(
            _safe_float(row.get("visible_nn_mean_mm")) for row in geot_failures
        )
        cutoff = visible_values[max(0, int(0.2 * (len(visible_values) - 1)))]
        plausible = [
            row
            for row in geot_failures
            if _safe_float(row.get("visible_nn_mean_mm")) <= cutoff
        ]
        row = min(
            plausible or geot_failures,
            key=lambda item: _safe_float(item.get("visible_nn_mean_mm")),
        )
        cases["plausible_but_wrong"] = _public_case(
            row,
            "plausible_but_wrong",
            difficulty_rows,
        )

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
            difficulty_rows,
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
        cases["severe_failure"] = _public_case(row, "severe_failure", difficulty_rows)

    model_maps = _model_record_map(records_by_model)
    all_fail_samples: list[tuple[str, float, list[dict[str, Any]]]] = []
    candidate_ids = set(difficulty_rows) or {
        str(row["sample_id"])
        for records in records_by_model.values()
        for row in records
    }
    for sample_id in candidate_ids:
        rows = [
            model_rows[sample_id]
            for model_rows in model_maps.values()
            if sample_id in model_rows
        ]
        if rows and all(int(row.get("success_10deg_10mm", 0)) == 0 for row in rows):
            difficulty = difficulty_rows.get(sample_id, {})
            proxy = _safe_float(difficulty.get("visible_nn_proxy_mm"), 0.0)
            all_fail_samples.append((sample_id, proxy, rows))
    if all_fail_samples:
        _, _, rows = max(
            all_fail_samples,
            key=lambda item: (
                item[1],
                max(_safe_float(row.get("visible_nn_mean_mm")) for row in item[2]),
            ),
        )
        row = max(rows, key=lambda item: _safe_float(item.get("visible_nn_mean_mm")))
        cases["hard_geometry_failure"] = _public_case(
            row,
            "hard_geometry_failure",
            difficulty_rows,
        )
    return cases


def write_qualitative_selection(
    records_by_model: dict[str, list[dict[str, Any]]],
) -> Path:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    selection = select_qualitative_cases(records_by_model)
    path = FIGURE_ROOT / "qualitative_selection_manifest.json"
    path.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def build_figures(
    rows: list[dict[str, Any]],
    records_by_model: dict[str, list[dict[str, Any]]],
) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    plot_rr_bar(rows)
    plot_cdf(
        records_by_model,
        "rre_deg",
        "RRE (deg)",
        "hard_rre_cdf.png",
        "RRE empirical CDF",
        180.0,
        (),
    )
    plot_cdf(
        records_by_model,
        "rte_mm",
        "RTE (mm)",
        "hard_rte_cdf.png",
        "RTE empirical CDF",
        600.0,
        (),
    )
    plot_scene_rr_heatmap(records_by_model)
    plot_main_benchmark_views()
    plot_difficulty_rr_rte(records_by_model)
    plot_failure_tags(records_by_model)
    plot_difficulty_failure_panels(records_by_model)


def main() -> int:
    records_by_model = _load_records()
    rows = build_combined_rows(records_by_model)
    fields = [
        "model_key",
        "display_name",
        "sample_count",
        "rr5_pct",
        "rr10_pct",
        "rot_hit_5deg_pct",
        "trans_hit_5mm_pct",
        "rot_hit_10deg_pct",
        "trans_hit_10mm_pct",
        "rre_deg_mean",
        "rre_deg_median",
        "rre_deg_p90",
        "rte_mm_mean",
        "rte_mm_median",
        "rte_mm_p90",
        "visible_nn_mean_mm",
        "trimmed_chamfer_mm_mean",
        "trimmed_chamfer_mm_median",
        "latency_ms_mean",
        "latency_ms_p90",
        "train_time_ms",
        "train_peak_memory_mb",
        "result_dir",
    ]
    _write_csv(PROTOCOL_ROOT / "combined_leaderboard.csv", rows, fields)
    write_markdown(rows)
    write_latex_tables(rows)
    build_figures(rows, records_by_model)
    qualitative_selection = write_qualitative_selection(records_by_model)
    payload = {
        "protocol_root": _rel(PROTOCOL_ROOT),
        "combined_csv": _rel(PROTOCOL_ROOT / "combined_leaderboard.csv"),
        "combined_md": _rel(PROTOCOL_ROOT / "combined_leaderboard.md"),
        "latex_main_table": _rel(PROTOCOL_ROOT / "latex_main_table.tex"),
        "latex_error_distribution_table": _rel(
            PROTOCOL_ROOT / "latex_error_distribution_table.tex"
        ),
        "qualitative_selection_manifest": _rel(qualitative_selection),
        "figures": [_rel(path) for path in sorted(FIGURE_ROOT.glob("hard_*.pdf"))],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
