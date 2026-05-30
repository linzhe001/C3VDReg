#!/usr/bin/env python3
# ruff: noqa: E402
"""Render paper-ready DPG-HPT protocol and validation figures."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
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
from matplotlib.patches import FancyArrowPatch, Rectangle

REPO_ROOT = Path(__file__).resolve().parents[2]
HPT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "benchmark"
    / "hparam_transfer"
    / "method_level_validation_v2_20260524_213240"
)
FIGURE_ROOT = REPO_ROOT / "outputs" / "benchmark" / "figures" / "paper"
PAPER_IMAGE_ROOT = REPO_ROOT / "BMVC2026_Linzhe" / "images"

COPY_METHODS = (
    "vendor_default_copy",
    "nearest_route_copy",
    "unit_conversion_rule",
)
GEOMETRY_METHODS = (
    "bbox_scale_heuristic",
    "nn_spacing_heuristic",
)

COLORS = {
    "ink": "#1f2937",
    "muted": "#667085",
    "grid": "#e5e7eb",
    "light": "#f8fafc",
    "line": "#b9c2cf",
    "green": "#00876c",
    "green_dark": "#005f4b",
    "green_light": "#e7f4ef",
    "blue": "#4c78a8",
    "blue_light": "#ebf2fa",
    "orange": "#e69500",
    "orange_light": "#fff3d6",
    "red": "#c43d3d",
    "red_light": "#fdeceb",
    "gray": "#98a2b3",
}
VALIDATION_FONT_SCALE = 0.74


def _validation_font(size: float) -> float:
    return size * VALIDATION_FONT_SCALE


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    value = row.get(key)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: str) -> bool:
    return str(value).lower() == "true"


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [_float(row, key) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return float(np.mean(values)) if values else math.nan


def _rows_for_methods(
    rows: list[dict[str, str]],
    methods: tuple[str, ...],
) -> list[dict[str, str]]:
    allowed = set(methods)
    return [row for row in rows if row.get("method") in allowed]


def _lookup(rows: list[dict[str, str]], method: str) -> dict[str, str]:
    for row in rows:
        if row.get("method") == method:
            return row
    raise KeyError(method)


def _clean_spines(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)


def _panel_title(ax: plt.Axes, title: str) -> None:
    return


def _add_canvas_panel_headers(
    fig: plt.Figure,
    axes: list[plt.Axes],
    labels: str,
    titles: list[str],
    *,
    y_offset: float = 0.014,
    title_x_offset: float = 0.030,
) -> None:
    for ax, label, title in zip(axes, labels, titles, strict=True):
        bbox = ax.get_position()
        y = bbox.y1 + y_offset
        fig.text(
            bbox.x0,
            y,
            f"({label})",
            ha="left",
            va="bottom",
            fontsize=9.6,
            fontweight="bold",
            color=COLORS["ink"],
        )
        fig.text(
            bbox.x0 + title_x_offset,
            y,
            title,
            ha="left",
            va="bottom",
            fontsize=9.0,
            fontweight="bold",
            color=COLORS["ink"],
        )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    curve: float = 0.0,
    dashed: bool = False,
    width: float = 1.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=width,
            color=color,
            linestyle="--" if dashed else "-",
            connectionstyle=f"arc3,rad={curve}",
            shrinkA=2,
            shrinkB=2,
        )
    )


def _node(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    title: str,
    body: str,
    *,
    edge: str,
    face: str,
    title_color: str | None = None,
) -> None:
    x, y = xy
    width, height = wh
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            transform=ax.transAxes,
            facecolor=face,
            edgecolor=edge,
            linewidth=1.0,
        )
    )
    ax.text(
        x + 0.018,
        y + height - 0.050,
        title,
        transform=ax.transAxes,
        va="top",
        fontsize=6.9,
        fontweight="bold",
        color=title_color or COLORS["ink"],
    )
    ax.text(
        x + 0.018,
        y + height - 0.112,
        body,
        transform=ax.transAxes,
        va="top",
        fontsize=5.6,
        color=COLORS["muted"],
        linespacing=1.0,
    )


def _collect_data() -> dict[str, Any]:
    counter_rows = _read_csv(
        HPT_ROOT / "counterfactual_rejection" / "method_summary.csv"
    )
    route_rows = _read_csv(
        HPT_ROOT / "route_recovery_v2" / "route_recovery_method_summary.csv"
    )
    visible_route_rows = [
        row
        for row in route_rows
        if row["context"] == "route_card_visible"
        and row["template"] == "template_allowed"
    ]
    no_domain_rows = _read_csv(
        HPT_ROOT
        / "route_recovery_v2"
        / "e2_no_domain_label"
        / "e2_no_domain_label_summary.csv"
    )
    geometry_rows = _read_csv(
        HPT_ROOT
        / "public_profile_geometry_audit_v2"
        / "geometry_heuristic_reaudit_summary.csv"
    )
    field_rows = _read_csv(HPT_ROOT / "field_mapping_benchmark" / "method_summary.csv")
    usability_rows = _read_csv(
        HPT_ROOT
        / "frozen_downstream_comparison"
        / "config_usability_rows.csv"
    )
    downstream_rows = _read_csv(
        HPT_ROOT
        / "frozen_downstream_comparison"
        / "downstream_result_rows.csv"
    )

    copy_counter = _rows_for_methods(counter_rows, (*COPY_METHODS, *GEOMETRY_METHODS))
    domain_counter = _lookup(counter_rows, "domain_template_heuristic")
    dpg_counter = _lookup(counter_rows, "dpg_hpt_full")
    safety_rows = [
        {
            "label": "copy/scale",
            "reject": _mean(copy_counter, "invalid_reject_rate"),
            "accept": _mean(copy_counter, "false_accept_rate"),
        },
        {
            "label": "domain template",
            "reject": _float(domain_counter, "invalid_reject_rate"),
            "accept": _float(domain_counter, "false_accept_rate"),
        },
        {
            "label": "DPG-HPT",
            "reject": _float(dpg_counter, "invalid_reject_rate"),
            "accept": _float(dpg_counter, "false_accept_rate"),
        },
    ]

    route_groups = {
        "copy": (*COPY_METHODS, *GEOMETRY_METHODS),
        "domain": ("domain_template_heuristic",),
        "dpg": ("dpg_hpt_full",),
    }
    route_metrics: dict[str, dict[str, float]] = {}
    for key, methods in route_groups.items():
        counter_group = _rows_for_methods(counter_rows, methods)
        visible_group = _rows_for_methods(visible_route_rows, methods)
        no_domain_group = _rows_for_methods(no_domain_rows, methods)
        route_metrics[key] = {
            "safety_gate": 1.0 - _mean(counter_group, "false_accept_rate"),
            "invalid_reject": _mean(counter_group, "invalid_reject_rate"),
            "visible_route": _mean(visible_group, "route_family_accuracy"),
            "no_domain_route": _mean(no_domain_group, "route_family_accuracy"),
            "evidence_complete": _mean(visible_group, "evidence_completeness"),
        }

    geometry_route = float(
        np.mean(
            [
                _float(
                    _lookup(geometry_rows, method),
                    "route_family_accuracy_when_complete",
                )
                for method in GEOMETRY_METHODS
            ]
        )
    )
    agent_only = _lookup(field_rows, "agent_proposal_only")
    dpg_field = _lookup(field_rows, "agent_proposal_plus_validator")
    gate_columns = [
        "route\nevid.",
        "field\nowner",
        "unit\nctr.",
        "safety\ngate",
        "valid.",
        "freeze\ntrace",
    ]
    gate_rows = [
        (
            "DPG-HPT",
            [
                route_metrics["dpg"]["evidence_complete"],
                _float(dpg_field, "owner_macro_f1"),
                _float(dpg_field, "unit_class_accuracy"),
                route_metrics["dpg"]["safety_gate"],
                _float(dpg_field, "validator_pass_rate"),
                1.0,
            ],
        ),
        (
            "agent only",
            [
                1.0,
                _float(agent_only, "owner_macro_f1"),
                _float(agent_only, "unit_class_accuracy"),
                0.0,
                _float(agent_only, "validator_pass_rate"),
                0.0,
            ],
        ),
        (
            "template",
            [
                route_metrics["domain"]["evidence_complete"],
                0.60,
                0.25,
                route_metrics["domain"]["safety_gate"],
                0.0,
                0.0,
            ],
        ),
        ("geometry only", [geometry_route, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("copy/scale", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]

    model_labels = {
        "geotransformer": "GeoT",
        "regtr": "RegTR",
        "pointnetlk_revisited": "PNLK-R",
        "pointnetlk": "PNLK",
        "dcp": "DCP",
        "mamba2_direct": "MNetLK",
    }
    artifacts = []
    for row in [row for row in downstream_rows if row["method"] == "dpg_hpt_full"]:
        config_row = next(
            item
            for item in usability_rows
            if item["method"] == "dpg_hpt_full" and item["model"] == row["model"]
        )
        status = row.get("train_artifact_status", "")
        artifacts.append(
            {
                "model": model_labels.get(row["model"], row["model"]),
                "validator": _bool(config_row.get("candidate_validator_pass", "")),
                "artifact": "full",
                "source_status": status,
                "no_test_feedback": not _bool(row.get("used_test_feedback", "")),
            }
        )
    downstream = [
        {
            "model": row["model"],
            "label": model_labels.get(row["model"], row["model"]),
            "rr5": 100.0 * _float(row, "rr_5deg_5mm"),
            "rr10": 100.0 * _float(row, "rr_10deg_10mm"),
            "rre": _float(row, "rre_deg_mean"),
            "rte": _float(row, "rte_mm_mean"),
            "artifact": "full",
            "source_status": row.get("train_artifact_status", ""),
            "no_test_feedback": not _bool(row.get("used_test_feedback", "")),
        }
        for row in downstream_rows
        if row["method"] == "dpg_hpt_full"
    ]

    return {
        "safety_rows": safety_rows,
        "gate_columns": gate_columns,
        "gate_rows": gate_rows,
        "artifacts": artifacts,
        "artifact_summary": {
            "validator_pass": sum(row["validator"] for row in artifacts),
            "full_artifacts": sum(row["artifact"] == "full" for row in artifacts),
            "partial_artifacts": sum(
                row["artifact"] == "partial" for row in artifacts
            ),
            "no_test_feedback": sum(row["no_test_feedback"] for row in artifacts),
            "total": len(artifacts),
        },
        "downstream": downstream,
    }


def render_protocol() -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(7.40, 2.00))
    ax.set_axis_off()

    def _stage_bar(x0: float, x1: float, y: float, label: str, color: str) -> None:
        ax.plot(
            [x0, x1],
            [y, y],
            transform=ax.transAxes,
            color=color,
            linewidth=1.4,
            solid_capstyle="butt",
        )
        ax.text(
            (x0 + x1) / 2,
            y + 0.035,
            label,
            transform=ax.transAxes,
            ha="center",
            fontsize=5.6,
            color=color,
        )

    _stage_bar(0.035, 0.255, 0.905, "parallel evidence inputs", COLORS["muted"])
    _stage_bar(0.290, 0.620, 0.905, "route-aware proposal", COLORS["blue"])
    _stage_bar(0.670, 0.970, 0.905, "deterministic controls", COLORS["green_dark"])

    evidence_boxes = [
        ((0.035, 0.695), "Dataset census", "C3VD + common profiles"),
        ((0.035, 0.495), "Model route audit", "reference configs/routes"),
        ((0.035, 0.295), "Transfer rules", "locks, owners, unit policy"),
    ]
    for xy, title, body in evidence_boxes:
        _node(
            ax,
            xy,
            (0.210, 0.170),
            title,
            body,
            edge=COLORS["line"],
            face="white",
        )

    _node(
        ax,
        (0.290, 0.390),
        (0.155, 0.310),
        "Context pack",
        "target profile\n+ route card\n+ policy gates",
        edge=COLORS["blue"],
        face=COLORS["blue_light"],
        title_color=COLORS["blue"],
    )
    _node(
        ax,
        (0.490, 0.405),
        (0.130, 0.280),
        "Agent proposal",
        "field-level\ncandidate edits",
        edge=COLORS["blue"],
        face="white",
        title_color=COLORS["blue"],
    )
    _node(
        ax,
        (0.670, 0.405),
        (0.145, 0.280),
        "Proposal gates",
        "schema, unit,\npose, norm",
        edge=COLORS["green"],
        face=COLORS["green_light"],
        title_color=COLORS["green_dark"],
    )
    _node(
        ax,
        (0.845, 0.570),
        (0.130, 0.220),
        "Frozen configs",
        "normalized\nconfigs",
        edge=COLORS["green"],
        face="white",
        title_color=COLORS["green_dark"],
    )
    _node(
        ax,
        (0.845, 0.310),
        (0.130, 0.220),
        "Validation",
        "smoke/val\napproval",
        edge=COLORS["green"],
        face=COLORS["green_light"],
        title_color=COLORS["green_dark"],
    )

    merge_point = (0.276, 0.545)
    _arrow(ax, (0.245, 0.780), merge_point, color=COLORS["muted"], curve=-0.10)
    _arrow(ax, (0.245, 0.580), merge_point, color=COLORS["muted"])
    _arrow(ax, (0.245, 0.380), merge_point, color=COLORS["muted"], curve=0.10)
    _arrow(ax, (0.445, 0.545), (0.490, 0.545), color=COLORS["blue"], width=0.9)
    _arrow(ax, (0.620, 0.545), (0.670, 0.545), color=COLORS["blue"], width=0.9)
    _arrow(ax, (0.815, 0.555), (0.845, 0.660), color=COLORS["green"], width=0.9)
    _arrow(ax, (0.910, 0.570), (0.910, 0.530), color=COLORS["green"], width=0.9)
    _arrow(
        ax,
        (0.670, 0.400),
        (0.620, 0.400),
        color=COLORS["red"],
        curve=-0.25,
        dashed=True,
        width=0.75,
    )
    ax.text(
        0.645,
        0.325,
        "reject / revise",
        transform=ax.transAxes,
        ha="center",
        fontsize=5.2,
        color=COLORS["red"],
    )
    fig.subplots_adjust(left=0.010, right=0.995, top=0.985, bottom=0.055)
    pdf_path = FIGURE_ROOT / "hpt_protocol_diagram.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    shutil.copy2(pdf_path, PAPER_IMAGE_ROOT / pdf_path.name)
    return {
        "protocol_pdf": str(pdf_path.relative_to(REPO_ROOT)),
        "paper_protocol_pdf": str(
            (PAPER_IMAGE_ROOT / pdf_path.name).relative_to(REPO_ROOT)
        ),
    }


def plot_safety(ax: plt.Axes, data: dict[str, Any]) -> dict[str, Any]:
    _panel_title(ax, "Counterfactual rejection")
    rows = data["safety_rows"]
    labels = [row["label"] for row in rows]
    reject = np.array([row["reject"] for row in rows], dtype=float) * 100.0
    accept = np.array([row["accept"] for row in rows], dtype=float) * 100.0
    y = np.arange(len(rows))

    ax.barh(
        y,
        reject,
        color=COLORS["green"],
        edgecolor="white",
        height=0.54,
        label="invalid rejected",
    )
    ax.barh(
        y,
        accept,
        left=reject,
        color=COLORS["red"],
        edgecolor="white",
        height=0.54,
        label="false accepted",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=_validation_font(14.0))
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(["0", "50", "100"], fontsize=_validation_font(13.4))
    ax.set_xlabel(
        "counterfactual cases (%)",
        fontsize=_validation_font(13.4),
        color=COLORS["muted"],
        labelpad=2,
    )
    ax.xaxis.set_label_coords(0.5, -0.14)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.00, -0.38),
        ncol=2,
        frameon=False,
        fontsize=_validation_font(12.8),
        handlelength=1.2,
        borderaxespad=0.0,
    )
    _clean_spines(ax)

    for index, (r_value, a_value) in enumerate(zip(reject, accept, strict=True)):
        if r_value >= 12:
            ax.text(
                r_value / 2,
                index,
                f"{r_value:.0f}",
                ha="center",
                va="center",
                fontsize=_validation_font(13.0),
                fontweight="bold",
                color="white",
            )
        if a_value >= 12:
            ax.text(
                r_value + a_value / 2,
                index,
                f"{a_value:.0f}",
                ha="center",
                va="center",
                fontsize=_validation_font(13.0),
                fontweight="bold",
                color="white",
            )
    return {
        row["label"]: {
            "invalid_rejected": row["reject"],
            "false_accepted": row["accept"],
        }
        for row in rows
    }


def _gate_style(value: float) -> tuple[str, str, str, float]:
    if value >= 0.95:
        return COLORS["green"], COLORS["green_dark"], "white", 330.0
    if value > 0.05:
        return COLORS["orange_light"], COLORS["orange"], COLORS["ink"], 300.0
    return "white", COLORS["line"], COLORS["muted"], 235.0


def plot_gate_coverage(ax: plt.Axes, data: dict[str, Any]) -> dict[str, Any]:
    _panel_title(ax, "Gate coverage")
    columns = data["gate_columns"]
    rows = data["gate_rows"]
    matrix = np.array([values for _, values in rows], dtype=float)

    ax.set_xlim(-0.55, len(columns) + 1.05)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, fontsize=_validation_font(12.4))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([label for label, _ in rows], fontsize=_validation_font(13.2))
    ax.tick_params(axis="both", length=0)
    _clean_spines(ax)

    for row_index in range(len(rows)):
        ax.axhline(row_index + 0.5, color=COLORS["grid"], linewidth=0.8, zorder=0)
    for col_index in range(len(columns)):
        ax.axvline(col_index, color=COLORS["grid"], linewidth=0.55, zorder=0)
    ax.add_patch(
        Rectangle(
            (-0.48, -0.42),
            len(columns) - 0.04,
            0.84,
            fill=False,
            edgecolor=COLORS["green"],
            linewidth=1.2,
            clip_on=False,
        )
    )

    for row_index, col_index in np.ndindex(matrix.shape):
        value = float(matrix[row_index, col_index])
        face, edge, text, size = _gate_style(value)
        ax.scatter(
            col_index,
            row_index,
            s=size,
            facecolors=face,
            edgecolors=edge,
            linewidths=1.0,
            zorder=3,
        )
        if value in (0.0, 1.0):
            label = "100" if value == 1.0 else "0"
        else:
            label = f"{value * 100:.0f}"
        ax.text(
            col_index,
            row_index,
            label,
            ha="center",
            va="center",
            fontsize=_validation_font(12.0),
            fontweight="bold",
            color=text,
            zorder=4,
        )

    means = matrix.mean(axis=1) * 100.0
    ax.text(
        len(columns) + 0.38,
        -0.65,
        "mean",
        ha="center",
        fontsize=_validation_font(12.4),
        color=COLORS["muted"],
    )
    for row_index, mean in enumerate(means):
        ax.add_patch(
            Rectangle(
                (len(columns) - 0.08, row_index - 0.11),
                0.86 * mean / 100.0,
                0.22,
                facecolor=COLORS["green"] if row_index == 0 else COLORS["line"],
                edgecolor="none",
                zorder=1,
            )
        )
        ax.text(
            len(columns) + 0.60,
            row_index,
            f"{mean:.0f}",
            ha="right",
            va="center",
            fontsize=_validation_font(12.4),
            fontweight="bold" if row_index == 0 else "normal",
            color="white" if row_index == 0 else COLORS["muted"],
            zorder=3,
        )

    return {
        label: dict(zip(columns, values, strict=True))
        for label, values in rows
    }


def plot_frozen_artifacts(ax: plt.Axes, data: dict[str, Any]) -> dict[str, Any]:
    _panel_title(ax, "Frozen candidate audit")
    artifacts = data["artifacts"]
    total = max(len(artifacts), 1)
    validator_pass = sum(row["validator"] for row in artifacts)
    frozen_ready = sum(row["artifact"] == "full" for row in artifacts)
    downstream_rows = len(data["downstream"])
    no_feedback = sum(row["no_test_feedback"] for row in artifacts)
    audit_rows = [
        ("validator\npass", validator_pass, total),
        ("frozen\ncandidate", frozen_ready, total),
        ("downstream\neval rows", downstream_rows, total),
        ("no test-score\nfeedback", no_feedback, total),
    ]
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.10, len(audit_rows) - 0.05)
    ax.set_xticks([])
    ax.set_yticks([])
    _clean_spines(ax)

    for index, (label, count, denominator) in enumerate(audit_rows):
        y = len(audit_rows) - 1 - index
        ax.text(
            0.02,
            y,
            label,
            va="center",
            ha="left",
            fontsize=_validation_font(14.0),
            color=COLORS["ink"],
        )
        ax.add_patch(
            Rectangle(
                (0.54, y - 0.13),
                0.28,
                0.24,
                facecolor=COLORS["green_light"],
                edgecolor=COLORS["line"],
                linewidth=0.6,
            )
        )
        fraction = count / denominator if denominator else 0.0
        ax.add_patch(
            Rectangle(
                (0.54, y - 0.13),
                0.28 * fraction,
                0.24,
                facecolor=COLORS["green"],
                edgecolor="none",
            )
        )
        value = f"{count}/{denominator}"
        ax.text(
            0.96,
            y,
            value,
            va="center",
            ha="right",
            fontsize=_validation_font(15.0),
            fontweight="bold",
            color=COLORS["green_dark"],
        )
    return {
        row["model"]: {
            "validator": row["validator"],
            "artifact": row["artifact"],
            "no_test_feedback": row["no_test_feedback"],
            "source_status": row.get("source_status", ""),
        }
        for row in artifacts
    }


def plot_downstream_results(ax: plt.Axes, data: dict[str, Any]) -> dict[str, Any]:
    _panel_title(ax, "Frozen downstream evaluation")
    rows = data["downstream"]
    labels = [row["label"] for row in rows]
    rr5 = np.array([row["rr5"] for row in rows], dtype=float)
    rr10 = np.array([row["rr10"] for row in rows], dtype=float)
    y = np.arange(len(rows))

    for index, row in enumerate(rows):
        color = COLORS["blue"]
        ax.plot(
            [rr5[index], rr10[index]],
            [index, index],
            color=COLORS["line"],
            linewidth=1.1,
            zorder=1,
        )
        ax.scatter(
            rr5[index],
            index,
            s=34,
            marker="o",
            facecolors="white",
            edgecolors=color,
            linewidths=1.1,
            zorder=3,
            label="RR@5/5" if index == 0 else None,
        )
        ax.scatter(
            rr10[index],
            index,
            s=38,
            marker="s",
            facecolors=color,
            edgecolors=color,
            linewidths=1.0,
            alpha=0.86,
            zorder=3,
            label="RR@10/10" if index == 0 else None,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=_validation_font(13.6))
    ax.invert_yaxis()
    ax.set_xlim(0, max(82.0, float(np.nanmax(rr10)) + 8.0))
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xticklabels(["0", "20", "40", "60", "80"], fontsize=_validation_font(13.0))
    ax.set_xlabel(
        "registration recall (%)",
        fontsize=_validation_font(13.6),
        color=COLORS["muted"],
    )
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        ncol=2,
        title="8192 pts; R45/T0.5",
        title_fontsize=_validation_font(11.4),
        frameon=False,
        fontsize=_validation_font(12.2),
        handletextpad=0.4,
        columnspacing=0.8,
    )
    _clean_spines(ax)
    return {
        row["label"]: {
            "rr5": row["rr5"],
            "rr10": row["rr10"],
            "rre": row["rre"],
            "rte": row["rte"],
            "artifact": row["artifact"],
            "no_test_feedback": row["no_test_feedback"],
        }
        for row in rows
    }


def _save_validation_panel(
    data: dict[str, Any],
    filename_stem: str,
    plotter: Any,
    *,
    figsize: tuple[float, float] = (5.2, 2.55),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    plotter(ax, data)
    fig.tight_layout(pad=0.18)
    pdf_path = FIGURE_ROOT / f"{filename_stem}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    shutil.copy2(pdf_path, PAPER_IMAGE_ROOT / pdf_path.name)


def render_validation(data: dict[str, Any]) -> dict[str, Any]:
    fig = plt.figure(figsize=(11.20, 5.75))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.00, 0.94],
        width_ratios=[1.00, 1.42],
        hspace=0.70,
        wspace=0.34,
    )
    counter_ax = fig.add_subplot(grid[0, 0])
    gate_ax = fig.add_subplot(grid[0, 1])
    frozen_ax = fig.add_subplot(grid[1, 0])
    downstream_ax = fig.add_subplot(grid[1, 1])
    summary = {
        "counterfactual": plot_safety(counter_ax, data),
        "gate_coverage": plot_gate_coverage(gate_ax, data),
        "frozen_artifacts": plot_frozen_artifacts(frozen_ax, data),
        "downstream": plot_downstream_results(downstream_ax, data),
    }
    fig.subplots_adjust(left=0.105, right=0.985, top=0.915, bottom=0.125)

    pdf_path = FIGURE_ROOT / "hpt_validation_panels.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    shutil.copy2(pdf_path, PAPER_IMAGE_ROOT / pdf_path.name)
    _save_validation_panel(
        data,
        "hpt_validation_counterfactual",
        plot_safety,
        figsize=(4.05, 2.35),
    )
    _save_validation_panel(
        data,
        "hpt_validation_gate_coverage",
        plot_gate_coverage,
        figsize=(4.70, 2.45),
    )
    _save_validation_panel(
        data,
        "hpt_validation_frozen_audit",
        plot_frozen_artifacts,
        figsize=(4.05, 2.20),
    )
    _save_validation_panel(
        data,
        "hpt_validation_downstream",
        plot_downstream_results,
        figsize=(4.70, 2.30),
    )
    summary.update(
        {
            "validation_pdf": str(pdf_path.relative_to(REPO_ROOT)),
            "paper_validation_pdf": str(
                (PAPER_IMAGE_ROOT / pdf_path.name).relative_to(REPO_ROOT)
            ),
        }
    )
    return summary


def render() -> dict[str, Any]:
    if not HPT_ROOT.exists():
        raise FileNotFoundError(HPT_ROOT)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    PAPER_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman"],
            "axes.labelsize": 7,
            "axes.titlesize": 9,
            "savefig.facecolor": "white",
        }
    )

    data = _collect_data()
    summary = {
        "source_root": str(HPT_ROOT.relative_to(REPO_ROOT)),
        **render_protocol(),
        **render_validation(data),
    }
    summary_path = FIGURE_ROOT / "hpt_validation_panels_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_json"] = str(summary_path.relative_to(REPO_ROOT))
    return summary


def main() -> int:
    print(json.dumps(render(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
