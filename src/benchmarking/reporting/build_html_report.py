"""Static HTML report builder."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _relative(path: str | Path, output_dir: Path) -> str:
    return str(Path(path).resolve().relative_to(output_dir.resolve()))


def _render_key_value_rows(
    payload: dict[str, Any],
    keys: list[str],
) -> str:
    return "".join(
        f"<tr><th>{html.escape(str(key))}</th>"
        f"<td>{html.escape(str(payload.get(key, 'n/a')))}</td></tr>"
        for key in keys
    )


def build_html_report(
    output_dir: str | Path,
    aggregate_summary: dict[str, Any],
    table_paths: dict[str, str],
    figure_paths: dict[str, str],
    run_card: dict[str, Any],
) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.html"

    overall = aggregate_summary["overall"]
    headline_items = _render_key_value_rows(
        overall,
        [
            "model_id",
            "preprocess_profile_id",
            "sample_count",
            "registration_recall@rre_5deg_rte_5mm",
            "rre_deg_mean",
            "rte_mm_mean",
            "trimmed_chamfer_mm_mean",
        ],
    )
    efficiency_items = _render_key_value_rows(
        overall,
        [
            "preprocess_time_ms_mean",
            "inference_time_ms_mean",
            "refinement_time_ms_mean",
            "latency_ms_mean",
            "latency_ms_p90",
            "peak_memory_mb_mean",
        ],
    )
    geometry_items = _render_key_value_rows(
        overall,
        [
            "visible_nn_mean_mm_mean",
            "trimmed_chamfer_mm_mean",
        ],
    )
    table_links = "".join(
        (
            f"<li><a href='{html.escape(_relative(path, output_dir))}'>"
            f"{html.escape(name)}</a></li>"
        )
        for name, path in sorted(table_paths.items())
    )
    figure_tags = "".join(
        (
            f"<figure><img src='{html.escape(_relative(path, output_dir))}' "
            f"alt='{html.escape(name)}' style='max-width: 640px; width: 100%;'/>"
            f"<figcaption>{html.escape(name)}</figcaption></figure>"
        )
        for name, path in sorted(figure_paths.items())
    )

    quick_start_items = """
    <li><a href="summary_overview.md">summary_overview.md</a></li>
    <li><a href="leaderboard/leaderboard_main.csv">
      leaderboard/leaderboard_main.csv
    </a></li>
    <li><a href="leaderboard/efficiency_summary.csv">
      leaderboard/efficiency_summary.csv
    </a></li>
    <li><a href="geometry/geometry_summary.csv">
      geometry/geometry_summary.csv
    </a></li>
  """

    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Benchmark Report</title>
  <style>
    body {{
      font-family: sans-serif;
      margin: 2rem auto;
      max-width: 960px;
      line-height: 1.5;
    }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; }}
    code {{ white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>Benchmark Report</h1>
  <h2>Quick Start</h2>
  <ol>{quick_start_items}</ol>
  <h2>Headline Metrics</h2>
  <table>{headline_items}</table>
  <h2>Efficiency</h2>
  <table>{efficiency_items}</table>
  <h2>Geometry</h2>
  <table>{geometry_items}</table>
  <h2>Tables</h2>
  <ul>{table_links}</ul>
  <h2>Figures</h2>
  {figure_tags}
  <h2>Run Card</h2>
  <pre><code>{html.escape(json.dumps(run_card, indent=2))}</code></pre>
</body>
</html>
"""
    report_path.write_text(document, encoding="utf-8")
    return str(report_path)
