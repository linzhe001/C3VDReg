#!/usr/bin/env python3
"""Build eval configs for the hard R25-90/T100-500mm protocol."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "benchmark" / "r25_90_t100_500mm_protocol"


@dataclass(frozen=True)
class SourceRun:
    model_key: str
    source_eval_dir: Path


SOURCE_RUNS = (
    SourceRun(
        "geotransformer",
        REPO_ROOT
        / "outputs"
        / "benchmark"
        / "r90_t500mm_protocol"
        / "geotransformer"
        / "eval_test",
    ),
    SourceRun(
        "regtr",
        REPO_ROOT
        / "outputs"
        / "benchmark"
        / "r90_t500mm_from_scratch_fixed_regtr_dcp"
        / "regtr"
        / "eval_test_latency_rerun2",
    ),
    SourceRun(
        "mamba2_direct",
        REPO_ROOT
        / "outputs"
        / "benchmark"
        / "mamba2_followup_point_order_pair_initializer"
        / "direct_sort_xyz_e5"
        / "eval_test_maxiter10",
    ),
    SourceRun(
        "pointnetlk_revisited",
        REPO_ROOT
        / "outputs"
        / "benchmark"
        / "r90_t500mm_protocol"
        / "pointnetlk_revisited"
        / "eval_test",
    ),
    SourceRun(
        "pointnetlk",
        REPO_ROOT
        / "outputs"
        / "benchmark"
        / "r90_t500mm_protocol"
        / "pointnetlk"
        / "eval_test",
    ),
    SourceRun(
        "dcp",
        REPO_ROOT
        / "outputs"
        / "benchmark"
        / "r90_t500mm_from_scratch_fixed_regtr_dcp"
        / "dcp"
        / "eval_test",
    ),
    SourceRun(
        "icp",
        REPO_ROOT
        / "outputs"
        / "benchmark"
        / "r90_t500mm_protocol"
        / "icp"
        / "eval_test",
    ),
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def build_config(source: SourceRun) -> Path:
    source_config_path = source.source_eval_dir / "normalized_eval_config.json"
    if not source_config_path.exists():
        raise FileNotFoundError(source_config_path)
    config = _read_json(source_config_path)
    config["perturbation"] = {
        **dict(config.get("perturbation", {})),
        "enabled": True,
        "rotation_deg": 90.0,
        "translation_m": 500.0,
        "min_rotation_deg": 25.0,
        "min_translation_m": 100.0,
        "noise_sigma": 0.0,
        "noise_clip": 0.0,
        "apply_noise_to": "source",
    }
    config.setdefault("benchmark", {})["split"] = "test"
    config["benchmark"]["subset_name"] = None
    config.setdefault("preprocess", {})["num_points_override"] = 8192
    config.setdefault("runtime", {})["output_dir"] = _rel(
        OUTPUT_ROOT / source.model_key / "eval_test"
    )
    config["runtime"]["export_html"] = False
    config.setdefault("analysis", {}).setdefault("qualitative", {})[
        "export_failure_gallery"
    ] = False
    config["analysis"].setdefault("export", {})["html"] = False

    output_path = OUTPUT_ROOT / "configs" / f"eval_{source.model_key}.yaml"
    _write_yaml(output_path, config)
    return output_path


def main() -> int:
    config_paths = [build_config(source) for source in SOURCE_RUNS]
    run_script = OUTPUT_ROOT / "run_eval_sequence.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd \"$(dirname \"$0\")/../../..\"",
        "",
    ]
    for path in config_paths:
        lines.append(
            "MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp "
            f"/home/linzhe/anaconda3/envs/PCLR_compare/bin/python "
            f"scripts/runners/eval_benchmark.py --config {_rel(path)}"
        )
    run_script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    run_script.chmod(0o755)

    manifest = {
        "protocol_id": "r25_90_t100_500mm_protocol",
        "rotation_deg_range": [25.0, 90.0],
        "translation_mm_range": [100.0, 500.0],
        "config_paths": [_rel(path) for path in config_paths],
        "run_script": _rel(run_script),
    }
    (OUTPUT_ROOT / "hard_protocol_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
