#!/usr/bin/env python3
"""Rerun hard-protocol eval configs and export per-sample pose transforms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmarking.runners.eval_runner import run_eval  # noqa: E402

PROTOCOL_ROOT = Path("outputs/benchmark/r25_90_t100_500mm_protocol")
OUTPUT_ROOT = PROTOCOL_ROOT / "pose_transform_exports"
RUN_CONFIGS = {
    "dcp": PROTOCOL_ROOT / "dcp/eval_test/normalized_eval_config.json",
    "geotransformer": PROTOCOL_ROOT
    / "geotransformer/eval_test/normalized_eval_config.json",
    "icp": PROTOCOL_ROOT / "icp/eval_test/normalized_eval_config.json",
    "mamba2_direct": PROTOCOL_ROOT
    / "mamba2_direct/eval_test/normalized_eval_config.json",
    "pointnetlk": PROTOCOL_ROOT / "pointnetlk/eval_test/normalized_eval_config.json",
    "pointnetlk_revisited": PROTOCOL_ROOT
    / "pointnetlk_revisited/eval_test/normalized_eval_config.json",
    "regtr": PROTOCOL_ROOT / "regtr/eval_test/normalized_eval_config.json",
}


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _prepare_config(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    config = json.loads(json.dumps(config))
    config["runtime"]["output_dir"] = str(output_dir)
    config["analysis"]["export_transforms"] = True
    config["analysis"]["geometry"]["sample_count"] = 256
    config["analysis"]["geometry"]["export_histogram"] = False
    config["analysis"]["geometry"]["export_cdf"] = False
    config["analysis"]["qualitative"]["topk_failures"] = 0
    config["analysis"]["qualitative"]["export_failure_gallery"] = False
    config["analysis"]["export"]["html"] = False
    config["analysis"]["export"]["png"] = False
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(RUN_CONFIGS),
        choices=sorted(RUN_CONFIGS),
        help="Hard-protocol model keys to rerun for pose transform export.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop at the first failed model instead of recording the failure.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for model_key in args.models:
        config_path = RUN_CONFIGS[model_key]
        output_dir = args.output_root / model_key / "eval_test"
        print(f"[export] {model_key}: {config_path} -> {output_dir}", flush=True)
        try:
            config = _prepare_config(_load_config(config_path), output_dir)
            summary = run_eval(config)
            summaries.append(
                {
                    "model_key": model_key,
                    "status": "ok",
                    "output_dir": str(output_dir),
                    "pose_transforms": str(output_dir / "pose_transforms.jsonl"),
                    "summary": summary,
                }
            )
        except Exception as exc:  # pragma: no cover - CLI diagnostics
            summaries.append(
                {
                    "model_key": model_key,
                    "status": "failed",
                    "output_dir": str(output_dir),
                    "error": repr(exc),
                }
            )
            print(f"[export] {model_key} failed: {exc!r}", flush=True)
            if args.stop_on_error:
                break

    manifest_path = args.output_root / "pose_transform_export_manifest.json"
    manifest_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")
    return 0 if all(row["status"] == "ok" for row in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
