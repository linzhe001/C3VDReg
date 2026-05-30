#!/usr/bin/env python3
"""Profile Transformer-like token stress memory, MT-PCR style."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmarking.diagnostics.token_stress import (  # noqa: E402
    DEFAULT_GEOTRANSFORMER_SPATIAL_SCALES,
    DEFAULT_TOKEN_LENGTHS,
    DEFAULT_TOKEN_STRESS_MODELS,
    profile_token_stress,
    write_token_stress_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-config",
        default="configs/benchmark/runtime/smoke_train.yaml",
        help="Base train runtime config.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/benchmark/transformer_token_stress",
        help="Output directory for token stress reports.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(DEFAULT_TOKEN_STRESS_MODELS),
        help="Transformer-like model ids. Currently supports dcp and geotransformer.",
    )
    parser.add_argument(
        "--token-lengths",
        nargs="+",
        type=int,
        default=list(DEFAULT_TOKEN_LENGTHS),
        help="Target active token lengths, e.g. 128 256 512 1024 1536.",
    )
    parser.add_argument(
        "--geotransformer-raw-points",
        type=int,
        default=65536,
        help="Raw C3VD sampled points used while varying GeoTransformer tokens.",
    )
    parser.add_argument(
        "--geotransformer-spatial-scales",
        nargs="*",
        type=float,
        default=list(DEFAULT_GEOTRANSFORMER_SPATIAL_SCALES),
        help="Candidate spatial_scale values used to approximate target tokens.",
    )
    parser.add_argument("--device", default=None, help="CUDA device override.")
    parser.add_argument("--memory-limit-gb", type=float, default=32.0)
    parser.add_argument("--safety-ratio", type=float, default=0.90)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--steps-per-case", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = (REPO_ROOT / args.output_root).resolve()
    report = profile_token_stress(
        repo_root=REPO_ROOT,
        runtime_config_path=(REPO_ROOT / args.runtime_config).resolve(),
        output_root=output_root,
        token_lengths=args.token_lengths,
        model_ids=args.models,
        geotransformer_raw_points=args.geotransformer_raw_points,
        geotransformer_spatial_scales=args.geotransformer_spatial_scales,
        device=args.device,
        memory_limit_gb=args.memory_limit_gb,
        safety_ratio=args.safety_ratio,
        micro_batch_size=args.micro_batch_size,
        steps_per_case=args.steps_per_case,
        timeout_seconds=args.timeout_seconds,
    )
    json_path, csv_path, markdown_path = write_token_stress_report(
        report,
        output_root,
    )
    if args.print_json:
        print(json.dumps(report, indent=2))
    else:
        ok_count = sum(1 for entry in report["entries"] if entry["status"] == "ok")
        total_count = len(report["entries"])
        print(
            "Transformer token stress complete: "
            f"{ok_count}/{total_count} cases profiled. "
            f"JSON={json_path} CSV={csv_path} MD={markdown_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
