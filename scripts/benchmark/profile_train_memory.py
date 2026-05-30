#!/usr/bin/env python3
"""Profile train-time CUDA memory across model/point-count sweeps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmarking.diagnostics.train_memory import (  # noqa: E402
    DEFAULT_TRAIN_MEMORY_MODELS,
    profile_train_memory,
    write_train_memory_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-config",
        default="configs/benchmark/runtime/smoke_train.yaml",
        help="Base runtime config used to seed per-model train configs.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/benchmark/train_memory_profile",
        help="Directory for memory profile reports.",
    )
    parser.add_argument(
        "--num-points",
        nargs="+",
        type=int,
        default=[512, 1024, 2048, 4096],
        help="Point-count sweep, for example: --num-points 1024 2048 4096",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(DEFAULT_TRAIN_MEMORY_MODELS),
        help="Optional subset of trainable model ids to profile.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="CUDA device override, for example cuda:0.",
    )
    parser.add_argument(
        "--memory-limit-gb",
        type=float,
        default=32.0,
        help="Nominal GPU memory limit used for safe/unsafe classification.",
    )
    parser.add_argument(
        "--safety-ratio",
        type=float,
        default=0.90,
        help="Fraction of memory limit treated as usable.",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
        help="Per-model train micro-batch size used during profiling.",
    )
    parser.add_argument(
        "--steps-per-case",
        type=int,
        default=1,
        help="Number of training steps per model/point-count case.",
    )
    parser.add_argument(
        "--poll-interval-sec",
        type=float,
        default=0.1,
        help="nvidia-smi polling interval while child training processes run.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Optional timeout per train command.",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional manifest override for local flow checks.",
    )
    parser.add_argument(
        "--subset-config-path",
        default=None,
        help="Optional subset config override for local flow checks.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional dataset root override for local flow checks.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the full profile JSON to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = (REPO_ROOT / args.output_root).resolve()
    report = profile_train_memory(
        repo_root=REPO_ROOT,
        runtime_config_path=(REPO_ROOT / args.runtime_config).resolve(),
        output_root=output_root,
        num_points_list=args.num_points,
        model_ids=args.models,
        device=args.device,
        memory_limit_gb=args.memory_limit_gb,
        safety_ratio=args.safety_ratio,
        micro_batch_size=args.micro_batch_size,
        steps_per_case=args.steps_per_case,
        poll_interval_sec=args.poll_interval_sec,
        timeout_seconds=args.timeout_seconds,
        manifest_path=args.manifest_path,
        subset_config_path=args.subset_config_path,
        dataset_root=args.dataset_root,
    )
    json_path, markdown_path = write_train_memory_report(report, output_root)

    if args.print_json:
        print(json.dumps(report, indent=2))
    else:
        ok_count = sum(1 for entry in report["entries"] if entry["status"] == "ok")
        total_count = len(report["entries"])
        print(
            f"Train memory profile complete: {ok_count}/{total_count} cases profiled. "
            f"JSON={json_path} MD={markdown_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
