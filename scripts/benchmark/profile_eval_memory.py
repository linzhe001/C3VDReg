#!/usr/bin/env python3
"""Profile eval-time CUDA memory across model/point-count sweeps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmarking.diagnostics.eval_memory import (  # noqa: E402
    profile_eval_memory,
    write_eval_memory_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-config",
        default="configs/benchmark/paper_r25_90_t100_500mm/base_eval.yaml",
        help="Base runtime config used to seed per-model eval configs.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/benchmark/eval_memory_profile",
        help="Directory for memory profile reports.",
    )
    parser.add_argument(
        "--num-points",
        nargs="+",
        type=int,
        required=True,
        help="Point-count sweep, for example: --num-points 1024 2048 4096",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of model ids to profile.",
    )
    parser.add_argument(
        "--samples-per-case",
        type=int,
        default=1,
        help="How many samples to profile per model/point-count case.",
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
        "--rotation-deg",
        type=float,
        default=0.0,
        help="Optional unified perturbation rotation magnitude in degrees.",
    )
    parser.add_argument(
        "--translation-m",
        type=float,
        default=0.0,
        help="Optional unified perturbation translation magnitude in meters.",
    )
    parser.add_argument(
        "--noise-sigma",
        type=float,
        default=0.0,
        help="Optional Gaussian noise sigma in meters.",
    )
    parser.add_argument(
        "--noise-clip",
        type=float,
        default=0.0,
        help="Optional Gaussian noise clip in meters.",
    )
    parser.add_argument(
        "--apply-noise-to",
        choices=("source", "both"),
        default="source",
        help="Whether unified Gaussian noise applies to source only or both clouds.",
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
    perturbation_overrides = {
        "enabled": any(
            value > 0.0
            for value in (
                args.rotation_deg,
                args.translation_m,
                args.noise_sigma,
                args.noise_clip,
            )
        ),
        "rotation_deg": args.rotation_deg,
        "translation_m": args.translation_m,
        "noise_sigma": args.noise_sigma,
        "noise_clip": args.noise_clip,
        "apply_noise_to": args.apply_noise_to,
    }
    report = profile_eval_memory(
        repo_root=REPO_ROOT,
        runtime_config_path=(REPO_ROOT / args.runtime_config).resolve(),
        output_root=output_root,
        num_points_list=args.num_points,
        model_ids=args.models,
        samples_per_case=args.samples_per_case,
        manifest_path=args.manifest_path,
        subset_config_path=args.subset_config_path,
        dataset_root=args.dataset_root,
        perturbation_overrides=perturbation_overrides,
    )
    json_path, markdown_path = write_eval_memory_report(report, output_root)

    if args.print_json:
        print(json.dumps(report, indent=2))
    else:
        ok_count = sum(1 for entry in report["entries"] if entry["status"] == "ok")
        total_count = len(report["entries"])
        print(
            f"Eval memory profile complete: {ok_count}/{total_count} cases profiled. "
            f"JSON={json_path} MD={markdown_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
