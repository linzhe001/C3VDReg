"""Validate DPG-HPT generated candidates with smoke/val runtime sanity checks."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.benchmarking.config.loader import load_benchmark_config
from src.benchmarking.hparam_transfer.candidate_validation import (
    load_candidate_bundle,
    validate_candidate_configs,
    write_validation_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-configs",
        required=True,
        help="Path to candidate config bundle YAML or JSON.",
    )
    parser.add_argument(
        "--runtime-config",
        required=True,
        help="Path to benchmark runtime config used for sanity validation.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where validation summaries will be written.",
    )
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="Only build candidate runtime configs and summary; do not execute eval.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    candidate_bundle = load_candidate_bundle(args.candidate_configs)
    runtime_config = load_benchmark_config(args.runtime_config)
    summary = validate_candidate_configs(
        candidate_bundle=candidate_bundle,
        runtime_config=runtime_config,
        output_root=Path(args.output_dir),
        execute_eval=not args.no_execute,
    )
    json_path, md_path = write_validation_summary(summary, args.output_dir)
    print(f"Wrote validation summary JSON to {json_path}")
    print(f"Wrote validation summary Markdown to {md_path}")


if __name__ == "__main__":
    main()
