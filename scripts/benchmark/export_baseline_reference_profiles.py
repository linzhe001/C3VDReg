"""Export route-aware baseline reference profiles for DPG-HPT."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.benchmarking.hparam_transfer.reference_profiles import (
    export_reference_profiles,
    write_reference_profile_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Model id in baseline_routes.yaml.",
    )
    parser.add_argument(
        "--routes",
        default="configs/benchmark/hparam_transfer/baseline_routes.yaml",
        help="Path to baseline route registry.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where JSON and Markdown artifacts will be written.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bundle = export_reference_profiles(
        model_id=args.model,
        route_registry_path=args.routes,
    )
    json_path, md_path = write_reference_profile_outputs(bundle, Path(args.output_dir))
    print(f"Wrote reference profile JSON to {json_path}")
    print(f"Wrote reference profile Markdown to {md_path}")


if __name__ == "__main__":
    main()
