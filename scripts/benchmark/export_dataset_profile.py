"""Export a measured dataset profile for DPG-HPT."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.benchmarking.hparam_transfer.dataset_profiles import (
    DEFAULT_SAMPLE_FRAMES,
    measure_dataset_profile,
    write_dataset_profile_outputs,
)

DEFAULT_DATASET_PROFILE_DIR = "src/benchmarking/hparam_transfer/dataset_profiles"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-config",
        required=True,
        help="Path to dataset config YAML.",
    )
    parser.add_argument(
        "--subset-config",
        default=None,
        help=(
            "Optional subset config JSON. Defaults to the path declared in "
            "dataset config."
        ),
    )
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Stable dataset profile id.",
    )
    parser.add_argument(
        "--sample-frames",
        nargs="+",
        default=list(DEFAULT_SAMPLE_FRAMES),
        choices=["first", "middle", "last"],
        help="Per-scene representative frames used for geometry measurement.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_DATASET_PROFILE_DIR,
        help="Directory where JSON and Markdown profile artifacts will be written.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    profile = measure_dataset_profile(
        dataset_config_path=args.dataset_config,
        subset_config_path=args.subset_config,
        dataset_id=args.dataset_id,
        sample_frames=args.sample_frames,
    )
    json_path, md_path = write_dataset_profile_outputs(profile, Path(args.output_dir))
    print(f"Wrote dataset profile JSON to {json_path}")
    print(f"Wrote dataset profile Markdown to {md_path}")


if __name__ == "__main__":
    main()
