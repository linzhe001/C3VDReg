"""Prepare a frozen DPG-HPT context pack for one model and target dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.benchmarking.hparam_transfer.context_pack import (
    build_context_pack,
    load_route_card,
    load_transfer_rules,
    write_context_pack_outputs,
)
from src.benchmarking.hparam_transfer.dataset_profiles import (
    DEFAULT_SAMPLE_FRAMES,
    load_profile_stub,
    measure_dataset_profile,
    write_dataset_profile_outputs,
)
from src.benchmarking.hparam_transfer.reference_profiles import (
    export_reference_profiles,
    write_reference_profile_outputs,
)

DEFAULT_DATASET_PROFILE_DIR = "src/benchmarking/hparam_transfer/dataset_profiles"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Model id in baseline_routes.yaml.",
    )
    parser.add_argument(
        "--target-dataset",
        required=True,
        help="Target dataset id.",
    )
    parser.add_argument(
        "--dataset-config",
        required=True,
        help="Target dataset config YAML.",
    )
    parser.add_argument(
        "--subset-config",
        default=None,
        help="Optional subset config JSON. Defaults to the dataset config declaration.",
    )
    parser.add_argument(
        "--routes",
        default="configs/benchmark/hparam_transfer/baseline_routes.yaml",
        help="Path to baseline route registry.",
    )
    parser.add_argument(
        "--rules",
        default="configs/benchmark/hparam_transfer/transfer_rules.yaml",
        help="Path to transfer rule registry.",
    )
    parser.add_argument(
        "--dataset-profile-registry",
        default=f"{DEFAULT_DATASET_PROFILE_DIR}/dataset_profiles.yaml",
        help="Path to durable dataset profile registry used for stub fallback.",
    )
    parser.add_argument(
        "--dataset-profile-output-dir",
        default=DEFAULT_DATASET_PROFILE_DIR,
        help=(
            "Repository-local directory where dataset profile JSON/Markdown "
            "artifacts are written."
        ),
    )
    parser.add_argument(
        "--reference-profile-output-dir",
        default=None,
        help=(
            "Directory for model-specific reference profile artifacts. Defaults "
            "to a sibling 'reference_profiles' directory next to context/."
        ),
    )
    parser.add_argument(
        "--allow-profile-stub-fallback",
        action="store_true",
        help=(
            "Fallback to dataset profile registry stub when manifest-based "
            "measurement is unavailable."
        ),
    )
    parser.add_argument(
        "--sample-frames",
        nargs="+",
        default=list(DEFAULT_SAMPLE_FRAMES),
        choices=["first", "middle", "last"],
        help="Per-scene representative frames used for target profile measurement.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where context_pack.json and context_pack.md will be written.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    dataset_profile_dir = Path(args.dataset_profile_output_dir)
    reference_profile_dir = (
        Path(args.reference_profile_output_dir)
        if args.reference_profile_output_dir is not None
        else output_dir.parent / "reference_profiles"
    )

    try:
        target_profile = measure_dataset_profile(
            dataset_config_path=args.dataset_config,
            subset_config_path=args.subset_config,
            dataset_id=args.target_dataset,
            sample_frames=args.sample_frames,
        )
    except (FileNotFoundError, ValueError):
        if not args.allow_profile_stub_fallback:
            raise
        target_profile = load_profile_stub(
            registry_path=args.dataset_profile_registry,
            dataset_id=args.target_dataset,
        )
    write_dataset_profile_outputs(target_profile, dataset_profile_dir)

    reference_profiles = export_reference_profiles(
        model_id=args.model,
        route_registry_path=args.routes,
    )
    write_reference_profile_outputs(reference_profiles, reference_profile_dir)

    route_card = load_route_card(args.routes, args.model)
    transfer_rules = load_transfer_rules(args.rules)
    context_pack = build_context_pack(
        model_id=args.model,
        target_dataset=args.target_dataset,
        target_profile=target_profile,
        reference_profiles=reference_profiles,
        route_card=route_card,
        transfer_rules=transfer_rules,
    )
    json_path, md_path = write_context_pack_outputs(context_pack, output_dir)
    print(f"Wrote context pack JSON to {json_path}")
    print(f"Wrote context pack Markdown to {md_path}")


if __name__ == "__main__":
    main()
