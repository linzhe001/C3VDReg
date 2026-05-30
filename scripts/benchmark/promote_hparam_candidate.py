"""Promote one DPG-HPT candidate into a durable benchmark model config."""

from __future__ import annotations

import argparse

from src.benchmarking.hparam_transfer.promotion import (
    build_promoted_model_config,
    load_base_model_config,
    load_candidate_bundle_payload,
    load_validation_summary,
    write_promoted_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-configs",
        required=True,
        help="Path to candidate config bundle YAML or JSON.",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="Candidate name to promote, e.g. default.",
    )
    parser.add_argument(
        "--base-config",
        required=True,
        help="Path to the durable benchmark model config used as merge base.",
    )
    parser.add_argument(
        "--validation-summary",
        default=None,
        help=(
            "Optional candidate validation summary JSON/YAML; when provided, "
            "only passed candidates can be promoted."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where preview promotion artifacts will be written.",
    )
    parser.add_argument(
        "--target-config",
        default=None,
        help="Durable config path to overwrite when --execute is set.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually write the promoted config to --target-config.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    candidate_bundle = load_candidate_bundle_payload(args.candidate_configs)
    base_config = load_base_model_config(args.base_config)
    validation_summary = load_validation_summary(args.validation_summary)
    promoted_config, manifest = build_promoted_model_config(
        candidate_bundle=candidate_bundle,
        base_config=base_config,
        candidate_name=args.candidate,
        candidate_bundle_path=args.candidate_configs,
        base_config_path=args.base_config,
        validation_summary=validation_summary,
        validation_summary_path=args.validation_summary,
    )
    artifact_paths = write_promoted_artifacts(
        promoted_config=promoted_config,
        manifest=manifest,
        output_dir=args.output_dir,
        candidate_name=args.candidate,
        target_config_path=args.target_config,
        execute=args.execute,
    )
    print(f"Wrote preview promoted config to {artifact_paths['preview_config_path']}")
    print(
        f"Wrote preview promotion manifest to {artifact_paths['preview_manifest_path']}"
    )
    if artifact_paths["target_config_path"] is not None:
        print(f"Wrote promoted config to {artifact_paths['target_config_path']}")
        print(f"Wrote promotion manifest to {artifact_paths['target_manifest_path']}")


if __name__ == "__main__":
    main()
