"""Validate one DPG-HPT agent proposal against a frozen context pack."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.benchmarking.hparam_transfer.proposal_validation import (
    build_transfer_trace,
    load_agent_proposal,
    load_context_pack,
    normalize_candidate_configs,
    validate_proposal,
    write_validation_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--proposal",
        required=True,
        help="Path to agent proposal YAML or JSON.",
    )
    parser.add_argument(
        "--context",
        required=True,
        help="Path to frozen context pack JSON or YAML.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where validation outputs will be written.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    proposal = load_agent_proposal(args.proposal)
    context_pack = load_context_pack(args.context)
    validation = validate_proposal(proposal, context_pack)
    candidate_configs = None
    if validation["passed"]:
        candidate_configs = normalize_candidate_configs(
            proposal,
            context_pack,
            validation,
        )
    trace = build_transfer_trace(
        proposal,
        context_pack,
        validation,
        candidate_configs=candidate_configs,
    )
    validation_path, candidate_path, trace_path = write_validation_outputs(
        validation,
        candidate_configs,
        trace,
        Path(args.output_dir),
    )
    print(f"Wrote proposal validation JSON to {validation_path}")
    if candidate_path is not None:
        print(f"Wrote candidate configs YAML to {candidate_path}")
    print(f"Wrote transfer trace JSON to {trace_path}")


if __name__ == "__main__":
    main()
