"""Render a human-readable DPG-HPT transfer report."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.benchmarking.hparam_transfer.proposal_validation import (
    load_agent_proposal,
    load_context_pack,
)
from src.benchmarking.hparam_transfer.report_rendering import (
    load_validation_payload,
    render_transfer_report,
    write_transfer_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--context",
        required=True,
        help="Path to frozen context pack JSON or YAML.",
    )
    parser.add_argument(
        "--proposal",
        required=True,
        help="Path to agent proposal YAML or JSON.",
    )
    parser.add_argument(
        "--validation",
        required=True,
        help="Path to proposal validation JSON or YAML.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where transfer_report.md will be written.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    context_pack = load_context_pack(args.context)
    proposal = load_agent_proposal(args.proposal)
    validation = load_validation_payload(args.validation)
    report = render_transfer_report(
        context_pack=context_pack,
        proposal=proposal,
        validation=validation,
    )
    report_path = write_transfer_report(report, Path(args.output_dir))
    print(f"Wrote transfer report Markdown to {report_path}")


if __name__ == "__main__":
    main()
