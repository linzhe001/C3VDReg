"""Initialize an agent_proposal.yaml skeleton from a frozen context pack."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.benchmarking.hparam_transfer.proposal_template import (
    build_agent_proposal_template,
    write_agent_proposal_template,
)
from src.benchmarking.hparam_transfer.proposal_validation import load_context_pack


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--context",
        required=True,
        help="Path to frozen context pack JSON or YAML.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where agent_proposal.yaml will be written.",
    )
    parser.add_argument(
        "--agent-name",
        default="codex_or_claude_code",
        help="Agent name to pre-fill in the template.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    context_pack = load_context_pack(args.context)
    proposal = build_agent_proposal_template(
        context_pack=context_pack,
        agent_name=args.agent_name,
    )
    proposal_path = write_agent_proposal_template(
        proposal=proposal,
        output_dir=Path(args.output_dir),
    )
    print(f"Wrote agent proposal template to {proposal_path}")


if __name__ == "__main__":
    main()
