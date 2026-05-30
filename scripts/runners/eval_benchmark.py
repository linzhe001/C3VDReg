#!/usr/bin/env python3
"""CLI wrapper for the benchmark eval runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmarking.config.loader import load_benchmark_config  # noqa: E402
from src.benchmarking.runners.eval_runner import run_eval  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Path to benchmark eval config."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_benchmark_config(args.config)
    summary = run_eval(config)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
