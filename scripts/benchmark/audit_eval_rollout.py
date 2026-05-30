#!/usr/bin/env python3
"""Audit benchmark eval rollout readiness and emit per-model configs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmarking.diagnostics.eval_rollout import (  # noqa: E402
    generate_eval_rollout_audit,
    write_eval_rollout_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-config",
        default="configs/benchmark/runtime/smoke_eval.yaml",
        help="Base runtime config used to seed per-model eval configs.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/benchmark/eval_rollout_audit",
        help="Directory for generated configs and audit reports.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of model ids to audit.",
    )
    parser.add_argument(
        "--no-write-configs",
        action="store_true",
        help="Skip emitting per-model eval config YAML files.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the full audit JSON to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = (REPO_ROOT / args.output_root).resolve()
    report = generate_eval_rollout_audit(
        repo_root=REPO_ROOT,
        runtime_config_path=(REPO_ROOT / args.runtime_config).resolve(),
        output_root=output_root,
        model_ids=args.models,
        write_configs=not args.no_write_configs,
    )
    json_path, markdown_path = write_eval_rollout_report(report, output_root)

    if args.print_json:
        print(json.dumps(report, indent=2))
    else:
        ready_count = sum(1 for entry in report["models"] if entry["ready_to_run"])
        total_count = len(report["models"])
        print(
            f"Audit complete: {ready_count}/{total_count} models ready. "
            f"JSON={json_path} MD={markdown_path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
