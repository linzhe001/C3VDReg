#!/usr/bin/env python3
"""Plan or execute benchmark train rollout runs from stable runtime configs."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmarking.diagnostics.train_rollout import (  # noqa: E402
    build_train_launch_plan,
    generate_train_rollout_audit,
    write_train_rollout_report,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-config",
        default="configs/benchmark/runtime/full_train.yaml",
        help="Base runtime config used to seed per-model train configs.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/benchmark/train_rollout_launch",
        help="Directory for generated configs, audits, plans, and execution logs.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of model ids to include in the rollout plan.",
    )
    parser.add_argument(
        "--ready-only",
        action="store_true",
        help="Only include ready-for-requested-mode train entries in the plan.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Sequentially execute ready train entries via train_benchmark.py.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep launching later ready entries after one execution fails.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the launch payload JSON to stdout instead of a short summary.",
    )
    return parser.parse_args(argv)


def _runner_command(config_path: str | Path) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "runners" / "train_benchmark.py"),
        "--config",
        str(Path(config_path).resolve()),
    ]


def _build_launch_payload(
    report: dict[str, Any],
    plan: list[dict[str, Any]],
    audit_json_path: Path,
    audit_markdown_path: Path,
    execution_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    execution_results = execution_results or []
    ready_count = sum(1 for entry in plan if entry["ready_for_requested_mode"])
    blocked_count = len(plan) - ready_count
    success_count = sum(1 for entry in execution_results if entry["status"] == "success")
    failed_count = sum(1 for entry in execution_results if entry["status"] == "failed")
    return {
        "runtime_config": report["runtime_config"],
        "requested_train_mode": report["requested_train_mode"],
        "stable_train_runner_mode": report["stable_train_runner_mode"],
        "environment": report["environment"],
        "audit_json_path": str(audit_json_path),
        "audit_markdown_path": str(audit_markdown_path),
        "plan_summary": {
            "planned_count": len(plan),
            "ready_count": ready_count,
            "blocked_count": blocked_count,
            "executed_count": len(execution_results),
            "success_count": success_count,
            "failed_count": failed_count,
        },
        "plan": plan,
        "execution_results": execution_results,
    }


def _build_launch_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Train Rollout Launch Plan",
        "",
        f"- Runtime config: `{payload['runtime_config']}`",
        f"- Requested train mode: `{payload['requested_train_mode']}`",
        f"- Stable train runner mode: `{payload['stable_train_runner_mode']}`",
        f"- CUDA available: `{payload['environment']['cuda_available']}`",
        f"- Planned entries: `{payload['plan_summary']['planned_count']}`",
        f"- Ready entries: `{payload['plan_summary']['ready_count']}`",
        f"- Blocked entries: `{payload['plan_summary']['blocked_count']}`",
        f"- Executed entries: `{payload['plan_summary']['executed_count']}`",
        "",
        "| model | ready | runtime | blockers | config |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in payload["plan"]:
        blockers = "; ".join(entry["blockers"]) if entry["blockers"] else "-"
        lines.append(
            f"| {entry['model_id']} | {entry['ready_for_requested_mode']} | "
            f"{entry['runtime_device']} | {blockers} | {entry['config_path']} |"
        )

    if payload["execution_results"]:
        lines.extend(
            [
                "",
                "| executed model | status | return code | log |",
                "| --- | --- | --- | --- |",
            ]
        )
        for entry in payload["execution_results"]:
            lines.append(
                f"| {entry['model_id']} | {entry['status']} | "
                f"{entry['return_code']} | {entry['log_path']} |"
            )

    lines.append("")
    return "\n".join(lines)


def _write_launch_artifacts(
    payload: dict[str, Any],
    output_root: Path,
) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "train_launch_plan.json"
    markdown_path = output_root / "train_launch_plan.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_build_launch_markdown(payload), encoding="utf-8")
    return json_path, markdown_path


def _execute_ready_entries(
    plan: list[dict[str, Any]],
    output_root: Path,
    continue_on_error: bool,
) -> list[dict[str, Any]]:
    execution_results: list[dict[str, Any]] = []
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    for index, entry in enumerate(plan, start=1):
        if not entry["ready_for_requested_mode"]:
            continue

        command = _runner_command(entry["config_path"])
        log_path = logs_dir / f"{index:02d}_{entry['model_id']}.log"
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(f"$ {' '.join(shlex.quote(token) for token in command)}\n\n")
            handle.flush()
            try:
                subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                execution_results.append(
                    {
                        "model_id": entry["model_id"],
                        "status": "failed",
                        "return_code": exc.returncode,
                        "log_path": str(log_path),
                    }
                )
                if not continue_on_error:
                    break
            else:
                execution_results.append(
                    {
                        "model_id": entry["model_id"],
                        "status": "success",
                        "return_code": 0,
                        "log_path": str(log_path),
                    }
                )

    return execution_results


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = (REPO_ROOT / args.output_root).resolve()
    report = generate_train_rollout_audit(
        repo_root=REPO_ROOT,
        runtime_config_path=(REPO_ROOT / args.runtime_config).resolve(),
        output_root=output_root,
        model_ids=args.models,
        write_configs=True,
    )
    audit_json_path, audit_markdown_path = write_train_rollout_report(report, output_root)
    plan = build_train_launch_plan(report, ready_only=args.ready_only)

    execution_results: list[dict[str, Any]] = []
    exit_code = 0
    if args.execute:
        execution_results = _execute_ready_entries(
            plan=plan,
            output_root=output_root,
            continue_on_error=args.continue_on_error,
        )
        if not any(entry["ready_for_requested_mode"] for entry in plan):
            exit_code = 2
        elif any(entry["status"] == "failed" for entry in execution_results):
            exit_code = 1

    payload = _build_launch_payload(
        report=report,
        plan=plan,
        audit_json_path=audit_json_path,
        audit_markdown_path=audit_markdown_path,
        execution_results=execution_results,
    )
    plan_json_path, plan_markdown_path = _write_launch_artifacts(payload, output_root)

    if args.print_json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"Launch plan ready: {payload['plan_summary']['ready_count']}/"
            f"{payload['plan_summary']['planned_count']} ready. "
            f"Audit={audit_json_path} Plan={plan_json_path} "
            f"PlanMD={plan_markdown_path}"
        )
        if args.execute:
            print(
                f"Execution summary: {payload['plan_summary']['success_count']}/"
                f"{payload['plan_summary']['executed_count']} succeeded."
            )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
