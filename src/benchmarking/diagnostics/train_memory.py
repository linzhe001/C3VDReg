"""Train-time CUDA memory profiling across model/point-count sweeps."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml

import src.benchmarking.runners.train_runner as train_runner
from src.benchmarking.diagnostics.compatibility import (
    assert_baseline_repo_clean,
    assert_runtime_policy_compatible,
)
from src.benchmarking.diagnostics.train_rollout import (
    TRAIN_BRIDGE_BUILDERS,
    _patched_train_runner_repo_root,
    build_train_config,
)
from src.benchmarking.registry.model_registry import ModelRegistry

DEFAULT_TRAIN_MEMORY_MODELS = (
    "bufferx",
    "dcp",
    "pointnetlk",
    "pointnetlk_revisited",
    "mamba3d",
    "mamba3d_mamba2",
    "mamba3d_mamba2_direct",
    "mambanetlk",
    "regtr",
    "geotransformer",
)


def _gpu_index(device: str) -> int:
    if ":" not in device:
        return 0
    return int(device.rsplit(":", 1)[-1])


def _memory_mb(value: int) -> float:
    return float(value) / (1024.0 * 1024.0)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {} if payload is None else dict(payload)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _tail_file(path: Path, max_lines: int = 60) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _descendant_pids(root_pid: int) -> set[int]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid="],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {root_pid}

    children_by_parent: dict[int, list[int]] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        children_by_parent.setdefault(ppid, []).append(pid)

    pids = {root_pid}
    stack = [root_pid]
    while stack:
        parent = stack.pop()
        for child in children_by_parent.get(parent, []):
            if child not in pids:
                pids.add(child)
                stack.append(child)
    return pids


def _query_compute_process_memory_mb(
    device_index: int,
    tracked_pids: set[int],
) -> tuple[float, dict[int, float]]:
    if not tracked_pids:
        return 0.0, {}
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={device_index}",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return 0.0, {}

    per_pid: dict[int, float] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
            used_mb = float(parts[1])
        except ValueError:
            continue
        if pid in tracked_pids:
            per_pid[pid] = used_mb
    return sum(per_pid.values()), per_pid


def _classify_command_failure(return_code: int, log_tail: str) -> str:
    lowered = log_tail.lower()
    if "out of memory" in lowered or "cuda oom" in lowered:
        return "oom"
    if return_code != 0:
        return "failed"
    return "ok"


def _run_monitored_command(
    command: list[str],
    cwd: Path,
    log_path: Path,
    device_index: int,
    poll_interval_sec: float,
    timeout_seconds: float | None,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    peak_used_mb = 0.0
    peak_per_pid: dict[int, float] = {}
    start = time.perf_counter()

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        timed_out = False
        while process.poll() is None:
            elapsed = time.perf_counter() - start
            if timeout_seconds is not None and elapsed > timeout_seconds:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                break
            used_mb, per_pid = _query_compute_process_memory_mb(
                device_index,
                _descendant_pids(process.pid),
            )
            if used_mb > peak_used_mb:
                peak_used_mb = used_mb
                peak_per_pid = per_pid
            time.sleep(max(float(poll_interval_sec), 0.01))

        return_code = process.wait()

    duration_ms = (time.perf_counter() - start) * 1000.0
    log_tail = _tail_file(log_path)
    status = (
        "timeout"
        if timed_out
        else _classify_command_failure(return_code, log_tail)
    )
    blocker = None
    if status != "ok":
        blocker = (
            f"command exited with return code {return_code}; "
            f"log tail:\n{log_tail}"
        )

    return {
        "status": status,
        "return_code": return_code,
        "peak_used_memory_mb": peak_used_mb if peak_used_mb > 0.0 else None,
        "peak_process_memory_mb": peak_per_pid,
        "duration_ms": duration_ms,
        "log_path": str(log_path),
        "blocker": blocker,
    }


def _torch_profile_command(
    repo_root: Path,
    script_command: list[str],
    metrics_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(repo_root / "scripts" / "benchmark" / "_profile_torch_command.py"),
        "--metrics-path",
        str(metrics_path),
        "--",
        *script_command,
    ]


def _load_torch_metrics(metrics_path: Path) -> dict[str, Any] | None:
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _resolve_existing_checkpoint(*candidates: Path) -> Path:
    return train_runner._resolve_existing_checkpoint(*candidates)


def _apply_profile_bridge_overrides(
    bridge_config: dict[str, Any],
    model_id: str,
    micro_batch_size: int,
    steps_per_case: int,
) -> None:
    micro_batch_size = max(int(micro_batch_size), 1)
    steps_per_case = max(int(steps_per_case), 1)
    min_pairs = micro_batch_size * steps_per_case
    dataset = bridge_config.get("dataset", {})
    if isinstance(dataset, dict):
        dataset["max_train_pairs"] = max(
            int(dataset.get("max_train_pairs", 0)),
            min_pairs,
        )
        dataset["max_val_pairs"] = max(
            int(dataset.get("max_val_pairs", 0)),
            micro_batch_size,
        )
        dataset["max_test_pairs"] = max(
            int(dataset.get("max_test_pairs", 0)),
            micro_batch_size,
        )

    if model_id == "dcp":
        training = bridge_config["training"]
        training["batch_size"] = micro_batch_size
        training["test_batch_size"] = micro_batch_size
        training["num_epochs"] = 1
        training["max_train_steps"] = steps_per_case
        training["max_test_steps"] = 1
        training["save_interval"] = 1
        training["val_interval"] = 999999
    elif model_id == "pointnetlk":
        training = bridge_config["training"]
        training["batch_size"] = max(micro_batch_size, 2)
        for stage in ("classifier", "pointnetlk"):
            training[stage]["epochs"] = 1
            training[stage]["max_train_steps"] = steps_per_case
            training[stage]["max_test_steps"] = 1
    elif model_id == "pointnetlk_revisited":
        training = bridge_config["training"]
        training["batch_size"] = micro_batch_size
        training["max_epochs"] = 1
        training["max_train_steps"] = steps_per_case
        training["max_val_steps"] = 1
        bridge_config["testing"]["batch_size"] = micro_batch_size
    elif model_id in {
        "mamba3d",
        "mamba3d_true",
        "mamba3d_mamba2",
        "mamba3d_mamba2_direct",
        "mambanetlk",
    }:
        training = bridge_config["training"]
        training["batch_size"] = micro_batch_size
        training["epochs"] = 1
        training["max_train_steps"] = steps_per_case
        training["max_test_steps"] = 1
    elif model_id == "regtr":
        bridge_config["dataset"]["train_batch_size"] = micro_batch_size
        bridge_config["dataset"]["val_batch_size"] = micro_batch_size
        bridge_config["dataset"]["test_batch_size"] = micro_batch_size
        bridge_config["train_options"]["niter"] = steps_per_case
        bridge_config["dataloader"]["num_workers"] = 0
        bridge_config["dataloader"]["persistent_workers"] = False
        bridge_config["dataloader"]["pin_memory"] = False
    elif model_id == "geotransformer":
        training = bridge_config["training"]
        training["batch_size"] = 1
        training["max_epochs"] = 1
        training["max_train_steps"] = steps_per_case
        training["max_val_steps"] = 1
        training["num_workers"] = 0
    elif model_id == "bufferx":
        training = bridge_config["bufferx"]["train"]
        training["batch_size"] = 1
        training["epoch"] = 1
        training["max_iter"] = steps_per_case
        training["num_workers"] = 0


def _build_case_config_and_bridge(
    repo_root: Path,
    runtime_config_path: Path,
    output_root: Path,
    model_id: str,
    num_points: int,
    device: str | None,
    micro_batch_size: int,
    steps_per_case: int,
    manifest_path: str | None,
    subset_config_path: str | None,
    dataset_root: str | None,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    registry = ModelRegistry()
    spec = registry.get(model_id)
    case_output = output_root / "runs" / f"{model_id}_{num_points}"
    config = build_train_config(
        repo_root=repo_root,
        runtime_config_path=runtime_config_path,
        output_root=case_output,
        spec=spec,
    )
    config["preprocess"]["num_points_override"] = int(num_points)
    config["runtime"]["num_workers"] = 0
    if device is not None:
        config["runtime"]["device"] = device
    if manifest_path is not None:
        config["data"]["manifest_path"] = manifest_path
    if subset_config_path is not None:
        config["data"]["subset_config_path"] = subset_config_path
    if dataset_root is not None:
        config["data"]["dataset_root"] = dataset_root

    builder = TRAIN_BRIDGE_BUILDERS[model_id]
    with _patched_train_runner_repo_root(repo_root):
        bridge_config = builder(config, case_output / "bridge_run")
    _apply_profile_bridge_overrides(
        bridge_config,
        model_id=model_id,
        micro_batch_size=micro_batch_size,
        steps_per_case=steps_per_case,
    )

    config_path = case_output / "benchmark_train_config.yaml"
    bridge_config_path = case_output / f"{model_id}_train_memory_bridge.yaml"
    _write_yaml(config_path, config)
    _write_yaml(bridge_config_path, bridge_config)
    return config, bridge_config, bridge_config_path


def _effective_micro_batch_size(model_id: str, bridge_config: dict[str, Any]) -> int:
    if model_id == "dcp":
        return int(bridge_config["training"]["batch_size"])
    if model_id == "pointnetlk":
        return int(bridge_config["training"]["batch_size"])
    if model_id == "pointnetlk_revisited":
        return int(bridge_config["training"]["batch_size"])
    if model_id in {
        "mamba3d",
        "mamba3d_true",
        "mamba3d_mamba2",
        "mamba3d_mamba2_direct",
        "mambanetlk",
    }:
        return int(bridge_config["training"]["batch_size"])
    if model_id == "regtr":
        return int(bridge_config["dataset"]["train_batch_size"])
    if model_id == "geotransformer":
        return int(bridge_config["training"]["batch_size"])
    if model_id == "bufferx":
        return int(bridge_config["bufferx"]["train"]["batch_size"])
    return 0


def _run_profile_bridge_commands(
    repo_root: Path,
    model_id: str,
    config: dict[str, Any],
    bridge_config: dict[str, Any],
    bridge_config_path: Path,
    device_index: int,
    poll_interval_sec: float,
    timeout_seconds: float | None,
) -> dict[str, Any]:
    case_output = bridge_config_path.parent
    command_results: list[dict[str, Any]] = []

    def run_one(name: str, script_command: list[str]) -> dict[str, Any]:
        metrics_path = case_output / "metrics" / f"{name}.json"
        result = _run_monitored_command(
            command=_torch_profile_command(repo_root, script_command, metrics_path),
            cwd=repo_root,
            log_path=case_output / "logs" / f"{name}.log",
            device_index=device_index,
            poll_interval_sec=poll_interval_sec,
            timeout_seconds=timeout_seconds,
        )
        torch_metrics = _load_torch_metrics(metrics_path)
        if torch_metrics is not None:
            torch_peak_reserved = torch_metrics.get("torch_peak_reserved_mb")
            if (
                result["peak_used_memory_mb"] is None
                and torch_peak_reserved is not None
            ):
                result["peak_used_memory_mb"] = float(torch_peak_reserved)
            result["torch_peak_allocated_mb"] = torch_metrics.get(
                "torch_peak_allocated_mb"
            )
            result["torch_peak_reserved_mb"] = torch_peak_reserved
            result["torch_metrics_path"] = str(metrics_path)
            exception = torch_metrics.get("exception")
            if exception and result["blocker"] is not None:
                result["blocker"] = (
                    f"{result['blocker']}\nWrapper exception:\n{exception}"
                )
        command_results.append(
            {"name": name, "command": script_command, **result}
        )
        return result

    if model_id == "dcp":
        run_one(
            "dcp_train",
            [
                str(
                    repo_root
                    / "src"
                    / "benchmarking"
                    / "bridges"
                    / "train_dcp_c3vd.py"
                ),
                "--config",
                str(bridge_config_path),
            ],
        )
    elif model_id == "pointnetlk":
        dataset_root = bridge_config["dataset"]["data_root"]
        script = (
            repo_root
            / "src"
            / "benchmarking"
            / "bridges"
            / "train_pointnetlk_c3vd.py"
        )
        first = run_one(
            "pointnetlk_classifier",
            [
                str(script),
                "--config",
                str(bridge_config_path),
                "--stage",
                "classifier",
                "--data-root",
                dataset_root,
            ],
        )
        if first["status"] == "ok":
            checkpoint_dir = Path(bridge_config["output"]["checkpoint_dir"])
            classifier_prefix = bridge_config["output"]["classifier_prefix"]
            transfer_from = _resolve_existing_checkpoint(
                checkpoint_dir / f"{classifier_prefix}_feat_best.pth",
                checkpoint_dir / f"{classifier_prefix}_feat_last.pth",
            )
            run_one(
                "pointnetlk_registration",
                [
                    str(script),
                    "--config",
                    str(bridge_config_path),
                    "--stage",
                    "pointnetlk",
                    "--data-root",
                    dataset_root,
                    "--transfer-from",
                    str(transfer_from),
                ],
            )
    elif model_id == "pointnetlk_revisited":
        run_one(
            "pointnetlk_revisited_train",
            [
                str(
                    repo_root
                    / "src"
                    / "benchmarking"
                    / "bridges"
                    / "train_pointnetlk_revisited_c3vd.py"
                ),
                "--config",
                str(bridge_config_path),
            ],
        )
    elif model_id in {
        "mamba3d",
        "mamba3d_true",
        "mamba3d_mamba2",
        "mamba3d_mamba2_direct",
        "mambanetlk",
    }:
        run_one(
            f"{model_id}_train",
            [
                str(
                    repo_root
                    / "src"
                    / "benchmarking"
                    / "bridges"
                    / "train_mamba3d_c3vd.py"
                ),
                "--config",
                str(bridge_config_path),
                "--data-root",
                str(config["data"]["dataset_root"]),
            ],
        )
    elif model_id == "regtr":
        logs_root = case_output / "train_bridge_logs"
        experiment_name = (
            f"memory_profile_{model_id}_"
            f"{bridge_config['dataset']['num_points']}"
        )
        run_one(
            "regtr_train",
            [
                str(
                    repo_root
                    / "src"
                    / "benchmarking"
                    / "bridges"
                    / "train_regtr_c3vd.py"
                ),
                "--config",
                str(bridge_config_path),
                "--logdir",
                str(logs_root),
                "--name",
                experiment_name,
                "--validate_every",
                "1",
                "--num_workers",
                "0",
                "--nb_sanity_val_steps",
                "1",
            ],
        )
    elif model_id == "geotransformer":
        run_one(
            "geotransformer_train",
            [
                str(
                    repo_root
                    / "src"
                    / "benchmarking"
                    / "bridges"
                    / "train_geotransformer_c3vd.py"
                ),
                "--config",
                str(bridge_config_path),
            ],
        )
    elif model_id == "bufferx":
        run_one(
            "bufferx_train",
            [
                str(
                    repo_root
                    / "src"
                    / "benchmarking"
                    / "bridges"
                    / "train_bufferx_c3vd.py"
                ),
                "--config",
                str(bridge_config_path),
            ],
        )
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"Unsupported train memory model id: {model_id}")

    peak_values = [
        result["peak_used_memory_mb"]
        for result in command_results
        if result.get("peak_used_memory_mb") is not None
    ]
    torch_allocated_values = [
        result["torch_peak_allocated_mb"]
        for result in command_results
        if result.get("torch_peak_allocated_mb") is not None
    ]
    torch_reserved_values = [
        result["torch_peak_reserved_mb"]
        for result in command_results
        if result.get("torch_peak_reserved_mb") is not None
    ]
    statuses = [result["status"] for result in command_results]
    if not command_results:
        status = "blocked"
        blocker = "No train command was executed."
    elif any(status == "oom" for status in statuses):
        status = "oom"
        blocker = next(
            result["blocker"] for result in command_results if result["status"] == "oom"
        )
    elif any(status != "ok" for status in statuses):
        status = "failed"
        blocker = next(
            result["blocker"] for result in command_results if result["status"] != "ok"
        )
    else:
        status = "ok"
        blocker = None

    return {
        "status": status,
        "blocker": blocker,
        "peak_used_memory_mb": max(peak_values) if peak_values else None,
        "torch_peak_allocated_mb": (
            max(torch_allocated_values) if torch_allocated_values else None
        ),
        "torch_peak_reserved_mb": (
            max(torch_reserved_values) if torch_reserved_values else None
        ),
        "duration_ms": sum(float(result["duration_ms"]) for result in command_results),
        "commands": command_results,
    }


def _build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Train Memory Profile",
        "",
        f"- Runtime config: `{report['runtime_config']}`",
        f"- CUDA available: `{report['environment']['cuda_available']}`",
        f"- Memory limit GB: `{report['memory_limit_gb']}`",
        f"- Safety ratio: `{report['safety_ratio']}`",
        f"- Usable limit MB: `{report['usable_limit_mb']:.2f}`",
        f"- Requested micro batch size: `{report['micro_batch_size']}`",
        "",
        "| model | max safe points | max effective batch | status | blocker |",
        "| --- | --- | --- | --- | --- |",
    ]
    for summary in report["summaries"]:
        blocker = summary["first_blocker"] or "-"
        max_safe = summary["max_safe_num_points"]
        lines.append(
            f"| {summary['model_id']} | {max_safe if max_safe is not None else '-'} | "
            f"{summary['max_safe_effective_micro_batch_size'] or '-'} | "
            f"{summary['status']} | {blocker} |"
        )

    lines.extend(
        [
            "",
            "| model | num_points | requested batch | effective batch | status | "
            "safe | peak MB | time ms | blocker |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for entry in report["entries"]:
        blocker = entry["blocker"] or "-"
        peak = (
            f"{entry['peak_used_memory_mb']:.2f}"
            if entry["peak_used_memory_mb"] is not None
            else "-"
        )
        time_ms = (
            f"{entry['duration_ms']:.1f}" if entry["duration_ms"] is not None else "-"
        )
        lines.append(
            f"| {entry['model_id']} | {entry['num_points']} | "
            f"{entry['micro_batch_size']} | {entry['effective_micro_batch_size']} | "
            f"{entry['status']} | {entry['safe_under_limit']} | {peak} | "
            f"{time_ms} | {blocker} |"
        )
    lines.append("")
    return "\n".join(lines)


def _raw_point_memory_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in report["entries"]:
        rows.append(
            {
                "model_id": entry["model_id"],
                "raw_points": entry["num_points"],
                "effective_batch_size": entry["effective_micro_batch_size"],
                "status": entry["status"],
                "safe_under_limit": entry["safe_under_limit"],
                "peak_gpu_memory_mb": entry["peak_used_memory_mb"],
                "torch_peak_allocated_mb": entry["torch_peak_allocated_mb"],
                "blocker": entry["blocker"],
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_raw_point_memory_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Raw-Point Memory Scalability",
        "",
        f"- Runtime config: `{report['runtime_config']}`",
        f"- Memory limit GB: `{report['memory_limit_gb']}`",
        f"- Safety ratio: `{report['safety_ratio']}`",
        "",
        "| model | raw_points | effective batch | status | safe | "
        "peak GPU MB | blocker |",
        "| --- | ---: | ---: | --- | --- | ---: | --- |",
    ]
    for row in _raw_point_memory_rows(report):
        peak = row["peak_gpu_memory_mb"]
        peak_text = f"{float(peak):.2f}" if peak is not None else "-"
        lines.append(
            f"| {row['model_id']} | {row['raw_points']} | "
            f"{row['effective_batch_size']} | {row['status']} | "
            f"{row['safe_under_limit']} | {peak_text} | {row['blocker'] or '-'} |"
        )
    lines.append("")
    return "\n".join(lines)


def _summarize_entries(
    model_ids: Iterable[str],
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for model_id in model_ids:
        model_entries = [entry for entry in entries if entry["model_id"] == model_id]
        safe_points = [
            int(entry["num_points"])
            for entry in model_entries
            if entry["status"] == "ok" and entry["safe_under_limit"] is True
        ]
        max_safe_effective_batch = None
        if safe_points:
            max_safe_point = max(safe_points)
            max_safe_effective_batch = next(
                int(entry["effective_micro_batch_size"])
                for entry in model_entries
                if int(entry["num_points"]) == max_safe_point
                and entry["status"] == "ok"
                and entry["safe_under_limit"] is True
            )
        first_blocker = next(
            (entry["blocker"] for entry in model_entries if entry["blocker"]),
            None,
        )
        if safe_points:
            status = "ok"
        elif any(entry["status"] == "blocked" for entry in model_entries):
            status = "blocked"
        elif any(entry["status"] == "oom" for entry in model_entries):
            status = "oom"
        elif model_entries:
            status = "failed"
        else:
            status = "not_run"
        summaries.append(
            {
                "model_id": model_id,
                "max_safe_num_points": max(safe_points) if safe_points else None,
                "max_safe_effective_micro_batch_size": max_safe_effective_batch,
                "status": status,
                "first_blocker": first_blocker,
            }
        )
    return summaries


def write_train_memory_report(
    report: dict[str, Any],
    output_root: Path,
) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "train_memory_profile.json"
    markdown_path = output_root / "train_memory_profile.md"
    raw_memory_csv = output_root / "raw_point_memory_scalability.csv"
    raw_memory_md = output_root / "raw_point_memory_scalability.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_build_markdown(report), encoding="utf-8")
    _write_csv(raw_memory_csv, _raw_point_memory_rows(report))
    raw_memory_md.write_text(
        _build_raw_point_memory_markdown(report),
        encoding="utf-8",
    )
    return json_path, markdown_path


def profile_train_memory(
    repo_root: Path,
    runtime_config_path: Path,
    output_root: Path,
    num_points_list: Iterable[int],
    model_ids: Iterable[str] | None = None,
    device: str | None = None,
    memory_limit_gb: float = 32.0,
    safety_ratio: float = 0.90,
    micro_batch_size: int = 1,
    steps_per_case: int = 1,
    poll_interval_sec: float = 0.1,
    timeout_seconds: float | None = None,
    manifest_path: str | None = None,
    subset_config_path: str | None = None,
    dataset_root: str | None = None,
) -> dict[str, Any]:
    """Profile train-time CUDA memory for C3VD model/point-count combinations."""

    repo_root = repo_root.resolve()
    runtime_config_path = runtime_config_path.resolve()
    output_root = output_root.resolve()
    registry = ModelRegistry()
    runtime_payload = _load_yaml(runtime_config_path)
    runtime_device = str(
        device or runtime_payload.get("runtime", {}).get("device", "cuda:0")
    )
    requested_ids = (
        list(model_ids)
        if model_ids is not None
        else list(DEFAULT_TRAIN_MEMORY_MODELS)
    )
    point_counts = [int(value) for value in num_points_list]
    usable_limit_mb = float(memory_limit_gb) * 1024.0 * float(safety_ratio)
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count())
    nvidia_smi_path = shutil.which("nvidia-smi")

    entries: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "repo_root": str(repo_root),
        "runtime_config": str(runtime_config_path),
        "environment": {
            "cuda_available": cuda_available,
            "cuda_device_count": device_count,
            "nvidia_smi": nvidia_smi_path,
        },
        "memory_limit_gb": float(memory_limit_gb),
        "safety_ratio": float(safety_ratio),
        "usable_limit_mb": usable_limit_mb,
        "micro_batch_size": int(micro_batch_size),
        "steps_per_case": int(steps_per_case),
        "entries": entries,
        "summaries": [],
    }

    for model_id in requested_ids:
        spec = registry.get(model_id)
        for num_points in point_counts:
            entry: dict[str, Any] = {
                "model_id": model_id,
                "num_points": int(num_points),
                "runtime_device": runtime_device,
                "micro_batch_size": int(micro_batch_size),
                "effective_micro_batch_size": int(micro_batch_size),
                "status": "blocked",
                "safe_under_limit": None,
                "peak_used_memory_mb": None,
                "torch_peak_allocated_mb": None,
                "torch_peak_reserved_mb": None,
                "duration_ms": None,
                "config_path": None,
                "bridge_config_path": None,
                "blocker": None,
                "commands": [],
            }

            if not spec.capabilities.supports_train:
                entry["blocker"] = "Model is eval-only in the current registry."
                entries.append(entry)
                continue
            if model_id not in TRAIN_BRIDGE_BUILDERS:
                entry["blocker"] = "Train bridge builder is not implemented."
                entries.append(entry)
                continue
            if not runtime_device.startswith("cuda"):
                entry["blocker"] = (
                    "Train memory profiling requires CUDA device, "
                    f"got {runtime_device!r}."
                )
                entries.append(entry)
                continue
            if not cuda_available:
                entry["blocker"] = (
                    "CUDA unavailable: torch.cuda.is_available() is False."
                )
                entries.append(entry)
                continue
            if nvidia_smi_path is None:
                entry["blocker"] = (
                    "nvidia-smi is required to monitor child-process GPU memory."
                )
                entries.append(entry)
                continue

            try:
                assert_runtime_policy_compatible(spec, runtime_device)
                assert_baseline_repo_clean(repo_root=repo_root, spec=spec)
                (
                    config,
                    bridge_config,
                    bridge_config_path,
                ) = _build_case_config_and_bridge(
                    repo_root=repo_root,
                    runtime_config_path=runtime_config_path,
                    output_root=output_root,
                    model_id=model_id,
                    num_points=num_points,
                    device=runtime_device,
                    micro_batch_size=micro_batch_size,
                    steps_per_case=steps_per_case,
                    manifest_path=manifest_path,
                    subset_config_path=subset_config_path,
                    dataset_root=dataset_root,
                )
                entry["config_path"] = str(
                    bridge_config_path.parent / "benchmark_train_config.yaml"
                )
                entry["bridge_config_path"] = str(bridge_config_path)
                entry["effective_micro_batch_size"] = _effective_micro_batch_size(
                    model_id,
                    bridge_config,
                )
                command_result = _run_profile_bridge_commands(
                    repo_root=repo_root,
                    model_id=model_id,
                    config=config,
                    bridge_config=bridge_config,
                    bridge_config_path=bridge_config_path,
                    device_index=_gpu_index(runtime_device),
                    poll_interval_sec=poll_interval_sec,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                entry["status"] = "failed"
                entry["blocker"] = str(exc)
                entries.append(entry)
                continue

            entry["status"] = command_result["status"]
            entry["blocker"] = command_result["blocker"]
            entry["peak_used_memory_mb"] = command_result["peak_used_memory_mb"]
            entry["torch_peak_allocated_mb"] = command_result["torch_peak_allocated_mb"]
            entry["torch_peak_reserved_mb"] = command_result["torch_peak_reserved_mb"]
            entry["duration_ms"] = command_result["duration_ms"]
            entry["commands"] = command_result["commands"]
            peak_mb = entry["peak_used_memory_mb"]
            entry["safe_under_limit"] = (
                bool(peak_mb <= usable_limit_mb)
                if entry["status"] == "ok" and peak_mb is not None
                else None
            )
            entries.append(entry)

    report["summaries"] = _summarize_entries(requested_ids, entries)
    return report
