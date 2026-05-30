#!/usr/bin/env python3
"""Run one Python script and write in-process torch CUDA peak-memory metrics."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
import time
import traceback
from pathlib import Path

import torch


def _memory_mb(value: int) -> float:
    return float(value) / (1024.0 * 1024.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("Expected Python script path after '--'.")

    metrics_path = Path(args.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    script_path = Path(command[0])
    original_argv = sys.argv
    sys.argv = command

    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    started_at = time.perf_counter()
    return_code = 0
    exception = None
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return_code = 0
        elif isinstance(code, int):
            return_code = code
        else:
            return_code = 1
            exception = str(code)
    except BaseException:  # noqa: BLE001 - profiler must record failures too.
        return_code = 1
        exception = traceback.format_exc()
    finally:
        if cuda_available:
            try:
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001 - best-effort metrics finalization.
                pass
        duration_ms = (time.perf_counter() - started_at) * 1000.0
        payload = {
            "script_path": str(script_path),
            "argv": command,
            "return_code": return_code,
            "cuda_available": cuda_available,
            "duration_ms": duration_ms,
            "torch_peak_allocated_mb": (
                _memory_mb(torch.cuda.max_memory_allocated())
                if cuda_available
                else None
            ),
            "torch_peak_reserved_mb": (
                _memory_mb(torch.cuda.max_memory_reserved())
                if cuda_available
                else None
            ),
            "exception": exception,
        }
        metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        sys.argv = original_argv

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
