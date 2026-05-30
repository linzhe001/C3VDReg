"""Compatibility checks for model/profile combinations and runtime policy."""

from __future__ import annotations

import subprocess
from pathlib import Path

import torch

from src.benchmarking.registry.model_registry import ModelRegistry, ModelSpec


def assert_model_preprocess_compatibility(
    registry: ModelRegistry,
    model_id: str,
    preprocess_profile_id: str,
) -> ModelSpec:
    registry.assert_compatible(model_id, preprocess_profile_id)
    return registry.get(model_id)


def assert_runtime_policy_compatible(spec: ModelSpec, device: str) -> None:
    runtime_device = str(device)
    if spec.runtime_policy == "cpu_only":
        if runtime_device != "cpu":
            raise ValueError(
                f"Model '{spec.model_id}' is benchmark-native CPU-only. "
                f"Received runtime.device={runtime_device!r}."
            )
        return

    if spec.runtime_policy == "cuda_required":
        if not runtime_device.startswith("cuda"):
            raise ValueError(
                f"Model '{spec.model_id}' requires a CUDA runtime under the "
                f"{spec.source_kind} bridge policy. "
                f"Received runtime.device={runtime_device!r}."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Model '{spec.model_id}' requires CUDA runtime, but "
                "torch.cuda.is_available() is False in the current environment."
            )
        return

    raise ValueError(
        f"Unsupported runtime policy {spec.runtime_policy!r} "
        f"for model '{spec.model_id}'."
    )


def assert_baseline_repo_clean(repo_root: Path, spec: ModelSpec) -> None:
    if spec.source_kind != "vendor_readonly":
        return
    if spec.baseline_repo_path is None:
        raise RuntimeError(
            f"Model '{spec.model_id}' is vendor-readonly but has no "
            "baseline_repo_path configured."
        )

    baseline_repo = repo_root / spec.baseline_repo_path
    if not baseline_repo.exists():
        raise FileNotFoundError(
            f"Baseline repository not found for model '{spec.model_id}': "
            f"{baseline_repo}"
        )
    if not (baseline_repo / ".git").exists():
        raise RuntimeError(
            f"Baseline repository for model '{spec.model_id}' must remain a git "
            f"repository under readonly policy: {baseline_repo}"
        )

    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=baseline_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    dirty_entries = [line for line in status.stdout.splitlines() if line.strip()]
    if dirty_entries:
        preview = "\n".join(dirty_entries[:10])
        raise RuntimeError(
            "Baseline readonly policy violated for model "
            f"'{spec.model_id}'. Revert tracked changes under {baseline_repo} "
            "before running benchmark train/eval.\n"
            f"{preview}"
        )
