"""Git and artifact traceability helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_bytes(path: Path | None) -> bytes | None:
    if path is None or not path.exists():
        return None
    return path.read_bytes()


def _sha256_bytes(payload: bytes | None) -> str | None:
    if payload is None:
        return None
    return hashlib.sha256(payload).hexdigest()


def _run_git(repo_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def git_snapshot(
    repo_root: str | Path,
    config_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    subset_config_path: str | Path | None = None,
    preprocess_profile_id: str | None = None,
    model_id: str | None = None,
    checkpoint_id: str | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Collect git and artifact digests for benchmark traceability."""

    repo_root = Path(repo_root).resolve()
    config_path = Path(config_path).resolve() if config_path else None
    manifest_path = Path(manifest_path).resolve() if manifest_path else None
    subset_config_path = (
        Path(subset_config_path).resolve() if subset_config_path else None
    )

    commit = _run_git(repo_root, "rev-parse", "HEAD") or "unavailable"
    branch = _run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD") or "unavailable"
    status_output = _run_git(repo_root, "status", "--porcelain")

    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": bool(status_output),
        "config_sha256": _sha256_bytes(_read_bytes(config_path)),
        "manifest_sha256": _sha256_bytes(_read_bytes(manifest_path)),
        "subset_config_sha256": _sha256_bytes(_read_bytes(subset_config_path)),
        "preprocess_profile_id": preprocess_profile_id,
        "model_id": model_id,
        "checkpoint_id": checkpoint_id,
        "non_release_traceability_warning": bool(status_output),
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")

    return snapshot
