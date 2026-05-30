"""Config loader for benchmark train/eval entrypoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.benchmarking.config.schema import validate_config


def _load_structured_file(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return {} if data is None else data
    raise ValueError(f"Unsupported config format: {path}")


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = _deep_merge(base[key], value)
        else:
            merged[key] = value
    return merged


def load_benchmark_config(config_path: str | Path) -> dict[str, Any]:
    """Load and normalize YAML/JSON benchmark config."""

    config_path = Path(config_path).resolve()
    repo_root = config_path.parents[3] if len(config_path.parents) >= 4 else Path.cwd()
    default_analysis_path = (
        repo_root / "configs" / "benchmark" / "analysis" / "default.yaml"
    )

    config = _load_structured_file(config_path)
    if default_analysis_path.exists():
        defaults = _load_structured_file(default_analysis_path)
        config = _deep_merge(defaults, config)

    return validate_config(config).to_dict()
