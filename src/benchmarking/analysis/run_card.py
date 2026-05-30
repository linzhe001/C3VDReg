"""Run card generation for traceable eval artifacts."""

from __future__ import annotations

from typing import Any


def build_run_card(
    eval_config: dict[str, object],
    git_snapshot: dict[str, object],
    aggregate_summary: dict[str, object],
) -> dict[str, Any]:
    benchmark = dict(eval_config.get("benchmark", {}))
    model = dict(eval_config.get("model", {}))
    preprocess = dict(eval_config.get("preprocess", {}))
    perturbation = dict(eval_config.get("perturbation", {}))
    runtime = dict(eval_config.get("runtime", {}))

    return {
        "benchmark": benchmark,
        "model": model,
        "preprocess": preprocess,
        "perturbation": perturbation,
        "runtime": runtime,
        "git": git_snapshot,
        "summary": aggregate_summary,
        "manifest_digest": git_snapshot.get("manifest_sha256"),
        "config_digest": git_snapshot.get("config_sha256"),
        "git_snapshot": git_snapshot.get("git_commit"),
        "non_release_traceability_warning": git_snapshot.get(
            "non_release_traceability_warning",
            False,
        ),
    }
