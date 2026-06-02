#!/usr/bin/env python3
"""Prepare and run the tiny GeoTransformer checkpoint smoke test."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_REPO = "https://github.com/qinzheng93/GeoTransformer.git"
DEFAULT_BASELINE_DIR = REPO_ROOT / "baselines" / "GeoTransformer"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "benchmark" / "tiny_8192" / (
    "eval_geotransformer.yaml"
)
DEFAULT_CHECKPOINT_TARGET = REPO_ROOT / "checkpoints" / "geotransformer" / (
    "geotransformer_c3vd_model_best.pth"
)
DEFAULT_CHECKPOINT_PARTS_DIR = DEFAULT_CHECKPOINT_TARGET.with_suffix(
    DEFAULT_CHECKPOINT_TARGET.suffix + ".parts"
)
COMMITTED_CHECKPOINT_SHA256 = (
    "1b48e3098b931d461e6c2466d295ea82ad14868987335d83f049c5342adb1863"
)
COMMITTED_CHECKPOINT_SIZE = 118_305_473
LOCAL_CHECKPOINT_CANDIDATES = (
    REPO_ROOT.parent
    / "C3VDReg_checkpoint"
    / "geotransformer"
    / "geotransformer_c3vd_model_best.pth",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--baseline-dir", default=str(DEFAULT_BASELINE_DIR))
    parser.add_argument("--baseline-repo", default=DEFAULT_BASELINE_REPO)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-target", default=str(DEFAULT_CHECKPOINT_TARGET))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--build-baseline",
        action="store_true",
        help="Run `python setup.py build develop` inside the external baseline repo.",
    )
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="Require the external baseline repo to already exist.",
    )
    parser.add_argument(
        "--no-copy-checkpoint",
        action="store_true",
        help="Use the configured checkpoint path without copying from --checkpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare/check paths but do not launch benchmark evaluation.",
    )
    return parser.parse_args()


def run_command(command: list[str], cwd: Path | None = None) -> None:
    display = " ".join(command)
    if cwd is not None:
        print(f"$ (cd {cwd} && {display})")
    else:
        print(f"$ {display}")
    subprocess.run(command, cwd=cwd, check=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_committed_checkpoint(path: Path) -> None:
    size = path.stat().st_size
    if size != COMMITTED_CHECKPOINT_SIZE:
        raise RuntimeError(
            "Unexpected GeoTransformer checkpoint size: "
            f"{path} has {size} bytes, expected {COMMITTED_CHECKPOINT_SIZE}."
        )
    digest = sha256_file(path)
    if digest != COMMITTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Unexpected GeoTransformer checkpoint SHA256: "
            f"{path} has {digest}, expected {COMMITTED_CHECKPOINT_SHA256}."
        )


def assemble_checkpoint_from_parts(parts_dir: Path, target: Path) -> Path:
    if not parts_dir.exists():
        raise FileNotFoundError(f"Checkpoint shard directory not found: {parts_dir}")
    parts = sorted(parts_dir.glob("part-*"))
    if not parts:
        raise FileNotFoundError(f"No checkpoint shards found under: {parts_dir}")

    target.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target.with_name(target.name + ".tmp")
    with temp_target.open("wb") as output:
        for part in parts:
            with part.open("rb") as input_file:
                shutil.copyfileobj(input_file, output, length=1024 * 1024)
    temp_target.replace(target)
    verify_committed_checkpoint(target)
    print(f"Assembled GeoTransformer checkpoint from shards: {parts_dir}")
    return target


def ensure_baseline_repo(args: argparse.Namespace) -> Path:
    baseline_dir = Path(args.baseline_dir).expanduser().resolve()
    if baseline_dir.exists():
        if not (baseline_dir / ".git").exists():
            raise RuntimeError(
                "GeoTransformer baseline path exists but is not a git repository: "
                f"{baseline_dir}"
            )
        print(f"GeoTransformer baseline found: {baseline_dir}")
    else:
        if args.skip_clone:
            raise FileNotFoundError(
                "GeoTransformer baseline is missing and --skip-clone was set: "
                f"{baseline_dir}"
            )
        baseline_dir.parent.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                "git",
                "clone",
                "--depth",
                "1",
                str(args.baseline_repo),
                str(baseline_dir),
            ]
        )

    experiment_dir = baseline_dir / "experiments" / (
        "geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn"
    )
    if not experiment_dir.exists():
        raise FileNotFoundError(
            "Expected GeoTransformer 3DMatch experiment directory not found: "
            f"{experiment_dir}"
        )

    if args.build_baseline:
        run_command(
            [str(args.python), "setup.py", "build", "develop"],
            cwd=baseline_dir,
        )
    return baseline_dir


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    target = Path(args.checkpoint_target).expanduser().resolve()
    explicit = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    parts_dir = target.with_suffix(target.suffix + ".parts")
    if target == DEFAULT_CHECKPOINT_TARGET.resolve():
        parts_dir = DEFAULT_CHECKPOINT_PARTS_DIR

    if args.no_copy_checkpoint:
        if not target.exists():
            raise FileNotFoundError(f"Checkpoint not found: {target}")
        if parts_dir.exists():
            verify_committed_checkpoint(target)
        return target

    if target.exists():
        if parts_dir.exists():
            verify_committed_checkpoint(target)
        print(f"GeoTransformer checkpoint found: {target}")
        return target

    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Checkpoint not found: {explicit}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(explicit, target)
        print(f"Copied GeoTransformer checkpoint:\n  from {explicit}\n  to   {target}")
        return target

    if parts_dir.exists():
        return assemble_checkpoint_from_parts(parts_dir, target)

    source_candidates = list(LOCAL_CHECKPOINT_CANDIDATES)
    source = next((path for path in source_candidates if path.exists()), None)
    if source is None:
        searched = "\n".join(str(path) for path in source_candidates)
        raise FileNotFoundError(
            "GeoTransformer checkpoint is missing. Restore the released checkpoint "
            "bundle, pass --checkpoint, or keep the committed checkpoint shards.\n"
            "Searched:\n"
            f"{searched}\nTarget path:\n{target}"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(f"Copied GeoTransformer checkpoint:\n  from {source}\n  to   {target}")
    if parts_dir.exists():
        verify_committed_checkpoint(target)
    return target


def print_result_summary(summary: dict[str, object]) -> None:
    output_dir = Path(str(summary["output_dir"]))
    overall = dict(summary.get("overall", {}))
    print("\nGeoTransformer tiny benchmark finished.")
    print(f"Output directory: {output_dir}")
    print(f"Samples: {summary.get('sample_count')}")
    print(
        "RR@5deg/5mm: "
        f"{overall.get('registration_recall@rre_5deg_rte_5mm', 'n/a')}"
    )
    print(f"RRE mean deg: {overall.get('rre_deg_mean', 'n/a')}")
    print(f"RTE mean mm: {overall.get('rte_mm_mean', 'n/a')}")

    expected_outputs = [
        output_dir / "report.html",
        output_dir / "summary_overview.md",
        output_dir / "leaderboard" / "leaderboard_main.csv",
        output_dir / "leaderboard" / "leaderboard_main.md",
        output_dir / "curves" / "rr_multithreshold.png",
        output_dir / "curves" / "success_latency_pareto.png",
        output_dir / "geometry" / "visible_distance_hist.png",
        output_dir / "geometry" / "visible_distance_cdf.png",
    ]
    print("\nPrimary result/visualization files:")
    for path in expected_outputs:
        if path.exists():
            print(f"  {path}")
    distance_render_dir = output_dir / "qualitative" / "distance_render"
    for path in sorted(distance_render_dir.glob("*")):
        print(f"  {path}")


def main() -> int:
    args = parse_args()
    ensure_baseline_repo(args)
    checkpoint = resolve_checkpoint(args)
    config = Path(args.config).expanduser().resolve()
    if not config.exists():
        raise FileNotFoundError(f"Config not found: {config}")
    print(f"Using config: {config}")
    print(f"Using checkpoint: {checkpoint}")

    if args.dry_run:
        return 0

    command = [
        str(args.python),
        "scripts/runners/eval_benchmark.py",
        "--config",
        str(config),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout,
            stderr=completed.stderr,
        )

    json_start = completed.stdout.rfind("\n{")
    if json_start == -1:
        json_start = completed.stdout.find("{")
    if json_start == -1:
        raise RuntimeError("Could not locate JSON summary in eval output.")
    summary = json.loads(completed.stdout[json_start:].strip())
    print_result_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
