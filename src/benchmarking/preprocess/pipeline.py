"""Benchmark-visible preprocess pipeline."""

from __future__ import annotations

from dataclasses import asdict, replace
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from src.benchmarking.preprocess.registry import PreprocessRegistry
from src.common.utils.benchmark_preprocess import (
    apply_pair_perturbation,
    joint_bbox_normalize_pair,
    sample_point_cloud,
)
from src.common.utils.sampling import clean_point_cloud


def _estimate_normals(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    return centered / norms

class PreprocessPipeline:
    """Run official benchmark-visible preprocessing on a sample dict."""

    def __init__(self, registry: PreprocessRegistry | None = None) -> None:
        self.registry = registry or PreprocessRegistry()

    def _resolve_profile(
        self,
        profile_id: str,
        sampling_override: str | None,
        num_points_override: int | None,
    ):
        profile = self.registry.get(profile_id)
        if sampling_override is not None:
            profile = replace(profile, sampling=sampling_override)
        if num_points_override is not None:
            profile = replace(profile, num_points=int(num_points_override))
        return profile

    def _sample(
        self,
        points: np.ndarray,
        sampling: str,
        num_points: int | None,
        seed: int,
    ) -> np.ndarray:
        return sample_point_cloud(
            points,
            sampling=sampling,
            num_points=num_points,
            seed=seed,
        )

    def _apply_perturbation(
        self,
        source: np.ndarray,
        target: np.ndarray,
        gt_transform: np.ndarray,
        perturbation_config: Mapping[str, Any] | None,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        if not perturbation_config or not bool(
            perturbation_config.get("enabled", False)
        ):
            return source, target, gt_transform, {}
        source_perturbed, target_perturbed, gt_updated, perturbation_meta = (
            apply_pair_perturbation(
                source=source,
                target=target,
                gt_transform=gt_transform,
                rotation_deg=float(perturbation_config.get("rotation_deg", 0.0)),
                translation_m=float(perturbation_config.get("translation_m", 0.0)),
                min_rotation_deg=float(
                    perturbation_config.get("min_rotation_deg", 0.0)
                ),
                min_translation_m=float(
                    perturbation_config.get("min_translation_m", 0.0)
                ),
                noise_sigma=float(perturbation_config.get("noise_sigma", 0.0)),
                noise_clip=float(perturbation_config.get("noise_clip", 0.0)),
                apply_noise_to=str(
                    perturbation_config.get("apply_noise_to", "source")
                ),
                seed=seed,
            )
        )
        perturbation_meta["enabled"] = True
        return source_perturbed, target_perturbed, gt_updated, perturbation_meta

    def run(
        self,
        sample: dict[str, Any],
        profile_id: str,
        seed: int,
        sampling_override: str | None = None,
        num_points_override: int | None = None,
        perturbation_config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        profile = self._resolve_profile(
            profile_id=profile_id,
            sampling_override=sampling_override,
            num_points_override=num_points_override,
        )
        start = perf_counter()

        source = np.asarray(sample["source_points"], dtype=np.float32)
        target = np.asarray(sample["target_points"], dtype=np.float32)
        gt_transform = np.asarray(
            sample.get("gt_transform", np.eye(4, dtype=np.float64)),
            dtype=np.float64,
        )

        if profile.clean_invalid:
            source = clean_point_cloud(source, min_points=min(len(source), 8))
            target = clean_point_cloud(target, min_points=min(len(target), 8))

        source = self._sample(source, profile.sampling, profile.num_points, seed)
        target = self._sample(target, profile.sampling, profile.num_points, seed + 1)

        source, target, gt_transform, perturbation_meta = self._apply_perturbation(
            source=source,
            target=target,
            gt_transform=gt_transform,
            perturbation_config=perturbation_config,
            seed=seed + 2,
        )

        normalize_meta: dict[str, Any] = {}
        if profile.normalize == "joint_bbox":
            source, target, normalize_meta, normalize_transform = (
                joint_bbox_normalize_pair(
                    source,
                    target,
                )
            )
            gt_transform = normalize_transform @ gt_transform @ np.linalg.inv(
                normalize_transform
            )
        elif profile.normalize == "unit_cube":
            source, target, normalize_meta, normalize_transform = (
                joint_bbox_normalize_pair(
                    source,
                    target,
                )
            )
            gt_transform = normalize_transform @ gt_transform @ np.linalg.inv(
                normalize_transform
            )
        elif profile.normalize not in {"none", "model_private"}:
            raise ValueError(f"Unsupported normalize mode '{profile.normalize}'.")

        processed = dict(sample)
        processed["source_points"] = source.astype(np.float32, copy=False)
        processed["target_points"] = target.astype(np.float32, copy=False)
        processed["gt_transform"] = gt_transform.astype(np.float64, copy=False)
        processed["preprocess_profile_id"] = profile.profile_id
        processed["leaderboard_track"] = profile.leaderboard_track
        processed["preprocess_metadata"] = {
            "profile": asdict(profile),
            **normalize_meta,
            "perturbation": perturbation_meta,
        }
        if profile.estimate_normals:
            processed["source_normals"] = _estimate_normals(processed["source_points"])
            processed["target_normals"] = _estimate_normals(processed["target_points"])

        processed["preprocess_time_ms"] = (perf_counter() - start) * 1000.0
        return processed
