"""Benchmark train runner with configurable smoke/full train bridges."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import torch
import yaml

from src.benchmarking.diagnostics.compatibility import (
    assert_baseline_repo_clean,
    assert_runtime_policy_compatible,
)
from src.benchmarking.registry.model_registry import ModelRegistry, ModelSpec
from src.utils.git_snapshot import git_snapshot


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_yaml_template(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_frozen_scene_splits(
    subset_config_path: str | None,
) -> tuple[list[str], list[str], list[str]] | None:
    if not subset_config_path:
        return None

    subset_payload = json.loads(Path(subset_config_path).read_text(encoding="utf-8"))
    scene_splits = subset_payload.get("scene_splits")
    if not isinstance(scene_splits, dict):
        return None

    train_scenes = scene_splits.get("train")
    if not isinstance(train_scenes, list) or not train_scenes:
        return None

    validation_scenes = scene_splits.get("val")
    if not isinstance(validation_scenes, list) or not validation_scenes:
        return None

    test_scenes = scene_splits.get("test")
    if not isinstance(test_scenes, list) or not test_scenes:
        test_scenes = validation_scenes

    return (
        [str(scene) for scene in train_scenes],
        [str(scene) for scene in validation_scenes],
        [str(scene) for scene in test_scenes],
    )


def _cuda_runtime_requested(config: dict[str, Any]) -> bool:
    return str(config["runtime"]["device"]).startswith("cuda")


def _train_mode(config: dict[str, Any]) -> str:
    train_mode = str(config["runtime"].get("train_mode", "smoke"))
    if train_mode not in {"smoke", "full"}:
        raise ValueError(
            "Benchmark train runner only supports runtime.train_mode "
            f"'smoke' or 'full'. Received {train_mode!r}."
        )
    return train_mode


def _is_smoke_train(config: dict[str, Any]) -> bool:
    return _train_mode(config) == "smoke"


def _require_cuda_runtime_device(config: dict[str, Any]) -> str:
    runtime_device = str(config["runtime"]["device"])
    if not _cuda_runtime_requested(config):
        raise ValueError(
            "Benchmark train bridge is GPU-only. "
            f"Received runtime.device={runtime_device!r}."
        )
    return runtime_device


def _cuda_device_index(runtime_device: str) -> int:
    if ":" not in runtime_device:
        return 0
    _, index = runtime_device.split(":", 1)
    return int(index)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _apply_runtime_dataset_overrides(
    dataset_section: dict[str, Any],
    config: dict[str, Any],
) -> None:
    dataset_overrides = dict(config.get("runtime", {}).get("dataset_overrides", {}))
    allowed_dataset_overrides = {
        "frame_stride",
        "max_train_pairs",
        "max_val_pairs",
        "max_test_pairs",
        "num_points",
        "sampling_mode",
    }
    unsupported = sorted(set(dataset_overrides) - allowed_dataset_overrides)
    if unsupported:
        raise ValueError(
            "Unsupported runtime.dataset_overrides keys: " + ", ".join(unsupported)
        )

    integer_fields = {
        "frame_stride",
        "max_train_pairs",
        "max_val_pairs",
        "max_test_pairs",
        "num_points",
    }
    nullable_fields = {"max_train_pairs", "max_val_pairs", "max_test_pairs"}
    for key, value in dataset_overrides.items():
        if value is None:
            if key in nullable_fields:
                dataset_section[key] = None
            else:
                dataset_section.pop(key, None)
        elif key in integer_fields:
            dataset_section[key] = int(value)
        else:
            dataset_section[key] = str(value)


def _apply_common_train_dataset_settings(
    dataset_section: dict[str, Any],
    config: dict[str, Any],
) -> None:
    dataset_root = config["data"].get("dataset_root")
    if not dataset_root:
        raise ValueError("Smoke training requires data.dataset_root.")

    dataset_section["data_root"] = dataset_root
    dataset_section["random_seed"] = int(config["preprocess"]["seed"])
    model_overrides = dict(config.get("model", {}).get("overrides", {}))
    normalize_mode = model_overrides.get("normalize_mode")
    if normalize_mode is None:
        spec = ModelRegistry().get(str(config["model"]["id"]))
        normalize_mode = spec.default_eval_normalize_mode or "none"
    dataset_section["normalize_mode"] = str(normalize_mode)

    frozen_scene_splits = _load_frozen_scene_splits(
        config["data"].get("subset_config_path")
    )
    if frozen_scene_splits is not None:
        train_scenes, validation_scenes, test_scenes = frozen_scene_splits
        dataset_section["train_scenes"] = train_scenes
        dataset_section["val_scenes"] = validation_scenes
        dataset_section["test_scenes"] = test_scenes

    preprocess = dict(config.get("preprocess", {}))
    sampling_mode = preprocess.get("sampling_override")
    if sampling_mode is not None:
        dataset_section["sampling_mode"] = str(sampling_mode)
    else:
        dataset_section.setdefault("sampling_mode", "voxel")

    num_points_override = preprocess.get("num_points_override")
    if num_points_override is not None:
        dataset_section["num_points"] = int(num_points_override)

    perturbation = dict(config.get("perturbation", {}))
    dataset_section["perturbation_enabled"] = bool(perturbation.get("enabled", False))
    dataset_section["rotation_deg"] = float(perturbation.get("rotation_deg", 0.0))
    dataset_section["translation_m"] = float(
        perturbation.get("translation_m", 0.0)
    )
    dataset_section["min_rotation_deg"] = float(
        perturbation.get("min_rotation_deg", 0.0)
    )
    dataset_section["min_translation_m"] = float(
        perturbation.get("min_translation_m", 0.0)
    )
    dataset_section["noise_sigma"] = float(perturbation.get("noise_sigma", 0.0))
    dataset_section["noise_clip"] = float(perturbation.get("noise_clip", 0.0))
    dataset_section["apply_noise_to"] = str(
        perturbation.get("apply_noise_to", "source")
    )
    _apply_runtime_dataset_overrides(dataset_section, config)


def _apply_smoke_dataset_settings(
    dataset_section: dict[str, Any],
    config: dict[str, Any],
) -> None:
    dataset_section["frame_stride"] = 1000
    dataset_section["max_train_pairs"] = 4
    dataset_section["max_test_pairs"] = 2
    if config.get("preprocess", {}).get("num_points_override") is None:
        dataset_section["num_points"] = 8192


def _bridge_run_name(config: dict[str, Any]) -> str:
    return f"benchmark_{_train_mode(config)}_{config['model']['id']}"


def _verify_checkpoint_loadable(checkpoint_path: Path) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    torch.load(checkpoint_path, map_location="cpu")


def _resolve_existing_checkpoint(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Unable to locate any checkpoint candidate: "
        + ", ".join(str(path) for path in candidates)
    )


def _verify_checkpoint_matches_eval_adapter(
    spec: ModelSpec,
    config: dict[str, Any],
    checkpoint_path: Path,
) -> None:
    module_name, class_name = spec.adapter_path.rsplit(".", 1)
    adapter_cls = getattr(importlib.import_module(module_name), class_name)
    default_model_config_path = (
        _repo_root() / "configs" / "benchmark" / "models" / f"{spec.model_id}.yaml"
    )
    adapter_overrides: dict[str, Any] = {}
    if default_model_config_path.exists():
        default_model_config = _load_yaml_template(default_model_config_path)
        adapter_overrides.update(
            default_model_config.get("model", {}).get("overrides", {})
        )
    adapter_overrides.update(config["model"].get("overrides", {}))
    adapter_args = SimpleNamespace(
        device="cpu",
        checkpoint_path=str(checkpoint_path),
        **adapter_overrides,
    )
    adapter = adapter_cls(adapter_args)
    adapter.load_model(str(checkpoint_path))


def _profile_wrapper_path() -> Path:
    return _repo_root() / "scripts" / "benchmark" / "_profile_torch_command.py"


def _python_script_args(command: list[str]) -> list[str]:
    if not command:
        raise ValueError("Expected non-empty Python command.")
    executable = Path(command[0]).name
    if executable.startswith("python"):
        return command[1:]
    return command


def _run_profiled_subprocess(
    name: str,
    command: list[str],
    cwd: Path,
    metrics_dir: Path,
) -> dict[str, Any]:
    """Run a Python train script and collect wall time plus torch CUDA peaks."""

    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{name}.json"
    script_args = _python_script_args(command)
    wrapped = [
        sys.executable,
        str(_profile_wrapper_path()),
        "--metrics-path",
        str(metrics_path),
        "--",
        *script_args,
    ]
    started_at = perf_counter()
    subprocess.run(wrapped, cwd=cwd, check=True)
    wall_time_ms = (perf_counter() - started_at) * 1000.0
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "name": name,
        "command": command,
        "metrics_path": str(metrics_path),
        "wall_time_ms": wall_time_ms,
        "duration_ms": metrics.get("duration_ms"),
        "torch_peak_allocated_mb": metrics.get("torch_peak_allocated_mb"),
        "torch_peak_reserved_mb": metrics.get("torch_peak_reserved_mb"),
        "return_code": metrics.get("return_code"),
    }


def _summarize_train_metrics(
    command_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    duration_values = [
        float(metric["duration_ms"])
        for metric in command_metrics
        if metric.get("duration_ms") is not None
    ]
    reserved_values = [
        float(metric["torch_peak_reserved_mb"])
        for metric in command_metrics
        if metric.get("torch_peak_reserved_mb") is not None
    ]
    allocated_values = [
        float(metric["torch_peak_allocated_mb"])
        for metric in command_metrics
        if metric.get("torch_peak_allocated_mb") is not None
    ]
    return {
        "train_time_ms": sum(duration_values) if duration_values else None,
        "train_peak_memory_mb": max(reserved_values) if reserved_values else None,
        "train_peak_allocated_mb": (
            max(allocated_values) if allocated_values else None
        ),
        "train_command_metrics": command_metrics,
    }


def _write_bridge_config(
    bridge_config: dict[str, Any],
    output_dir: Path,
    filename: str,
) -> Path:
    bridge_config_path = output_dir / filename
    bridge_config_path.write_text(
        yaml.safe_dump(bridge_config, sort_keys=False),
        encoding="utf-8",
    )
    return bridge_config_path


def _build_dcp_bridge_config(
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    runtime_device = _require_cuda_runtime_device(config)
    template_name = "c3vd_dcp_test.yaml" if _is_smoke_train(config) else "c3vd_dcp.yaml"
    template = _load_yaml_template(
        _repo_root() / "src" / "benchmarking" / "bridges" / "configs" / template_name
    )

    overrides = dict(config["model"].get("overrides", {}))
    dataset = template["dataset"]
    _apply_common_train_dataset_settings(dataset, config)
    if _is_smoke_train(config):
        _apply_smoke_dataset_settings(dataset, config)
    template["model"].update(
        {key: value for key, value in overrides.items() if key in template["model"]}
    )

    training = template["training"]
    testing = template["testing"]
    training["exp_name"] = _bridge_run_name(config)
    training["use_cuda"] = True
    training["cuda_device"] = _cuda_device_index(runtime_device)
    training["num_workers"] = int(config["runtime"].get("num_workers", 0))
    training["checkpoint_dir"] = str(output_dir / "train_bridge")
    training["seed"] = int(config["preprocess"]["seed"])
    training["device"] = runtime_device
    testing["num_workers"] = int(config["runtime"].get("num_workers", 0))
    training_overrides = dict(config["runtime"].get("training_overrides", {}))
    allowed_training_overrides = {
        "num_epochs",
        "batch_size",
        "test_batch_size",
        "lr",
        "lr_decay_epochs",
        "lr_decay_rate",
        "save_interval",
        "val_interval",
        "resume_from",
        "resume_start_epoch",
        "resume_best_test_loss",
    }
    unsupported = sorted(set(training_overrides) - allowed_training_overrides)
    if unsupported:
        raise ValueError(
            "Unsupported DCP runtime.training_overrides keys: "
            + ", ".join(unsupported)
        )
    training.update(training_overrides)

    if _is_smoke_train(config):
        training["num_epochs"] = 1
        training["save_interval"] = 1
        training["val_interval"] = 1
        training["max_train_steps"] = 4
        training["max_test_steps"] = 2
        training["batch_size"] = 16
        training["test_batch_size"] = 8

    return template


def _build_pointnetlk_bridge_config(
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    runtime_device = _require_cuda_runtime_device(config)
    template = _load_yaml_template(
        _repo_root()
        / "src"
        / "benchmarking"
        / "bridges"
        / "configs"
        / "c3vd_pointnetlk.yaml"
    )
    dataset = template["dataset"]
    _apply_common_train_dataset_settings(dataset, config)
    if _is_smoke_train(config):
        _apply_smoke_dataset_settings(dataset, config)

    training = template["training"]
    training["device"] = runtime_device
    training["num_workers"] = int(config["runtime"].get("num_workers", 0))
    training_overrides = dict(config["runtime"].get("training_overrides", {}))
    classifier_overrides = dict(training_overrides.pop("classifier", {}))
    pointnetlk_overrides = dict(training_overrides.pop("pointnetlk", {}))
    allowed_training_overrides = {"batch_size"}
    unsupported = sorted(set(training_overrides) - allowed_training_overrides)
    if unsupported:
        raise ValueError(
            "Unsupported PointNetLK runtime.training_overrides keys: "
            + ", ".join(unsupported)
        )
    allowed_classifier_overrides = {
        "epochs",
        "grad_clip_norm",
        "learning_rate",
        "max_train_steps",
        "max_test_steps",
        "min_lr",
    }
    allowed_pointnetlk_overrides = allowed_classifier_overrides | {
        "max_loss",
        "max_skipped_batches",
        "min_lr",
        "pointnet_mode",
        "resume_best_test_loss",
        "resume_from",
        "resume_optimizer",
        "resume_scheduler",
        "resume_start_epoch",
        "scheduler_cooldown",
        "scheduler_factor",
        "scheduler_patience",
        "scheduler_threshold",
        "scheduler_threshold_mode",
    }
    unsupported_classifier = sorted(
        set(classifier_overrides) - allowed_classifier_overrides
    )
    if unsupported_classifier:
        raise ValueError(
            "Unsupported PointNetLK runtime.training_overrides.classifier keys: "
            + ", ".join(unsupported_classifier)
        )
    unsupported_pointnetlk = sorted(
        set(pointnetlk_overrides) - allowed_pointnetlk_overrides
    )
    if unsupported_pointnetlk:
        raise ValueError(
            "Unsupported PointNetLK runtime.training_overrides.pointnetlk keys: "
            + ", ".join(unsupported_pointnetlk)
        )
    training.update(training_overrides)
    training["classifier"].update(classifier_overrides)
    training["pointnetlk"].update(pointnetlk_overrides)

    output = template["output"]
    output["checkpoint_dir"] = str(output_dir / "train_bridge")
    output["log_dir"] = str(output_dir / "train_bridge_logs")

    if _is_smoke_train(config):
        training["batch_size"] = 8
        training["classifier"]["epochs"] = 1
        training["classifier"]["max_train_steps"] = 2
        training["classifier"]["max_test_steps"] = 1
        training["pointnetlk"]["epochs"] = 1
        training["pointnetlk"]["max_train_steps"] = 2
        training["pointnetlk"]["max_test_steps"] = 1

    return template


def _build_pointnetlk_revisited_bridge_config(
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    runtime_device = _require_cuda_runtime_device(config)
    template = _load_yaml_template(
        _repo_root()
        / "src"
        / "benchmarking"
        / "bridges"
        / "configs"
        / "c3vd_pointnetlk_revisited.yaml"
    )
    dataset = template["dataset"]
    _apply_common_train_dataset_settings(dataset, config)
    if _is_smoke_train(config):
        _apply_smoke_dataset_settings(dataset, config)

    model_overrides = dict(config["model"].get("overrides", {}))
    template["model"].update(
        {
            key: value
            for key, value in model_overrides.items()
            if key in template["model"]
        }
    )
    template["pointnetlk"].update(
        {
            key: value
            for key, value in model_overrides.items()
            if key in template["pointnetlk"]
        }
    )

    training = template["training"]
    training["checkpoint_dir"] = str(output_dir / "train_bridge")
    training["num_workers"] = int(config["runtime"].get("num_workers", 0))
    training["device"] = runtime_device
    if not _is_smoke_train(config) and config["runtime"].get("batch_size") is not None:
        training["batch_size"] = int(config["runtime"]["batch_size"])

    training_overrides = dict(config["runtime"].get("training_overrides", {}))
    if "max_loss" in model_overrides and "max_loss" not in training_overrides:
        training_overrides["max_loss"] = model_overrides["max_loss"]
    for key in (
        "batch_size",
        "max_epochs",
        "start_epoch",
        "optimizer",
        "lr",
        "decay_rate",
        "grad_clip_norm",
        "save_interval",
        "max_train_steps",
        "max_val_steps",
        "max_loss",
        "eval_max_loss",
        "max_skipped_batches",
        "scheduler_factor",
        "scheduler_patience",
        "scheduler_threshold",
        "scheduler_threshold_mode",
        "scheduler_cooldown",
        "min_lr",
    ):
        if key in training_overrides:
            training[key] = training_overrides[key]
    if "resume_from" in training_overrides:
        template["resume"]["checkpoint"] = str(training_overrides["resume_from"])
    if "pretrained_from" in training_overrides:
        template["resume"]["pretrained"] = str(training_overrides["pretrained_from"])

    template["testing"]["batch_size"] = training["batch_size"]
    template["testing"]["num_workers"] = training["num_workers"]
    template["testing"]["device"] = training["device"]

    if _is_smoke_train(config):
        training["batch_size"] = 8
        training["max_epochs"] = 1
        training["save_interval"] = 1
        training["max_train_steps"] = 2
        training["max_val_steps"] = 1
        template["testing"]["batch_size"] = training["batch_size"]

    return template


def _build_mamba3d_bridge_config(
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    runtime_device = _require_cuda_runtime_device(config)
    overrides = dict(config["model"].get("overrides", {}))
    model_id = str(config["model"]["id"])
    override_model_type = str(overrides.get("model_type", ""))
    if (
        model_id == "mambanetlk"
        or override_model_type == "mambanetlk"
    ):
        mamba_route = "mambanetlk"
        template_name = "c3vd_mambanetlk.yaml"
    elif (
        model_id == "mamba3d_mamba2_direct"
        or override_model_type == "mamba3d_mamba2_direct"
    ):
        mamba_route = "mamba3d_mamba2_direct"
        template_name = "c3vd_mamba3d_mamba2_direct.yaml"
    elif model_id == "mamba3d_mamba2" or override_model_type == "mamba3d_mamba2":
        mamba_route = "mamba3d_mamba2"
        template_name = "c3vd_mamba3d_mamba2.yaml"
    elif model_id == "mamba3d_true" or override_model_type == "mamba3d_true":
        mamba_route = "mamba3d_true"
        template_name = "c3vd_mamba3d_true.yaml"
    else:
        mamba_route = "mamba3d_v1"
        template_name = "c3vd_mamba3d.yaml"
    use_pointmamba_style = mamba_route in {
        "mamba3d_true",
        "mamba3d_mamba2",
        "mambanetlk",
    }
    preserve_template_num_points = mamba_route in {
        "mamba3d_true",
        "mamba3d_mamba2",
        "mamba3d_mamba2_direct",
        "mambanetlk",
    }
    template = _load_yaml_template(
        _repo_root()
        / "src"
        / "benchmarking"
        / "bridges"
        / "configs"
        / template_name
    )
    dataset = template["dataset"]
    default_num_points = int(dataset.get("num_points", 8192))
    default_max_train_pairs = dataset.get("max_train_pairs")
    default_max_val_pairs = dataset.get("max_val_pairs")
    default_frame_stride = dataset.get("frame_stride")
    _apply_common_train_dataset_settings(dataset, config)
    if _is_smoke_train(config):
        _apply_smoke_dataset_settings(dataset, config)
        if preserve_template_num_points and config.get("preprocess", {}).get(
            "num_points_override"
        ) is None:
            dataset["num_points"] = default_num_points
        if mamba_route == "mamba3d_mamba2_direct":
            if default_max_train_pairs is not None:
                dataset["max_train_pairs"] = int(default_max_train_pairs)
            if default_max_val_pairs is not None:
                dataset["max_val_pairs"] = int(default_max_val_pairs)
            dataset.pop("max_test_pairs", None)
            if default_frame_stride is None:
                dataset.pop("frame_stride", None)
            else:
                dataset["frame_stride"] = int(default_frame_stride)

    model = template["model"]
    if mamba_route != "mamba3d_v1":
        model["model_type"] = mamba_route
    else:
        model["model_type"] = model.get("model_type", "mamba3d_v1")
    for key, value in overrides.items():
        if key != "normalize_mode":
            model[key] = value

    training = template["training"]
    training["num_workers"] = int(config["runtime"].get("num_workers", 0))
    training_overrides = dict(config["runtime"].get("training_overrides", {}))
    allowed_training_overrides = {
        "epochs",
        "batch_size",
        "lr",
        "min_lr",
        "weight_decay",
        "optimizer",
        "pointnet_tune",
        "rot_factor",
        "trans_mag",
        "loss_mode",
        "composition_loss_scale",
        "pose_rotation_weight",
        "pose_translation_weight",
        "feature_residual_weight",
        "init_pose_weight",
        "init_rotation_weight",
        "init_translation_weight",
        "grad_clip_norm",
        "max_loss",
        "eval_max_loss",
        "max_skipped_batches",
        "skip_nonfinite_loss",
        "abort_after_nonfinite_steps",
        "log_grad_norm",
        "scheduler_factor",
        "scheduler_patience",
        "scheduler_threshold",
        "scheduler_threshold_mode",
        "scheduler_cooldown",
        "max_train_steps",
        "max_test_steps",
        "resume_from",
    }
    unsupported = sorted(set(training_overrides) - allowed_training_overrides)
    if unsupported:
        raise ValueError(
            "Unsupported Mamba3D runtime.training_overrides keys: "
            + ", ".join(unsupported)
        )
    training.update(training_overrides)

    template["output"]["checkpoint_dir"] = str(output_dir / "train_bridge")
    template["output"]["log_dir"] = str(output_dir / "train_bridge_logs")
    template["device"] = runtime_device

    if _is_smoke_train(config):
        training["epochs"] = 1
        if mamba_route == "mamba3d_mamba2_direct":
            training["batch_size"] = int(training.get("batch_size", 16))
        else:
            training["batch_size"] = 2 if use_pointmamba_style else 8
        training["max_train_steps"] = 2
        training["max_test_steps"] = 1

    return template


def _apply_bufferx_hparam_overrides(
    template: dict[str, Any],
    overrides: dict[str, Any],
) -> None:
    """Map BUFFER-X benchmark overrides onto the bridge config."""

    dataset = template["dataset"]
    bufferx = template["bufferx"]

    if "reference_dataset" in overrides:
        bufferx["reference_dataset"] = str(overrides["reference_dataset"])
    if "root_dir" in overrides:
        bufferx["root_dir"] = str(overrides["root_dir"])
    if "normalize_mode" in overrides:
        dataset["normalize_mode"] = str(overrides["normalize_mode"])
    if "heuristic_mode" in overrides:
        dataset["heuristic_mode"] = str(overrides["heuristic_mode"])
    if "sampling_mode" in overrides:
        dataset["sampling_mode"] = str(overrides["sampling_mode"])
    if "num_points" in overrides:
        dataset["num_points"] = int(overrides["num_points"])
    if "first_downsample" in overrides:
        dataset["first_downsample"] = float(overrides["first_downsample"])
        bufferx["data"]["downsample"] = float(overrides["first_downsample"])
    if "second_downsample" in overrides:
        value = float(overrides["second_downsample"])
        dataset["second_downsample"] = value
        bufferx["data"]["voxel_size_0"] = value
        bufferx["data"]["voxel_size_1"] = value

    dotted_map = {
        "data.downsample": ("data", "downsample", float),
        "data.voxel_size_0": ("data", "voxel_size_0", float),
        "data.voxel_size_1": ("data", "voxel_size_1", float),
        "data.max_numPts": ("data", "max_numPts", int),
        "patch.des_r": ("patch", "des_r", float),
        "patch.num_points_per_patch": ("patch", "num_points_per_patch", int),
        "patch.num_fps": ("patch", "num_fps", int),
        "patch.num_scales": ("patch", "num_scales", int),
        "patch.search_radius_thresholds": (
            "patch",
            "search_radius_thresholds",
            lambda value: [float(item) for item in value],
        ),
        "patch.num_points_radius_estimate": (
            "patch",
            "num_points_radius_estimate",
            int,
        ),
        "patch.fixed_des_radii": (
            "patch",
            "fixed_des_radii",
            lambda value: [float(item) for item in value],
        ),
        "match.dist_th": ("match", "dist_th", float),
        "match.inlier_th": ("match", "inlier_th", float),
        "match.pose_estimator": ("match", "pose_estimator", str),
        "match.kiss_resolution": ("match", "kiss_resolution", float),
        "match.enable_early_exit": ("match", "enable_early_exit", _coerce_bool),
        "match.early_exit_min_inliers": ("match", "early_exit_min_inliers", int),
        "test.pose_refine": ("test", "pose_refine", _coerce_bool),
    }
    for key, (section, field, caster) in dotted_map.items():
        if key in overrides:
            bufferx[section][field] = caster(overrides[key])

    dataset["first_downsample"] = float(bufferx["data"]["downsample"])
    dataset["second_downsample"] = float(bufferx["data"]["voxel_size_0"])
    dataset["max_num_points"] = int(bufferx["data"]["max_numPts"])


def _build_bufferx_bridge_config(
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    runtime_device = _require_cuda_runtime_device(config)
    template = _load_yaml_template(
        _repo_root()
        / "src"
        / "benchmarking"
        / "bridges"
        / "configs"
        / "c3vd_bufferx.yaml"
    )
    dataset = template["dataset"]
    dataset["manifest_path"] = config["data"]["manifest_path"]
    dataset["dataset_root"] = config["data"].get("dataset_root")
    dataset["random_seed"] = int(config["preprocess"]["seed"])
    dataset["num_points"] = int(
        config["preprocess"].get("num_points_override") or dataset["num_points"]
    )
    dataset["sampling_mode"] = str(
        config["preprocess"].get("sampling_override") or dataset["sampling_mode"]
    )
    perturbation = dict(config.get("perturbation", {}))
    dataset["perturbation_enabled"] = bool(perturbation.get("enabled", False))
    dataset["rotation_deg"] = float(perturbation.get("rotation_deg", 0.0))
    dataset["translation_m"] = float(perturbation.get("translation_m", 0.0))
    dataset["min_rotation_deg"] = float(perturbation.get("min_rotation_deg", 0.0))
    dataset["min_translation_m"] = float(
        perturbation.get("min_translation_m", 0.0)
    )
    dataset["noise_sigma"] = float(perturbation.get("noise_sigma", 0.0))
    dataset["noise_clip"] = float(perturbation.get("noise_clip", 0.0))
    dataset["apply_noise_to"] = str(perturbation.get("apply_noise_to", "source"))

    frozen_scene_splits = _load_frozen_scene_splits(
        config["data"].get("subset_config_path")
    )
    if frozen_scene_splits is not None:
        train_scenes, validation_scenes, test_scenes = frozen_scene_splits
        dataset["train_scenes"] = train_scenes
        dataset["val_scenes"] = validation_scenes
        dataset["test_scenes"] = test_scenes

    _apply_bufferx_hparam_overrides(
        template,
        dict(config["model"].get("overrides", {})),
    )

    training = template["training"]
    training["device"] = runtime_device
    training["checkpoint_dir"] = str(output_dir / "train_bridge")
    training["tensorboard_dir"] = str(output_dir / "train_bridge_tensorboard")
    training["log_dir"] = str(output_dir / "train_bridge_logs")

    bufferx_train = template["bufferx"]["train"]
    bufferx_train["num_workers"] = int(config["runtime"].get("num_workers", 0))
    training_overrides = dict(config["runtime"].get("training_overrides", {}))
    allowed = {
        "epoch",
        "max_iter",
        "batch_size",
        "num_workers",
        "pos_num",
        "augmentation_noise",
        "max_train_pairs",
        "max_val_pairs",
        "max_test_pairs",
    }
    unsupported = sorted(set(training_overrides) - allowed)
    if unsupported:
        raise ValueError(
            "Unsupported BUFFER-X runtime.training_overrides keys: "
            + ", ".join(unsupported)
        )
    for key in ("epoch", "max_iter", "batch_size", "num_workers", "pos_num"):
        if key in training_overrides:
            bufferx_train[key] = int(training_overrides[key])
    if "augmentation_noise" in training_overrides:
        bufferx_train["augmentation_noise"] = float(
            training_overrides["augmentation_noise"]
        )
    for key in ("max_train_pairs", "max_val_pairs", "max_test_pairs"):
        if key in training_overrides:
            dataset[key] = int(training_overrides[key])

    if _is_smoke_train(config):
        bufferx_train["epoch"] = 1
        bufferx_train["max_iter"] = 2
        dataset["max_train_pairs"] = 2
        dataset["max_val_pairs"] = 1
        dataset["max_test_pairs"] = 1

    return template


def _apply_regtr_hparam_transfer_overrides(
    template: dict[str, Any],
    overrides: dict[str, Any],
) -> None:
    """Map DPG-HPT dot-key overrides onto the RegTR bridge config."""

    if "data.voxel_size" in overrides or "first_subsampling_dl" in overrides:
        template["kpconv_options"]["first_subsampling_dl"] = float(
            overrides.get("data.voxel_size", overrides["first_subsampling_dl"])
        )
    if "model.matching_radius" in overrides or "overlap_radius" in overrides:
        matching_radius = float(
            overrides.get("model.matching_radius", overrides["overlap_radius"])
        )
        template["dataset"]["overlap_radius"] = matching_radius
        template["kpconv_options"]["overlap_radius"] = matching_radius
    if "losses.r_p" in overrides or "r_p" in overrides:
        template["losses"]["r_p"] = float(overrides.get("losses.r_p", overrides["r_p"]))
    if "losses.r_n" in overrides or "r_n" in overrides:
        template["losses"]["r_n"] = float(overrides.get("losses.r_n", overrides["r_n"]))
    if "eval.acceptance_radius" in overrides or "reg_success_thresh_trans" in overrides:
        template["validation"]["reg_success_thresh_trans"] = float(
            overrides.get(
                "eval.acceptance_radius",
                overrides["reg_success_thresh_trans"],
            )
        )
    if "reg_success_thresh_rot" in overrides:
        template["validation"]["reg_success_thresh_rot"] = float(
            overrides["reg_success_thresh_rot"]
        )


def _build_regtr_bridge_config(
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    _require_cuda_runtime_device(config)
    template = _load_yaml_template(
        _repo_root()
        / "src"
        / "benchmarking"
        / "bridges"
        / "configs"
        / "c3vd_regtr.yaml"
    )
    dataset = template["dataset"]
    _apply_common_train_dataset_settings(dataset, config)
    if dataset.get("perturbation_enabled"):
        dataset["augment_noise"] = 0.0
        dataset["perturb_pose"] = None
    _apply_regtr_hparam_transfer_overrides(
        template,
        dict(config["model"].get("overrides", {})),
    )
    template["dataloader"]["num_workers"] = int(config["runtime"].get("num_workers", 0))
    template["general"]["expt_name"] = _bridge_run_name(config)
    training_overrides = dict(config["runtime"].get("training_overrides", {}))
    allowed_training_overrides = {
        "base_lr",
        "grad_clip",
        "niter",
        "resume_from",
        "test_batch_size",
        "train_batch_size",
        "val_batch_size",
        "validate_every",
        "nb_sanity_val_steps",
    }
    unsupported = sorted(set(training_overrides) - allowed_training_overrides)
    if unsupported:
        raise ValueError(
            "Unsupported RegTR runtime.training_overrides keys: "
            + ", ".join(unsupported)
        )
    for key in ("train_batch_size", "val_batch_size", "test_batch_size"):
        if key in training_overrides:
            dataset[key] = int(training_overrides[key])
    if "niter" in training_overrides:
        template["train_options"]["niter"] = int(training_overrides["niter"])
    if "base_lr" in training_overrides:
        template["solver"]["base_lr"] = float(training_overrides["base_lr"])
    if "grad_clip" in training_overrides:
        template["solver"]["grad_clip"] = float(training_overrides["grad_clip"])

    if _is_smoke_train(config):
        _apply_smoke_dataset_settings(dataset, config)
        dataset["train_batch_size"] = 1
        dataset["val_batch_size"] = 1
        dataset["test_batch_size"] = 1
        template["train_options"]["niter"] = 2
        template["dataloader"]["persistent_workers"] = False
        template["dataloader"]["pin_memory"] = False
        template["losses"]["wt_feature"] = 0.0
        template["losses"]["wt_feature_un"] = 0.0
        template["losses"]["feature_loss_on"] = []

    return template


def _regtr_validate_every(config: dict[str, Any]) -> int:
    training_overrides = dict(config["runtime"].get("training_overrides", {}))
    if "validate_every" in training_overrides:
        return int(training_overrides["validate_every"])
    return 1 if _is_smoke_train(config) else -1


def _regtr_sanity_val_steps(config: dict[str, Any]) -> int:
    training_overrides = dict(config["runtime"].get("training_overrides", {}))
    if "nb_sanity_val_steps" in training_overrides:
        return int(training_overrides["nb_sanity_val_steps"])
    return 1


def _build_geotransformer_bridge_config(
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    runtime_device = _require_cuda_runtime_device(config)
    template = _load_yaml_template(
        _repo_root()
        / "src"
        / "benchmarking"
        / "bridges"
        / "configs"
        / "c3vd_geotransformer.yaml"
    )
    dataset = template["dataset"]
    _apply_common_train_dataset_settings(dataset, config)
    if _is_smoke_train(config):
        _apply_smoke_dataset_settings(dataset, config)

    training = template["training"]
    training["experiment_name"] = _bridge_run_name(config)
    training["device"] = runtime_device
    training["num_workers"] = int(config["runtime"].get("num_workers", 0))
    training["checkpoint_dir"] = str(output_dir / "train_bridge")
    training["log_dir"] = str(output_dir / "train_bridge_logs")
    training["seed"] = int(config["preprocess"]["seed"])

    overrides = dict(config["model"].get("overrides", {}))
    model = template["model"]
    if "experiment_name" in overrides:
        model["experiment_name"] = str(overrides["experiment_name"])
    if "spatial_scale" in overrides:
        model["spatial_scale"] = float(overrides["spatial_scale"])
    if "num_points_in_patch" in overrides:
        model["num_points_in_patch"] = int(overrides["num_points_in_patch"])
    if "angle_k" in overrides:
        model["angle_k"] = int(overrides["angle_k"])
    if "neighbor_limits" in overrides:
        training["neighbor_limits"] = [
            int(value) for value in overrides["neighbor_limits"]
        ]
    training_overrides = dict(config["runtime"].get("training_overrides", {}))
    allowed_training_overrides = {
        "batch_size",
        "grad_acc_steps",
        "lr",
        "lr_decay",
        "lr_decay_steps",
        "max_epochs",
        "max_train_steps",
        "max_val_steps",
        "resume_from",
        "resume_optimizer",
        "resume_strict",
        "save_interval",
        "validate_every",
        "weight_decay",
    }
    unsupported = sorted(set(training_overrides) - allowed_training_overrides)
    if unsupported:
        raise ValueError(
            "Unsupported GeoTransformer runtime.training_overrides keys: "
            + ", ".join(unsupported)
        )
    for key in (
        "batch_size",
        "max_epochs",
        "max_train_steps",
        "max_val_steps",
        "resume_from",
        "resume_optimizer",
        "resume_strict",
        "save_interval",
        "validate_every",
    ):
        if key in training_overrides:
            training[key] = training_overrides[key]
    if "lr" in training_overrides:
        template["optim"]["lr"] = float(training_overrides["lr"])
    if "lr_decay" in training_overrides:
        template["optim"]["lr_decay"] = float(training_overrides["lr_decay"])
    if "lr_decay_steps" in training_overrides:
        template["optim"]["lr_decay_steps"] = int(training_overrides["lr_decay_steps"])
    if "weight_decay" in training_overrides:
        template["optim"]["weight_decay"] = float(training_overrides["weight_decay"])
    if "grad_acc_steps" in training_overrides:
        template["optim"]["grad_acc_steps"] = int(training_overrides["grad_acc_steps"])

    if _is_smoke_train(config):
        training["batch_size"] = 1
        training["max_epochs"] = 1
        training["max_train_steps"] = 2
        training["max_val_steps"] = 1
        training["save_interval"] = 1
        training["validate_every"] = 1

    return template


def _resolve_checkpoint_from_manager_dir(checkpoint_dir: Path) -> Path:
    checkpoints_file = checkpoint_dir / "checkpoints.txt"
    if not checkpoints_file.exists():
        raise FileNotFoundError(f"Checkpoint manifest not found: {checkpoints_file}")

    best_step_line = checkpoints_file.read_text(encoding="utf-8").splitlines()[0]
    best_step = best_step_line.split(":", 1)[1].strip()
    if best_step and best_step.lower() != "none":
        best_path = checkpoint_dir / f"model-{best_step}.pth"
        if best_path.exists():
            return best_path

    candidates = sorted(checkpoint_dir.glob("model-*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under {checkpoint_dir}")
    return candidates[-1]


def _run_dcp_bridge(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    bridge_config = _build_dcp_bridge_config(config, output_dir)
    bridge_config_path = _write_bridge_config(
        bridge_config, output_dir, "dcp_smoke_bridge_config.yaml"
    )

    repo_root = _repo_root()
    command_metrics = [
        _run_profiled_subprocess(
            "dcp_train",
            [
                sys.executable,
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
            cwd=repo_root,
            metrics_dir=output_dir / "train_bridge_metrics",
        )
    ]

    checkpoint_dir = Path(bridge_config["training"]["checkpoint_dir"])
    checkpoint_path = _resolve_existing_checkpoint(
        checkpoint_dir / "models" / "model_best.pth",
        checkpoint_dir / "models" / "model_epoch_1.pth",
    )
    _verify_checkpoint_loadable(checkpoint_path)
    return {
        "status": "completed",
        "train_bridge": "src.benchmarking.bridges.train_dcp_c3vd",
        "bridge_config_path": str(bridge_config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_load_verified": True,
        **_summarize_train_metrics(command_metrics),
    }


def _run_pointnetlk_bridge(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    bridge_config = _build_pointnetlk_bridge_config(config, output_dir)
    bridge_config_path = _write_bridge_config(
        bridge_config, output_dir, "pointnetlk_smoke_bridge_config.yaml"
    )

    repo_root = _repo_root()
    script_path = (
        repo_root
        / "src"
        / "benchmarking"
        / "bridges"
        / "train_pointnetlk_c3vd.py"
    )
    dataset_root = bridge_config["dataset"]["data_root"]

    command_metrics = [
        _run_profiled_subprocess(
            "pointnetlk_classifier",
            [
                sys.executable,
                str(script_path),
                "--config",
                str(bridge_config_path),
                "--stage",
                "classifier",
                "--data-root",
                dataset_root,
            ],
            cwd=repo_root,
            metrics_dir=output_dir / "train_bridge_metrics",
        )
    ]

    checkpoint_dir = Path(bridge_config["output"]["checkpoint_dir"])
    classifier_prefix = bridge_config["output"]["classifier_prefix"]
    transfer_from = _resolve_existing_checkpoint(
        checkpoint_dir / f"{classifier_prefix}_feat_best.pth",
        checkpoint_dir / f"{classifier_prefix}_feat_last.pth",
    )

    command_metrics.append(
        _run_profiled_subprocess(
            "pointnetlk",
            [
                sys.executable,
                str(script_path),
                "--config",
                str(bridge_config_path),
                "--stage",
                "pointnetlk",
                "--data-root",
                dataset_root,
                "--transfer-from",
                str(transfer_from),
            ],
            cwd=repo_root,
            metrics_dir=output_dir / "train_bridge_metrics",
        )
    )

    pointnetlk_prefix = bridge_config["output"]["pointnetlk_prefix"]
    checkpoint_path = _resolve_existing_checkpoint(
        checkpoint_dir / f"{pointnetlk_prefix}_model_best.pth",
        checkpoint_dir / f"{pointnetlk_prefix}_model_last.pth",
        checkpoint_dir / f"{pointnetlk_prefix}_snap_best.pth",
        checkpoint_dir / f"{pointnetlk_prefix}_snap_last.pth",
    )
    _verify_checkpoint_loadable(checkpoint_path)
    return {
        "status": "completed",
        "train_bridge": "src.benchmarking.bridges.train_pointnetlk_c3vd",
        "bridge_config_path": str(bridge_config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_load_verified": True,
        "pretrain_checkpoint_path": str(transfer_from),
        **_summarize_train_metrics(command_metrics),
    }


def _run_pointnetlk_revisited_bridge(
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    bridge_config = _build_pointnetlk_revisited_bridge_config(config, output_dir)
    bridge_config_path = _write_bridge_config(
        bridge_config, output_dir, "pointnetlk_revisited_smoke_bridge_config.yaml"
    )

    repo_root = _repo_root()
    command_metrics = [
        _run_profiled_subprocess(
            "pointnetlk_revisited_train",
            [
                sys.executable,
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
            cwd=repo_root,
            metrics_dir=output_dir / "train_bridge_metrics",
        )
    ]

    checkpoint_dir = Path(bridge_config["training"]["checkpoint_dir"])
    checkpoint_path = _resolve_existing_checkpoint(
        checkpoint_dir / "pointnetlk_c3vd_model_best.pth",
        checkpoint_dir / "pointnetlk_c3vd_snap_last.pth",
    )
    _verify_checkpoint_loadable(checkpoint_path)
    return {
        "status": "completed",
        "train_bridge": "src.benchmarking.bridges.train_pointnetlk_revisited_c3vd",
        "bridge_config_path": str(bridge_config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_load_verified": True,
        **_summarize_train_metrics(command_metrics),
    }


def _run_mamba3d_bridge(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    bridge_config = _build_mamba3d_bridge_config(config, output_dir)
    model_id = str(config["model"]["id"])
    bridge_config_path = _write_bridge_config(
        bridge_config, output_dir, f"{model_id}_smoke_bridge_config.yaml"
    )

    repo_root = _repo_root()
    command_metrics = [
        _run_profiled_subprocess(
            f"{model_id}_train",
            [
                sys.executable,
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
                config["data"]["dataset_root"],
            ],
            cwd=repo_root,
            metrics_dir=output_dir / "train_bridge_metrics",
        )
    ]

    checkpoint_dir = Path(bridge_config["output"]["checkpoint_dir"])
    checkpoint_path = _resolve_existing_checkpoint(
        checkpoint_dir / "mamba3d_pointlk_model_best.pth",
        checkpoint_dir / "checkpoint_epoch_1.pth",
    )
    _verify_checkpoint_loadable(checkpoint_path)
    return {
        "status": "completed",
        "train_bridge": "src.benchmarking.bridges.train_mamba3d_c3vd",
        "bridge_config_path": str(bridge_config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_load_verified": True,
        **_summarize_train_metrics(command_metrics),
    }


def _run_bufferx_bridge(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    bridge_config = _build_bufferx_bridge_config(config, output_dir)
    bridge_config_path = _write_bridge_config(
        bridge_config,
        output_dir,
        "bufferx_bridge_config.yaml",
    )

    repo_root = _repo_root()
    command_metrics = [
        _run_profiled_subprocess(
            "bufferx_train",
            [
                sys.executable,
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
            cwd=repo_root,
            metrics_dir=output_dir / "train_bridge_metrics",
        )
    ]

    checkpoint_dir = Path(bridge_config["training"]["checkpoint_dir"])
    checkpoint_path = _resolve_existing_checkpoint(
        checkpoint_dir / "Pose" / "best.pth",
        checkpoint_dir / "Pose" / "0.pth",
        checkpoint_dir / "Desc" / "best.pth",
    )
    _verify_checkpoint_loadable(checkpoint_path)
    return {
        "status": "completed",
        "train_bridge": "src.benchmarking.bridges.train_bufferx_c3vd",
        "bridge_config_path": str(bridge_config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_load_verified": True,
        **_summarize_train_metrics(command_metrics),
    }


def _run_regtr_bridge(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    bridge_config = _build_regtr_bridge_config(config, output_dir)
    bridge_config_path = _write_bridge_config(
        bridge_config, output_dir, "regtr_smoke_bridge_config.yaml"
    )

    repo_root = _repo_root()
    logs_root = output_dir / "train_bridge_logs"
    experiment_name = _bridge_run_name(config)
    command_metrics = [
        _run_profiled_subprocess(
            "regtr_train",
            [
                sys.executable,
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
                str(_regtr_validate_every(config)),
                "--num_workers",
                str(config["runtime"].get("num_workers", 0)),
                "--nb_sanity_val_steps",
                str(_regtr_sanity_val_steps(config)),
            ]
            + (
                [
                    "--resume",
                    str(config["runtime"]["training_overrides"]["resume_from"]),
                ]
                if config["runtime"].get("training_overrides", {}).get("resume_from")
                else []
            ),
            cwd=repo_root,
            metrics_dir=output_dir / "train_bridge_metrics",
        )
    ]

    run_dirs = sorted((logs_root / "c3vd").glob(f"*_{experiment_name}"))
    if not run_dirs:
        raise FileNotFoundError(
            f"Unable to locate RegTR run directory under {logs_root}"
        )

    run_dir = run_dirs[-1]
    checkpoint_path = _resolve_checkpoint_from_manager_dir(run_dir / "ckpt")
    _verify_checkpoint_loadable(checkpoint_path)
    return {
        "status": "completed",
        "train_bridge": "src.benchmarking.bridges.train_regtr_c3vd",
        "bridge_config_path": str(bridge_config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_load_verified": True,
        "bridge_run_dir": str(run_dir),
        **_summarize_train_metrics(command_metrics),
    }


def _run_geotransformer_bridge(
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    bridge_config = _build_geotransformer_bridge_config(config, output_dir)
    bridge_config_path = _write_bridge_config(
        bridge_config, output_dir, "geotransformer_smoke_bridge_config.yaml"
    )

    repo_root = _repo_root()
    command_metrics = [
        _run_profiled_subprocess(
            "geotransformer_train",
            [
                sys.executable,
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
            cwd=repo_root,
            metrics_dir=output_dir / "train_bridge_metrics",
        )
    ]

    checkpoint_dir = Path(bridge_config["training"]["checkpoint_dir"])
    checkpoint_path = _resolve_existing_checkpoint(
        checkpoint_dir / "geotransformer_c3vd_model_best.pth",
        checkpoint_dir / "epoch-1.pth.tar",
    )
    _verify_checkpoint_loadable(checkpoint_path)
    return {
        "status": "completed",
        "train_bridge": "src.benchmarking.bridges.train_geotransformer_c3vd",
        "bridge_config_path": str(bridge_config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_load_verified": True,
        **_summarize_train_metrics(command_metrics),
    }


def _wandb_smoke_check(config: dict[str, Any]) -> dict[str, Any]:
    try:
        import wandb  # type: ignore
    except ImportError:
        return {"status": "unavailable"}

    run = wandb.init(
        project="c3vd-raycasting-benchmark",
        mode="disabled",
        config={
            "model_id": config["model"]["id"],
            "preprocess_profile": config["preprocess"]["profile"],
        },
    )
    run.finish()
    return {"status": "initialized"}


def run_train(config: dict[str, Any]) -> dict[str, Any]:
    """Run the configured train bridge and validate checkpoint traceability."""

    output_dir = Path(config["runtime"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = ModelRegistry()
    spec = registry.get(config["model"]["id"])
    assert_runtime_policy_compatible(spec, str(config["runtime"]["device"]))
    assert_baseline_repo_clean(repo_root=_repo_root(), spec=spec)
    normalized_config_path = output_dir / "normalized_train_config.json"
    normalized_config_path.write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    snapshot = git_snapshot(
        repo_root=_repo_root(),
        config_path=normalized_config_path,
        manifest_path=config["data"]["manifest_path"],
        subset_config_path=config["data"].get("subset_config_path"),
        preprocess_profile_id=config["preprocess"]["profile"],
        model_id=config["model"]["id"],
        checkpoint_id=config["model"].get("checkpoint_path"),
        output_path=output_dir / "git_snapshot.json",
    )

    if not spec.capabilities.supports_train:
        bridge_summary = {
            "status": "not_supported",
            "train_bridge": spec.train_bridge,
            "adapter_load_verified": False,
        }
    else:
        bridge_runners = {
            "dcp": _run_dcp_bridge,
            "pointnetlk": _run_pointnetlk_bridge,
            "pointnetlk_revisited": _run_pointnetlk_revisited_bridge,
            "mamba3d": _run_mamba3d_bridge,
            "mamba3d_true": _run_mamba3d_bridge,
            "mamba3d_mamba2": _run_mamba3d_bridge,
            "mamba3d_mamba2_direct": _run_mamba3d_bridge,
            "mambanetlk": _run_mamba3d_bridge,
            "bufferx": _run_bufferx_bridge,
            "regtr": _run_regtr_bridge,
            "geotransformer": _run_geotransformer_bridge,
        }
        bridge_runner = bridge_runners.get(spec.model_id)
        if bridge_runner is None:
            bridge_summary = {
                "status": "bridge_pending" if spec.train_bridge else "not_supported",
                "train_bridge": spec.train_bridge,
                "adapter_load_verified": False,
            }
        else:
            bridge_summary = bridge_runner(config, output_dir)
            checkpoint_path = Path(bridge_summary["checkpoint_path"])
            _verify_checkpoint_matches_eval_adapter(spec, config, checkpoint_path)
            bridge_summary["adapter_load_verified"] = True

    summary = {
        **bridge_summary,
        "model_id": spec.model_id,
        "git_snapshot": snapshot,
        "wandb": _wandb_smoke_check(config),
        "output_dir": str(output_dir),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
