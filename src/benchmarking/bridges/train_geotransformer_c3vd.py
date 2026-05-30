#!/usr/bin/env python
"""Train GeoTransformer on C3VD through the benchmark bridge."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parents[1]
REPO_ROOT = SRC_ROOT.parent
GEOTRANSFORMER_ROOT = REPO_ROOT / "baselines" / "GeoTransformer"
DEFAULT_EXPERIMENT_NAME = (
    "geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn"
)

sys.path.insert(0, str(SRC_ROOT))

from common.datasets.c3vd_for_geotransformer import C3VDForGeoTransformer  # noqa: E402


def _module_origin(module: object) -> str:
    module_file = getattr(module, "__file__", None)
    if module_file:
        return os.path.abspath(module_file)
    module_path = getattr(module, "__path__", None)
    if module_path:
        try:
            return os.path.abspath(next(iter(module_path)))
        except StopIteration:
            return ""
    return ""


def _purge_conflicting_geotransformer_modules(
    package_root: Path,
    experiment_root: Path,
) -> None:
    package_root_str = os.path.abspath(str(package_root))
    experiment_root_str = os.path.abspath(str(experiment_root))
    experiment_modules = {"backbone", "config", "dataset", "loss", "model"}
    for module_name, module in list(sys.modules.items()):
        top_level = module_name.split(".", 1)[0]
        if top_level != "geotransformer" and module_name not in experiment_modules:
            continue
        origin = _module_origin(module)
        if not origin:
            continue
        if top_level == "geotransformer" and not origin.startswith(package_root_str):
            sys.modules.pop(module_name, None)
        elif (
            module_name in experiment_modules
            and not origin.startswith(experiment_root_str)
        ):
            sys.modules.pop(module_name, None)


def _ensure_import_paths(experiment_root: Path) -> None:
    path_entries = (str(experiment_root), str(GEOTRANSFORMER_ROOT))
    for entry in path_entries:
        if entry in sys.path:
            sys.path.remove(entry)
    for entry in reversed(path_entries):
        sys.path.insert(0, entry)


def _load_module_from_path(module_name: str, module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create module spec for '{module_path}'.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _move_to_device(payload: Any, device: torch.device) -> Any:
    if torch.is_tensor(payload):
        return payload.to(device)
    if isinstance(payload, list):
        return [_move_to_device(item, device) for item in payload]
    if isinstance(payload, dict):
        return {key: _move_to_device(value, device) for key, value in payload.items()}
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {} if payload is None else dict(payload)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _prepare_experiment_modules(experiment_name: str) -> tuple[Any, Any, Any]:
    experiment_root = GEOTRANSFORMER_ROOT / "experiments" / experiment_name
    if not experiment_root.exists():
        raise FileNotFoundError(
            f"GeoTransformer experiment directory not found: {experiment_root}"
        )
    _purge_conflicting_geotransformer_modules(GEOTRANSFORMER_ROOT, experiment_root)
    _ensure_import_paths(experiment_root)
    suffix = experiment_name.replace(".", "_").replace("-", "_")
    config_module = _load_module_from_path(
        f"_c3vd_geotransformer_config_{suffix}",
        experiment_root / "config.py",
    )
    model_module = _load_module_from_path(
        f"_c3vd_geotransformer_model_{suffix}",
        experiment_root / "model.py",
    )
    loss_module = _load_module_from_path(
        f"_c3vd_geotransformer_loss_{suffix}",
        experiment_root / "loss.py",
    )
    return config_module, model_module, loss_module


def _make_vendor_cfg(config: dict[str, Any], output_dir: Path) -> Any:
    model_config = dict(config.get("model", {}))
    experiment_name = str(
        model_config.get("experiment_name", DEFAULT_EXPERIMENT_NAME)
    )
    config_module, _, _ = _prepare_experiment_modules(experiment_name)
    cfg = config_module.make_cfg()

    training = dict(config.get("training", {}))
    optim_config = dict(config.get("optim", {}))

    cfg.seed = int(training.get("seed", cfg.seed))
    cfg.output_dir = str(output_dir)
    cfg.snapshot_dir = str(output_dir / "snapshots")
    cfg.log_dir = str(output_dir / "logs")
    cfg.event_dir = str(output_dir / "events")
    cfg.feature_dir = str(output_dir / "features")
    cfg.registration_dir = str(output_dir / "registration")

    cfg.train.batch_size = int(training.get("batch_size", cfg.train.batch_size))
    cfg.train.num_workers = int(training.get("num_workers", cfg.train.num_workers))
    cfg.train.point_limit = config.get("dataset", {}).get("num_points")
    cfg.train.use_augmentation = False
    cfg.test.batch_size = int(training.get("batch_size", cfg.test.batch_size))
    cfg.test.num_workers = int(training.get("num_workers", cfg.test.num_workers))
    cfg.test.point_limit = config.get("dataset", {}).get("num_points")

    cfg.optim.lr = float(optim_config.get("lr", cfg.optim.lr))
    cfg.optim.lr_decay = float(optim_config.get("lr_decay", cfg.optim.lr_decay))
    cfg.optim.lr_decay_steps = int(
        optim_config.get("lr_decay_steps", cfg.optim.lr_decay_steps)
    )
    cfg.optim.weight_decay = float(
        optim_config.get("weight_decay", cfg.optim.weight_decay)
    )
    cfg.optim.max_epoch = int(training.get("max_epochs", cfg.optim.max_epoch))
    cfg.optim.grad_acc_steps = int(
        optim_config.get("grad_acc_steps", cfg.optim.grad_acc_steps)
    )

    spatial_scale = float(model_config.get("spatial_scale", 1.0))
    if "num_points_in_patch" in model_config:
        cfg.model.num_points_in_patch = int(model_config["num_points_in_patch"])
    if "angle_k" in model_config:
        cfg.geotransformer.angle_k = int(model_config["angle_k"])
    if spatial_scale != 1.0:
        cfg.backbone.init_voxel_size *= spatial_scale
        cfg.backbone.init_radius = (
            cfg.backbone.base_radius * cfg.backbone.init_voxel_size
        )
        cfg.backbone.init_sigma = (
            cfg.backbone.base_sigma * cfg.backbone.init_voxel_size
        )
        cfg.model.ground_truth_matching_radius *= spatial_scale
        cfg.fine_loss.positive_radius *= spatial_scale
        cfg.fine_matching.acceptance_radius *= spatial_scale
        cfg.eval.acceptance_radius *= spatial_scale
        cfg.eval.rmse_threshold *= spatial_scale
        cfg.ransac.distance_threshold *= spatial_scale
        cfg.geotransformer.sigma_d *= spatial_scale
    return cfg


def _make_loader(
    config: dict[str, Any],
    split: str,
    cfg: Any,
    neighbor_limits: list[int],
    max_pairs_key: str,
) -> DataLoader:
    from geotransformer.utils.data import registration_collate_fn_stack_mode

    dataset_config = dict(config["dataset"])
    dataset = C3VDForGeoTransformer(
        data_root=str(dataset_config["data_root"]),
        split=split,
        num_points=dataset_config.get("num_points"),
        sampling_mode=str(dataset_config.get("sampling_mode", "voxel")),
        normalize_mode=str(dataset_config.get("normalize_mode", "none")),
        perturbation_enabled=bool(dataset_config.get("perturbation_enabled", False)),
        rotation_deg=float(dataset_config.get("rotation_deg", 0.0)),
        translation_m=float(dataset_config.get("translation_m", 0.0)),
        noise_sigma=float(dataset_config.get("noise_sigma", 0.0)),
        noise_clip=float(dataset_config.get("noise_clip", 0.0)),
        apply_noise_to=str(dataset_config.get("apply_noise_to", "source")),
        synthetic_rotation_deg=float(
            dataset_config.get("synthetic_rotation_deg", 45.0)
        ),
        synthetic_translation_m=float(
            dataset_config.get("synthetic_translation_m", 0.5)
        ),
        train_ratio=float(dataset_config.get("train_ratio", 0.7)),
        random_seed=int(dataset_config.get("random_seed", 42)),
        train_scenes=dataset_config.get("train_scenes"),
        val_scenes=dataset_config.get("val_scenes"),
        test_scenes=dataset_config.get("test_scenes"),
        frame_stride=int(dataset_config.get("frame_stride", 1)),
        max_pairs=dataset_config.get(max_pairs_key)
        or (
            dataset_config.get("max_test_pairs")
            if max_pairs_key == "max_val_pairs"
            else None
        ),
    )

    def collate_fn(items: list[dict[str, object]]) -> dict[str, object]:
        return registration_collate_fn_stack_mode(
            items,
            cfg.backbone.num_stages,
            cfg.backbone.init_voxel_size,
            cfg.backbone.init_radius,
            neighbor_limits,
        )

    training = dict(config.get("training", {}))
    return DataLoader(
        dataset,
        batch_size=int(training.get("batch_size", 1)),
        shuffle=(split == "train"),
        num_workers=int(training.get("num_workers", 0)),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )


def _loss_value_to_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def _run_epoch(
    model: torch.nn.Module,
    loss_func: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_acc_steps: int,
    max_steps: int | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    if is_train:
        optimizer.zero_grad(set_to_none=True)

    totals: dict[str, float] = {}
    steps = 0
    for step, data_dict in enumerate(loader, start=1):
        data_dict = _move_to_device(data_dict, device)
        with torch.set_grad_enabled(is_train):
            output_dict = model(data_dict)
            loss_dict = loss_func(output_dict, data_dict)
            loss = loss_dict["loss"]
            if is_train:
                (loss / grad_acc_steps).backward()
                if step % grad_acc_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        for key, value in loss_dict.items():
            totals[key] = totals.get(key, 0.0) + _loss_value_to_float(value)
        steps += 1
        if max_steps is not None and step >= max_steps:
            break

    if is_train and steps % grad_acc_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if steps == 0:
        return {"loss": float("nan")}
    return {key: value / steps for key, value in totals.items()}


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def _move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _load_resume_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    training: dict[str, Any],
    device: torch.device,
) -> None:
    resume_from = training.get("resume_from")
    if not resume_from:
        return

    resume_path = Path(str(resume_from)).expanduser().resolve()
    if not resume_path.exists():
        raise FileNotFoundError(
            f"GeoTransformer resume checkpoint not found: {resume_path}"
        )

    checkpoint = torch.load(resume_path, map_location="cpu")
    strict = bool(training.get("resume_strict", True))
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=strict)
        if bool(training.get("resume_optimizer", False)) and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            _move_optimizer_state_to_device(optimizer, device)
    else:
        model.load_state_dict(checkpoint, strict=strict)
    print(f"Resumed GeoTransformer weights from {resume_path}")


def _is_better_metric(current: float, best: float, mode: str) -> bool:
    if not np.isfinite(current):
        return False
    if mode == "min":
        return current < best
    if mode == "max":
        return current > best
    raise ValueError(f"Unsupported best_metric_mode '{mode}'.")


def _json_metric(value: float) -> float | None:
    return value if np.isfinite(value) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to bridge YAML config.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = _load_yaml(config_path)
    training = dict(config.get("training", {}))
    model_config = dict(config.get("model", {}))
    experiment_name = str(
        model_config.get("experiment_name", DEFAULT_EXPERIMENT_NAME)
    )
    output_dir = Path(training.get("checkpoint_dir", "c3vd_geotransformer")).resolve()
    log_dir = Path(training.get("log_dir", output_dir / "logs")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    cfg = _make_vendor_cfg(config, output_dir)
    _, model_module, loss_module = _prepare_experiment_modules(experiment_name)
    _seed_everything(int(training.get("seed", cfg.seed)))

    device = torch.device(str(training.get("device", "cuda:0")))
    if not torch.cuda.is_available() or device.type != "cuda":
        raise RuntimeError("GeoTransformer C3VD training requires CUDA.")
    torch.cuda.set_device(device)

    neighbor_limits = [
        int(value)
        for value in training.get("neighbor_limits", [38, 36, 36, 38])
    ]
    train_loader = _make_loader(
        config,
        split="train",
        cfg=cfg,
        neighbor_limits=neighbor_limits,
        max_pairs_key="max_train_pairs",
    )
    val_loader = _make_loader(
        config,
        split="val",
        cfg=cfg,
        neighbor_limits=neighbor_limits,
        max_pairs_key="max_val_pairs",
    )

    model = model_module.create_model(cfg).to(device)
    loss_func = loss_module.OverallLoss(cfg).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(cfg.optim.lr),
        weight_decay=float(cfg.optim.weight_decay),
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        int(cfg.optim.lr_decay_steps),
        gamma=float(cfg.optim.lr_decay),
    )
    _load_resume_checkpoint(model, optimizer, training, device)

    max_epochs = int(training.get("max_epochs", cfg.optim.max_epoch))
    max_train_steps = training.get("max_train_steps")
    max_val_steps = training.get("max_val_steps")
    save_interval = int(training.get("save_interval", 1))
    validate_every = int(training.get("validate_every", 1))
    grad_acc_steps = max(int(cfg.optim.grad_acc_steps), 1)
    best_metric_name = str(training.get("best_metric", "loss"))
    best_metric_mode = str(training.get("best_metric_mode", "min"))
    if best_metric_mode not in {"min", "max"}:
        raise ValueError(f"Unsupported best_metric_mode '{best_metric_mode}'.")
    best_metric = float("inf") if best_metric_mode == "min" else -float("inf")
    best_checkpoint_path = output_dir / "geotransformer_c3vd_model_best.pth"
    best_epoch: int | None = None

    metrics: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        train_metrics = _run_epoch(
            model,
            loss_func,
            train_loader,
            device,
            optimizer,
            grad_acc_steps,
            int(max_train_steps) if max_train_steps is not None else None,
        )
        val_metrics = None
        if validate_every > 0 and epoch % validate_every == 0:
            with torch.no_grad():
                val_metrics = _run_epoch(
                    model,
                    loss_func,
                    val_loader,
                    device,
                    optimizer=None,
                    grad_acc_steps=1,
                    max_steps=int(max_val_steps)
                    if max_val_steps is not None
                    else None,
                )
        scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": float(scheduler.get_last_lr()[0]),
            "best_metric_name": best_metric_name,
            "best_metric_mode": best_metric_mode,
            "best_metric": _json_metric(best_metric),
            "best_epoch": best_epoch,
        }
        best_updated = False
        if val_metrics is not None and best_metric_name in val_metrics:
            current_metric = float(val_metrics[best_metric_name])
            if _is_better_metric(current_metric, best_metric, best_metric_mode):
                best_metric = current_metric
                best_epoch = epoch
                best_updated = True
                _save_checkpoint(
                    best_checkpoint_path,
                    model,
                    optimizer,
                    epoch,
                    config,
                )
        epoch_record["best_metric"] = _json_metric(best_metric)
        epoch_record["best_epoch"] = best_epoch
        epoch_record["best_checkpoint_updated"] = best_updated
        metrics.append(epoch_record)
        print(json.dumps(epoch_record, sort_keys=True))

        if epoch % save_interval == 0 or epoch == max_epochs:
            _save_checkpoint(
                output_dir / f"epoch-{epoch}.pth.tar",
                model,
                optimizer,
                epoch,
                config,
            )

    if best_epoch is None:
        _save_checkpoint(
            best_checkpoint_path,
            model,
            optimizer,
            max_epochs,
            config,
        )

    (log_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
