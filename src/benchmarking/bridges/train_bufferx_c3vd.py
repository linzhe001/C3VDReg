"""Train BUFFER-X on C3VD through the benchmark bridge."""

from __future__ import annotations

import argparse
import os
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree
from torch import optim
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
BUFFERX_ROOT = REPO_ROOT / "baselines" / "BUFFER-X"
BUFFERX_NAMESPACE_PACKAGES = {"dataset", "loss", "models", "utils"}
if str(BUFFERX_ROOT) not in sys.path:
    sys.path.insert(0, str(BUFFERX_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _module_belongs_to_bufferx(module: object) -> bool:
    root = os.path.abspath(str(BUFFERX_ROOT))
    module_file = getattr(module, "__file__", None)
    if module_file and os.path.abspath(module_file).startswith(root):
        return True
    return any(
        os.path.abspath(path).startswith(root)
        for path in getattr(module, "__path__", [])
    )


for _name, _module in list(sys.modules.items()):
    _top = _name.split(".", 1)[0]
    if _top in {"config", "dataset", "loss", "models", "trainer", "utils"}:
        if not _module_belongs_to_bufferx(_module):
            sys.modules.pop(_name, None)

for _package in BUFFERX_NAMESPACE_PACKAGES:
    _namespace = types.ModuleType(_package)
    _namespace.__path__ = [  # type: ignore[attr-defined]
        str(BUFFERX_ROOT / _package)
    ]
    _namespace.__package__ = _package
    _namespace.__spec__ = ModuleSpec(_package, loader=None, is_package=True)
    sys.modules[_package] = _namespace

from config import make_cfg  # type: ignore  # noqa: E402
from models.BUFFERX import BufferX  # type: ignore  # noqa: E402
from trainer import Trainer  # type: ignore  # noqa: E402
from utils.tools import setup_logger  # type: ignore  # noqa: E402

from src.common.datasets.c3vd_for_bufferx import (  # noqa: E402
    BufferXC3VDConfig,
    C3VDForBufferX,
    collate_bufferx_pair,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {} if payload is None else payload


def _set_nested_attr(root: Any, dotted_key: str, value: Any) -> None:
    current = root
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = getattr(current, part)
    setattr(current, parts[-1], value)


def _apply_section(cfg: Any, prefix: str, section: dict[str, Any]) -> None:
    for key, value in section.items():
        if isinstance(value, dict):
            _apply_section(cfg, f"{prefix}.{key}", value)
        else:
            _set_nested_attr(cfg, f"{prefix}.{key}", value)


def _build_vendor_cfg(config: dict[str, Any]) -> Any:
    bufferx = dict(config.get("bufferx", {}))
    cfg = make_cfg(
        bufferx.get("reference_dataset", "3DMatch"),
        Path(bufferx.get("root_dir", "../datasets")),
    )
    for section_name in ("data", "train", "optim", "patch", "match", "test"):
        section = bufferx.get(section_name, {})
        if isinstance(section, dict):
            _apply_section(cfg, section_name, section)
    return cfg


def _dataset_config(
    config: dict[str, Any],
    split: str,
) -> BufferXC3VDConfig:
    dataset = dict(config["dataset"])
    return BufferXC3VDConfig(
        manifest_path=str(dataset["manifest_path"]),
        dataset_root=dataset.get("dataset_root"),
        split=split,
        train_scenes=tuple(dataset.get("train_scenes") or ()),
        val_scenes=tuple(dataset.get("val_scenes") or ()),
        test_scenes=tuple(dataset.get("test_scenes") or ()),
        num_points=int(dataset.get("num_points", 8192)),
        sampling_mode=str(dataset.get("sampling_mode", "voxel")),
        normalize_mode=str(dataset.get("normalize_mode", "none")),
        random_seed=int(dataset.get("random_seed", 42)),
        max_pairs=dataset.get(f"max_{split}_pairs"),
        first_downsample=float(dataset.get("first_downsample", 0.02)),
        second_downsample=float(dataset.get("second_downsample", 0.035)),
        max_num_points=int(dataset.get("max_num_points", 30000)),
        heuristic_mode=str(dataset.get("heuristic_mode", "native")),
        num_fps=int(config["bufferx"]["patch"].get("num_fps", 1500)),
        num_points_radius_estimate=int(
            config["bufferx"]["patch"].get("num_points_radius_estimate", 2000)
        ),
        perturbation_enabled=bool(dataset.get("perturbation_enabled", False)),
        rotation_deg=float(dataset.get("rotation_deg", 0.0)),
        translation_m=float(dataset.get("translation_m", 0.0)),
        noise_sigma=float(dataset.get("noise_sigma", 0.0)),
        noise_clip=float(dataset.get("noise_clip", 0.0)),
        apply_noise_to=str(dataset.get("apply_noise_to", "source")),
    )


def _make_loader(config: dict[str, Any], split: str, num_workers: int) -> DataLoader:
    dataset = C3VDForBufferX(_dataset_config(config, split))
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate_bufferx_pair,
        drop_last=True,
    )


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return points @ rotation.T + translation


def _count_positive_correspondences(item: dict[str, Any]) -> int:
    source = np.asarray(item["src_sds_pts"], dtype=np.float64)
    target = np.asarray(item["tgt_sds_pts"], dtype=np.float64)
    transform = np.asarray(item["relt_pose"], dtype=np.float64)
    warped_source = _transform_points(source, transform)
    distances, _ = cKDTree(target).query(warped_source, k=1)
    return int(np.count_nonzero(distances < float(item["voxel_size"])))


def _preflight_positive_correspondences(
    config: dict[str, Any],
    logger: Any,
    *,
    max_samples: int = 3,
) -> None:
    dataset = C3VDForBufferX(_dataset_config(config, "train"))
    sample_count = min(int(max_samples), len(dataset))
    counts = [
        _count_positive_correspondences(dataset[index])
        for index in range(sample_count)
    ]
    logger.info(
        "BUFFER-X positive correspondence preflight: "
        f"samples={sample_count}, counts={counts}"
    )
    if counts and max(counts) > 0:
        return
    voxel_size = float(config["dataset"].get("second_downsample", 0.0))
    raise RuntimeError(
        "BUFFER-X C3VD bridge found zero positive correspondences before "
        "training. Current second_downsample/voxel_size_0="
        f"{voxel_size} is likely incompatible with C3VD mm_like spacing; "
        "adjust the C3VD hparams or use a validated DPG-HPT candidate."
    )


def _patch_fixed_radius_estimation(cfg: Any) -> None:
    import models.BUFFERX as bufferx_module  # type: ignore

    fixed = getattr(cfg.patch, "fixed_des_radii", None)
    if fixed is None:
        fixed = [float(cfg.patch.des_r)] * int(cfg.patch.num_scales)
    fixed_values = [float(value) for value in fixed]

    def _fixed_radius_estimation(
        *args: Any,
        thresholds: list[float] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        count = len(thresholds or fixed_values)
        if len(fixed_values) >= count:
            return fixed_values[:count]
        return [*fixed_values, *([fixed_values[-1]] * (count - len(fixed_values)))]

    bufferx_module.density_aware_radius_estimation = _fixed_radius_estimation


def _train_stage(config: dict[str, Any], cfg: Any, stage: str, logger: Any) -> None:
    training = dict(config["training"])
    checkpoint_dir = Path(training["checkpoint_dir"]).resolve()
    tensorboard_dir = Path(training["tensorboard_dir"]).resolve()
    cfg.stage = stage
    cfg.snapshot_root = str(checkpoint_dir)
    cfg.tensorboard_root = str(tensorboard_dir / stage)
    cfg.train.num_workers = int(config["bufferx"]["train"].get("num_workers", 0))

    model = BufferX(cfg)
    for other_stage in [item for item in cfg.train.all_stage if item != stage]:
        weight_path = checkpoint_dir / other_stage / "best.pth"
        if weight_path.exists():
            state_dict = torch.load(weight_path, map_location="cpu")
            filtered = {k: v for k, v in state_dict.items() if other_stage in k}
            model_state = model.state_dict()
            model_state.update(filtered)
            model.load_state_dict(model_state, strict=False)
            for parameter in getattr(model, other_stage).parameters():
                parameter.requires_grad = False

    optimizer = optim.Adam(
        model.get_parameter(),
        lr=cfg.optim.lr[stage],
        weight_decay=cfg.optim.weight_decay,
    )
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=cfg.optim.lr_decay,
    )
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=[0])
    args = SimpleNamespace(
        cfg=cfg,
        logger=logger,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_interval=cfg.optim.scheduler_interval[stage],
        train_loader=_make_loader(config, "train", cfg.train.num_workers),
        val_loader=_make_loader(config, "val", cfg.train.num_workers),
        save_dir=str(checkpoint_dir / stage),
        tboard_dir=cfg.tensorboard_root,
        evaluate_interval=1,
    )
    Trainer(args).train()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = _load_yaml(Path(args.config))
    training = dict(config["training"])
    log_dir = Path(training["log_dir"]).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(str(log_dir / "bufferx_train.log"))

    device = str(training.get("device", "cuda:0"))
    if not torch.cuda.is_available():
        raise RuntimeError("BUFFER-X training requires CUDA.")
    if ":" in device:
        torch.cuda.set_device(int(device.split(":", 1)[1]))

    seed = int(config["dataset"].get("random_seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cfg = _build_vendor_cfg(config)
    if config["dataset"].get("heuristic_mode") == "fixed":
        fixed = config["bufferx"]["patch"].get("fixed_des_radii")
        if fixed is not None:
            cfg.patch.fixed_des_radii = [float(value) for value in fixed]
        _patch_fixed_radius_estimation(cfg)
    _preflight_positive_correspondences(config, logger)

    for stage in cfg.train.all_stage:
        _train_stage(config, cfg, stage, logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
