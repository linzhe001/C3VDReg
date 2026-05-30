"""BUFFER-X adapter for benchmark evaluation."""

from __future__ import annotations

import os
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from src.common.datasets.c3vd_for_bufferx import (
    BufferXC3VDConfig,
    collate_bufferx_pair,
    prepare_bufferx_pair,
)
from src.unified_testing.core.base_adapter import BaseAdapter

BUFFERX_ROOT = Path(__file__).resolve().parents[3] / "baselines" / "BUFFER-X"
BUFFERX_NAMESPACE_PACKAGES = {"dataset", "loss", "models", "utils"}


def _module_origin(module: object) -> str:
    module_file = getattr(module, "__file__", None)
    return os.path.abspath(module_file) if module_file else ""


def _module_paths(module: object) -> list[str]:
    return [os.path.abspath(path) for path in getattr(module, "__path__", [])]


def _prepare_bufferx_imports() -> None:
    root = os.path.abspath(str(BUFFERX_ROOT))
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    for name, module in list(sys.modules.items()):
        top = name.split(".", 1)[0]
        if top not in {"config", "dataset", "loss", "models", "trainer", "utils"}:
            continue
        origin = _module_origin(module)
        module_paths = _module_paths(module)
        belongs_to_bufferx = origin.startswith(root) or any(
            path.startswith(root) for path in module_paths
        )
        if not belongs_to_bufferx:
            sys.modules.pop(name, None)
    for package in BUFFERX_NAMESPACE_PACKAGES:
        module = types.ModuleType(package)
        module.__path__ = [str(BUFFERX_ROOT / package)]  # type: ignore[attr-defined]
        module.__package__ = package
        module.__spec__ = ModuleSpec(package, loader=None, is_package=True)
        sys.modules[package] = module


def _load_bufferx_symbols() -> tuple[Any, Any]:
    _prepare_bufferx_imports()
    from config import make_cfg  # type: ignore
    from models.BUFFERX import BufferX  # type: ignore

    return make_cfg, BufferX


def _set_nested_attr(root: Any, dotted_key: str, value: Any) -> None:
    current = root
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = getattr(current, part)
    setattr(current, parts[-1], value)


def _apply_overrides(cfg: Any, overrides: dict[str, Any]) -> None:
    mapping = {
        "data.downsample": "data.downsample",
        "data.voxel_size_0": "data.voxel_size_0",
        "data.voxel_size_1": "data.voxel_size_1",
        "data.max_numPts": "data.max_numPts",
        "patch.des_r": "patch.des_r",
        "patch.num_points_per_patch": "patch.num_points_per_patch",
        "patch.num_fps": "patch.num_fps",
        "patch.num_scales": "patch.num_scales",
        "patch.search_radius_thresholds": "patch.search_radius_thresholds",
        "patch.num_points_radius_estimate": "patch.num_points_radius_estimate",
        "patch.fixed_des_radii": "patch.fixed_des_radii",
        "match.dist_th": "match.dist_th",
        "match.inlier_th": "match.inlier_th",
        "match.pose_estimator": "match.pose_estimator",
        "match.kiss_resolution": "match.kiss_resolution",
        "match.enable_early_exit": "match.enable_early_exit",
        "match.early_exit_min_inliers": "match.early_exit_min_inliers",
        "test.pose_refine": "test.pose_refine",
    }
    for key, attr in mapping.items():
        if key in overrides:
            _set_nested_attr(cfg, attr, overrides[key])


def _patch_fixed_radius_estimation(cfg: Any) -> None:
    _prepare_bufferx_imports()
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


def _resolve_stage_root(model_path: str | None) -> Path | None:
    if not model_path:
        return None
    path = Path(model_path).expanduser().resolve()
    if path.is_dir():
        return path
    if path.name == "best.pth" and path.parent.name in {"Desc", "Pose"}:
        return path.parent.parent
    if path.parent.name in {"Desc", "Pose"}:
        return path.parent.parent
    return path.parent


def _get_arg(args: SimpleNamespace, key: str, default: Any = None) -> Any:
    return getattr(args, key) if hasattr(args, key) else default


class BufferXAdapter(BaseAdapter):
    """Adapter around the BUFFER-X vendor model."""

    def __init__(self, args: SimpleNamespace) -> None:
        super().__init__(args)
        self.reference_dataset = getattr(args, "reference_dataset", "3DMatch")
        self.root_dir = Path(getattr(args, "root_dir", "../datasets"))
        self.heuristic_mode = getattr(args, "heuristic_mode", "native")
        self.normalize_mode = getattr(args, "normalize_mode", "none")
        self.num_points = int(getattr(args, "num_points", 8192))
        self.sampling_mode = getattr(args, "sampling_mode", "voxel")
        self.first_downsample = float(
            _get_arg(args, "first_downsample", _get_arg(args, "data.downsample", 0.02))
        )
        self.second_downsample = float(
            _get_arg(
                args,
                "second_downsample",
                _get_arg(args, "data.voxel_size_0", 0.035),
            )
        )
        self.random_seed = int(getattr(args, "random_seed", 42))
        self._sample_index = 0
        self.cfg = None

    def _build_cfg(self) -> Any:
        make_cfg, _ = _load_bufferx_symbols()
        cfg = make_cfg(self.reference_dataset, self.root_dir)
        cfg.stage = "test"
        overrides = vars(self.args).copy()
        _apply_overrides(cfg, overrides)
        self.first_downsample = float(
            _get_arg(self.args, "first_downsample", cfg.data.downsample)
        )
        self.second_downsample = float(
            _get_arg(self.args, "second_downsample", cfg.data.voxel_size_0)
        )
        cfg.data.downsample = self.first_downsample
        cfg.data.voxel_size_0 = self.second_downsample
        cfg.data.voxel_size_1 = float(
            _get_arg(self.args, "data.voxel_size_1", self.second_downsample)
        )
        cfg.match.enable_early_exit = bool(
            getattr(cfg.match, "enable_early_exit", False)
        )
        if self.heuristic_mode == "fixed":
            fixed = _get_arg(
                self.args,
                "fixed_des_radii",
                _get_arg(self.args, "patch.fixed_des_radii"),
            )
            if fixed is not None:
                cfg.patch.fixed_des_radii = [float(value) for value in fixed]
            _patch_fixed_radius_estimation(cfg)
        return cfg

    def load_model(self, model_path: str | None) -> None:
        _, BufferX = _load_bufferx_symbols()
        self.cfg = self._build_cfg()
        self.model = BufferX(self.cfg)
        stage_root = _resolve_stage_root(model_path)
        if stage_root is not None:
            for stage in self.cfg.train.all_stage:
                checkpoint = stage_root / stage / "best.pth"
                if not checkpoint.exists():
                    print(f"Warning: BUFFER-X checkpoint not found: {checkpoint}")
                    continue
                state_dict = torch.load(checkpoint, map_location="cpu")
                filtered = {k: v for k, v in state_dict.items() if stage in k}
                model_dict = self.model.state_dict()
                model_dict.update(filtered)
                self.model.load_state_dict(model_dict, strict=False)
                print(f"Loaded BUFFER-X {stage} checkpoint: {checkpoint}")
        else:
            print("Warning: no BUFFER-X checkpoint provided; using random weights.")
        self.model.to(self.device)
        self.model.eval()

    def preprocess(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> tuple[dict[str, Any], None]:
        if self.cfg is None:
            raise RuntimeError(
                "BufferXAdapter.load_model() must run before preprocess()."
            )
        config = BufferXC3VDConfig(
            manifest_path="",
            dataset_root=None,
            split="test",
            num_points=self.num_points,
            sampling_mode=self.sampling_mode,
            normalize_mode=self.normalize_mode,
            random_seed=self.random_seed,
            first_downsample=float(self.cfg.data.downsample),
            second_downsample=float(self.cfg.data.voxel_size_0),
            max_num_points=int(self.cfg.data.max_numPts),
            heuristic_mode=self.heuristic_mode,
            num_fps=int(self.cfg.patch.num_fps),
            num_points_radius_estimate=int(self.cfg.patch.num_points_radius_estimate),
        )
        item = prepare_bufferx_pair(
            source=source,
            target=target,
            gt_transform=np.eye(4, dtype=np.float64),
            config=config,
            index=self._sample_index,
            split="test",
            sample_id=f"eval_{self._sample_index:06d}",
            scene_id="benchmark_eval",
        )
        self._sample_index += 1
        data_source = collate_bufferx_pair([item])
        for key, value in list(data_source.items()):
            if torch.is_tensor(value):
                data_source[key] = value.to(self.device)
        return data_source, None

    def forward(self, data_source: dict[str, Any], _unused: None = None) -> Any:
        with torch.no_grad():
            return self.model(data_source)

    def extract_transformation(self, output: Any) -> tuple[np.ndarray, np.ndarray]:
        transform = output[0] if isinstance(output, tuple) else output
        if transform is None:
            transform = np.eye(4, dtype=np.float64)
        transform = np.asarray(transform, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError(f"Unexpected BUFFER-X transform shape: {transform.shape}")
        return transform[:3, :3], transform[:3, 3]

    def get_algorithm_name(self) -> str:
        return "BUFFER-X"
