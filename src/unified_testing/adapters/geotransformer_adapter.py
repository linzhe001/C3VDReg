"""
GeoTransformer adapter for benchmark/unified evaluation.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..core.base_adapter import BaseAdapter

GEOTRANSFORMER_ROOT = (
    Path(__file__).resolve().parents[3] / "baselines" / "GeoTransformer"
)
DEFAULT_EXPERIMENT_NAME = (
    "geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn"
)
DEFAULT_NEIGHBOR_LIMITS = [38, 36, 36, 38]


def _apply_spatial_scale_to_cfg(cfg: Any, spatial_scale: float) -> None:
    if spatial_scale == 1.0:
        return
    cfg.backbone.init_voxel_size *= spatial_scale
    cfg.backbone.init_radius = cfg.backbone.base_radius * cfg.backbone.init_voxel_size
    cfg.backbone.init_sigma = cfg.backbone.base_sigma * cfg.backbone.init_voxel_size
    cfg.model.ground_truth_matching_radius *= spatial_scale
    cfg.fine_loss.positive_radius *= spatial_scale
    cfg.fine_matching.acceptance_radius *= spatial_scale
    cfg.eval.acceptance_radius *= spatial_scale
    cfg.eval.rmse_threshold *= spatial_scale
    cfg.ransac.distance_threshold *= spatial_scale
    cfg.geotransformer.sigma_d *= spatial_scale


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
    package_root: Path, experiment_root: Path
) -> None:
    package_root_str = os.path.abspath(str(package_root))
    experiment_root_str = os.path.abspath(str(experiment_root))
    package_prefixes = ("geotransformer",)
    experiment_modules = {"backbone", "config", "dataset", "loss", "model"}

    for module_name, module in list(sys.modules.items()):
        top_level = module_name.split(".", 1)[0]
        if top_level not in package_prefixes and module_name not in experiment_modules:
            continue

        origin = _module_origin(module)
        if not origin:
            continue

        if top_level in package_prefixes and not origin.startswith(package_root_str):
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
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

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


class GeoTransformerAdapter(BaseAdapter):
    """Thin adapter around the GeoTransformer 3DMatch experiment model."""

    def __init__(self, args):
        super().__init__(args)
        self.experiment_name = getattr(
            args, "experiment_name", DEFAULT_EXPERIMENT_NAME
        )
        self.experiment_root = (
            GEOTRANSFORMER_ROOT / "experiments" / self.experiment_name
        )
        if not self.experiment_root.exists():
            raise FileNotFoundError(
                f"GeoTransformer experiment directory not found: {self.experiment_root}"
            )

        neighbor_limits = getattr(args, "neighbor_limits", DEFAULT_NEIGHBOR_LIMITS)
        self.neighbor_limits = [int(limit) for limit in neighbor_limits]
        if len(self.neighbor_limits) != 4:
            raise ValueError(
                "GeoTransformer expects four neighbor limits, "
                f"received {self.neighbor_limits!r}."
            )

        suffix = self.experiment_name.replace(".", "_").replace("-", "_")
        self._config_module_name = f"_geotransformer_config_{suffix}"
        self._model_module_name = f"_geotransformer_model_{suffix}"
        self.cfg = None

        print("GeoTransformer adapter initialized:")
        print(f"  Experiment: {self.experiment_name}")
        print(f"  Neighbor limits: {self.neighbor_limits}")

    def _prepare_runtime(self) -> None:
        _purge_conflicting_geotransformer_modules(
            GEOTRANSFORMER_ROOT, self.experiment_root
        )
        _ensure_import_paths(self.experiment_root)

    def _load_config(self) -> Any:
        self._prepare_runtime()
        config_module = _load_module_from_path(
            self._config_module_name, self.experiment_root / "config.py"
        )
        cfg = config_module.make_cfg()

        point_limit = getattr(self.args, "point_limit", None)
        if point_limit is not None:
            cfg.test.point_limit = point_limit

        spatial_scale = float(getattr(self.args, "spatial_scale", 1.0))
        num_points_in_patch = getattr(self.args, "num_points_in_patch", None)
        if num_points_in_patch is not None:
            cfg.model.num_points_in_patch = int(num_points_in_patch)
        _apply_spatial_scale_to_cfg(cfg, spatial_scale)
        return cfg

    def load_model(self, model_path):
        self.cfg = self._load_config()
        model_module = _load_module_from_path(
            self._model_module_name, self.experiment_root / "model.py"
        )
        self.model = model_module.create_model(self.cfg)

        if model_path and os.path.exists(model_path):
            print(f"Loading GeoTransformer model from: {model_path}")
            checkpoint = torch.load(model_path, map_location="cpu")
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            missing_keys, unexpected_keys = self.model.load_state_dict(
                state_dict, strict=False
            )
            if missing_keys:
                print(
                    "  Warning: Missing keys in checkpoint: "
                    f"{len(missing_keys)} keys"
                )
            if unexpected_keys:
                print(
                    "  Warning: Unexpected keys in checkpoint: "
                    f"{len(unexpected_keys)} keys"
                )
            print("✓ Loaded GeoTransformer model (with strict=False)")
        else:
            print(f"Warning: Model path not found: {model_path}")
            print("Using randomly initialized weights")

        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, source, target):
        if self.cfg is None:
            raise RuntimeError("GeoTransformerAdapter.load_model() must run first.")

        self._prepare_runtime()
        from geotransformer.utils.data import registration_collate_fn_stack_mode

        data_dict = {
            "ref_points": np.asarray(target, dtype=np.float32),
            "src_points": np.asarray(source, dtype=np.float32),
            "ref_feats": np.ones((target.shape[0], 1), dtype=np.float32),
            "src_feats": np.ones((source.shape[0], 1), dtype=np.float32),
            # Vendor forward computes gt_node_corr_indices even during eval,
            # but the value is not consumed in inference mode.
            "transform": np.eye(4, dtype=np.float32),
        }

        collated = registration_collate_fn_stack_mode(
            [data_dict],
            self.cfg.backbone.num_stages,
            self.cfg.backbone.init_voxel_size,
            self.cfg.backbone.init_radius,
            self.neighbor_limits,
        )
        return _move_to_device(collated, self.device), None

    def forward(self, data_dict, _unused=None):
        with torch.no_grad():
            return self.model(data_dict)

    def extract_transformation(self, output):
        transform = output["estimated_transform"]
        if torch.is_tensor(transform):
            transform = self.to_numpy(transform)
        else:
            transform = np.asarray(transform, dtype=np.float64)

        if transform.shape == (1, 4, 4):
            transform = transform[0]
        if transform.shape != (4, 4):
            raise ValueError(
                "Unexpected GeoTransformer transform shape: "
                f"{tuple(transform.shape)}"
            )

        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        return rotation, translation

    def get_algorithm_name(self):
        return "GeoTransformer"
