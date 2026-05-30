"""
RegTR adapter for benchmark/unified evaluation.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import torch
import yaml
from easydict import EasyDict

from ..core.base_adapter import BaseAdapter

REGTR_SRC_PATH = Path(__file__).resolve().parents[3] / "baselines" / "RegTR" / "src"


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


def _purge_conflicting_regtr_modules(expected_root: Path) -> None:
    expected_root_str = os.path.abspath(str(expected_root))
    managed_prefixes = (
        "benchmark",
        "cvhelpers",
        "data_loaders",
        "models",
        "trainer",
        "utils",
    )
    for module_name, module in list(sys.modules.items()):
        if module_name.split(".", 1)[0] not in managed_prefixes:
            continue
        origin = _module_origin(module)
        if origin and not origin.startswith(expected_root_str):
            sys.modules.pop(module_name, None)


if str(REGTR_SRC_PATH) in sys.path:
    sys.path.remove(str(REGTR_SRC_PATH))
sys.path.insert(0, str(REGTR_SRC_PATH))
_purge_conflicting_regtr_modules(REGTR_SRC_PATH)


def _load_default_config() -> EasyDict:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "benchmarking"
        / "bridges"
        / "configs"
        / "c3vd_regtr.yaml"
    )
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    merged: dict[str, object] = {}
    for section in raw_config.values():
        if isinstance(section, dict):
            merged.update(section)
    return EasyDict(merged)


class RegTRAdapter(BaseAdapter):
    """Thin adapter around the RegTR baseline model."""

    def __init__(self, args):
        super().__init__(args)
        self.cfg = _load_default_config()

        overrides = vars(args) if hasattr(args, "__dict__") else dict(args)
        self.cfg.model = "regtr.RegTR"
        for key, value in overrides.items():
            if key in {"device", "checkpoint_path"}:
                continue
            self.cfg[key] = value

        self.cfg.setdefault("dataset", "c3vd")
        self.cfg.setdefault("reg_success_thresh_rot", 10)
        self.cfg.setdefault("reg_success_thresh_trans", 0.1)

        self.normalize_mode = getattr(args, "normalize_mode", "none")
        print("RegTR adapter initialized:")
        print(f"  Normalization mode: {self.normalize_mode}")
        print(f"  Embedding dim: {self.cfg.d_embed}")
        print(f"  Encoder layers: {self.cfg.num_encoder_layers}")

    def load_model(self, model_path):
        _purge_conflicting_regtr_modules(REGTR_SRC_PATH)
        model_module = importlib.import_module("models")
        model_class = model_module.get_model(self.cfg.model)
        if model_class is None:
            raise RuntimeError(f"Unable to resolve RegTR model '{self.cfg.model}'.")

        self.model = model_class(self.cfg)
        if model_path and os.path.exists(model_path):
            print(f"Loading RegTR model from: {model_path}")
            checkpoint = torch.load(model_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            if isinstance(state_dict, dict):
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
            else:
                raise RuntimeError("Unsupported RegTR checkpoint format.")
            print("✓ Loaded RegTR model (with strict=False)")
        else:
            print(f"Warning: Model path not found: {model_path}")
            print("Using randomly initialized weights")

        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, source, target):
        source_tensor = torch.from_numpy(source).float().to(self.device)
        target_tensor = torch.from_numpy(target).float().to(self.device)
        return source_tensor, target_tensor

    def forward(self, source_tensor, target_tensor):
        batch = {
            "src_xyz": [source_tensor],
            "tgt_xyz": [target_tensor],
        }
        with torch.no_grad():
            return self.model(batch)

    def extract_transformation(self, output):
        pose = output["pose"]
        if pose.ndim == 4:
            pose = pose[-1, 0]
        elif pose.ndim == 3:
            pose = pose[0]
        else:
            raise ValueError(f"Unexpected RegTR pose shape: {tuple(pose.shape)}")

        rotation = pose[:3, :3]
        translation = pose[:3, 3]
        return self.to_numpy(rotation), self.to_numpy(translation)

    def get_algorithm_name(self):
        return "RegTR"
