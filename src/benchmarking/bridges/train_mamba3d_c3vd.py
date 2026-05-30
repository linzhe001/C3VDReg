#!/usr/bin/env python
# ruff: noqa: E402,I001
"""
Training script for PointNetLK with Mamba3D backbone on C3VD dataset.

Follows the same unified preprocessing as other models in common/:
1. Clean point clouds (remove NaN/Inf)
2. VoxelGrid downsampling
3. Baseline-aware model-private normalization

Usage:
    python src/benchmarking/bridges/train_mamba3d_c3vd.py \\
        --config src/benchmarking/bridges/configs/c3vd_mamba3d.yaml \\
        --data-root /path/to/C3VD_datasets \\
        --resume experiments/checkpoints/mamba3d_pointlk_resume_0706_0435_model_best.pth
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import argparse
import numpy as np
import torch
import torch.utils.data
import yaml

# Add the local PointNetLK_c3vd fork to path for Mamba3D models
BRIDGE_DIR = Path(__file__).resolve().parent
SRC_ROOT = BRIDGE_DIR.parents[1]
REPO_ROOT = SRC_ROOT.parent
pointnetlk_c3vd_path = REPO_ROOT / "forks" / "PointNetLK_c3vd"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(pointnetlk_c3vd_path))

# Import Mamba3D models
from ptlk import (
    mamba3d_mamba2,
    mamba3d_mamba2_direct,
    mamba3d_true,
    mamba3d_v1,
    mambanetlk,
    pointlk,
    pointlk_cached,
)

# Import common utilities
from common.utils.benchmark_preprocess import (
    apply_pair_perturbation,
    normalize_point_cloud_pair,
    sample_point_cloud,
)
from common.utils.sampling import clean_point_cloud

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_CACHED_LK_ORDER_MODES = {
    "cached_recompute_order": "recompute",
    "cached_reuse_order": "reuse",
}
_LK_MODES = {"exact", *_CACHED_LK_ORDER_MODES}


def _extract_state_dict(payload):
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    return payload


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _optional_float(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


def _lk_mode(model_config):
    lk_mode = str(model_config.get("lk_mode", "exact"))
    if lk_mode not in _LK_MODES:
        raise ValueError(
            f"Unsupported Mamba3D lk_mode={lk_mode!r}. "
            "Expected 'exact', 'cached_recompute_order', or 'cached_reuse_order'."
        )
    return lk_mode


def _create_feature_extractor(model_config):
    model_type = str(model_config.get("model_type", "mamba3d_v1"))
    if model_type == "mambanetlk":
        logger.info("Creating MambaNetLK pair-pose + LK model.")
        model = mambanetlk.MambaNetLK(
            dim_k=model_config["dim_k"],
            sym_fn=model_config.get("symfn", "max"),
            num_groups=int(model_config.get("num_groups", 128)),
            group_size=int(model_config.get("group_size", 32)),
            trans_dim=int(model_config.get("trans_dim", 384)),
            depth=int(model_config.get("depth", 4)),
            d_state=int(model_config.get("d_state", 64)),
            d_conv=int(model_config.get("d_conv", 4)),
            expand=model_config.get("expand", 2),
            headdim=int(model_config.get("headdim", 64)),
            ngroups=int(model_config.get("ngroups", 1)),
            chunk_size=int(model_config.get("chunk_size", 128)),
            rmsnorm=_as_bool(model_config.get("rmsnorm", True)),
            drop_path_rate=float(model_config.get("drop_path_rate", 0.05)),
            drop_out=float(model_config.get("drop_out", 0.0)),
            grid_size=float(model_config.get("grid_size", 0.02)),
            knn_backend=str(model_config.get("knn_backend", "auto")),
            lk_mode=_lk_mode(model_config),
            delta=float(model_config.get("delta", 1.0e-2)),
            learn_delta=_as_bool(model_config.get("learn_delta", False)),
            lm_damping=float(model_config.get("lm_damping", 1.0e-3)),
            dx_clip_norm=_optional_float(model_config.get("dx_clip_norm", 0.25)),
            pair_dim=int(model_config.get("pair_dim", 256)),
            pair_hidden_dim=int(model_config.get("pair_hidden_dim", 512)),
            pair_dropout=float(model_config.get("pair_dropout", 0.0)),
            pose_rotation_scale=float(model_config.get("pose_rotation_scale", 1.5708)),
            pose_translation_scale=float(
                model_config.get("pose_translation_scale", 1.0)
            ),
            use_mem_eff_path=_as_bool(model_config.get("use_mem_eff_path", True)),
        )
        pretrained_path = model_config.get("pretrained_path")
        if pretrained_path:
            strict = _as_bool(model_config.get("pretrained_strict", False))
            logger.info(
                "Loading MambaNetLK backbone pretrained weights from %s (strict=%s)",
                pretrained_path,
                strict,
            )
            model.load_pretrained_weights(
                str(pretrained_path),
                strict=strict,
            )
        return model

    if model_type == "mamba3d_mamba2_direct":
        logger.info("Creating Mamba3DMamba2Direct feature extractor.")
        feature_extractor = mamba3d_mamba2_direct.Mamba3DMamba2Direct_features(
            dim_k=model_config["dim_k"],
            sym_fn=model_config.get("sym_fn", model_config.get("symfn", "max")),
            d_model=model_config.get("d_model", 128),
            depth=model_config.get("depth", 2),
            d_state=model_config.get("d_state", 16),
            d_conv=model_config.get("d_conv", 4),
            expand=model_config.get("expand", 1),
            headdim=model_config.get("headdim", 64),
            ngroups=model_config.get("ngroups", 1),
            chunk_size=model_config.get("chunk_size", 128),
            rmsnorm=_as_bool(model_config.get("rmsnorm", True)),
            drop_path_rate=model_config.get("drop_path_rate", 0.0),
            drop_out=model_config.get("drop_out", 0.0),
            layer_scale_init=model_config.get("layer_scale_init", 1.0e-4),
            stem_hidden_dim=model_config.get("stem_hidden_dim", 64),
            pos_hidden_dim=model_config.get("pos_hidden_dim", 64),
            ffn_ratio=model_config.get("ffn_ratio", 2.0),
            mlp_hidden_dim=model_config.get("mlp_hidden_dim", 256),
            mlp_norm=model_config.get("mlp_norm", "layernorm"),
            pooling=model_config.get("pooling", "maxavg"),
            descriptor_norm=model_config.get("descriptor_norm", "layernorm"),
            point_order=model_config.get("point_order", "identity"),
        )
        pretrained_from = model_config.get("pretrained_from")
        if pretrained_from:
            strict = _as_bool(model_config.get("pretrained_strict", False))
            logger.info(
                "Loading Mamba3DMamba2Direct pretrained weights from %s (strict=%s)",
                pretrained_from,
                strict,
            )
            feature_extractor.load_pretrained_weights(
                str(pretrained_from),
                strict=strict,
                verbose=True,
            )
        return feature_extractor

    if model_type == "mamba3d_mamba2":
        logger.info("Creating Mamba3DMamba2 feature extractor.")
        feature_extractor = mamba3d_mamba2.Mamba3DMamba2_features(
            dim_k=model_config["dim_k"],
            sym_fn=model_config.get("sym_fn", model_config.get("symfn", "max")),
            num_groups=model_config.get("num_groups", 64),
            group_size=model_config.get("group_size", 32),
            trans_dim=model_config.get("trans_dim", 384),
            depth=model_config.get("depth", 4),
            d_state=model_config.get("d_state", 64),
            d_conv=model_config.get("d_conv", 4),
            expand=model_config.get("expand", 2),
            headdim=model_config.get("headdim", 64),
            ngroups=model_config.get("ngroups", 1),
            chunk_size=model_config.get("chunk_size", 128),
            rmsnorm=_as_bool(model_config.get("rmsnorm", True)),
            drop_path_rate=model_config.get("drop_path_rate", 0.05),
            drop_out=model_config.get("drop_out", 0.0),
            grid_size=model_config.get("grid_size", 0.02),
            knn_backend=model_config.get("knn_backend", "auto"),
        )
        pretrained_from = model_config.get("pretrained_from")
        if pretrained_from:
            strict = _as_bool(model_config.get("pretrained_strict", False))
            logger.info(
                "Loading Mamba3DMamba2 pretrained weights from %s (strict=%s)",
                pretrained_from,
                strict,
            )
            feature_extractor.load_pretrained_weights(
                str(pretrained_from),
                strict=strict,
                verbose=True,
            )
        return feature_extractor

    if model_type == "mamba3d_true":
        logger.info("Creating Mamba3DTrue feature extractor.")
        feature_extractor = mamba3d_true.Mamba3DTrue_features(
            dim_k=model_config["dim_k"],
            sym_fn=model_config.get("sym_fn", model_config.get("symfn", "max")),
            num_groups=model_config.get("num_groups", 64),
            group_size=model_config.get("group_size", 32),
            trans_dim=model_config.get("trans_dim", 384),
            depth=model_config.get("depth", 6),
            d_state=model_config.get("d_state", 16),
            d_conv=model_config.get("d_conv", 4),
            expand=model_config.get("expand", 2),
            drop_path_rate=model_config.get("drop_path_rate", 0.1),
            drop_out=model_config.get("drop_out", 0.0),
            rms_norm=_as_bool(model_config.get("rms_norm", False)),
            grid_size=model_config.get("grid_size", 0.02),
            knn_backend=model_config.get("knn_backend", "auto"),
        )
        pretrained_from = model_config.get("pretrained_from")
        if pretrained_from:
            strict = _as_bool(model_config.get("pretrained_strict", False))
            logger.info(
                "Loading Mamba3DTrue pretrained weights from %s (strict=%s)",
                pretrained_from,
                strict,
            )
            feature_extractor.load_pretrained_weights(
                str(pretrained_from),
                strict=strict,
                verbose=True,
            )
        return feature_extractor

    if model_type in {"mamba3d", "mamba3d_v1"}:
        logger.info("Creating legacy mamba3d_v1 feature extractor.")
        return mamba3d_v1.Mamba3D_features(
            dim_k=model_config["dim_k"],
            num_mamba_blocks=model_config.get("num_mamba_blocks", 1),
            d_state=model_config.get("d_state", 8),
            expand=model_config.get("expand", 1.5),
        )

    raise ValueError(
        f"Unsupported Mamba3D model_type={model_type!r}. "
        "Expected 'mamba3d_v1', 'mamba3d_true', 'mamba3d_mamba2', "
        "'mamba3d_mamba2_direct', or 'mambanetlk'."
    )


def _create_optimizer(parameters, training_config):
    optimizer_name = str(training_config.get("optimizer", "Adam")).lower()
    lr = float(training_config["lr"])
    weight_decay = float(training_config.get("weight_decay", 0.0))
    if optimizer_name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(
        f"Unsupported optimizer={training_config.get('optimizer')!r}. "
        "Expected Adam or AdamW."
    )


def _create_pointlk_model(ptnet, model_config):
    if getattr(ptnet, "is_mambanetlk", False):
        logger.info("Using MambaNetLK wrapper with internal LK refinement.")
        return ptnet

    lk_mode = _lk_mode(model_config)
    common_kwargs = {
        "ptnet": ptnet,
        "delta": model_config["delta"],
        "learn_delta": _as_bool(model_config.get("learn_delta", False)),
    }
    if lk_mode == "exact":
        logger.info("Creating exact PointLK wrapper.")
        return pointlk.PointLK(**common_kwargs)
    if not hasattr(ptnet, "prepare_token_cache") or not hasattr(
        ptnet,
        "forward_from_cache",
    ):
        raise ValueError(
            f"lk_mode={lk_mode!r} requires a feature extractor with token cache "
            "APIs. Use lk_mode='exact' for feature extractors without cache "
            "support."
        )

    order_mode = _CACHED_LK_ORDER_MODES[lk_mode]
    logger.info(
        "Creating cached approximate PointLK wrapper "
        "(lk_mode=%s, order_mode=%s, lm_damping=%s, dx_clip_norm=%s).",
        lk_mode,
        order_mode,
        model_config.get("lm_damping", 1.0e-3),
        model_config.get("dx_clip_norm"),
    )
    return pointlk_cached.ApproxCachedPointLK(
        **common_kwargs,
        order_mode=order_mode,
        lm_damping=float(model_config.get("lm_damping", 1.0e-3)),
        dx_clip_norm=_optional_float(model_config.get("dx_clip_norm")),
    )


def _compute_grad_norm(parameters):
    grads = [
        parameter.grad.detach()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not grads:
        return torch.tensor(0.0)
    device = grads[0].device
    norms = torch.stack([torch.norm(grad, p=2).to(device) for grad in grads])
    return torch.norm(norms, p=2)


def _loss_is_usable(loss, max_loss=None):
    if loss is None or not torch.isfinite(loss):
        return False
    if max_loss is not None and float(loss.detach().item()) > float(max_loss):
        return False
    return True


def _pointlk_forward(model, template, source, model_config):
    wrapper_cls = (
        pointlk_cached.ApproxCachedPointLK
        if isinstance(model, pointlk_cached.ApproxCachedPointLK)
        else pointlk.PointLK
    )
    return wrapper_cls.do_forward(
        model,
        template,
        source,
        maxiter=int(model_config["max_iter"]),
        xtol=float(model_config.get("xtol", 1.0e-7)),
        p0_zero_mean=_as_bool(model_config.get("p0_zero_mean", True)),
        p1_zero_mean=_as_bool(model_config.get("p1_zero_mean", True)),
    )


def _rotation_geodesic_loss(rotation_error):
    trace = (
        rotation_error[:, 0, 0]
        + rotation_error[:, 1, 1]
        + rotation_error[:, 2, 2]
    )
    cos_theta = torch.clamp((trace - 1.0) * 0.5, -1.0 + 1.0e-6, 1.0 - 1.0e-6)
    theta = torch.acos(cos_theta)
    return torch.mean(theta * theta)


def _compute_registration_loss(model, residual, igt, training_config):
    loss_mode = str(training_config.get("loss_mode", "composition_mse"))
    iters = int(getattr(model, "itr", -1))
    metrics = {
        "loss_mode": loss_mode,
        "iters": iters,
        "loss_r": None,
        "loss_g": None,
        "loss_rot": None,
        "loss_t": None,
        "loss_init": None,
        "loss_init_rot": None,
        "loss_init_t": None,
    }
    if residual is None or getattr(model, "g", None) is None or iters < 0:
        return None, metrics

    composition = torch.matmul(model.g, igt)

    if loss_mode == "pointnetlk":
        wrapper_cls = (
            pointlk_cached.ApproxCachedPointLK
            if isinstance(model, pointlk_cached.ApproxCachedPointLK)
            else pointlk.PointLK
        )
        loss_r = wrapper_cls.rsq(residual)
        loss_g = wrapper_cls.comp(model.g, igt)
        metrics["loss_r"] = float(loss_r.detach().cpu().item())
        metrics["loss_g"] = float(loss_g.detach().cpu().item())
        return loss_r + loss_g, metrics

    if loss_mode == "composition_mse":
        identity = torch.eye(4, device=igt.device).unsqueeze(0).expand_as(composition)
        scale = float(training_config.get("composition_loss_scale", 4.0))
        loss = (
            torch.nn.functional.mse_loss(
                composition,
                identity,
                reduction="mean",
            )
            * scale
        )
        metrics["loss_g"] = float(loss.detach().cpu().item())
        return loss, metrics

    if loss_mode == "pose_geodesic":
        rot_weight = float(training_config.get("pose_rotation_weight", 16.0))
        trans_weight = float(training_config.get("pose_translation_weight", 1.0))
        residual_weight = float(training_config.get("feature_residual_weight", 0.0))
        init_pose_weight = float(training_config.get("init_pose_weight", 0.0))
        init_rot_weight = float(training_config.get("init_rotation_weight", rot_weight))
        init_trans_weight = float(
            training_config.get("init_translation_weight", trans_weight)
        )

        loss_rot = _rotation_geodesic_loss(composition[:, :3, :3])
        loss_t = torch.mean(torch.sum(composition[:, :3, 3] ** 2, dim=1))
        loss = rot_weight * loss_rot + trans_weight * loss_t

        if residual_weight > 0.0:
            wrapper_cls = (
                pointlk_cached.ApproxCachedPointLK
                if isinstance(model, pointlk_cached.ApproxCachedPointLK)
                else pointlk.PointLK
            )
            loss_r = wrapper_cls.rsq(residual) * residual_weight
            metrics["loss_r"] = float(loss_r.detach().cpu().item())
            loss = loss + loss_r

        initial_g = getattr(model, "initial_g", None)
        if init_pose_weight > 0.0 and initial_g is not None:
            init_composition = torch.matmul(initial_g, igt)
            loss_init_rot = _rotation_geodesic_loss(init_composition[:, :3, :3])
            loss_init_t = torch.mean(
                torch.sum(init_composition[:, :3, 3] ** 2, dim=1)
            )
            loss_init = (
                init_rot_weight * loss_init_rot + init_trans_weight * loss_init_t
            )
            loss = loss + init_pose_weight * loss_init
            metrics["loss_init"] = float(loss_init.detach().cpu().item())
            metrics["loss_init_rot"] = float(loss_init_rot.detach().cpu().item())
            metrics["loss_init_t"] = float(loss_init_t.detach().cpu().item())

        metrics["loss_g"] = float(loss.detach().cpu().item())
        metrics["loss_rot"] = float(loss_rot.detach().cpu().item())
        metrics["loss_t"] = float(loss_t.detach().cpu().item())
        return loss, metrics

    raise ValueError(
        f"Unsupported Mamba3D training loss_mode={loss_mode!r}. "
        "Expected 'composition_mse', 'pointnetlk', or 'pose_geodesic'."
    )


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


class C3VDMamba3DDataset(torch.utils.data.Dataset):
    """
    C3VD dataset for Mamba3D training with unified preprocessing.

    Preprocessing follows common/C3VD_PREPROCESSING_UNIFIED.md:
    1. Clean point clouds (remove NaN/Inf)
    2. VoxelGrid downsampling
    3. Baseline-aware model-private normalization
    """

    def __init__(
        self,
        data_root,
        num_points=8192,
        split="train",
        train_ratio=0.7,
        random_seed=42,
        rot_factor=1.0,
        trans_mag=0.8,
        sampling_mode="voxel",
        normalize_mode="unit_cube",
        perturbation_enabled=False,
        rotation_deg=0.0,
        translation_m=0.0,
        noise_sigma=0.0,
        noise_clip=0.0,
        apply_noise_to="source",
        train_scenes=None,
        val_scenes=None,
        test_scenes=None,
        frame_stride=1,
        max_pairs=None,
    ):
        """
        Args:
            data_root: Root directory of C3VD dataset
            num_points: Number of points to sample
            split: 'train' or 'test'
            train_ratio: Ratio of training data
            random_seed: Random seed for reproducibility
            rot_factor: Rotation magnitude factor (in radians)
            trans_mag: Translation magnitude
        """
        self.data_root = data_root
        self.num_points = num_points
        self.split = split
        self.random_seed = random_seed
        self.rot_factor = rot_factor
        self.trans_mag = trans_mag
        self.sampling_mode = sampling_mode
        self.normalize_mode = normalize_mode
        self.perturbation_enabled = bool(perturbation_enabled)
        self.rotation_deg = float(rotation_deg)
        self.translation_m = float(translation_m)
        self.noise_sigma = float(noise_sigma)
        self.noise_clip = float(noise_clip)
        self.apply_noise_to = apply_noise_to
        self.frame_stride = max(int(frame_stride), 1)
        self.max_pairs = None if max_pairs is None else max(int(max_pairs), 0)

        # Load point cloud pairs
        self.source_files = []
        self.target_files = []

        source_dir = os.path.join(data_root, "C3VD_ply_source")
        target_dir = os.path.join(data_root, "visible_point_cloud_ply_depth")

        # Collect all scenes
        scenes = sorted(
            [
                d
                for d in os.listdir(source_dir)
                if os.path.isdir(os.path.join(source_dir, d))
            ]
        )

        # Scene-based split
        if train_scenes is not None and (
            val_scenes is not None or test_scenes is not None
        ):
            if split == "train":
                selected_scenes = sorted(train_scenes)
            elif split == "val":
                selected_scenes = sorted(
                    val_scenes if val_scenes is not None else test_scenes
                )
            else:
                selected_scenes = sorted(
                    test_scenes if test_scenes is not None else val_scenes
                )
        else:
            np.random.seed(random_seed)
            np.random.shuffle(scenes)
            n_train = int(len(scenes) * train_ratio)

            if split == "train":
                selected_scenes = scenes[:n_train]
            else:
                selected_scenes = scenes[n_train:]

        # Collect file pairs
        for scene in selected_scenes:
            source_scene_dir = os.path.join(source_dir, scene)
            target_scene_dir = os.path.join(target_dir, scene)

            source_files = sorted(
                [f for f in os.listdir(source_scene_dir) if f.endswith(".ply")]
            )

            for source_file in source_files:
                # Match source and target files
                frame_num = source_file.split("_")[0]
                if int(frame_num) % self.frame_stride != 0:
                    continue
                target_file = f"frame_{frame_num}_visible.ply"

                source_path = os.path.join(source_scene_dir, source_file)
                target_path = os.path.join(target_scene_dir, target_file)

                if os.path.exists(target_path):
                    self.source_files.append(source_path)
                    self.target_files.append(target_path)
                    if (
                        self.max_pairs is not None
                        and len(self.source_files) >= self.max_pairs
                    ):
                        break
            if (
                self.max_pairs is not None
                and len(self.source_files) >= self.max_pairs
            ):
                break

        logger.info(f"Loaded {len(self.source_files)} pairs for {split} set")
        logger.info(f"Sampling mode: {self.sampling_mode}")
        logger.info(f"Normalize mode: {self.normalize_mode}")

    def __len__(self):
        return len(self.source_files)

    def __getitem__(self, idx):
        """
        Load and preprocess a point cloud pair.

        Returns:
            template: [N, 3] target point cloud (normalized)
            transformed_source: [N, 3] transformed source (normalized)
            igt: [4, 4] ground truth transformation matrix
        """
        import open3d as o3d

        # Load point clouds
        source_pcd = o3d.io.read_point_cloud(self.source_files[idx])
        target_pcd = o3d.io.read_point_cloud(self.target_files[idx])

        source = np.asarray(source_pcd.points)
        target = np.asarray(target_pcd.points)

        # Step 1: Clean point clouds (remove NaN/Inf)
        source = clean_point_cloud(source, min_points=100)
        target = clean_point_cloud(target, min_points=100)

        sample_seed = self.random_seed + idx * 9973
        source = sample_point_cloud(
            source,
            sampling=self.sampling_mode,
            num_points=self.num_points,
            seed=sample_seed,
        )
        target = sample_point_cloud(
            target,
            sampling=self.sampling_mode,
            num_points=self.num_points,
            seed=sample_seed + 1,
        )

        if self.perturbation_enabled:
            perturb_seed = None if self.split == "train" else sample_seed + 2
            (
                transformed_source_raw,
                target_raw,
                _,
                perturb_meta,
            ) = apply_pair_perturbation(
                source,
                target,
                rotation_deg=self.rotation_deg,
                translation_m=self.translation_m,
                noise_sigma=self.noise_sigma,
                noise_clip=self.noise_clip,
                apply_noise_to=self.apply_noise_to,
                seed=perturb_seed,
            )
            transformed_source, target, _, _, target_norm_transform = (
                normalize_point_cloud_pair(
                    transformed_source_raw,
                    target_raw,
                    self.normalize_mode,
                )
            )

            # The benchmark pipeline applies perturbation in raw C3VD units and
            # adapters normalize afterward. Train the same task, but conjugate
            # the raw rigid transform into the normalized frame used by PointLK.
            rigid_transform = np.asarray(
                perturb_meta["rigid_transform"],
                dtype=np.float64,
            )
            norm_transform = np.asarray(target_norm_transform, dtype=np.float64)
            igt_np = norm_transform @ rigid_transform @ np.linalg.inv(norm_transform)
            igt = torch.from_numpy(igt_np.astype(np.float32, copy=False)).float()
        else:
            source, target, _, _, _ = normalize_point_cloud_pair(
                source,
                target,
                self.normalize_mode,
            )
            # Convert to tensors
            source_tensor = torch.from_numpy(source).float()

            # Generate random transformation
            anglex = np.random.uniform() * np.pi * self.rot_factor
            angley = np.random.uniform() * np.pi * self.rot_factor
            anglez = np.random.uniform() * np.pi * self.rot_factor
            cosx = np.cos(anglex)
            cosy = np.cos(angley)
            cosz = np.cos(anglez)
            sinx = np.sin(anglex)
            siny = np.sin(angley)
            sinz = np.sin(anglez)

            Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
            Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
            Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])
            R_ab = Rx @ Ry @ Rz
            t_ab = np.random.uniform(-self.trans_mag, self.trans_mag, 3)

            igt_np = np.eye(4, dtype=np.float32)
            igt_np[:3, :3] = R_ab
            igt_np[:3, 3] = t_ab
            igt = torch.from_numpy(igt_np).float()
            transformed_source = (
                source_tensor @ torch.from_numpy(R_ab).float().T
                + torch.from_numpy(t_ab).float()
            ).numpy()

        # Return: template (target), transformed source, ground truth transform
        return (
            torch.from_numpy(target).float(),
            torch.from_numpy(transformed_source).float(),
            igt,
        )


def train_mamba3d(config, args):
    """Main training function for Mamba3D."""

    # Setup output directory
    output_dir = Path(config["output"]["checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(config["output"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup file logging
    log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Log file: {log_file}")

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = C3VDMamba3DDataset(
        data_root=args.data_root,
        num_points=config["dataset"]["num_points"],
        split="train",
        train_ratio=config["dataset"]["train_ratio"],
        random_seed=config["dataset"]["random_seed"],
        rot_factor=config["training"]["rot_factor"],
        trans_mag=config["training"]["trans_mag"],
        sampling_mode=config["dataset"].get("sampling_mode", "voxel"),
        normalize_mode=config["dataset"].get("normalize_mode", "unit_cube"),
        perturbation_enabled=config["dataset"].get("perturbation_enabled", False),
        rotation_deg=config["dataset"].get("rotation_deg", 0.0),
        translation_m=config["dataset"].get("translation_m", 0.0),
        noise_sigma=config["dataset"].get("noise_sigma", 0.0),
        noise_clip=config["dataset"].get("noise_clip", 0.0),
        apply_noise_to=config["dataset"].get("apply_noise_to", "source"),
        train_scenes=config["dataset"].get("train_scenes"),
        val_scenes=config["dataset"].get("val_scenes"),
        test_scenes=config["dataset"].get("test_scenes"),
        frame_stride=config["dataset"].get("frame_stride", 1),
        max_pairs=config["dataset"].get("max_train_pairs"),
    )

    test_dataset = C3VDMamba3DDataset(
        data_root=args.data_root,
        num_points=config["dataset"]["num_points"],
        split="val",
        train_ratio=config["dataset"]["train_ratio"],
        random_seed=config["dataset"]["random_seed"],
        rot_factor=config["training"]["rot_factor"],
        trans_mag=config["training"]["trans_mag"],
        sampling_mode=config["dataset"].get("sampling_mode", "voxel"),
        normalize_mode=config["dataset"].get("normalize_mode", "unit_cube"),
        perturbation_enabled=config["dataset"].get("perturbation_enabled", False),
        rotation_deg=config["dataset"].get("rotation_deg", 0.0),
        translation_m=config["dataset"].get("translation_m", 0.0),
        noise_sigma=config["dataset"].get("noise_sigma", 0.0),
        noise_clip=config["dataset"].get("noise_clip", 0.0),
        apply_noise_to=config["dataset"].get("apply_noise_to", "source"),
        train_scenes=config["dataset"].get("train_scenes"),
        val_scenes=config["dataset"].get("val_scenes"),
        test_scenes=config["dataset"].get("test_scenes"),
        frame_stride=config["dataset"].get("frame_stride", 1),
        max_pairs=config["dataset"].get("max_val_pairs")
        or config["dataset"].get("max_test_pairs"),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(test_dataset)} samples")
    if len(train_loader) == 0:
        raise ValueError(
            "Mamba3D train loader produced zero batches. "
            "Increase dataset.max_train_pairs or reduce training.batch_size; "
            "drop_last=True requires at least one full training batch."
        )

    # Create model
    logger.info("Creating Mamba3D PointLK model...")
    device = torch.device(config["device"])
    ptnet = _create_feature_extractor(config["model"])
    model = _create_pointlk_model(ptnet, config["model"])

    # Load checkpoint if resuming
    start_epoch = 0
    best_loss = float("inf")

    resume_path = args.resume or config["training"].get("resume_from", "")
    if resume_path:
        logger.info(f"Loading checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(_extract_state_dict(checkpoint), strict=False)
        logger.info("✓ Loaded checkpoint")

    model.to(device)

    # Setup optimizer
    if config["training"]["pointnet_tune"]:
        params = list(model.parameters())
    else:
        params = [p for n, p in model.named_parameters() if "ptnet" not in n]

    optimizer = _create_optimizer(params, config["training"])

    # Add learning rate scheduler to prevent gradient explosion
    # ReduceLROnPlateau: reduce lr when validation loss plateaus or increases
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",  # minimize validation loss
        factor=float(config["training"].get("scheduler_factor", 0.5)),
        patience=int(config["training"].get("scheduler_patience", 3)),
        threshold=float(config["training"].get("scheduler_threshold", 1.0e-4)),
        threshold_mode=str(config["training"].get("scheduler_threshold_mode", "rel")),
        cooldown=int(config["training"].get("scheduler_cooldown", 0)),
        min_lr=float(config["training"].get("min_lr", 1e-6)),
    )
    logger.info("✓ Using ReduceLROnPlateau scheduler")
    max_train_steps = config["training"].get("max_train_steps")
    max_test_steps = config["training"].get("max_test_steps")
    grad_clip_norm = config["training"].get("grad_clip_norm")
    skip_nonfinite_loss = _as_bool(
        config["training"].get("skip_nonfinite_loss", False)
    )
    abort_after_nonfinite_steps = int(
        config["training"].get("abort_after_nonfinite_steps", 0)
    )
    log_grad_norm = _as_bool(config["training"].get("log_grad_norm", False))
    max_loss = config["training"].get("max_loss")
    eval_max_loss = config["training"].get("eval_max_loss")
    max_skipped_batches = int(config["training"].get("max_skipped_batches", 100))
    consecutive_nonfinite_steps = 0
    if max_loss is not None:
        logger.info("Using train max usable loss guard: %s", max_loss)
    if eval_max_loss is not None:
        logger.info("Using eval max usable loss guard: %s", eval_max_loss)

    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, config["training"]["epochs"]):
        model.train()
        train_loss = 0.0
        train_steps = 0
        skipped_batches = 0

        for batch_idx, (template, source, igt) in enumerate(train_loader):
            if max_train_steps is not None and batch_idx >= int(max_train_steps):
                break
            template = template.to(device)
            source = source.to(device)
            igt = igt.to(device)

            optimizer.zero_grad(set_to_none=True)

            # Forward pass with PointNetLK
            residual = _pointlk_forward(model, template, source, config["model"])
            loss, loss_metrics = _compute_registration_loss(
                model,
                residual,
                igt,
                config["training"],
            )

            if not _loss_is_usable(loss, max_loss):
                skipped_batches += 1
                if loss is None or not torch.isfinite(loss):
                    consecutive_nonfinite_steps += 1
                    loss_repr = "None" if loss is None else loss.detach().cpu().item()
                else:
                    loss_repr = loss.detach().cpu().item()
                logger.warning(
                    "Skipping unstable Mamba3D batch at epoch=%d batch=%d "
                    "loss=%s loss_r=%s loss_g=%s loss_init=%s iters=%s",
                    epoch + 1,
                    batch_idx,
                    loss_repr,
                    loss_metrics["loss_r"],
                    loss_metrics["loss_g"],
                    loss_metrics["loss_init"],
                    loss_metrics["iters"],
                )
                if skipped_batches > max_skipped_batches:
                    raise RuntimeError(
                        "Mamba3D training exceeded "
                        f"max_skipped_batches={max_skipped_batches}."
                    )
                if (
                    abort_after_nonfinite_steps > 0
                    and consecutive_nonfinite_steps >= abort_after_nonfinite_steps
                ):
                    raise RuntimeError(
                        "Aborting Mamba3D training after "
                        f"{consecutive_nonfinite_steps} consecutive non-finite steps."
                    )
                if skip_nonfinite_loss or (loss is not None and torch.isfinite(loss)):
                    continue
                raise RuntimeError("Non-finite Mamba3D loss encountered.")

            loss.backward()
            if grad_clip_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    params,
                    max_norm=float(grad_clip_norm),
                    error_if_nonfinite=False,
                )
            else:
                grad_norm = _compute_grad_norm(params)

            if not torch.isfinite(grad_norm):
                consecutive_nonfinite_steps += 1
                logger.warning(
                    "Skipping non-finite grad norm at epoch=%d batch=%d grad_norm=%s",
                    epoch + 1,
                    batch_idx,
                    grad_norm.detach().cpu().item(),
                )
                optimizer.zero_grad(set_to_none=True)
                if (
                    abort_after_nonfinite_steps > 0
                    and consecutive_nonfinite_steps >= abort_after_nonfinite_steps
                ):
                    raise RuntimeError(
                        "Aborting Mamba3D training after "
                        f"{consecutive_nonfinite_steps} consecutive non-finite steps."
                    )
                if skip_nonfinite_loss:
                    continue
                raise RuntimeError("Non-finite Mamba3D gradient norm encountered.")

            optimizer.step()
            consecutive_nonfinite_steps = 0

            train_loss += loss.item()
            train_steps += 1

            if batch_idx % 10 == 0:
                message = (
                    f"Epoch [{epoch + 1}/{config['training']['epochs']}] "
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.6f}"
                )
                if loss_metrics["loss_r"] is not None:
                    message += f" LossR: {loss_metrics['loss_r']:.6f}"
                if loss_metrics["loss_g"] is not None:
                    message += f" LossG: {loss_metrics['loss_g']:.6f}"
                if loss_metrics["loss_rot"] is not None:
                    message += f" LossRot: {loss_metrics['loss_rot']:.6f}"
                if loss_metrics["loss_t"] is not None:
                    message += f" LossT: {loss_metrics['loss_t']:.6f}"
                if loss_metrics["loss_init"] is not None:
                    message += f" LossInit: {loss_metrics['loss_init']:.6f}"
                message += f" Iters: {loss_metrics['iters']}"
                if log_grad_norm:
                    message += f" GradNorm: {float(grad_norm):.6f}"
                logger.info(message)

        train_loss = train_loss / train_steps if train_steps else float("inf")
        if skipped_batches:
            logger.info(f"Skipped unstable Mamba3D train batches: {skipped_batches}")

        # Validation
        model.eval()
        test_loss = 0.0
        test_steps = 0

        with torch.no_grad():
            for batch_idx, (template, source, igt) in enumerate(test_loader):
                if max_test_steps is not None and batch_idx >= int(max_test_steps):
                    break
                template = template.to(device)
                source = source.to(device)
                igt = igt.to(device)

                residual = _pointlk_forward(model, template, source, config["model"])
                loss, loss_metrics = _compute_registration_loss(
                    model,
                    residual,
                    igt,
                    config["training"],
                )
                if not _loss_is_usable(loss, eval_max_loss):
                    logger.warning(
                        "Skipping unstable Mamba3D eval batch=%d loss=%s "
                        "loss_r=%s loss_g=%s loss_rot=%s loss_t=%s "
                        "loss_init=%s iters=%s",
                        batch_idx,
                        "None" if loss is None else loss.detach().cpu().item(),
                        loss_metrics["loss_r"],
                        loss_metrics["loss_g"],
                        loss_metrics["loss_rot"],
                        loss_metrics["loss_t"],
                        loss_metrics["loss_init"],
                        loss_metrics["iters"],
                    )
                    continue

                test_loss += loss.item()
                test_steps += 1

        test_loss = test_loss / test_steps if test_steps else float("inf")

        logger.info(
            f"Epoch [{epoch + 1}/{config['training']['epochs']}] "
            f"Train Loss: {train_loss:.6f} Test Loss: {test_loss:.6f}"
        )

        # Update learning rate based on validation loss
        if np.isfinite(test_loss):
            scheduler.step(test_loss)
        else:
            logger.warning("Skipping scheduler step because test loss is non-finite.")

        # Save checkpoint
        if test_loss < best_loss:
            best_loss = test_loss
            save_path = output_dir / "mamba3d_pointlk_model_best.pth"
            torch.save(model.state_dict(), save_path)
            logger.info(f"✓ Saved best model (test loss: {test_loss:.6f})")

        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:
            save_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), save_path)
            logger.info("✓ Saved checkpoint")

    logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train Mamba3D on C3VD dataset")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--data-root", required=True, help="Path to C3VD dataset")
    parser.add_argument(
        "--resume", default="", help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Train
    train_mamba3d(config, args)


if __name__ == "__main__":
    main()
