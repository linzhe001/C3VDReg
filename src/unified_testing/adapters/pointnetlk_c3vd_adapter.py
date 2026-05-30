"""
PointNetLK_c3vd algorithm adapter

Supports multiple backbones: PointNet, Attention, Mamba3D (v1-v4), etc.
"""

import os
import sys

import numpy as np
import torch

from ..core.base_adapter import BaseAdapter


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


def _purge_conflicting_ptlk_modules(expected_root: str) -> None:
    """Ensure PointNetLK_c3vd imports do not reuse baseline PointNetLK modules."""
    for module_name, module in list(sys.modules.items()):
        if module_name != "ptlk" and not module_name.startswith("ptlk."):
            continue
        origin = _module_origin(module)
        if origin and not origin.startswith(expected_root):
            sys.modules.pop(module_name, None)


# add PointNetLK_c3vd fork path
POINTNETLK_C3VD_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "forks",
        "PointNetLK_c3vd",
    )
)
if POINTNETLK_C3VD_PATH in sys.path:
    sys.path.remove(POINTNETLK_C3VD_PATH)
sys.path.insert(0, POINTNETLK_C3VD_PATH)
_purge_conflicting_ptlk_modules(POINTNETLK_C3VD_PATH)

try:
    import ptlk.pointlk as pointlk
    import ptlk.pointlk_cached as pointlk_cached
    from ptlk import (
        attention_v1,
        cformer,
        fast_point_attention,
        mamba3d_mamba2,
        mamba3d_mamba2_direct,
        mamba3d_true,
        mamba3d_v1,
        mamba3d_v2,
        mamba3d_v3,
        mamba3d_v4,
        mambanetlk,
        pointnet,
    )
except ImportError as e:
    print(f"Error importing PointNetLK_c3vd modules: {e}")
    print(f"Make sure PointNetLK_c3vd is available at: {POINTNETLK_C3VD_PATH}")
    raise


# Mapping of model types to their classes
MODEL_CLASSES = {
    "pointnet": pointnet.PointNet_features,
    "attention": attention_v1.AttentionNet_features,
    "mamba3d_v1": mamba3d_v1.Mamba3D_features,
    "mamba3d_v2": mamba3d_v2.Mamba3D_features,
    "mamba3d_v3": mamba3d_v3.Mamba3D_features,
    "mamba3d_v4": mamba3d_v4.Mamba3D_features,
    "mamba3d_true": mamba3d_true.Mamba3DTrue_features,
    "mamba3d_mamba2": mamba3d_mamba2.Mamba3DMamba2_features,
    "mamba3d_mamba2_direct": mamba3d_mamba2_direct.Mamba3DMamba2Direct_features,
    "mambanetlk": mambanetlk.MambaNetLK,
    "fast_attention": fast_point_attention.FastPointAttention_features,
    "cformer": cformer.CFormer_features,
}


def _extract_state_dict(payload):
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    return payload


def _is_pointlk_state_dict(state_dict):
    if not hasattr(state_dict, "keys"):
        return False
    return any(key == "dt" or key.startswith("ptnet.") for key in state_dict)


_LK_MODE_TO_ORDER_MODE = {
    "cached_recompute_order": "recompute",
    "cached_reuse_order": "reuse",
}
_LK_MODES = {"exact", *_LK_MODE_TO_ORDER_MODE}


def _optional_float(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


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


class PointNetLKC3VDAdapter(BaseAdapter):
    """PointNetLK_c3vd adapter with multiple backbone support"""

    def __init__(self, args):
        super().__init__(args)

        # get backbone type
        self.backbone_type = getattr(args, "model_type", "pointnet")

        # get PointLK parameters
        self.max_iter = getattr(args, "max_iter", 20)
        self.delta = getattr(args, "delta", 1.0e-2)
        self.learn_delta = _as_bool(getattr(args, "learn_delta", False))
        self.xtol = float(getattr(args, "xtol", 1.0e-7))
        self.p0_zero_mean = _as_bool(getattr(args, "p0_zero_mean", True))
        self.p1_zero_mean = _as_bool(getattr(args, "p1_zero_mean", True))
        self.dim_k = getattr(args, "dim_k", 1024)
        self.use_tnet = getattr(args, "use_tnet", True)
        self.lk_mode = getattr(args, "lk_mode", "exact")
        if self.lk_mode not in _LK_MODES:
            raise ValueError(
                f"Unknown lk_mode: {self.lk_mode}. "
                f"Supported modes: {sorted(_LK_MODES)}"
            )
        self.lm_damping = float(getattr(args, "lm_damping", 1.0e-3))
        self.dx_clip_norm = _optional_float(getattr(args, "dx_clip_norm", None))

        # Mamba3D-specific parameters
        is_mamba2 = self.backbone_type == "mamba3d_mamba2"
        is_mamba2_direct = self.backbone_type == "mamba3d_mamba2_direct"
        is_mambanetlk = self.backbone_type == "mambanetlk"
        self.num_mamba_blocks = getattr(args, "num_mamba_blocks", 3)
        self.d_model = getattr(args, "d_model", 128)
        self.d_state = getattr(
            args,
            "d_state",
            16 if is_mamba2_direct else 64 if (is_mamba2 or is_mambanetlk) else 16,
        )
        self.d_conv = getattr(args, "d_conv", 4)
        self.expand = getattr(args, "expand", 1 if is_mamba2_direct else 2.0)
        self.headdim = getattr(args, "headdim", 64)
        self.ngroups = getattr(args, "ngroups", 1)
        self.chunk_size = getattr(args, "chunk_size", 128)
        self.sym_fn = getattr(args, "sym_fn", getattr(args, "symfn", "max"))
        self.num_groups = getattr(args, "num_groups", 64)
        self.group_size = getattr(args, "group_size", 32)
        self.trans_dim = getattr(args, "trans_dim", 384)
        self.depth = getattr(
            args,
            "depth",
            2 if is_mamba2_direct else 4 if (is_mamba2 or is_mambanetlk) else 6,
        )
        self.drop_path_rate = getattr(
            args,
            "drop_path_rate",
            0.0
            if is_mamba2_direct
            else 0.05
            if (is_mamba2 or is_mambanetlk)
            else 0.1,
        )
        self.drop_out = getattr(args, "drop_out", 0.0)
        self.rms_norm = getattr(args, "rms_norm", False)
        self.rmsnorm = getattr(
            args,
            "rmsnorm",
            True
            if (is_mamba2 or is_mamba2_direct or is_mambanetlk)
            else self.rms_norm,
        )
        self.grid_size = getattr(args, "grid_size", 0.02)
        self.knn_backend = getattr(args, "knn_backend", "auto")
        self.layer_scale_init = getattr(args, "layer_scale_init", 1.0e-4)
        self.stem_hidden_dim = getattr(args, "stem_hidden_dim", 64)
        self.pos_hidden_dim = getattr(args, "pos_hidden_dim", 64)
        self.ffn_ratio = getattr(args, "ffn_ratio", 2.0)
        self.mlp_hidden_dim = getattr(args, "mlp_hidden_dim", 256)
        self.mlp_norm = getattr(
            args,
            "mlp_norm",
            "layernorm" if is_mamba2_direct else "batchnorm",
        )
        self.pooling = getattr(args, "pooling", "maxavg" if is_mamba2_direct else None)
        self.descriptor_norm = getattr(args, "descriptor_norm", "layernorm")
        self.point_order = getattr(args, "point_order", "identity")
        self.pair_dim = getattr(args, "pair_dim", 256)
        self.pair_hidden_dim = getattr(args, "pair_hidden_dim", 512)
        self.pair_dropout = getattr(args, "pair_dropout", 0.0)
        self.pose_rotation_scale = getattr(args, "pose_rotation_scale", 1.5708)
        self.pose_translation_scale = getattr(args, "pose_translation_scale", 1.0)
        self.use_mem_eff_path = _as_bool(getattr(args, "use_mem_eff_path", True))

        # Normalization mode
        # - 'none': no normalization
        # - 'unit_cube': apply target/template unit-cube transform to both clouds
        # - 'joint': joint normalization (source and target together)
        self.normalize_mode = getattr(args, "normalize_mode", "unit_cube")

        print("PointNetLK_c3vd adapter initialized:")
        print(f"  Backbone: {self.backbone_type}")
        print(f"  Max iterations: {self.max_iter}")
        print(f"  Delta: {self.delta}")
        print(f"  Learn delta: {self.learn_delta}")
        print(f"  X tolerance: {self.xtol}")
        print(f"  Dim k: {self.dim_k}")
        print(f"  LK mode: {self.lk_mode}")
        print(f"  Zero-mean template/source: {self.p0_zero_mean}/{self.p1_zero_mean}")
        if self.lk_mode != "exact":
            print(f"  LM damping: {self.lm_damping}")
            print(f"  dx clip norm: {self.dx_clip_norm}")
        if (
            self.backbone_type.startswith("mamba3d")
            or self.backbone_type == "mambanetlk"
        ):
            print(f"  Mamba3D blocks: {self.num_mamba_blocks}")
            print(f"  Mamba3D d_state: {self.d_state}")
            print(f"  Mamba3D expand: {self.expand}")
            if self.backbone_type in {"mamba3d_true", "mamba3d_mamba2"}:
                print(f"  Mamba3D groups: {self.num_groups} x {self.group_size}")
                print(f"  Mamba3D trans_dim/depth: {self.trans_dim}/{self.depth}")
            if self.backbone_type == "mambanetlk":
                print(f"  MambaNetLK groups: {self.num_groups} x {self.group_size}")
                print(f"  MambaNetLK trans_dim/depth: {self.trans_dim}/{self.depth}")
                print(f"  MambaNetLK pair dim: {self.pair_dim}")
            if self.backbone_type == "mamba3d_mamba2_direct":
                print(f"  Mamba3D Direct d_model/depth: {self.d_model}/{self.depth}")
                print(f"  Mamba3D Direct MLP norm: {self.mlp_norm}")
                print(f"  Mamba3D Direct pooling: {self.pooling}")
                print(f"  Mamba3D Direct descriptor norm: {self.descriptor_norm}")
                print(f"  Mamba3D Direct point order: {self.point_order}")
            if self.backbone_type == "mamba3d_mamba2":
                print(
                    "  Mamba3D Mamba2 headdim/ngroups/chunk: "
                    f"{self.headdim}/{self.ngroups}/{self.chunk_size}"
                )
        print(f"  Normalization mode: {self.normalize_mode}")

        # Enable normalized-space perturbation for models trained with normalization
        self._supports_normalized_perturbation = self.normalize_mode in [
            "joint",
            "unit_cube",
        ]

    def supports_normalized_perturbation(self):
        """Check if this adapter supports applying perturbations in normalized space."""
        return self._supports_normalized_perturbation

    def load_model(self, model_path):
        """
        Load PointNetLK model

        Args:
            model_path: path to PointNetLK checkpoint (.pth file)
        """
        # select model class based on backbone type
        if self.backbone_type not in MODEL_CLASSES:
            raise ValueError(
                f"Unknown backbone type: {self.backbone_type}. "
                f"Supported types: {list(MODEL_CLASSES.keys())}"
            )

        model_class = MODEL_CLASSES[self.backbone_type]

        # create feature extractor
        # Note: only PointNet supports use_tnet parameter
        if self.backbone_type == "pointnet":
            self.model = model_class(dim_k=self.dim_k, use_tnet=self.use_tnet)
        elif self.backbone_type == "mamba3d_true":
            self.model = model_class(
                dim_k=self.dim_k,
                sym_fn=self.sym_fn,
                num_groups=self.num_groups,
                group_size=self.group_size,
                trans_dim=self.trans_dim,
                depth=self.depth,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                drop_path_rate=self.drop_path_rate,
                drop_out=self.drop_out,
                rms_norm=self.rms_norm,
                grid_size=self.grid_size,
                knn_backend=self.knn_backend,
            )
        elif self.backbone_type == "mamba3d_mamba2":
            self.model = model_class(
                dim_k=self.dim_k,
                sym_fn=self.sym_fn,
                num_groups=self.num_groups,
                group_size=self.group_size,
                trans_dim=self.trans_dim,
                depth=self.depth,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                headdim=self.headdim,
                ngroups=self.ngroups,
                chunk_size=self.chunk_size,
                rmsnorm=self.rmsnorm,
                drop_path_rate=self.drop_path_rate,
                drop_out=self.drop_out,
                grid_size=self.grid_size,
                knn_backend=self.knn_backend,
            )
        elif self.backbone_type == "mambanetlk":
            self.model = model_class(
                dim_k=self.dim_k,
                sym_fn=self.sym_fn,
                num_groups=self.num_groups,
                group_size=self.group_size,
                trans_dim=self.trans_dim,
                depth=self.depth,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                headdim=self.headdim,
                ngroups=self.ngroups,
                chunk_size=self.chunk_size,
                rmsnorm=self.rmsnorm,
                drop_path_rate=self.drop_path_rate,
                drop_out=self.drop_out,
                grid_size=self.grid_size,
                knn_backend=self.knn_backend,
                lk_mode=self.lk_mode,
                delta=self.delta,
                learn_delta=self.learn_delta,
                lm_damping=self.lm_damping,
                dx_clip_norm=self.dx_clip_norm,
                pair_dim=self.pair_dim,
                pair_hidden_dim=self.pair_hidden_dim,
                pair_dropout=self.pair_dropout,
                pose_rotation_scale=self.pose_rotation_scale,
                pose_translation_scale=self.pose_translation_scale,
                use_mem_eff_path=self.use_mem_eff_path,
            )
        elif self.backbone_type == "mamba3d_mamba2_direct":
            self.model = model_class(
                dim_k=self.dim_k,
                sym_fn=self.sym_fn,
                d_model=self.d_model,
                depth=self.depth,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                headdim=self.headdim,
                ngroups=self.ngroups,
                chunk_size=self.chunk_size,
                rmsnorm=self.rmsnorm,
                drop_path_rate=self.drop_path_rate,
                drop_out=self.drop_out,
                layer_scale_init=self.layer_scale_init,
                stem_hidden_dim=self.stem_hidden_dim,
                pos_hidden_dim=self.pos_hidden_dim,
                ffn_ratio=self.ffn_ratio,
                mlp_hidden_dim=self.mlp_hidden_dim,
                mlp_norm=self.mlp_norm,
                pooling=self.pooling,
                descriptor_norm=self.descriptor_norm,
                point_order=self.point_order,
            )
        elif self.backbone_type.startswith("mamba3d"):
            # Legacy Mamba3D models support the original compact parameter set.
            self.model = model_class(
                dim_k=self.dim_k,
                num_mamba_blocks=self.num_mamba_blocks,
                d_state=self.d_state,
                expand=self.expand,
            )
        else:
            # Other backbones don't have use_tnet parameter
            self.model = model_class(dim_k=self.dim_k)

        # load classifier weights (if specified for transfer learning)
        transfer_from = getattr(self.args, "transfer_from", None)
        if transfer_from and os.path.exists(transfer_from):
            print(f"Loading classifier features from: {transfer_from}")
            if self.backbone_type == "mambanetlk":
                self.model.load_pretrained_weights(transfer_from, strict=False)
            else:
                state_dict = torch.load(transfer_from, map_location="cpu")
                self.model.load_state_dict(
                    _extract_state_dict(state_dict),
                    strict=False,
                )
            print("✓ Loaded classifier features")

        checkpoint_state_dict = None
        if model_path and os.path.exists(model_path):
            print(f"Loading PointNetLK model from: {model_path}")
            state_dict = torch.load(model_path, map_location="cpu")
            checkpoint_state_dict = _extract_state_dict(state_dict)
            if not _is_pointlk_state_dict(checkpoint_state_dict):
                self.model.load_state_dict(checkpoint_state_dict, strict=False)
                checkpoint_state_dict = None
                print("✓ Loaded feature extractor checkpoint")
        else:
            print(f"Warning: Model path not found or not specified: {model_path}")
            print("Using randomly initialized weights (classifier transfer only)")

        # Create PointLK wrapper for iterative optimization
        if self.backbone_type == "mambanetlk":
            self.pointlk = self.model
            print("✓ Created MambaNetLK wrapper with internal LK refinement")
        elif self.lk_mode == "exact":
            self.pointlk = pointlk.PointLK(
                ptnet=self.model,
                delta=self.delta,
                learn_delta=self.learn_delta,
            )
            print(f"✓ Created PointLK wrapper (delta={self.delta})")
        else:
            order_mode = _LK_MODE_TO_ORDER_MODE[self.lk_mode]
            if not hasattr(self.model, "prepare_token_cache") or not hasattr(
                self.model,
                "forward_from_cache",
            ):
                raise ValueError(
                    f"lk_mode={self.lk_mode!r} requires a feature extractor with "
                    "token cache APIs. Use lk_mode='exact' for feature "
                    "extractors without cache support."
                )
            self.pointlk = pointlk_cached.ApproxCachedPointLK(
                ptnet=self.model,
                delta=self.delta,
                learn_delta=self.learn_delta,
                order_mode=order_mode,
                lm_damping=self.lm_damping,
                dx_clip_norm=self.dx_clip_norm,
            )
            print(
                "✓ Created ApproxCachedPointLK wrapper "
                f"(delta={self.delta}, order_mode={order_mode})"
            )
        if checkpoint_state_dict is not None:
            self.pointlk.load_state_dict(checkpoint_state_dict, strict=False)
            print("✓ Loaded PointLK checkpoint")

        # move to device and set to eval mode
        self.pointlk.to(self.device)
        self.pointlk.eval()

    def preprocess(self, source, target):
        """
        Preprocess point clouds for PointNetLK

        PointNetLK expects: [B, N, 3] format

        Args:
            source: [N, 3] numpy array (perturbed source)
            target: [N, 3] numpy array (target/template)

        Returns:
            source_tensor: [1, N, 3] torch.Tensor
            target_tensor: [1, N, 3] torch.Tensor
        """
        source_norm, target_norm, _ = self._normalize_pair(source, target)

        # convert to tensor and add batch dimension
        source_tensor = torch.from_numpy(source_norm).float().unsqueeze(0)  # [1, N, 3]
        target_tensor = torch.from_numpy(target_norm).float().unsqueeze(0)  # [1, N, 3]

        # move to device
        source_tensor = source_tensor.to(self.device)
        target_tensor = target_tensor.to(self.device)

        return source_tensor, target_tensor

    def preprocess_for_perturbation(self, source, target):
        """Preprocess point clouds for perturbation (normalize without perturbation)."""
        source_norm, target_norm, preprocess_info = self._normalize_pair(source, target)

        return source_norm, target_norm, preprocess_info

    def predict_after_perturbation(
        self, source_perturbed, target_norm, preprocess_info
    ):
        """Predict transformation after perturbation in normalized space."""
        self.set_eval_mode()

        # Convert to tensors
        source_tensor = torch.from_numpy(source_perturbed).float().unsqueeze(0)
        target_tensor = torch.from_numpy(target_norm).float().unsqueeze(0)
        source_tensor = source_tensor.to(self.device)
        target_tensor = target_tensor.to(self.device)

        # Forward pass
        output = self.forward(source_tensor, target_tensor)

        # Extract transformation
        g_pred = output.detach().cpu().numpy()[0]
        R = g_pred[:3, :3]
        t = g_pred[:3, 3]

        # IMPORTANT: Return translation in NORMALIZED space
        # Do NOT denormalize translation here. Error computation should be done
        # in the same space (normalized space) to ensure consistency.
        # The old code incorrectly denormalized t, causing mixed-unit error calculation.

        return R, t

    def forward(self, source_tensor, target_tensor):
        """
        Forward inference through PointNetLK

        Args:
            source_tensor: [1, N, 3] perturbed source
            target_tensor: [1, N, 3] template (target)

        Returns:
            g_pred: [1, 4, 4] predicted SE(3) transformation
        """
        with torch.no_grad():
            # PointNetLK iterative refinement
            # Note: PointLK expects (template, source) order
            # Use PointLK wrapper's do_forward (static method)
            if self.backbone_type == "mambanetlk":
                do_forward = pointlk.PointLK.do_forward
            else:
                do_forward = (
                    pointlk.PointLK.do_forward
                    if self.lk_mode == "exact"
                    else pointlk_cached.ApproxCachedPointLK.do_forward
                )
            _ = do_forward(
                self.pointlk,  # Use PointLK wrapper, not just feature extractor
                target_tensor,  # p0 (template)
                source_tensor,  # p1 (source)
                maxiter=self.max_iter,
                xtol=self.xtol,
                p0_zero_mean=self.p0_zero_mean,
                p1_zero_mean=self.p1_zero_mean,
            )

            # get predicted transformation (stored in pointlk.g)
            g_pred = self.pointlk.g  # [1, 4, 4]

        return g_pred

    def extract_transformation(self, output):
        """
        Extract rotation and translation from SE(3) matrix

        Args:
            output: [1, 4, 4] SE(3) matrix

        Returns:
            R: [3, 3] rotation matrix (numpy)
            t: [3,] translation vector (numpy)
        """
        g_pred = output.detach().cpu().numpy()[0]  # [4, 4]

        R = g_pred[:3, :3]
        t_norm = g_pred[:3, 3]

        transform_norm = np.eye(4, dtype=np.float64)
        transform_norm[:3, :3] = R
        transform_norm[:3, 3] = t_norm
        transform_raw = self._recover_raw_transform(transform_norm)
        t = transform_raw[:3, 3]
        R = transform_raw[:3, :3]

        return R, t

    def get_data_format(self):
        """Return data format"""
        return "BN3"  # [B, N, 3]

    def get_algorithm_name(self):
        """Return algorithm name with backbone"""
        return f"PointNetLK_c3vd_{self.backbone_type}"
