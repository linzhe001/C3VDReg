"""Real mamba_ssm based Mamba3D feature extractor for PointNetLK.

This module keeps the PointNetLK feature extractor contract:
input points are [B, N, 3] and the default output is [B, dim_k].
The sequence mixer is mamba_ssm.modules.mamba_simple.Mamba, while FPS/KNN,
Hilbert serialization, and pooling adapt point clouds into short token
sequences suitable for PointLK finite-difference iterations.
"""

from __future__ import annotations

import logging
import math
from typing import Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    from mamba_ssm.modules.mamba_simple import Mamba

    _MAMBA_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - exercised only without optional dep.
    Mamba = None  # type: ignore[assignment]
    _MAMBA_IMPORT_ERROR = exc

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm
except ImportError:  # pragma: no cover - depends on optional Triton kernels.
    RMSNorm = None  # type: ignore[assignment]

try:
    if torch.cuda.is_available():
        from knn_cuda import KNN

        _KNN_IMPORT_ERROR: Exception | None = None
    else:
        KNN = None  # type: ignore[assignment]
        _KNN_IMPORT_ERROR = RuntimeError("torch.cuda.is_available() is False")
except Exception as exc:  # pragma: no cover - optional CUDA extension.
    KNN = None  # type: ignore[assignment]
    _KNN_IMPORT_ERROR = exc


def symfn_max(x: torch.Tensor) -> torch.Tensor:
    """Max-pool token features over the sequence dimension."""
    return torch.max(x, dim=1)[0]


def symfn_avg(x: torch.Tensor) -> torch.Tensor:
    """Average-pool token features over the sequence dimension."""
    return torch.mean(x, dim=1)


class DropPath(nn.Module):
    """Per-sample stochastic depth used by the Mamba residual blocks."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Encoder(nn.Module):
    """Local PointNet-style encoder for each grouped neighborhood."""

    def __init__(self, encoder_channel: int = 384) -> None:
        super().__init__()
        self.encoder_channel = int(encoder_channel)
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups: torch.Tensor) -> torch.Tensor:
        batch_size, num_groups, group_size, _ = point_groups.shape
        x = point_groups.reshape(batch_size * num_groups, group_size, 3)
        x = self.first_conv(x.transpose(2, 1))
        global_feature = torch.max(x, dim=2, keepdim=True)[0]
        x = torch.cat([global_feature.expand(-1, -1, group_size), x], dim=1)
        x = self.second_conv(x)
        x = torch.max(x, dim=2, keepdim=False)[0]
        return x.reshape(batch_size, num_groups, self.encoder_channel)


class Group(nn.Module):
    """FPS centers plus KNN local neighborhoods."""

    def __init__(
        self,
        num_group: int = 64,
        group_size: int = 32,
        knn_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_group = int(num_group)
        self.group_size = int(group_size)
        self.knn_backend = str(knn_backend)
        self.knn = None

        if self.num_group <= 0 or self.group_size <= 0:
            raise ValueError("num_group and group_size must be positive.")
        if self.knn_backend not in {"auto", "cuda", "torch"}:
            raise ValueError("knn_backend must be one of 'auto', 'cuda', or 'torch'.")

        if self.knn_backend in {"auto", "cuda"} and KNN is not None:
            self.knn = KNN(k=self.group_size, transpose_mode=True)
        elif self.knn_backend == "cuda":
            raise ImportError(
                "knn_cuda was requested but is unavailable. "
                f"Import failure: {_KNN_IMPORT_ERROR}"
            )
        else:
            logger.warning("Using PyTorch KNN fallback for Mamba3DTrue grouping.")

    def forward(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError(
                f"Expected points shaped [B, N, 3], got {tuple(xyz.shape)}"
            )

        batch_size, num_points, _ = xyz.shape
        if self.num_group > num_points:
            raise ValueError(
                f"num_group={self.num_group} exceeds input point count {num_points}."
            )
        if self.group_size > num_points:
            raise ValueError(
                f"group_size={self.group_size} exceeds input point count {num_points}."
            )

        center = self._fps(xyz, self.num_group)
        if self.knn is not None:
            _, idx = self.knn(xyz, center)
        else:
            idx = self._knn_pytorch(xyz, center, self.group_size)

        idx_base = (
            torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        )
        idx = (idx + idx_base).reshape(-1)
        neighborhood = xyz.reshape(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.reshape(
            batch_size, self.num_group, self.group_size, 3
        ).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center

    def _fps(self, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        device = xyz.device
        batch_size, num_points, channels = xyz.shape
        centroids = torch.zeros(batch_size, npoint, channels, device=device)
        distance = torch.ones(batch_size, num_points, device=device) * 1.0e10
        farthest = torch.randint(0, num_points, (batch_size,), dtype=torch.long).to(
            device
        )
        batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)

        for i in range(npoint):
            centroids[:, i, :] = xyz[batch_indices, farthest, :]
            centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, dim=-1)
            distance = torch.where(dist < distance, dist, distance)
            farthest = torch.max(distance, dim=-1)[1]

        return centroids

    def _knn_pytorch(
        self,
        xyz: torch.Tensor,
        query: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        dist = torch.cdist(query, xyz)
        _, idx = torch.topk(dist, k, dim=2, largest=False, sorted=True)
        return idx


class MambaResidualBlock(nn.Module):
    """LayerNorm/RMSNorm + real mamba_ssm Mamba + residual connection."""

    def __init__(
        self,
        dim: int,
        *,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int | float = 2,
        layer_idx: int = 0,
        drop_path: float = 0.0,
        rms_norm: bool = False,
    ) -> None:
        super().__init__()
        if _MAMBA_IMPORT_ERROR is not None or Mamba is None:
            raise ImportError(
                "mamba_ssm is required for Mamba3DTrue_features. "
                "Install/import mamba_ssm in the PCLR_compare environment."
            ) from _MAMBA_IMPORT_ERROR
        if rms_norm and RMSNorm is None:
            raise ImportError("rms_norm=True requires mamba_ssm Triton RMSNorm.")

        norm_cls = RMSNorm if rms_norm else nn.LayerNorm
        self.norm = norm_cls(dim, eps=1.0e-5)
        self.mixer = Mamba(
            d_model=dim,
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=expand,
            layer_idx=layer_idx,
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return x + self.drop_path(self.mixer(self.norm(x)))
        except RuntimeError as exc:
            message = str(exc)
            if "is_cuda" in message or "Expected x.is_cuda" in message:
                raise RuntimeError(
                    "mamba_ssm Mamba forward requires a CUDA tensor in this "
                    "environment. Move Mamba3DTrue_features and inputs to a CUDA "
                    "device for shape/backward/training smoke tests."
                ) from exc
            raise


class MambaTokenMixer(nn.Module):
    """Stack of real Mamba blocks over serialized group tokens."""

    def __init__(
        self,
        dim: int,
        depth: int,
        *,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int | float = 2,
        drop_path_rate: float = 0.0,
        drop_out: float = 0.0,
        rms_norm: bool = False,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive.")
        dpr = [value.item() for value in torch.linspace(0, drop_path_rate, depth)]
        self.layers = nn.ModuleList(
            [
                MambaResidualBlock(
                    dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    layer_idx=i,
                    drop_path=dpr[i],
                    rms_norm=rms_norm,
                )
                for i in range(depth)
            ]
        )
        norm_cls = RMSNorm if rms_norm else nn.LayerNorm
        self.norm = norm_cls(dim, eps=1.0e-5)
        self.drop_out = nn.Dropout(drop_out) if drop_out > 0.0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            if module.bias is not None and not getattr(
                module.bias,
                "_no_reinit",
                False,
            ):
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        for name, param in module.named_parameters(recurse=False):
            if name in {"out_proj.weight", "fc2.weight"}:
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                with torch.no_grad():
                    param /= math.sqrt(len(self.layers))

    def forward(self, tokens: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        x = tokens + pos
        for layer in self.layers:
            x = self.drop_out(layer(x))
        return self.norm(x)


def _init_order_scale(dim: int) -> tuple[nn.Parameter, nn.Parameter]:
    gamma = nn.Parameter(torch.ones(dim))
    beta = nn.Parameter(torch.zeros(dim))
    nn.init.normal_(gamma, mean=1.0, std=0.02)
    nn.init.normal_(beta, std=0.02)
    return gamma, beta


def _apply_order_scale(
    x: torch.Tensor,
    gamma: nn.Parameter,
    beta: nn.Parameter,
) -> torch.Tensor:
    return x * gamma + beta


def _pooling_from_name(symfn: str | Callable[[torch.Tensor], torch.Tensor] | None):
    if symfn is None or symfn == "max":
        return symfn_max
    if symfn == "avg":
        return symfn_avg
    if callable(symfn):
        return symfn
    raise ValueError("sym_fn must be None, 'max', 'avg', or a callable.")


class Mamba3DTrue_features(nn.Module):
    """True Mamba/PointMamba-style backbone for PointNetLK registration."""

    def __init__(
        self,
        dim_k: int = 1024,
        sym_fn: str | Callable[[torch.Tensor], torch.Tensor] | None = "max",
        scale: int | float = 1,
        num_groups: int = 64,
        group_size: int = 32,
        trans_dim: int = 384,
        depth: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int | float = 2,
        drop_path_rate: float = 0.1,
        drop_out: float = 0.0,
        rms_norm: bool = False,
        grid_size: float = 0.02,
        knn_backend: str = "auto",
        return_tokens: bool = False,
    ) -> None:
        super().__init__()
        if _MAMBA_IMPORT_ERROR is not None or Mamba is None:
            raise ImportError(
                "mamba_ssm is required for Mamba3DTrue_features. "
                "Install/import mamba_ssm in the PCLR_compare environment."
            ) from _MAMBA_IMPORT_ERROR

        self.dim_k = int(dim_k / scale)
        self.trans_dim = int(trans_dim)
        self.depth = int(depth)
        self.num_group = int(num_groups)
        self.group_size = int(group_size)
        self.grid_size = float(grid_size)
        self.return_tokens = bool(return_tokens)
        self.sy = _pooling_from_name(sym_fn)

        self.group_divider = Group(
            num_group=self.num_group,
            group_size=self.group_size,
            knn_backend=knn_backend,
        )
        self.encoder = Encoder(encoder_channel=self.trans_dim)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )
        self.blocks = MambaTokenMixer(
            dim=self.trans_dim,
            depth=self.depth,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            drop_path_rate=drop_path_rate,
            drop_out=drop_out,
            rms_norm=rms_norm,
        )
        self.order_scale_gamma_1, self.order_scale_beta_1 = _init_order_scale(
            self.trans_dim
        )
        self.order_scale_gamma_2, self.order_scale_beta_2 = _init_order_scale(
            self.trans_dim
        )
        self.feature_proj = nn.Linear(self.trans_dim, self.dim_k)

        self.t_out_h1 = None
        self.t_out_t2 = None

    def forward(
        self,
        points: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from .serialization import serialization_func

        neighborhood, center = self.group_divider(points)
        group_tokens = self.encoder(neighborhood)
        pos = self.pos_embed(center)

        _, _, _, tokens_fwd, pos_fwd = serialization_func(
            center,
            group_tokens,
            pos,
            "hilbert",
            grid_size=self.grid_size,
        )
        _, _, _, tokens_bwd, pos_bwd = serialization_func(
            center,
            group_tokens,
            pos,
            "hilbert-trans",
            grid_size=self.grid_size,
        )
        tokens_fwd = _apply_order_scale(
            tokens_fwd,
            self.order_scale_gamma_1,
            self.order_scale_beta_1,
        )
        tokens_bwd = _apply_order_scale(
            tokens_bwd,
            self.order_scale_gamma_2,
            self.order_scale_beta_2,
        )

        tokens = torch.cat([tokens_fwd, tokens_bwd], dim=1)
        pos_tokens = torch.cat([pos_fwd, pos_bwd], dim=1)
        token_features = self.blocks(tokens, pos_tokens)
        self.t_out_h1 = token_features.transpose(1, 2)

        pooled = self.sy(token_features)
        global_features = self.feature_proj(pooled)
        if self.return_tokens:
            return global_features, token_features
        return global_features

    def load_pretrained_weights(
        self,
        ckpt_path: str,
        *,
        strict: bool = False,
        verbose: bool = True,
    ) -> tuple[torch.nn.modules.module._IncompatibleKeys, float]:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "base_model" in ckpt:
            state_dict = ckpt["base_model"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        cleaned = {}
        for key, value in state_dict.items():
            clean_key = key.replace("module.", "").replace("MAE_encoder.", "")
            if "cls_head_finetune" not in clean_key:
                cleaned[clean_key] = value

        own_state = self.state_dict()
        matched = {
            key: value
            for key, value in cleaned.items()
            if key in own_state and value.shape == own_state[key].shape
        }
        incompatible = self.load_state_dict(matched, strict=strict)
        matched_params = sum(own_state[key].numel() for key in matched)
        total_params = sum(param.numel() for param in own_state.values())
        utilization = matched_params / total_params * 100.0 if total_params else 0.0

        if verbose:
            logger.info(
                "Loaded %d/%d compatible Mamba3DTrue tensors from %s "
                "(%.1f%% parameter utilization).",
                len(matched),
                len(own_state),
                ckpt_path,
                utilization,
            )
        return incompatible, utilization


Mamba3D_features = Mamba3DTrue_features


class Mamba3DTrue_classifier(nn.Module):
    """Classifier wrapper kept for compatibility with PointNet-style scripts."""

    def __init__(self, num_c: int, ptfeat: Mamba3DTrue_features, dim_k: int) -> None:
        super().__init__()
        self.features = ptfeat
        self.classifier = nn.Sequential(
            nn.Linear(dim_k, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_c),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        features = self.features(points)
        if isinstance(features, tuple):
            features = features[0]
        return self.classifier(features)

    def loss(
        self,
        out: torch.Tensor,
        target: torch.Tensor,
        w: float = 0.001,
    ) -> torch.Tensor:
        del w
        return torch.nn.functional.nll_loss(
            torch.nn.functional.log_softmax(out, dim=1),
            target,
            reduction="mean",
        )
