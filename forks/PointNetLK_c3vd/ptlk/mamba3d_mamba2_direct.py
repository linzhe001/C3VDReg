"""Direct-point Mamba2 feature extractor for PointNetLK.

This compact route keeps the original PointNetLK-Mamba style: embed raw points,
run a small stack of Mamba2 blocks directly over the point sequence, transform
per-point features with a shared MLP, then pool to a global descriptor.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba3d_true import DropPath, symfn_avg, symfn_max

logger = logging.getLogger(__name__)

try:
    from mamba_ssm.modules.mamba2 import Mamba2

    _MAMBA2_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - exercised only without optional dep.
    Mamba2 = None  # type: ignore[assignment]
    _MAMBA2_IMPORT_ERROR = exc


@dataclass
class DirectPointMamba2Cache:
    """Cached direct-point support used by ApproxCachedPointLK."""

    points: torch.Tensor
    ordered_points: torch.Tensor | None = None


def _pooling_from_name(symfn: str | Callable[[torch.Tensor], torch.Tensor] | None):
    if symfn is None or symfn == "max":
        return symfn_max
    if symfn == "avg":
        return symfn_avg
    if callable(symfn):
        return symfn
    raise ValueError("sym_fn must be None, 'max', 'avg', or a callable.")


def _pooling_name(
    pooling: str | None,
    symfn: str | Callable[[torch.Tensor], torch.Tensor] | None,
) -> str:
    if pooling is not None:
        return str(pooling).lower()
    if callable(symfn):
        return "custom"
    return str("max" if symfn is None else symfn).lower()


def _integer_expand(expand: int | float) -> int:
    """Mamba2 in this environment requires an integer expansion ratio."""
    if isinstance(expand, bool):
        raise ValueError("expand must be a positive integer, not bool.")
    if isinstance(expand, int):
        expand_int = expand
    elif isinstance(expand, float) and expand.is_integer():
        expand_int = int(expand)
    else:
        raise ValueError(
            "mamba3d_mamba2_direct expand must be a positive integer in this "
            "environment; fractional values make Mamba2 projection dimensions "
            "non-integral."
        )
    if expand_int <= 0:
        raise ValueError("expand must be positive.")
    return expand_int


class DescriptorNorm(nn.Module):
    """Normalize pooled descriptors without changing the PointLK dim_k contract."""

    def __init__(self, dim: int, norm_type: str = "layernorm") -> None:
        super().__init__()
        norm_type = str(norm_type).lower()
        self.norm_type = norm_type
        if norm_type in {"none", "identity"}:
            self.norm = nn.Identity()
            self.apply_l2 = False
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(dim)
            self.apply_l2 = False
        elif norm_type == "l2":
            self.norm = nn.Identity()
            self.apply_l2 = True
        elif norm_type in {"layernorm_l2", "ln_l2"}:
            self.norm = nn.LayerNorm(dim)
            self.apply_l2 = True
        else:
            raise ValueError(
                "descriptor_norm must be one of 'layernorm', 'l2', "
                "'layernorm_l2', or 'none'."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        if self.apply_l2:
            x = F.normalize(x, p=2, dim=1)
        return x


class _ChannelNorm(nn.Module):
    def __init__(self, channels: int, norm_type: str = "batchnorm") -> None:
        super().__init__()
        norm_type = str(norm_type).lower()
        self.norm_type = norm_type
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(channels)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(channels)
        elif norm_type == "groupnorm":
            groups = min(8, channels)
            while channels % groups != 0 and groups > 1:
                groups -= 1
            self.norm = nn.GroupNorm(groups, channels)
        elif norm_type in {"none", "identity"}:
            self.norm = nn.Identity()
        else:
            raise ValueError(
                "mlp_norm must be one of 'batchnorm', 'layernorm', "
                "'groupnorm', or 'none'."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_type == "layernorm":
            return self.norm(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x)


class PointwiseFeatureMLP(nn.Module):
    """Shared Conv1d MLP over point features shaped [B, C, N]."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        *,
        norm_type: str = "batchnorm",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 1),
            _ChannelNorm(hidden_channels, norm_type),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Conv1d(hidden_channels, out_channels, 1),
            _ChannelNorm(out_channels, norm_type),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DirectPointMamba2Block(nn.Module):
    """PreNorm residual Mamba2 block with a conservative layer scale."""

    def __init__(
        self,
        dim: int,
        *,
        d_state: int = 32,
        d_conv: int = 4,
        expand: int | float = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 128,
        rmsnorm: bool = True,
        ffn_ratio: int | float = 2.0,
        layer_idx: int = 0,
        drop_path: float = 0.0,
        drop_out: float = 0.0,
        layer_scale_init: float = 1.0e-2,
        use_mem_eff_path: bool = True,
    ) -> None:
        super().__init__()
        if _MAMBA2_IMPORT_ERROR is not None or Mamba2 is None:
            raise ImportError(
                "mamba_ssm.modules.mamba2.Mamba2 is required for "
                "Mamba3DMamba2Direct_features. Install/import mamba_ssm with "
                "Mamba2 in the PCLR_compare environment."
            ) from _MAMBA2_IMPORT_ERROR

        expand_int = _integer_expand(expand)
        self.norm1 = nn.LayerNorm(dim, eps=1.0e-5)
        self.mixer = Mamba2(
            d_model=dim,
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=expand_int,
            headdim=int(headdim),
            ngroups=int(ngroups),
            chunk_size=int(chunk_size),
            rmsnorm=bool(rmsnorm),
            layer_idx=layer_idx,
            use_mem_eff_path=bool(use_mem_eff_path),
        )
        self.norm2 = nn.LayerNorm(dim, eps=1.0e-5)
        hidden_dim = max(dim, int(round(dim * float(ffn_ratio))))
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_out) if drop_out > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop_out) if drop_out > 0.0 else nn.Identity(),
        )
        self.drop_path = DropPath(drop_path)
        self.layer_scale_mamba = nn.Parameter(
            torch.ones(dim) * float(layer_scale_init)
        )
        self.layer_scale_ffn = nn.Parameter(torch.ones(dim) * float(layer_scale_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            x = x + self.drop_path(self.mixer(self.norm1(x)) * self.layer_scale_mamba)
        except RuntimeError as exc:
            message = str(exc)
            cuda_markers = (
                "is_cuda",
                "Expected x.is_cuda",
                "CUDA",
                "cuda",
                "causal_conv1d",
                "mamba_chunk_scan",
                "mamba_split_conv1d_scan",
            )
            if any(marker in message for marker in cuda_markers):
                raise RuntimeError(
                    "mamba_ssm Mamba2 forward requires a compatible CUDA runtime "
                    "for Mamba3DMamba2Direct_features. Constructor/import checks "
                    "can run on CPU; move the model and inputs to CUDA for "
                    "forward/backward/training."
                ) from exc
            raise
        x = x + self.drop_path(self.ffn(self.norm2(x)) * self.layer_scale_ffn)
        return x


class Mamba3DMamba2Direct_features(nn.Module):
    """Compact direct-point Mamba2 backbone for PointNetLK registration."""

    def __init__(
        self,
        dim_k: int = 1024,
        sym_fn: str | Callable[[torch.Tensor], torch.Tensor] | None = "max",
        scale: int | float = 1,
        d_model: int = 128,
        depth: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int | float = 1,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 128,
        rmsnorm: bool = True,
        drop_path_rate: float = 0.0,
        drop_out: float = 0.0,
        layer_scale_init: float = 1.0e-4,
        stem_hidden_dim: int = 64,
        pos_hidden_dim: int = 64,
        ffn_ratio: int | float = 2.0,
        mlp_hidden_dim: int = 256,
        mlp_norm: str = "layernorm",
        pooling: str | None = "maxavg",
        descriptor_norm: str = "layernorm",
        point_order: str = "identity",
        return_points: bool = False,
        use_mem_eff_path: bool = True,
    ) -> None:
        super().__init__()
        if _MAMBA2_IMPORT_ERROR is not None or Mamba2 is None:
            raise ImportError(
                "mamba_ssm.modules.mamba2.Mamba2 is required for "
                "Mamba3DMamba2Direct_features. This direct route does not fall "
                "back to Mamba-1."
            ) from _MAMBA2_IMPORT_ERROR
        if depth <= 0:
            raise ValueError("depth must be positive.")
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        expand_int = _integer_expand(expand)

        self.dim_k = int(dim_k / scale)
        self.d_model = int(d_model)
        self.depth = int(depth)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = expand_int
        self.headdim = int(headdim)
        self.ngroups = int(ngroups)
        self.chunk_size = int(chunk_size)
        self.rmsnorm = bool(rmsnorm)
        self.layer_scale_init = float(layer_scale_init)
        self.stem_hidden_dim = int(stem_hidden_dim)
        self.pos_hidden_dim = int(pos_hidden_dim)
        self.ffn_ratio = ffn_ratio
        self.mlp_hidden_dim = int(mlp_hidden_dim)
        self.mlp_norm = str(mlp_norm)
        self.pooling = _pooling_name(pooling, sym_fn)
        self.descriptor_norm_type = str(descriptor_norm)
        self.point_order = str(point_order)
        self.return_points = bool(return_points)
        self.sy = _pooling_from_name(sym_fn) if self.pooling == "custom" else None
        if self.point_order not in {"identity", "sort_xyz"}:
            raise ValueError("point_order must be 'identity' or 'sort_xyz'.")
        if self.pooling not in {"max", "avg", "maxavg", "custom"}:
            raise ValueError("pooling must be 'max', 'avg', 'maxavg', or custom.")

        self.coord_embed = nn.Sequential(
            nn.Linear(3, self.stem_hidden_dim),
            nn.LayerNorm(self.stem_hidden_dim),
            nn.GELU(),
            nn.Linear(self.stem_hidden_dim, self.d_model),
        )
        self.pos_embed = nn.Sequential(
            nn.Linear(3, self.pos_hidden_dim),
            nn.LayerNorm(self.pos_hidden_dim),
            nn.GELU(),
            nn.Linear(self.pos_hidden_dim, self.d_model),
        )
        dpr = [value.item() for value in torch.linspace(0, drop_path_rate, self.depth)]
        self.blocks = nn.ModuleList(
            [
                DirectPointMamba2Block(
                    self.d_model,
                    d_state=self.d_state,
                    d_conv=self.d_conv,
                    expand=self.expand,
                    headdim=self.headdim,
                    ngroups=self.ngroups,
                    chunk_size=self.chunk_size,
                    rmsnorm=self.rmsnorm,
                    ffn_ratio=self.ffn_ratio,
                    layer_idx=i,
                    drop_path=dpr[i],
                    drop_out=drop_out,
                    layer_scale_init=self.layer_scale_init,
                    use_mem_eff_path=use_mem_eff_path,
                )
                for i in range(self.depth)
            ]
        )
        self.final_norm = nn.LayerNorm(self.d_model, eps=1.0e-5)
        self.feature_transform = PointwiseFeatureMLP(
            self.d_model,
            self.mlp_hidden_dim,
            self.dim_k,
            norm_type=self.mlp_norm,
            dropout=drop_out,
        )
        if self.pooling == "maxavg":
            self.pool_project = nn.Sequential(
                nn.LayerNorm(self.dim_k * 2),
                nn.Linear(self.dim_k * 2, self.dim_k),
                nn.GELU(),
                nn.Linear(self.dim_k, self.dim_k),
            )
        else:
            self.pool_project = nn.Identity()
        self.descriptor_norm = DescriptorNorm(
            self.dim_k,
            norm_type=self.descriptor_norm_type,
        )

        self.t_out_h1 = None
        self.t_out_t2 = None

    def _order_points(self, points: torch.Tensor) -> torch.Tensor:
        if self.point_order == "identity":
            return points

        ordered = points
        for axis in (2, 1, 0):
            order = torch.argsort(ordered[..., axis], dim=1, stable=True)
            gather_index = order.unsqueeze(-1).expand(-1, -1, 3)
            ordered = torch.gather(ordered, dim=1, index=gather_index)
        return ordered

    def _pool_features(self, point_features: torch.Tensor) -> torch.Tensor:
        if self.pooling == "max":
            descriptor = symfn_max(point_features)
        elif self.pooling == "avg":
            descriptor = symfn_avg(point_features)
        elif self.pooling == "maxavg":
            descriptor = torch.cat(
                [symfn_max(point_features), symfn_avg(point_features)],
                dim=1,
            )
            descriptor = self.pool_project(descriptor)
        else:
            descriptor = self.sy(point_features)
        return self.descriptor_norm(descriptor)

    @staticmethod
    def _validate_points(points: torch.Tensor) -> None:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(
                "Expected points shaped [B, N, 3], "
                f"got {tuple(points.shape)}"
            )

    @staticmethod
    def _transform_points(
        points: torch.Tensor,
        transform: torch.Tensor | None,
    ) -> torch.Tensor:
        if transform is None:
            return points
        if transform.ndim == 4 and transform.shape[1] == 1:
            transform = transform[:, 0]
        if transform.ndim != 3 or transform.shape[-2:] != (4, 4):
            raise ValueError(
                "Expected transform shaped [B, 4, 4] or [B, 1, 4, 4], "
                f"got {tuple(transform.shape)}."
            )
        if transform.shape[0] != points.shape[0]:
            raise ValueError(
                "Transform batch size must match cached points batch size: "
                f"{transform.shape[0]} != {points.shape[0]}."
            )

        rotation_t = transform[:, :3, :3].transpose(1, 2)
        translation = transform[:, :3, 3]
        return torch.bmm(points, rotation_t) + translation[:, None, :]

    def _encode_ordered_points(
        self,
        points: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.coord_embed(points) + self.pos_embed(points)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)

        point_features = self.feature_transform(x.transpose(1, 2)).transpose(1, 2)
        self.t_out_h1 = point_features.transpose(1, 2)

        global_features = self._pool_features(point_features)
        if self.return_points:
            return global_features, point_features
        return global_features

    def prepare_token_cache(
        self,
        points: torch.Tensor,
        *,
        cache_order: bool = True,
    ) -> DirectPointMamba2Cache:
        self._validate_points(points)
        ordered_points = self._order_points(points) if cache_order else None
        return DirectPointMamba2Cache(points=points, ordered_points=ordered_points)

    def forward_from_cache(
        self,
        cache: DirectPointMamba2Cache,
        *,
        transform: torch.Tensor | None = None,
        order_mode: str = "reuse",
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if order_mode not in {"reuse", "recompute"}:
            raise ValueError(
                "order_mode must be one of 'recompute' or 'reuse', "
                f"got {order_mode!r}."
            )
        if order_mode == "reuse" and cache.ordered_points is not None:
            points = cache.ordered_points
            points = self._transform_points(points, transform)
        else:
            points = self._transform_points(cache.points, transform)
            points = self._order_points(points)
        return self._encode_ordered_points(points)

    def forward(
        self,
        points: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self._validate_points(points)
        points = self._order_points(points)
        return self._encode_ordered_points(points)

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

        cleaned = {
            key.replace("module.", ""): value
            for key, value in state_dict.items()
            if "cls_head_finetune" not in key
        }
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
                "Loaded %d/%d compatible Mamba3DMamba2Direct tensors from %s "
                "(%.1f%% parameter utilization).",
                len(matched),
                len(own_state),
                ckpt_path,
                utilization,
            )
        return incompatible, utilization


Mamba3D_features = Mamba3DMamba2Direct_features


class Mamba3DMamba2Direct_classifier(nn.Module):
    """Classifier wrapper kept for compatibility with PointNet-style scripts."""

    def __init__(
        self,
        num_c: int,
        ptfeat: Mamba3DMamba2Direct_features,
        dim_k: int,
    ) -> None:
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
