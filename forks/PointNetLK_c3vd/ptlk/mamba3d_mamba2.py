"""Mamba2-first PointMamba-style feature extractor for PointNetLK.

This module keeps the PointNetLK feature extractor contract: input points are
``[B, N, 3]`` and the default output is ``[B, dim_k]``. It reuses the
FPS/KNN grouping, local PointNet encoder, Hilbert serialization, and pooling
helpers from ``mamba3d_true`` while using ``mamba_ssm.modules.mamba2.Mamba2``
as the sequence mixer.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from .mamba3d_true import (
    DropPath,
    Encoder,
    Group,
    _apply_order_scale,
    _init_order_scale,
    _pooling_from_name,
    symfn_avg,  # noqa: F401 - preserved for legacy module-level access.
    symfn_max,  # noqa: F401 - preserved for legacy module-level access.
)

logger = logging.getLogger(__name__)

try:
    from mamba_ssm.modules.mamba2 import Mamba2

    _MAMBA2_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - exercised only without optional dep.
    Mamba2 = None  # type: ignore[assignment]
    _MAMBA2_IMPORT_ERROR = exc


@dataclass
class Mamba2TokenCache:
    """Cached local support set for approximate LK feature evaluations."""

    neighborhood: torch.Tensor
    center: torch.Tensor
    order_fwd: torch.Tensor | None = None
    order_bwd: torch.Tensor | None = None
    inverse_order_fwd: torch.Tensor | None = None
    inverse_order_bwd: torch.Tensor | None = None


def _gather_ordered(x: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3 or order.ndim != 2:
        raise ValueError(
            "Expected x shaped [B, G, C] and order shaped [B, G], "
            f"got {tuple(x.shape)} and {tuple(order.shape)}."
        )
    idx = order.to(device=x.device, dtype=torch.long).unsqueeze(-1)
    idx = idx.expand(-1, -1, x.shape[-1])
    return torch.gather(x, dim=1, index=idx)


class Mamba2ResidualBlock(nn.Module):
    """LayerNorm + real mamba_ssm Mamba2 + residual connection."""

    def __init__(
        self,
        dim: int,
        *,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int | float = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 128,
        rmsnorm: bool = True,
        layer_idx: int = 0,
        drop_path: float = 0.0,
        use_mem_eff_path: bool = True,
    ) -> None:
        super().__init__()
        if _MAMBA2_IMPORT_ERROR is not None or Mamba2 is None:
            raise ImportError(
                "mamba_ssm.modules.mamba2.Mamba2 is required for "
                "Mamba3DMamba2_features. Install a mamba_ssm version that exposes "
                "the Mamba2 module in the PCLR_compare environment."
            ) from _MAMBA2_IMPORT_ERROR

        self.norm = nn.LayerNorm(dim, eps=1.0e-5)
        self.mixer = Mamba2(
            d_model=dim,
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=expand,
            headdim=int(headdim),
            ngroups=int(ngroups),
            chunk_size=int(chunk_size),
            rmsnorm=bool(rmsnorm),
            layer_idx=layer_idx,
            use_mem_eff_path=bool(use_mem_eff_path),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return x + self.drop_path(self.mixer(self.norm(x)))
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
                    "in this environment. Constructor/import checks can run on CPU; "
                    "move Mamba3DMamba2_features and inputs to CUDA for "
                    "shape/backward/training smoke tests."
                ) from exc
            raise


class Mamba2TokenMixer(nn.Module):
    """Stack of real Mamba2 blocks over serialized group tokens."""

    def __init__(
        self,
        dim: int,
        depth: int,
        *,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int | float = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 128,
        rmsnorm: bool = True,
        drop_path_rate: float = 0.05,
        drop_out: float = 0.0,
        use_mem_eff_path: bool = True,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive.")

        dpr = [value.item() for value in torch.linspace(0, drop_path_rate, depth)]
        self.layers = nn.ModuleList(
            [
                Mamba2ResidualBlock(
                    dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    headdim=headdim,
                    ngroups=ngroups,
                    chunk_size=chunk_size,
                    rmsnorm=rmsnorm,
                    layer_idx=i,
                    drop_path=dpr[i],
                    use_mem_eff_path=use_mem_eff_path,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim, eps=1.0e-5)
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


class Mamba3DMamba2_features(nn.Module):
    """Mamba2/PointMamba-style backbone for PointNetLK registration."""

    def __init__(
        self,
        dim_k: int = 1024,
        sym_fn: str | Callable[[torch.Tensor], torch.Tensor] | None = "max",
        scale: int | float = 1,
        num_groups: int = 64,
        group_size: int = 32,
        trans_dim: int = 384,
        depth: int = 4,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int | float = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 128,
        rmsnorm: bool = True,
        drop_path_rate: float = 0.05,
        drop_out: float = 0.0,
        grid_size: float = 0.02,
        knn_backend: str = "auto",
        return_tokens: bool = False,
        use_mem_eff_path: bool = True,
    ) -> None:
        super().__init__()
        if _MAMBA2_IMPORT_ERROR is not None or Mamba2 is None:
            raise ImportError(
                "mamba_ssm.modules.mamba2.Mamba2 is required for "
                "Mamba3DMamba2_features. Install/import mamba_ssm with Mamba2 in "
                "the PCLR_compare environment. This route does not fall back to "
                "Mamba-1."
            ) from _MAMBA2_IMPORT_ERROR

        self.dim_k = int(dim_k / scale)
        self.trans_dim = int(trans_dim)
        self.depth = int(depth)
        self.num_group = int(num_groups)
        self.group_size = int(group_size)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = expand
        self.headdim = int(headdim)
        self.ngroups = int(ngroups)
        self.chunk_size = int(chunk_size)
        self.rmsnorm = bool(rmsnorm)
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
        self.blocks = Mamba2TokenMixer(
            dim=self.trans_dim,
            depth=self.depth,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            headdim=self.headdim,
            ngroups=self.ngroups,
            chunk_size=self.chunk_size,
            rmsnorm=self.rmsnorm,
            drop_path_rate=drop_path_rate,
            drop_out=drop_out,
            use_mem_eff_path=use_mem_eff_path,
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
        cache = self.prepare_token_cache(points, cache_order=False)
        return self.forward_from_cache(
            cache,
            transform=None,
            order_mode="recompute",
        )

    def prepare_token_cache(
        self,
        points: torch.Tensor,
        *,
        cache_order: bool = True,
    ) -> Mamba2TokenCache:
        from .serialization import serialization_func

        neighborhood, center = self.group_divider(points)

        if not cache_order:
            return Mamba2TokenCache(neighborhood=neighborhood, center=center)

        _, order_fwd, inverse_order_fwd, _, _ = serialization_func(
            center,
            order="hilbert",
            grid_size=self.grid_size,
        )
        _, order_bwd, inverse_order_bwd, _, _ = serialization_func(
            center,
            order="hilbert-trans",
            grid_size=self.grid_size,
        )
        return Mamba2TokenCache(
            neighborhood=neighborhood,
            center=center,
            order_fwd=order_fwd,
            order_bwd=order_bwd,
            inverse_order_fwd=inverse_order_fwd,
            inverse_order_bwd=inverse_order_bwd,
        )

    def forward_from_cache(
        self,
        cache: Mamba2TokenCache,
        *,
        transform: torch.Tensor | None = None,
        order_mode: str = "reuse",
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        neighborhood, center = self._transform_cached_support(cache, transform)
        token_features = self._encode_serialized_tokens(
            neighborhood,
            center,
            cache=cache,
            order_mode=order_mode,
        )
        return self._pool_project(token_features)

    def _transform_cached_support(
        self,
        cache: Mamba2TokenCache,
        transform: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        neighborhood = cache.neighborhood
        center = cache.center
        if transform is None:
            return neighborhood, center

        if transform.ndim == 4 and transform.shape[1] == 1:
            transform = transform[:, 0]
        if transform.ndim != 3 or transform.shape[-2:] != (4, 4):
            raise ValueError(
                "Expected transform shaped [B, 4, 4] or [B, 1, 4, 4], "
                f"got {tuple(transform.shape)}."
            )
        if transform.shape[0] != center.shape[0]:
            raise ValueError(
                "Transform batch size must match cached support batch size: "
                f"{transform.shape[0]} != {center.shape[0]}."
            )

        rotation_t = transform[:, :3, :3].transpose(1, 2)
        translation = transform[:, :3, 3]
        batch_size, num_groups, group_size, _ = neighborhood.shape

        flat_neighborhood = neighborhood.reshape(batch_size, num_groups * group_size, 3)
        flat_neighborhood_t = torch.bmm(flat_neighborhood, rotation_t)
        neighborhood_t = flat_neighborhood_t.reshape(
            batch_size,
            num_groups,
            group_size,
            3,
        )
        center_t = torch.bmm(center, rotation_t) + translation[:, None, :]
        return neighborhood_t.contiguous(), center_t.contiguous()

    def _encode_serialized_tokens(
        self,
        neighborhood: torch.Tensor,
        center: torch.Tensor,
        *,
        cache: Mamba2TokenCache | None = None,
        order_mode: str = "recompute",
    ) -> torch.Tensor:
        from .serialization import serialization_func

        if order_mode not in {"recompute", "reuse"}:
            raise ValueError(
                "order_mode must be one of 'recompute' or 'reuse', "
                f"got {order_mode!r}."
            )

        group_tokens = self.encoder(neighborhood)
        pos = self.pos_embed(center)

        if order_mode == "reuse":
            if cache is None or cache.order_fwd is None or cache.order_bwd is None:
                raise ValueError(
                    "order_mode='reuse' requires a cache prepared with "
                    "cache_order=True."
                )
            tokens_fwd = _gather_ordered(group_tokens, cache.order_fwd)
            pos_fwd = _gather_ordered(pos, cache.order_fwd)
            tokens_bwd = _gather_ordered(group_tokens, cache.order_bwd)
            pos_bwd = _gather_ordered(pos, cache.order_bwd)
        else:
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
        return self.blocks(tokens, pos_tokens)

    def _pool_project(
        self,
        token_features: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
                "Loaded %d/%d compatible Mamba3DMamba2 tensors from %s "
                "(%.1f%% parameter utilization).",
                len(matched),
                len(own_state),
                ckpt_path,
                utilization,
            )
        return incompatible, utilization


Mamba3D_features = Mamba3DMamba2_features


class Mamba3DMamba2_classifier(nn.Module):
    """Classifier wrapper kept for compatibility with PointNet-style scripts."""

    def __init__(self, num_c: int, ptfeat: Mamba3DMamba2_features, dim_k: int) -> None:
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
