"""MambaNetLK registration wrapper.

This route keeps the PointNetLK contract but adds a learned Mamba2 pair-pose
initializer before the final LK refinement.  The final exported transform is
always the refined transform, so benchmark/eval code can continue to read
``model.g`` exactly like PointLK.
"""

from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn as nn

from . import pointlk, pointlk_cached, se3
from .mamba3d_mamba2 import Mamba3DMamba2_features

logger = logging.getLogger(__name__)


_LK_MODE_TO_ORDER_MODE = {
    "cached_recompute_order": "recompute",
    "cached_reuse_order": "reuse",
}
_LK_MODES = {"exact", *_LK_MODE_TO_ORDER_MODE}


def _optional_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


def _global_and_tokens(
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(output, tuple):
        raise ValueError(
            "MambaNetLK requires a backbone configured with return_tokens."
        )
    return output


class PairPoseHead(nn.Module):
    """Predict a source-to-template SE(3) initialization from paired tokens."""

    def __init__(
        self,
        *,
        global_dim: int,
        token_dim: int,
        pair_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.0,
        rotation_scale: float = 1.0,
        translation_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.rotation_scale = float(rotation_scale)
        self.translation_scale = float(translation_scale)
        if self.rotation_scale <= 0.0 or self.translation_scale <= 0.0:
            raise ValueError("pose scales must be positive.")

        self.global_proj = nn.Sequential(
            nn.LayerNorm(global_dim),
            nn.Linear(global_dim, pair_dim),
            nn.GELU(),
        )
        self.token_proj = nn.Sequential(
            nn.LayerNorm(token_dim * 2),
            nn.Linear(token_dim * 2, pair_dim),
            nn.GELU(),
        )
        pair_features = pair_dim * 8
        self.head = nn.Sequential(
            nn.LayerNorm(pair_features),
            nn.Linear(pair_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, 6),
        )
        self._init_identity_bias()

    def _init_identity_bias(self) -> None:
        final = self.head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def _token_summary(self, tokens: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat(
            [torch.max(tokens, dim=1)[0], torch.mean(tokens, dim=1)],
            dim=1,
        )
        return self.token_proj(pooled)

    def forward(
        self,
        template_global: torch.Tensor,
        source_global: torch.Tensor,
        template_tokens: torch.Tensor,
        source_tokens: torch.Tensor,
    ) -> torch.Tensor:
        tg = self.global_proj(template_global)
        sg = self.global_proj(source_global)
        tt = self._token_summary(template_tokens)
        st = self._token_summary(source_tokens)
        pair = torch.cat(
            [
                tg,
                sg,
                tg - sg,
                tg * sg,
                tt,
                st,
                tt - st,
                tt * st,
            ],
            dim=1,
        )
        raw = self.head(pair)
        rotation = torch.tanh(raw[:, :3]) * self.rotation_scale
        translation = torch.tanh(raw[:, 3:]) * self.translation_scale
        return torch.cat([rotation, translation], dim=1)


class MambaNetLK(nn.Module):
    """Mamba2 pair-pose initializer followed by mandatory LK refinement."""

    is_mambanetlk = True

    def __init__(
        self,
        *,
        dim_k: int = 1024,
        sym_fn: str | Callable[[torch.Tensor], torch.Tensor] | None = "max",
        scale: int | float = 1,
        num_groups: int = 128,
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
        lk_mode: str = "cached_reuse_order",
        delta: float = 1.0e-2,
        learn_delta: bool = False,
        lm_damping: float = 1.0e-3,
        dx_clip_norm: float | None = 0.25,
        pair_dim: int = 256,
        pair_hidden_dim: int = 512,
        pair_dropout: float = 0.0,
        pose_rotation_scale: float = 1.5708,
        pose_translation_scale: float = 1.0,
        use_mem_eff_path: bool = True,
    ) -> None:
        super().__init__()
        if lk_mode not in _LK_MODES:
            raise ValueError(
                f"Unknown lk_mode={lk_mode!r}. Supported modes: {sorted(_LK_MODES)}"
            )

        backbone = Mamba3DMamba2_features(
            dim_k=dim_k,
            sym_fn=sym_fn,
            scale=scale,
            num_groups=num_groups,
            group_size=group_size,
            trans_dim=trans_dim,
            depth=depth,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            chunk_size=chunk_size,
            rmsnorm=rmsnorm,
            drop_path_rate=drop_path_rate,
            drop_out=drop_out,
            grid_size=grid_size,
            knn_backend=knn_backend,
            return_tokens=True,
            use_mem_eff_path=use_mem_eff_path,
        )
        if lk_mode == "exact":
            self.refiner = pointlk.PointLK(
                ptnet=backbone,
                delta=delta,
                learn_delta=learn_delta,
            )
        else:
            self.refiner = pointlk_cached.ApproxCachedPointLK(
                ptnet=backbone,
                delta=delta,
                learn_delta=learn_delta,
                order_mode=_LK_MODE_TO_ORDER_MODE[lk_mode],
                lm_damping=lm_damping,
                dx_clip_norm=_optional_float(dx_clip_norm),
            )
        self.pose_head = PairPoseHead(
            global_dim=int(dim_k / scale),
            token_dim=trans_dim,
            pair_dim=pair_dim,
            hidden_dim=pair_hidden_dim,
            dropout=pair_dropout,
            rotation_scale=pose_rotation_scale,
            translation_scale=pose_translation_scale,
        )
        self.exp = se3.Exp
        self.transform = se3.transform
        self.lk_mode = lk_mode

        self.initial_g = None
        self.initial_twist = None
        self.refine_g = None
        self.g = None
        self.g_series = None
        self.prev_r = None
        self.last_err = None
        self.itr = 0

    @property
    def backbone(self) -> Mamba3DMamba2_features:
        return self.refiner.ptnet

    @staticmethod
    def rsq(r: torch.Tensor) -> torch.Tensor:
        return pointlk.PointLK.rsq(r)

    @staticmethod
    def comp(g: torch.Tensor, igt: torch.Tensor) -> torch.Tensor:
        return pointlk.PointLK.comp(g, igt)

    def predict_initial_transform(
        self,
        template: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        template_global, template_tokens = _global_and_tokens(self.backbone(template))
        source_global, source_tokens = _global_and_tokens(self.backbone(source))
        twist = self.pose_head(
            template_global,
            source_global,
            template_tokens,
            source_tokens,
        )
        self.initial_twist = twist
        self.initial_g = self.exp(twist)
        return self.initial_g

    def load_pretrained_weights(
        self,
        ckpt_path: str,
        *,
        strict: bool = False,
        verbose: bool = True,
    ):
        return self.backbone.load_pretrained_weights(
            ckpt_path,
            strict=strict,
            verbose=verbose,
        )

    def forward(
        self,
        p0: torch.Tensor,
        p1: torch.Tensor,
        maxiter: int = 10,
        xtol: float = 1.0e-7,
    ) -> torch.Tensor | None:
        initial_g = self.predict_initial_transform(p0, p1)
        aligned_source = self.transform(initial_g.unsqueeze(1), p1)

        residual = self.refiner(p0, aligned_source, maxiter=maxiter, xtol=xtol)
        refine_g = self.refiner.g
        self.refine_g = refine_g
        self.prev_r = getattr(self.refiner, "prev_r", None)
        self.last_err = getattr(self.refiner, "last_err", None)
        self.itr = int(getattr(self.refiner, "itr", -1))

        if refine_g is None:
            self.g = initial_g
            self.g_series = initial_g.unsqueeze(0)
            return residual

        self.g = refine_g.bmm(initial_g)
        refiner_series = getattr(self.refiner, "g_series", None)
        if refiner_series is None:
            self.g_series = self.g.unsqueeze(0)
        else:
            self.g_series = refiner_series.matmul(initial_g.unsqueeze(0))
        return residual


MambaNetLK_features = MambaNetLK
