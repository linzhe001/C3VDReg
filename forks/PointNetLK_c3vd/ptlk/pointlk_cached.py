"""Cached approximate PointLK wrapper for Mamba2 token-cache backbones."""

from __future__ import annotations

from dataclasses import fields, is_dataclass

import torch

from . import invmat, se3


class ApproxCachedPointLK(torch.nn.Module):
    """Approximate inverse-compositional LK over cached Mamba2 support tokens."""

    def __init__(
        self,
        ptnet: torch.nn.Module,
        delta: float = 1.0e-2,
        learn_delta: bool = False,
        order_mode: str = "reuse",
        lm_damping: float = 1.0e-3,
        dx_clip_norm: float | None = None,
    ) -> None:
        super().__init__()
        if not (
            hasattr(ptnet, "prepare_token_cache")
            and hasattr(ptnet, "forward_from_cache")
        ):
            raise TypeError(
                "ApproxCachedPointLK requires a backbone with prepare_token_cache "
                "and forward_from_cache. Use lk_mode=exact for non-cached "
                "backbones."
            )
        if order_mode not in {"reuse", "recompute"}:
            raise ValueError(
                "order_mode must be one of 'reuse' or 'recompute', "
                f"got {order_mode!r}."
            )

        self.ptnet = ptnet
        self.inverse = invmat.InvMatrix.apply
        self.exp = se3.Exp
        self.transform = se3.transform
        self.order_mode = order_mode
        self.lm_damping = float(lm_damping)
        self.dx_clip_norm = (
            None if dx_clip_norm is None else float(dx_clip_norm)
        )
        if self.dx_clip_norm is not None and self.dx_clip_norm <= 0.0:
            raise ValueError("dx_clip_norm must be positive when provided.")

        twist = torch.Tensor([delta, delta, delta, delta, delta, delta])
        self.dt = torch.nn.Parameter(twist.view(1, 6), requires_grad=learn_delta)

        self.last_err = None
        self.g_series = None
        self.prev_r = None
        self.g = None
        self.itr = 0

    @staticmethod
    def rsq(r: torch.Tensor) -> torch.Tensor:
        z = torch.zeros_like(r)
        return torch.nn.functional.mse_loss(r, z, reduction="sum")

    @staticmethod
    def comp(g: torch.Tensor, igt: torch.Tensor) -> torch.Tensor:
        assert g.size(0) == igt.size(0)
        assert g.size(1) == igt.size(1) and g.size(1) == 4
        assert g.size(2) == igt.size(2) and g.size(2) == 4
        a = g.matmul(igt)
        eye = torch.eye(4).to(a).view(1, 4, 4).expand(a.size(0), 4, 4)
        return torch.nn.functional.mse_loss(a, eye, reduction="mean") * 16

    @staticmethod
    def _global_feature(
        output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        return output[0] if isinstance(output, tuple) else output

    @staticmethod
    def _repeat_cache(cache, repeats: int):
        if isinstance(cache, torch.Tensor):
            return cache.repeat_interleave(repeats, dim=0)
        if not is_dataclass(cache):
            raise TypeError(
                "Cached LK can batch Jacobian evaluation only for tensor or "
                "dataclass cache objects."
            )

        values = {}
        for field in fields(cache):
            value = getattr(cache, field.name)
            if isinstance(value, torch.Tensor):
                value = value.repeat_interleave(repeats, dim=0)
            values[field.name] = value
        return type(cache)(**values)

    @staticmethod
    def do_forward(
        net,
        p0: torch.Tensor,
        p1: torch.Tensor,
        maxiter: int = 10,
        xtol: float = 1.0e-7,
        p0_zero_mean: bool = True,
        p1_zero_mean: bool = True,
    ):
        a0 = torch.eye(4).view(1, 4, 4).expand(p0.size(0), 4, 4).to(p0).clone()
        a1 = torch.eye(4).view(1, 4, 4).expand(p1.size(0), 4, 4).to(p1).clone()
        if p0_zero_mean:
            p0_m = p0.mean(dim=1)
            a0[:, 0:3, 3] = p0_m
            q0 = p0 - p0_m.unsqueeze(1)
        else:
            q0 = p0

        if p1_zero_mean:
            p1_m = p1.mean(dim=1)
            a1[:, 0:3, 3] = -p1_m
            q1 = p1 - p1_m.unsqueeze(1)
        else:
            q1 = p1

        r = net(q0, q1, maxiter=maxiter, xtol=xtol)

        if p0_zero_mean or p1_zero_mean:
            est_g = net.g
            if p0_zero_mean:
                est_g = a0.to(est_g).bmm(est_g)
            if p1_zero_mean:
                est_g = est_g.bmm(a1.to(est_g))
            net.g = est_g

            est_gs = net.g_series
            if p0_zero_mean:
                est_gs = a0.unsqueeze(0).contiguous().to(est_gs).matmul(est_gs)
            if p1_zero_mean:
                est_gs = est_gs.matmul(a1.unsqueeze(0).contiguous().to(est_gs))
            net.g_series = est_gs

        return r

    def forward(
        self,
        p0: torch.Tensor,
        p1: torch.Tensor,
        maxiter: int = 10,
        xtol: float = 1.0e-7,
    ):
        g0 = torch.eye(4).to(p0).view(1, 4, 4)
        g0 = g0.expand(p0.size(0), 4, 4).contiguous()

        r, g, itr = self.iclk(g0, p0, p1, maxiter, xtol)

        self.g = g
        self.itr = itr
        return r

    def update(self, g: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
        dg = self.exp(dx)
        return dg.matmul(g)

    def approx_Jic_cached(
        self,
        template_cache,
        f0: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = f0.size(0)
        perturbations = -torch.diag_embed(dt).reshape(batch_size * 6, 6)
        transform = self.exp(perturbations)
        repeated_cache = self._repeat_cache(template_cache, repeats=6)
        feature_out = self.ptnet.forward_from_cache(
            repeated_cache,
            transform=transform,
            order_mode=self.order_mode,
        )
        f = self._global_feature(feature_out)
        f = f.reshape(batch_size, 6, -1).transpose(1, 2)
        df = f0.unsqueeze(-1) - f
        return df / dt.unsqueeze(1)

    def _damped_pinv(self, j: torch.Tensor) -> torch.Tensor:
        jt = j.transpose(1, 2)
        h = jt.bmm(j)
        if self.lm_damping > 0.0:
            eye = torch.eye(6, dtype=h.dtype, device=h.device).unsqueeze(0)
            h = h + self.lm_damping * eye
        return self.inverse(h).bmm(jt)

    def _clip_dx(self, dx: torch.Tensor) -> torch.Tensor:
        if self.dx_clip_norm is None:
            return dx
        norm = dx.norm(p=2, dim=1, keepdim=True).clamp_min(1.0e-12)
        scale = torch.clamp(self.dx_clip_norm / norm, max=1.0)
        return dx * scale

    def _cache_order_enabled(self) -> bool:
        return self.order_mode == "reuse"

    def iclk(
        self,
        g0: torch.Tensor,
        p0: torch.Tensor,
        p1: torch.Tensor,
        maxiter: int,
        xtol: float,
    ):
        training = self.ptnet.training
        batch_size = p0.size(0)
        cache_order = self._cache_order_enabled()

        g = g0
        self.g_series = torch.zeros(
            maxiter + 1,
            *g0.size(),
            dtype=g0.dtype,
            device=g0.device,
        )
        self.g_series[0] = g0.clone()

        if training:
            template_cache = self.ptnet.prepare_token_cache(
                p0,
                cache_order=cache_order,
            )
            source_cache = self.ptnet.prepare_token_cache(
                p1,
                cache_order=cache_order,
            )
            self._global_feature(
                self.ptnet.forward_from_cache(
                    template_cache,
                    order_mode=self.order_mode,
                )
            )
            self._global_feature(
                self.ptnet.forward_from_cache(
                    source_cache,
                    order_mode=self.order_mode,
                )
            )

        self.ptnet.eval()

        template_cache = self.ptnet.prepare_token_cache(
            p0,
            cache_order=cache_order,
        )
        source_cache = self.ptnet.prepare_token_cache(
            p1,
            cache_order=cache_order,
        )
        f0_out = self.ptnet.forward_from_cache(
            template_cache,
            transform=None,
            order_mode=self.order_mode,
        )
        f0 = self._global_feature(f0_out)

        dt = self.dt.to(p0).expand(batch_size, 6)
        try:
            j = self.approx_Jic_cached(template_cache, f0, dt)
        except Exception as exc:
            self.last_err = exc
            self.ptnet.train(training)
            return None, g0, -1

        self.last_err = None
        itr = -1
        try:
            pinv = self._damped_pinv(j)
        except RuntimeError as err:
            self.last_err = err
            f1_out = self.ptnet.forward_from_cache(
                source_cache,
                transform=g,
                order_mode=self.order_mode,
            )
            f1 = self._global_feature(f1_out)
            r = f1 - f0
            self.ptnet.train(training)
            return r, g, itr

        itr = 0
        r = None
        for itr in range(maxiter):
            self.prev_r = r
            f_out = self.ptnet.forward_from_cache(
                source_cache,
                transform=g,
                order_mode=self.order_mode,
            )
            f = self._global_feature(f_out)
            r = f - f0

            dx = -pinv.bmm(r.unsqueeze(-1)).view(batch_size, 6)
            dx = self._clip_dx(dx)
            check = dx.norm(p=2, dim=1, keepdim=True).max()

            if float(check.detach()) < xtol:
                if itr == 0:
                    self.last_err = 0
                break

            g = self.update(g, dx)
            self.g_series[itr + 1] = g.clone()

        rep = len(range(itr, maxiter))
        self.g_series[(itr + 1) :] = g.clone().unsqueeze(0).repeat(rep, 1, 1, 1)

        self.ptnet.train(training)
        return r, g, (itr + 1)


# EOF
