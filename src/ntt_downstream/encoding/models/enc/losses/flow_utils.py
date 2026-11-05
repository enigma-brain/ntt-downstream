"""Shared rectified-flow utilities for the flow-matching heads.

All three flow heads (pointwise, full-grid, LDNS) train the same rectified
flow — straight-line interpolant ``z_t = (1 - t) * noise + t * target`` with
constant velocity target ``target - noise`` and the convention t=0 -> noise,
t=1 -> data. This module centralizes that math plus four optional upgrades:

- logit-normal time sampling (SD3): concentrates training time steps in the
  mid-trajectory region where the velocity field is hardest to learn.
- minibatch OT noise coupling: permutes the noise batch so each target gets
  its optimal-transport noise partner, straightening paths for better
  few-step sampling. The noise marginal is unchanged (it's a permutation).
- inference timestep shift: non-uniform integration grid. With our t
  convention the SD3-equivalent map is t(u) = u / (shift - (shift - 1) * u);
  shift > 1 concentrates steps near the noise end (t=0).
- classifier-free guidance: train-time conditioning dropout against a learned
  null embedding, sample-time guided velocity
  ``v_uncond + cfg_scale * (v_cond - v_uncond)``.

Shapes are handled generically: time ``t`` is always ``(B,)`` and gets
broadcast over the trailing dims of the state tensor; the conditioning
tensor's LAST dim must be the conditioning dim (true for all three heads).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# The pointwise flow path can see very large flattened batches
# (B * S * T * diffusion_batch_mul — tens of thousands of rows), so we solve
# OT on chunks. Rows are randomly ordered within a batch, so chunked OT is
# just minibatch OT with a smaller effective minibatch. The coupling is solved
# with an on-device Sinkhorn (entropic OT) rather than an exact CPU Hungarian:
# `scipy.linear_sum_assignment` is O(n^3) on the host and needs a GPU->CPU
# transfer per chunk. Each Sinkhorn chunk is a few (n x n) matmuls, cheap
# on-device.
_OT_MAX_CHUNK = 1024
# Sinkhorn iterations and entropic regularization (as a fraction of the mean
# chunk cost). 50 iters with reg=0.05 gives a near-deterministic plan, so the
# argmax assignment is close to a true permutation.
_OT_SINKHORN_ITERS = 50
_OT_SINKHORN_REG = 0.05

_TIME_SAMPLING_MODES = ("uniform", "logit_normal")


@dataclass(frozen=True)
class FlowOptions:
    """Train-time options for rectified-flow heads.

    Args:
        time_sampling: "uniform" or "logit_normal" (SD3-style
            ``sigmoid(mean + std * randn())``).
        logit_normal_mean: location in logit space (0.0 = symmetric).
        logit_normal_std: scale in logit space.
        ot_coupling: pair noise to targets with minibatch optimal transport.
        cond_dropout: probability of replacing a sample's conditioning with
            the head's learned null embedding. Must be > 0 to use CFG
            (``cfg_scale != 1``) at sampling time.
    """

    time_sampling: str = "uniform"
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 1.0
    ot_coupling: bool = False
    cond_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.time_sampling not in _TIME_SAMPLING_MODES:
            raise ValueError(f"time_sampling must be one of {_TIME_SAMPLING_MODES}, got {self.time_sampling!r}")
        if not 0.0 <= self.cond_dropout < 1.0:
            raise ValueError(f"cond_dropout must be in [0, 1), got {self.cond_dropout}")


def sample_flow_time(n: int, opts: FlowOptions, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Draw n training time steps in (0, 1) per ``opts.time_sampling``."""
    if opts.time_sampling == "logit_normal":
        z = torch.randn(n, device=device, dtype=dtype)
        return torch.sigmoid(opts.logit_normal_mean + opts.logit_normal_std * z)
    return torch.rand(n, device=device, dtype=dtype)


def make_time_grid(n_steps: int, shift: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Integration grid of ``n_steps + 1`` points from t=0 (noise) to t=1 (data).

    ``shift=1`` is uniform. ``shift > 1`` applies the SD3 timestep shift
    (converted to our t convention), concentrating steps near the noise end.
    """
    u = torch.linspace(0.0, 1.0, n_steps + 1, device=device, dtype=dtype)
    if shift == 1.0:
        return u
    return u / (shift - (shift - 1.0) * u)


def _sinkhorn_assignment(cost: torch.Tensor) -> torch.Tensor:
    """Argmax of the entropic-OT (Sinkhorn) plan for a square cost matrix.

    Returns, for each row i, the column j of its OT partner. Runs entirely on
    ``cost``'s device in the log domain (numerically stable). Uniform marginals;
    their constant log term cancels under argmax and is omitted.
    """
    n = cost.shape[0]
    reg = cost.mean().clamp_min(1e-8) * _OT_SINKHORN_REG
    log_K = -cost / reg
    log_u = torch.zeros(n, device=cost.device, dtype=log_K.dtype)
    log_v = torch.zeros(n, device=cost.device, dtype=log_K.dtype)
    for _ in range(_OT_SINKHORN_ITERS):
        log_u = -torch.logsumexp(log_K + log_v[None, :], dim=1)
        log_v = -torch.logsumexp(log_K + log_u[:, None], dim=0)
    log_plan = log_K + log_u[:, None] + log_v[None, :]
    return log_plan.argmax(dim=1)


def ot_pair_noise(noise: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Permute noise rows so noise[i] is target[i]'s minibatch-OT partner.

    Cost is Euclidean distance over flattened non-batch dims. The coupling is
    solved with an on-device entropic-OT (Sinkhorn) on chunks of at most
    ``_OT_MAX_CHUNK`` rows, then taken as a hard argmax assignment. This is the
    standard fast minibatch-OT used in OT-CFM; it is preferred over the
    exact Hungarian solver, which is O(n^3) on the host and requires a GPU->CPU
    copy per chunk. argmax is
    not guaranteed to be a strict permutation, so the noise marginal is
    approximately — not exactly — preserved (collisions are rare with a sharp
    plan).
    """
    B = noise.shape[0]
    noise_flat = noise.reshape(B, -1).float()
    target_flat = target.reshape(B, -1).float()
    perm = torch.empty(B, dtype=torch.long, device=noise.device)
    for start in range(0, B, _OT_MAX_CHUNK):
        end = min(start + _OT_MAX_CHUNK, B)
        cost = torch.cdist(target_flat[start:end], noise_flat[start:end])
        perm[start:end] = start + _sinkhorn_assignment(cost)
    return noise[perm]


def _expand_null_cond(null_cond: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
    """Broadcast a (cond_dim,) null embedding to conditioning's full shape."""
    view_shape = (1,) * (conditioning.ndim - 1) + (-1,)
    return null_cond.to(conditioning.dtype).view(view_shape).expand_as(conditioning)


def flow_matching_loss(
    forward_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    target: torch.Tensor,
    conditioning: torch.Tensor,
    opts: FlowOptions,
    null_cond: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rectified-flow training loss, shared by all heads.

    Args:
        forward_fn: velocity field ``(x_t, t, conditioning) -> v`` with t of
            shape (B,).
        target: clean latents, any shape (B, ...).
        conditioning: conditioning tensor, shape (B, ..., cond_dim); its
            batch dim must match target's.
        opts: training options (time sampling, OT coupling, cond dropout).
        null_cond: learned (cond_dim,) null embedding; required when
            ``opts.cond_dropout > 0``.
    """
    B = target.shape[0]
    noise = torch.randn_like(target)
    if opts.ot_coupling:
        noise = ot_pair_noise(noise, target)

    if opts.cond_dropout > 0.0:
        if null_cond is None:
            raise ValueError("cond_dropout > 0 requires the head to own a null_cond embedding")
        keep = torch.rand(B, device=conditioning.device) >= opts.cond_dropout
        keep = keep.view((B,) + (1,) * (conditioning.ndim - 1))
        conditioning = torch.where(keep, conditioning, _expand_null_cond(null_cond, conditioning))

    t = sample_flow_time(B, opts, target.device, target.dtype)
    t_b = t.view((B,) + (1,) * (target.ndim - 1))
    z_t = (1.0 - t_b) * noise + t_b * target
    v_target = target - noise
    v_pred = forward_fn(z_t, t, conditioning)
    return F.mse_loss(v_pred, v_target)


def flow_sample(
    forward_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    conditioning: torch.Tensor,
    *,
    n_steps: int = 5,
    solver: str = "euler",
    shift: float = 1.0,
    cfg_scale: float = 1.0,
    null_cond: torch.Tensor | None = None,
) -> torch.Tensor:
    """Integrate the flow ODE from initial noise ``x`` (t=0) to data (t=1).

    Supports a shifted time grid and classifier-free guidance. With
    ``cfg_scale != 1`` each velocity evaluation runs conditional and
    null-conditioned predictions in one batched forward.
    """
    if cfg_scale != 1.0 and null_cond is None:
        raise ValueError(
            "cfg_scale != 1 requires a head trained with cond_dropout > 0 (no null_cond embedding is available)."
        )

    B = x.shape[0]
    grid = make_time_grid(n_steps, shift, x.device, x.dtype)

    def velocity(x_cur: torch.Tensor, t_val: torch.Tensor) -> torch.Tensor:
        t = t_val.expand(B)
        if cfg_scale == 1.0:
            return forward_fn(x_cur, t, conditioning)
        assert null_cond is not None  # checked above; narrows for pyright
        null = _expand_null_cond(null_cond, conditioning)
        v_both = forward_fn(
            torch.cat([x_cur, x_cur], dim=0),
            torch.cat([t, t], dim=0),
            torch.cat([conditioning, null], dim=0),
        )
        v_cond, v_uncond = v_both.chunk(2, dim=0)
        return v_uncond + cfg_scale * (v_cond - v_uncond)

    for i in range(n_steps):
        dt = grid[i + 1] - grid[i]
        v = velocity(x, grid[i])
        if solver == "heun" and i < n_steps - 1:
            x_euler = x + v * dt
            v_next = velocity(x_euler, grid[i + 1])
            x = x + 0.5 * (v + v_next) * dt
        else:
            x = x + v * dt

    return x
