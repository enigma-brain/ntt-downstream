"""Energy-score loss (EAR) for continuous-valued token generation.

Reference:
    Shao, Meng, Zhou. "Continuous Visual Autoregressive Generation via Score
    Maximization" (ICML 2025, arXiv:2505.07812). EAR replaces MAR's diffusion
    loss with an energy-score loss derived from strictly proper scoring rules,
    enabling single-pass sampling.

Adapted from the EAR code (https://github.com/shaochenze/EAR, ``models/scoreloss.py``;
MIT License, Copyright (c) 2024 Tianhong Li, inherited from MAR). ``modulate``,
``_ResBlock``, ``_FinalLayer`` and the weight initialisation originate in DiT
(https://github.com/facebookresearch/DiT) via MAR (https://github.com/LTH14/mar, MIT).
Modified for ntt-downstream: the generator takes noise as its input and is conditioned
on the backbone feature through a separate embedding, the energy score is computed
over two independent samples per target with masking and diagnostics, and alpha is
validated.

No count wrapper (softplus mapping latents to rates) is needed here because the
``tokenizer_wrapper`` decodes tokenizer latents to rates outside this module.

Design notes:
- The generator is an implicit MLP with AdaLN blocks: input is noise
  ``eps ~ U[-0.5, 0.5]^noise_channels``, output is a sample in the tokenizer
  latent space conditioned on a backbone-derived feature ``z``.
- The energy-score training loss uses TWO independent samples per target so
  the diversity term ``||x1 - x2||^alpha`` is unbiased.
- alpha is restricted to [1, 2) — the paper shows alpha < 1 causes gradient
  explosion and alpha = 2 (proper but NOT strictly proper) collapses to mean.
"""

from __future__ import annotations

import warnings
from typing import cast

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class _ResBlock(nn.Module):
    """AdaLN-modulated residual MLP block."""

    def __init__(self, channels: int):
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 3 * channels))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift, scale)
        h = self.mlp(h)
        return x + gate * h


class _FinalLayer(nn.Module):
    def __init__(self, model_channels: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 2 * model_channels))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class EnergyGenerator(nn.Module):
    """Implicit generator ``(noise, conditioning) -> sample``.

    Mirrors EAR's ``SimpleMLPAdaLN`` but drops the timestep embedding — EAR has
    no diffusion state. Noise is fed both as the residual-stream input and
    as an additional embedding into the AdaLN conditioning (following EAR).
    """

    def __init__(
        self,
        noise_channels: int,
        target_channels: int,
        z_channels: int,
        width: int,
        depth: int,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.noise_channels = noise_channels
        self.target_channels = target_channels
        self.z_channels = z_channels
        self.width = width
        self.grad_checkpointing = grad_checkpointing

        self.input_proj = nn.Linear(noise_channels, width)
        self.noise_embed = nn.Linear(noise_channels, width)
        self.cond_embed = nn.Linear(z_channels, width)
        self.res_blocks = nn.ModuleList(_ResBlock(width) for _ in range(depth))
        self.final_layer = _FinalLayer(width, target_channels)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero-init AdaLN + final layer so each block starts as identity.
        for block in self.res_blocks:
            # ModuleList iteration yields plain ``Module``; narrow for pyright.
            adaln = cast(nn.Linear, cast(nn.Sequential, block.adaLN_modulation)[-1])
            nn.init.constant_(adaln.weight, 0)
            nn.init.constant_(adaln.bias, 0)
        final_adaln = cast(nn.Linear, self.final_layer.adaLN_modulation[-1])
        nn.init.constant_(final_adaln.weight, 0)
        nn.init.constant_(final_adaln.bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, noise: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        weight_dtype = self.input_proj.weight.dtype
        noise = noise.to(weight_dtype)
        c = c.to(weight_dtype)

        x = self.input_proj(noise)
        y = self.noise_embed(noise) + self.cond_embed(c)

        if self.grad_checkpointing and not torch.jit.is_scripting():  # pyright: ignore[reportPrivateImportUsage] # stub omits re-export
            for block in self.res_blocks:
                x = checkpoint(block, x, y, use_reentrant=False)
        else:
            for block in self.res_blocks:
                x = block(x, y)
        return self.final_layer(x, y)


class EnergyLoss(nn.Module):
    """Energy-score loss module.

    Loss for target ``y`` and conditioning ``z``:

        L = ||x1 - y||^alpha + ||x2 - y||^alpha - train_temperature * ||x1 - x2||^alpha

    where ``x1, x2 ~ G(., z)`` are two independent samples from the implicit
    generator. The third term is the strict-propriety / diversity term:
    removing it (train_temperature=0) makes the loss a plain MSE-alpha and
    collapses the generator to a point estimate.

    Args:
        target_channels: dim of the tokenizer latent ``y``.
        z_channels: dim of the conditioning ``z``.
        width: generator MLP hidden dim. EAR defaults 1024 (image); small
            neural models can use 128–256.
        depth: number of AdaLN residual blocks. EAR defaults 3.
        noise_channels: dim of the uniform noise input. EAR default 64.
        alpha: energy-score exponent. Must be in [1, 2). EAR found alpha < 1
            unstable and alpha = 2 collapses. Typical values 1.0, 1.25, 1.5.
        train_temperature: multiplier on the diversity term during training.
            Default 1.0 (pure energy score). Raise it to push diversity harder
            (useful if the generator is collapsing).
        grad_checkpointing: enable residual-block checkpointing.
    """

    def __init__(
        self,
        target_channels: int,
        z_channels: int,
        width: int = 256,
        depth: int = 3,
        noise_channels: int = 64,
        alpha: float = 1.0,
        train_temperature: float = 1.0,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        if not (1.0 <= alpha < 2.0):
            raise ValueError(
                f"alpha must be in [1.0, 2.0); got {alpha}. Per EAR (arXiv:2505.07812) "
                "Sec. 4: alpha < 1 causes gradient explosion and alpha = 2 is proper "
                "but NOT strictly proper and collapses to mean."
            )
        if alpha == 1.0:
            warnings.warn(
                "alpha=1.0 is on the boundary of strict propriety; if training is unstable try alpha=1.25 or 1.5.",
                stacklevel=2,
            )
        if train_temperature <= 0.0:
            raise ValueError(
                f"train_temperature must be > 0; got {train_temperature}. "
                "Setting it to 0 removes the diversity term and collapses the generator."
            )

        self.target_channels = target_channels
        self.z_channels = z_channels
        self.noise_channels = noise_channels
        self.alpha = alpha
        self.train_temperature = train_temperature
        self.net = EnergyGenerator(
            noise_channels=noise_channels,
            target_channels=target_channels,
            z_channels=z_channels,
            width=width,
            depth=depth,
            grad_checkpointing=grad_checkpointing,
        )

    def _pow_norm(self, diff: torch.Tensor) -> torch.Tensor:
        """``||diff||_2^alpha`` along last dim, with eps floor to keep the
        gradient finite at diff=0 when alpha < 2."""
        norm = torch.linalg.norm(diff, ord=2, dim=-1)
        return torch.pow(norm.clamp_min(1e-6), self.alpha)

    def forward(
        self,
        target: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ):
        """Compute the energy-score loss.

        Args:
            target: (B, target_channels) ground-truth tokenizer latent.
            z: (B, z_channels) conditioning.
            mask: (B,) optional per-sample mask. Positions with mask=0 are
                excluded from the loss average.
            return_diagnostics: if True, return (loss, diag) where diag contains
                'diversity_abs' (mean ||x1-x2||, alpha-independent for cross-alpha
                comparison) and 'diversity_ratio' (||x1-x2|| / (||x1-y||+||x2-y||)/2,
                alpha-invariant since numerator and denominator share the alpha power).

        Returns:
            Scalar loss, or (scalar loss, diag dict).
        """
        B = target.shape[0]
        weight_dtype = self.net.input_proj.weight.dtype
        noise = torch.rand(2 * B, self.noise_channels, device=z.device, dtype=weight_dtype) - 0.5
        z_dup = torch.cat([z, z], dim=0)
        samples = self.net(noise, z_dup)  # (2B, D)
        x1, x2 = samples[:B], samples[B:]

        target = target.to(x1.dtype)
        d1 = self._pow_norm(x1 - target)
        d2 = self._pow_norm(x2 - target)
        dd = self._pow_norm(x1 - x2)

        per_sample = d1 + d2 - self.train_temperature * dd  # (B,)

        if mask is not None:
            mask = mask.to(per_sample.dtype)
            loss = (per_sample * mask).sum() / mask.sum().clamp_min(1.0)
        else:
            loss = per_sample.mean()

        if return_diagnostics:
            with torch.no_grad():
                denom = (d1.mean() + d2.mean()).clamp_min(1e-8)
                # Recover raw L2 distance from ||.||^alpha so the metric is
                # comparable across runs with different alpha.
                dd_l2 = torch.linalg.norm(x1 - x2, ord=2, dim=-1).mean()
                diag = {
                    "diversity_abs": dd_l2.detach(),
                    "diversity_ratio": (2 * dd.mean() / denom).detach(),
                }
            return loss, diag
        return loss

    @torch.no_grad()
    def sample(
        self,
        z: torch.Tensor,
        n_samples: int = 1,
        temperature: float = 1.0,
        cfg: float = 1.0,
    ) -> torch.Tensor:
        """Draw samples from the conditional generator.

        Args:
            z: (B, z_channels) conditioning.
            n_samples: samples per conditioning vector. Output is batched with
                samples for z[i] at indices [i*n_samples, (i+1)*n_samples).
                NOTE: this parameter is specific to EnergyLoss — DiffLoss.sample
                does not accept it. Do not pass it at polymorphic call sites.
            temperature: scalar multiplier on the uniform noise magnitude.
                Defaults to 1.0 (standard U[-0.5, 0.5] as in EAR).
            cfg: classifier-free-guidance scale. Retained for drop-in compat
                with DiffLoss.sample — NOT implemented. A value != 1.0
                emits a warning and is treated as 1.0.

        Returns:
            (B * n_samples, target_channels) samples.
        """
        if cfg != 1.0:
            warnings.warn(
                f"EnergyLoss.sample: cfg {cfg!r} not implemented. Proceeding with unconditional-identity (cfg=1.0).",
                stacklevel=2,
            )
        if n_samples > 1:
            z = z.repeat_interleave(n_samples, dim=0)
        weight_dtype = self.net.input_proj.weight.dtype
        noise = torch.rand(z.shape[0], self.noise_channels, device=z.device, dtype=weight_dtype) - 0.5
        noise = noise * float(temperature)
        return self.net(noise, z)
