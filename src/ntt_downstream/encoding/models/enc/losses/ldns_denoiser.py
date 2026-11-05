"""Sequence-aware denoisers for compress_n tokenizers.

Adapted from LDNS (Latent Diffusion for Neural Spiking data; https://github.com/mackelab/LDNS,
MIT License, Copyright (c) 2024 mackelab): ``TimestepEmbedder`` and
``TemporalDenoiserBlock`` follow its timestep embedding and ``DenoiserBlock``. These
modules replace the pointwise MLP denoisers with temporal-aware architectures that
preserve and exploit temporal structure in latent trajectories. Modified for
ntt-downstream: the S4/FFTConv time mixer is replaced by a depthwise 1D convolution.

Each block alternates:
  - Time mixer: depthwise 1D convolution across T (per channel)
  - Channel mixer: MLP across C (per time step)
Both conditioned via AdaLN on diffusion timestep + image conditioning.
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn

from .diffusion import create_diffusion
from .flow_utils import FlowOptions, flow_matching_loss, flow_sample

# ---------------------------------------------------------------------------
# Timestep embedding (shared with diffloss.py, kept local to avoid coupling)
# ---------------------------------------------------------------------------


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(cast(nn.Linear, self.mlp[0]).weight.dtype)
        return self.mlp(t_freq)


# ---------------------------------------------------------------------------
# Core blocks
# ---------------------------------------------------------------------------


class TemporalDenoiserBlock(nn.Module):
    """Single denoiser block with temporal mixing + channel mixing + AdaLN.

    Mirrors LDNS DenoiserBlock but replaces S4/FFTConv with depthwise 1D conv.

    Args:
        hidden_channels: Number of hidden channels (C).
        cond_dim: Dimension of the conditioning vector (timestep + image).
        kernel_size: Kernel size for the depthwise temporal convolution.
    """

    def __init__(self, hidden_channels: int, cond_dim: int, kernel_size: int = 15):
        super().__init__()
        C = hidden_channels

        # Norms (affine=False because AdaLN provides shift/scale)
        self.norm1 = nn.InstanceNorm1d(C, affine=False)
        self.norm2 = nn.InstanceNorm1d(C, affine=False)

        # Time mixer: depthwise conv
        self.time_mixer = nn.Conv1d(C, C, kernel_size, padding="same", groups=C)

        # Channel mixer: pointwise MLP applied per time step
        self.channel_mixer = nn.Sequential(
            nn.Linear(C, C * 2),
            nn.GELU(),
            nn.Linear(C * 2, C),
        )

        self.act1 = nn.GELU()

        # AdaLN: conditioning → 6*C (shift/scale/gate for each mixer)
        self.ada_ln = nn.Sequential(
            nn.GELU(),
            nn.Linear(cond_dim, C * 6, bias=True),
        )
        # Zero-init so block is identity at init (like LDNS)
        ada_out = cast(nn.Linear, self.ada_ln[-1])
        ada_out.weight.data.zero_()
        ada_out.bias.data.zero_()

    @staticmethod
    def _affine(x, shift, scale):
        # x: (B, C, L), shift/scale: (B, C)
        return x * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)

    def forward(self, x, cond):
        """
        Args:
            x: (B, C, T) hidden states
            cond: (B, cond_dim) conditioning vector
        Returns:
            (B, C, T)
        """
        shift_tm, scale_tm, gate_tm, shift_cm, scale_cm, gate_cm = self.ada_ln(cond).chunk(6, dim=1)

        # --- Time mixer ---
        y = self.norm1(x)
        y = self._affine(y, shift_tm, scale_tm)
        y = self.time_mixer(y)
        x = x + gate_tm.unsqueeze(-1) * self.act1(y)

        # --- Channel mixer ---
        y = self.norm2(x)
        y = self._affine(y, shift_cm, scale_cm)
        y = y.permute(0, 2, 1)  # (B, C, L) -> (B, L, C)
        y = self.channel_mixer(y)
        y = y.permute(0, 2, 1)  # (B, L, C) -> (B, C, L)
        x = x + gate_cm.unsqueeze(-1) * y

        return x


# ---------------------------------------------------------------------------
# Sequence-aware flow matching head
# ---------------------------------------------------------------------------


class LDNSFlowMatchingHead(nn.Module):
    """Flow matching head that operates on full latent trajectories.

    Instead of denoising each time step independently, this processes
    ``(B, latent_n, T)`` sequences through temporal + channel mixing blocks.

    Args:
        latent_n: Number of latent channels (C_in).
        hidden_channels: Internal hidden dimension (C).
        conditioning_dim: Dimension of the per-chunk conditioning vector.
        num_blocks: Number of TemporalDenoiserBlocks.
        kernel_size: Temporal convolution kernel size.
        flow_opts: Rectified-flow training options (see ``flow_utils``).
    """

    def __init__(
        self,
        latent_n: int,
        hidden_channels: int,
        conditioning_dim: int,
        num_blocks: int = 4,
        kernel_size: int = 15,
        flow_opts: FlowOptions | None = None,
    ):
        super().__init__()
        self.latent_n = latent_n
        self.hidden_channels = hidden_channels
        self.flow_opts = flow_opts or FlowOptions()
        cond_dim = hidden_channels // 4

        # CFG null embedding — registered only when conditioning dropout is
        # on, so default configs keep an unchanged state_dict.
        if self.flow_opts.cond_dropout > 0.0:
            self.null_cond = nn.Parameter(torch.zeros(conditioning_dim))
        else:
            self.register_parameter("null_cond", None)

        # Conditioning embeddings
        self.t_emb = TimestepEmbedder(cond_dim)
        self.c_emb = nn.Sequential(
            nn.Linear(conditioning_dim, cond_dim, bias=True),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim, bias=True),
        )

        # Input/output projections (grouped like LDNS when possible)
        in_groups = latent_n if hidden_channels % latent_n == 0 and hidden_channels > latent_n else 1
        self.conv_in = nn.Conv1d(latent_n, hidden_channels, 1, groups=in_groups)
        self.conv_out = nn.Conv1d(hidden_channels, latent_n, 1, groups=in_groups)

        # Zero-init output conv so block starts as identity
        nn.init.zeros_(self.conv_out.weight)
        if self.conv_out.bias is not None:
            nn.init.zeros_(self.conv_out.bias)

        # Denoiser blocks
        self.blocks = nn.ModuleList(
            [TemporalDenoiserBlock(hidden_channels, cond_dim, kernel_size) for _ in range(num_blocks)]
        )

        self._initialize_weights()

    def _initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.t_emb.apply(_basic_init)
        self.c_emb.apply(_basic_init)
        nn.init.normal_(cast(nn.Linear, self.t_emb.mlp[0]).weight, std=0.02)
        nn.init.normal_(cast(nn.Linear, self.t_emb.mlp[2]).weight, std=0.02)

    def forward(self, x_t, t, conditioning):
        """Predict velocity field.

        Args:
            x_t: (B, latent_n, T) noisy trajectory
            t: (B,) continuous time in [0, 1]
            conditioning: (B, conditioning_dim) per-chunk conditioning
        Returns:
            (B, latent_n, T) predicted velocity
        """
        t_scaled = t * 999.0
        cond = self.t_emb(t_scaled) + self.c_emb(conditioning)

        x = self.conv_in(x_t)
        for block in self.blocks:
            x = block(x, cond)
        return self.conv_out(x)

    def compute_loss(self, target, conditioning):
        """Rectified-flow training loss (options per ``self.flow_opts``).

        Args:
            target: (B, latent_n, T) ground-truth latent trajectory
            conditioning: (B, conditioning_dim) per-chunk conditioning
        Returns:
            scalar loss
        """
        return flow_matching_loss(self.forward, target, conditioning, self.flow_opts, self.null_cond)

    @torch.no_grad()
    def sample(self, conditioning, T, n_steps=5, temperature=1.0, solver="euler", shift=1.0, cfg_scale=1.0):
        """Generate latent trajectories via ODE integration.

        Args:
            conditioning: (B, conditioning_dim)
            T: temporal length of the output sequence
            n_steps: number of ODE integration steps
            temperature: scale for initial noise
            solver: "euler" or "heun"
            shift: inference time-grid shift (1.0 = uniform)
            cfg_scale: classifier-free guidance scale (1.0 = off; requires a
                head trained with ``cond_dropout > 0``)
        Returns:
            (B, latent_n, T)
        """
        B = conditioning.shape[0]
        x = torch.randn(B, self.latent_n, T, device=conditioning.device, dtype=conditioning.dtype) * temperature
        return flow_sample(
            self.forward,
            x,
            conditioning,
            n_steps=n_steps,
            solver=solver,
            shift=shift,
            cfg_scale=cfg_scale,
            null_cond=self.null_cond,
        )


# ---------------------------------------------------------------------------
# Sequence-aware DDPM diffusion loss
# ---------------------------------------------------------------------------


class _SequenceNetWrapper(nn.Module):
    """Wraps the sequence denoiser to match gaussian_diffusion's call convention.

    gaussian_diffusion calls ``model(x_t, t, **model_kwargs)`` where
    ``model_kwargs`` contains ``c`` (conditioning). This wrapper delegates to
    the actual sequence network.

    For learned variance, outputs ``(B, 2 * latent_n, T)``.
    """

    def __init__(
        self,
        latent_n: int,
        hidden_channels: int,
        conditioning_dim: int,
        num_blocks: int = 4,
        kernel_size: int = 15,
    ):
        super().__init__()
        self.latent_n = latent_n
        cond_dim = hidden_channels // 4

        self.t_emb = TimestepEmbedder(cond_dim)
        self.c_emb = nn.Sequential(
            nn.Linear(conditioning_dim, cond_dim, bias=True),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim, bias=True),
        )

        in_groups = latent_n if hidden_channels % latent_n == 0 and hidden_channels > latent_n else 1
        self.conv_in = nn.Conv1d(latent_n, hidden_channels, 1, groups=in_groups)
        # Output 2x channels for learned variance
        self.conv_out = nn.Conv1d(hidden_channels, latent_n * 2, 1)

        self.blocks = nn.ModuleList(
            [TemporalDenoiserBlock(hidden_channels, cond_dim, kernel_size) for _ in range(num_blocks)]
        )

        self._initialize_weights()

    def _initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.t_emb.apply(_basic_init)
        self.c_emb.apply(_basic_init)
        nn.init.normal_(cast(nn.Linear, self.t_emb.mlp[0]).weight, std=0.02)
        nn.init.normal_(cast(nn.Linear, self.t_emb.mlp[2]).weight, std=0.02)

        # Zero-init output
        nn.init.zeros_(self.conv_out.weight)
        if self.conv_out.bias is not None:
            nn.init.zeros_(self.conv_out.bias)

        # Zero-init AdaLN in blocks
        for block in self.blocks:
            ada_out = cast(nn.Linear, cast(nn.Sequential, block.ada_ln)[-1])
            nn.init.zeros_(ada_out.weight)
            nn.init.zeros_(ada_out.bias)

    def forward(self, x, t, c):
        """
        Args:
            x: (B, latent_n, T) noisy input
            t: (B,) integer timesteps
            c: (B, conditioning_dim) conditioning
        Returns:
            (B, 2 * latent_n, T) epsilon prediction + variance
        """
        weight_dtype = self.conv_in.weight.dtype
        x = x.to(weight_dtype)
        c = c.to(weight_dtype)

        cond = self.t_emb(t.to(weight_dtype)) + self.c_emb(c)

        h = self.conv_in(x)
        for block in self.blocks:
            h = block(h, cond)
        return self.conv_out(h)

    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, : self.latent_n], model_out[:, self.latent_n :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class LDNSDiffLoss(nn.Module):
    """Sequence-aware DDPM diffusion loss for compress_n tokenizers.

    Uses the same gaussian_diffusion infrastructure as DiffLoss but with a
    temporal sequence network instead of a pointwise MLP.

    Args:
        latent_n: Number of latent channels.
        hidden_channels: Internal hidden dimension.
        conditioning_dim: Dimension of per-chunk conditioning.
        num_blocks: Number of TemporalDenoiserBlocks.
        kernel_size: Temporal convolution kernel size.
        num_sampling_steps: Respaced timesteps for generation (e.g. "100").
    """

    def __init__(
        self,
        latent_n: int,
        hidden_channels: int,
        conditioning_dim: int,
        num_blocks: int = 4,
        kernel_size: int = 15,
        num_sampling_steps: str = "100",
    ):
        super().__init__()
        self.latent_n = latent_n

        self.net = _SequenceNetWrapper(
            latent_n=latent_n,
            hidden_channels=hidden_channels,
            conditioning_dim=conditioning_dim,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
        )

        self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule="cosine")
        self.gen_diffusion = create_diffusion(timestep_respacing=num_sampling_steps, noise_schedule="cosine")

    def forward(self, target, conditioning, mask=None):
        """Compute training loss.

        Args:
            target: (B, latent_n, T) ground-truth latent trajectory
            conditioning: (B, conditioning_dim) per-chunk conditioning
            mask: optional (B,) mask for loss weighting
        Returns:
            scalar loss
        """
        t = torch.randint(0, self.train_diffusion.num_timesteps, (target.shape[0],), device=target.device)
        model_kwargs = {"c": conditioning}
        loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
        loss = loss_dict["loss"]
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return loss.mean()

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        device = next(self.parameters()).device
        self.train_diffusion.to(device)
        self.gen_diffusion.to(device)
        return result

    @torch.no_grad()
    def sample(self, conditioning, T, temperature=1.0, cfg=1.0):
        """Generate latent trajectories.

        Args:
            conditioning: (B, conditioning_dim)
            T: temporal length of the output sequence
            temperature: noise scale
            cfg: classifier-free guidance scale (1.0 = no guidance)
        Returns:
            (B, latent_n, T)
        """
        weight_dtype = next(self.parameters()).dtype
        B = conditioning.shape[0]

        if cfg != 1.0:
            noise = torch.randn(B // 2, self.latent_n, T, device=conditioning.device, dtype=weight_dtype)
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = {"c": conditioning.to(weight_dtype), "cfg_scale": cfg}
            sample_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(B, self.latent_n, T, device=conditioning.device, dtype=weight_dtype)
            model_kwargs = {"c": conditioning.to(weight_dtype)}
            sample_fn = self.net.forward

        sampled = self.gen_diffusion.p_sample_loop(
            sample_fn,
            noise.shape,
            noise,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            temperature=temperature,
            device=conditioning.device,
        )

        return sampled
