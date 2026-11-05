"""Full-grid denoisers for tokenizer latent trajectories.

This head is the non-MAR generative path for latents shaped ``(B, S, T, D)``.
It reuses the Perceiver tokenizer's M-space block: attention mixes the S slots
at each time bin, and the temporal mixer operates along T for each slot.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .diffloss import TimestepEmbedder
from .flow_utils import FlowOptions, flow_matching_loss, flow_sample


class FullGridFlowMatchingHead(nn.Module):
    """Conditional rectified-flow head over the full ``(S, T)`` latent grid.

    Args:
        latent_dim: Target latent feature dimension D.
        conditioning_dim: Image-conditioning feature dimension.
        hidden_channels: Internal channel dimension used by MSpaceBlock.
        num_blocks: Number of Perceiver M-space blocks.
        num_heads: Number of attention heads in each block.
        mlp_ratio: Channel MLP expansion ratio.
        max_spatial: Maximum S supported by learned slot embeddings.
        max_time: Maximum T supported by learned time embeddings.
        use_s4: Whether to use the tokenizer block's S4/FFT time mixer.
        kernel_size: Depthwise conv kernel size when ``use_s4=False``.
        flow_opts: Rectified-flow training options (see ``flow_utils``).
    """

    def __init__(
        self,
        latent_dim: int,
        conditioning_dim: int,
        hidden_channels: int = 64,
        num_blocks: int = 4,
        num_heads: int = 4,
        mlp_ratio: int = 2,
        max_spatial: int = 64,
        max_time: int = 256,
        use_s4: bool = False,
        kernel_size: int = 7,
        flow_opts: FlowOptions | None = None,
    ):
        super().__init__()
        if hidden_channels % num_heads != 0:
            raise ValueError(f"hidden_channels={hidden_channels} must be divisible by num_heads={num_heads}")

        try:
            from ntt.contrib.tokenizer.perceiver.blocks import MSpaceBlock
        except ImportError as exc:  # pragma: no cover - depends on external ntt checkout
            raise ImportError(
                "FullGridFlowMatchingHead requires ntt.contrib.tokenizer.perceiver.blocks.MSpaceBlock. "
                "Use the ntt perceiver-tokenizer branch/environment."
            ) from exc

        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.max_spatial = max_spatial
        self.max_time = max_time
        self.flow_opts = flow_opts or FlowOptions()

        # CFG null embedding — registered only when conditioning dropout is
        # on, so default configs keep an unchanged state_dict.
        if self.flow_opts.cond_dropout > 0.0:
            self.null_cond = nn.Parameter(torch.zeros(conditioning_dim))
        else:
            self.register_parameter("null_cond", None)

        self.input_proj = nn.Linear(latent_dim, hidden_channels)
        self.cond_proj = nn.Linear(conditioning_dim, hidden_channels)
        self.time_embed = TimestepEmbedder(hidden_channels)
        self.slot_embed = nn.Parameter(torch.zeros(1, max_spatial, hidden_channels, 1))
        self.temporal_embed = nn.Parameter(torch.zeros(1, 1, hidden_channels, max_time))

        self.blocks = nn.ModuleList(
            [
                MSpaceBlock(
                    hidden_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_s4=use_s4,
                    kernel_size=kernel_size,
                    bidirectional=True,
                    use_layernorm=True,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_proj = nn.Linear(hidden_channels, latent_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in (self.input_proj, self.cond_proj, self.output_proj):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.slot_embed, std=0.02)
        nn.init.normal_(self.temporal_embed, std=0.02)
        nn.init.zeros_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    def _check_shape(self, x: torch.Tensor, conditioning: torch.Tensor) -> tuple[int, int, int, int]:
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape (B,S,T,D), got {tuple(x.shape)}")
        if conditioning.shape[:3] != x.shape[:3]:
            raise ValueError(
                "Conditioning grid must share (B,S,T) with target/noisy latents: "
                f"conditioning={tuple(conditioning.shape)}, x={tuple(x.shape)}"
            )
        B, S, T, D = x.shape
        if D != self.latent_dim:
            raise ValueError(f"Expected latent dim {self.latent_dim}, got {D}")
        if S > self.max_spatial or T > self.max_time:
            raise ValueError(
                f"Latent grid S={S}, T={T} exceeds configured max_spatial={self.max_spatial}, max_time={self.max_time}"
            )
        return B, S, T, D

    def forward(self, x_t: torch.Tensor, tau: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Predict the flow velocity for a noisy full latent grid."""
        B, S, T, _D = self._check_shape(x_t, conditioning)

        weight_dtype = self.input_proj.weight.dtype
        x_t = x_t.to(weight_dtype)
        conditioning = conditioning.to(weight_dtype)
        tau = tau.to(weight_dtype)

        h = self.input_proj(x_t) + self.cond_proj(conditioning)  # (B,S,T,H)
        h = h.permute(0, 1, 3, 2).contiguous()  # (B,S,H,T), MSpaceBlock layout

        t_emb = self.time_embed(tau * 999.0).view(B, 1, self.hidden_channels, 1)
        h = h + t_emb + self.slot_embed[:, :S] + self.temporal_embed[:, :, :, :T]

        for block in self.blocks:
            h = block(h)

        h = h.permute(0, 1, 3, 2).contiguous()  # (B,S,T,H)
        return self.output_proj(h)

    def compute_loss(self, target: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Rectified-flow loss on the full grid (options per ``self.flow_opts``)."""
        return flow_matching_loss(self.forward, target, conditioning, self.flow_opts, self.null_cond)

    @torch.no_grad()
    def sample(
        self,
        conditioning: torch.Tensor,
        n_steps: int = 5,
        temperature: float = 1.0,
        solver: str = "euler",
        shift: float = 1.0,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """Generate a full latent grid via ODE integration."""
        if conditioning.ndim != 4:
            raise ValueError(f"Expected conditioning with shape (B,S,T,D), got {tuple(conditioning.shape)}")
        B, S, T, _ = conditioning.shape
        x = torch.randn(B, S, T, self.latent_dim, device=conditioning.device, dtype=conditioning.dtype) * temperature
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
