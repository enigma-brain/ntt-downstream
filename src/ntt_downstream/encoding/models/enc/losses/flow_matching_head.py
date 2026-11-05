"""Flow-matching head: AdaLN MLP predicting velocity for rectified-flow training.

A sibling of DiffLoss and EnergyLoss. This class has no MAR-specific
dependency — it is a generic conditional velocity field usable by any encoder that emits a per-token
conditioning vector z and a continuous tokenizer latent target.

Training uses rectified flow (conditional flow matching with independent
noise-data coupling by default; minibatch-OT coupling, logit-normal time
sampling, conditioning dropout for CFG, and a shifted inference grid are
available via ``FlowOptions`` — see ``flow_utils``).
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from .diffloss import FinalLayer, ResBlock, TimestepEmbedder
from .flow_utils import FlowOptions, flow_matching_loss, flow_sample


class FlowMatchingHead(nn.Module):
    """AdaLN MLP predicting velocity for rectified flow.

    Uses the same ResBlock/FinalLayer architecture as DiffLoss but with:
    - out_channels = token_dim (velocity only, no variance)
    - continuous t in [0, 1] scaled to [0, 999] for TimestepEmbedder
    """

    def __init__(
        self,
        token_dim: int,
        conditioning_dim: int,
        width: int = 256,
        depth: int = 3,
        flow_opts: FlowOptions | None = None,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.flow_opts = flow_opts or FlowOptions()

        self.time_embed = TimestepEmbedder(width)
        self.cond_embed = nn.Linear(conditioning_dim, width)
        self.input_proj = nn.Linear(token_dim, width)

        self.res_blocks = nn.ModuleList([ResBlock(width) for _ in range(depth)])
        self.final_layer = FinalLayer(width, token_dim)

        # CFG null embedding — registered only when conditioning dropout is
        # on, so default configs keep an unchanged state_dict.
        if self.flow_opts.cond_dropout > 0.0:
            self.null_cond = nn.Parameter(torch.zeros(conditioning_dim))
        else:
            self.register_parameter("null_cond", None)

        self._initialize_weights()

    def _initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(cast(nn.Linear, self.time_embed.mlp[0]).weight, std=0.02)
        nn.init.normal_(cast(nn.Linear, self.time_embed.mlp[2]).weight, std=0.02)

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

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Predict velocity.

        Args:
            x_t: noisy tokens (B_flat, token_dim)
            t: continuous time in [0, 1] (B_flat,)
            conditioning: transformer hidden states (B_flat, conditioning_dim)
        Returns:
            velocity (B_flat, token_dim)
        """
        t_scaled = t * 999.0
        x = self.input_proj(x_t)
        t_emb = self.time_embed(t_scaled)
        c_emb = self.cond_embed(conditioning)
        y = t_emb + c_emb

        for block in self.res_blocks:
            x = block(x, y)

        return self.final_layer(x, y)

    def compute_loss(self, target: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Rectified-flow training loss (options applied per ``self.flow_opts``).

        Note: targets arrive pre-flattened to (B_flat, D), so conditioning
        dropout for CFG is per token, not per image — consistent with
        per-token guidance at sampling time.
        """
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
        """Generate tokens via ODE integration (optionally shifted grid + CFG)."""
        B = conditioning.shape[0]
        x = torch.randn(B, self.token_dim, device=conditioning.device, dtype=conditioning.dtype) * temperature
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
