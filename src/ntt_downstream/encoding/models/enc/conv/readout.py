"""Gaussian mask readout for mapping 2D feature maps to neural responses.

Adapted from open-retina GaussianMaskReadout and MultiGaussianMaskReadout
(openretina/modules/readout/; https://github.com/open-retina/open-retina, MIT License,
Copyright (c) 2025 open-retina). Modified for ntt-downstream: simplified for static 2D
images.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianMaskReadout2D(nn.Module):
    """Per-session readout using 2D Gaussian spatial masks.

    Each neuron has a learnable isotropic 2D Gaussian mask that spatially
    integrates over the core feature map, combined with per-channel feature
    weights. Output is passed through softplus to ensure positive rates.

    Adapted from open-retina's ``GaussianMaskReadout`` (Hoefling et al., 2024).
    """

    def __init__(
        self,
        in_channels: int,
        spatial_size: tuple[int, int],
        n_neurons: int,
        gaussian_mean_scale: float = 6.0,
        gaussian_var_scale: float = 4.0,
        positive: bool = False,
        scale: bool = True,
        bias: bool = True,
        mask_l1_reg: float = 0.0,
        feature_weights_l1_reg: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.spatial_size = spatial_size
        self.n_neurons = n_neurons
        self.gaussian_mean_scale = gaussian_mean_scale
        self.gaussian_var_scale = gaussian_var_scale
        self.positive = positive
        self.mask_l1_reg = mask_l1_reg
        self.feature_weights_l1_reg = feature_weights_l1_reg

        # Learnable Gaussian parameters
        self.mask_mean = nn.Parameter(torch.zeros(n_neurons, 2))
        self.mask_log_var = nn.Parameter(torch.zeros(n_neurons))

        # Fixed spatial grid
        w, h = spatial_size
        self.register_buffer("grid", self._make_grid(n_neurons, w, h))

        # Per-channel feature weights
        self.features = nn.Parameter(torch.empty(1, in_channels, n_neurons))
        nn.init.normal_(self.features, mean=1.0 / in_channels, std=0.01)

        # Scale
        if scale:
            self.scale_param = nn.Parameter(torch.ones(n_neurons))
            nn.init.normal_(self.scale_param, mean=1.0, std=0.01)
        else:
            self.register_buffer("scale_param", torch.ones(n_neurons))

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(n_neurons))
        else:
            self.register_buffer("bias", torch.zeros(n_neurons))

    @staticmethod
    def _make_grid(n_neurons: int, w: int, h: int) -> torch.Tensor:
        grid_w = torch.linspace(-1 * w / max(w, h), 1 * w / max(w, h), w)
        grid_h = torch.linspace(-1 * h / max(w, h), 1 * h / max(w, h), h)
        xx, yy = torch.meshgrid(grid_w, grid_h, indexing="ij")
        grid = torch.stack([xx, yy], dim=2)[None, ...]  # (1, W, H, 2)
        return grid.expand(n_neurons, -1, -1, -1)  # (N, W, H, 2)

    def get_mask(self) -> torch.Tensor:
        """Compute normalized isotropic Gaussian masks. Returns (N, W, H)."""
        scaled_log_var = self.mask_log_var * self.gaussian_var_scale
        mask_var = torch.exp(torch.clamp(scaled_log_var, min=-20, max=20)).view(-1, 1, 1)

        pdf = cast(torch.Tensor, self.grid) - self.mask_mean.view(self.n_neurons, 1, 1, -1) * self.gaussian_mean_scale
        pdf = torch.sum(pdf**2, dim=-1) / (mask_var + 1e-8)
        pdf = torch.exp(-0.5 * torch.clamp(pdf, max=20))
        normalisation = torch.sum(pdf, dim=(1, 2), keepdim=True)
        return torch.nan_to_num(pdf / normalisation)

    @property
    def masks(self) -> torch.Tensor:
        """Gaussian masks reshaped for einsum: (W, H, N)."""
        return self.get_mask().permute(1, 2, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map feature maps to per-neuron rates.

        Args:
            x: (B, C, H, W) feature maps from core.
        Returns:
            (B, N) predicted rates (after softplus).
        """
        masks = self.masks  # (W, H, N)
        y = torch.einsum("bchw,hwn->bcn", x, masks)

        if self.positive:
            self.features.data.clamp_(0)
        y = (y * self.features).sum(1)  # (B, N)

        return F.softplus(y * self.scale_param + self.bias)

    def regularizer_components(self) -> dict[str, torch.Tensor]:
        """Return individual regularization terms as a dict."""
        zero = torch.tensor(0.0, device=self.mask_mean.device)
        mask_reg = zero
        if self.mask_l1_reg > 0:
            mask_reg = self.mask_l1_reg * (
                torch.exp(self.mask_log_var * self.gaussian_var_scale).sum()
                + (self.mask_mean * self.gaussian_mean_scale).pow(2).sum()
            )
        feature_reg = zero
        if self.feature_weights_l1_reg > 0:
            feature_reg = self.feature_weights_l1_reg * self.features.abs().sum()
        return {"mask": mask_reg, "features": feature_reg}

    def regularizer(self) -> torch.Tensor:
        c = self.regularizer_components()
        return c["mask"] + c["features"]


class MultiSessionGaussianReadout(nn.ModuleDict):
    """Multi-session wrapper: one ``GaussianMaskReadout2D`` per session."""

    def __init__(
        self,
        in_channels: int,
        spatial_size: tuple[int, int],
        session_info: dict[str, int],
        **readout_kwargs,
    ):
        super().__init__()
        for session_id, n_neurons in session_info.items():
            self.add_module(
                session_id,
                GaussianMaskReadout2D(
                    in_channels=in_channels,
                    spatial_size=spatial_size,
                    n_neurons=n_neurons,
                    **readout_kwargs,
                ),
            )

    def forward(self, x: torch.Tensor, session_key: str) -> torch.Tensor:
        return cast(GaussianMaskReadout2D, self[session_key])(x)

    def regularizer_components(self, session_key: str) -> dict[str, torch.Tensor]:
        return cast(GaussianMaskReadout2D, self[session_key]).regularizer_components()

    def regularizer(self, session_key: str) -> torch.Tensor:
        return cast(GaussianMaskReadout2D, self[session_key]).regularizer()
