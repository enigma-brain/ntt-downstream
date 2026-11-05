"""Regularizers and utility layers for convolutional encoding models.

Taken from open-retina (openretina/modules/layers/regularizers.py;
https://github.com/open-retina/open-retina, MIT License, Copyright (c) 2025 open-retina),
with minor modifications for ntt-downstream.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 3x3 Laplace filter kernel
LAPLACE_3x3 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).astype(np.float32)[None, None, ...]

EPS = 1e-6


class Laplace(nn.Module):
    """2D Laplace filter for spatial regularization of convolutional weights."""

    def __init__(self, padding: int | None = None):
        super().__init__()
        self.register_buffer("filter", torch.from_numpy(LAPLACE_3x3))
        self.padding_size = LAPLACE_3x3.shape[-1] // 2 if padding is None else padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, cast(torch.Tensor, self.filter), bias=None, padding=self.padding_size)


class FlatLaplaceL2norm(nn.Module):
    """Normalized Laplace regularizer for 2D convolutional weights.

    Computes ``|laplace(filters)|^2 / (|filters|^2 + eps)``.
    """

    def __init__(self, padding: int | None = None):
        super().__init__()
        self.laplace = Laplace(padding=padding)

    def forward(self, x: torch.Tensor, avg: bool = False) -> torch.Tensor:
        agg_fn = torch.mean if avg else torch.sum
        oc, ic, k1, k2 = x.size()
        return agg_fn(self.laplace(x.view(oc * ic, 1, k1, k2)).pow(2)) / (
            agg_fn(x.view(oc * ic, 1, k1, k2).pow(2)) + EPS
        )


class Bias2DLayer(nn.Module):
    """Learnable per-channel bias for 2D feature maps."""

    def __init__(self, channels: int, initial: float = 0.0):
        super().__init__()
        self.bias = nn.Parameter(torch.empty(1, channels, 1, 1).fill_(initial))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias
