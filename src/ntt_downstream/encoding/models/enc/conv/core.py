"""Stacked 2D convolutional core for static image encoding.

Adapted from open-retina SimpleCoreWrapper (openretina/modules/core/base_core.py;
https://github.com/open-retina/open-retina, MIT License, Copyright (c) 2025 open-retina).
Modified for ntt-downstream: simplified for static 2D images (no temporal dimension).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import cast

import torch
import torch.nn as nn

from .layers import Bias2DLayer, FlatLaplaceL2norm


class Stacked2DCore(nn.Module):
    """Multi-layer 2D convolutional core with batch normalization and ELU activation.

    Each layer: Conv2D -> BatchNorm2d -> Bias -> ELU (-> optional Dropout).

    Regularization:
      - Spatial Laplace on the first layer (``gamma_input``).
      - Group sparsity on hidden layers (``gamma_hidden``).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int],
        spatial_kernel_sizes: list[int],
        gamma_input: float = 0.0,
        gamma_hidden: float = 0.0,
        input_padding: int | str = 0,
        hidden_padding: int | str = "same",
        dropout_rate: float = 0.0,
    ):
        if len(hidden_channels) != len(spatial_kernel_sizes):
            raise ValueError(
                f"hidden_channels and spatial_kernel_sizes must have same length. "
                f"Got {len(hidden_channels)} and {len(spatial_kernel_sizes)}"
            )

        super().__init__()
        self.gamma_input = gamma_input
        self.gamma_hidden = gamma_hidden
        self._input_regularizer = FlatLaplaceL2norm(padding=0)

        channels = [in_channels, *hidden_channels]
        self.features = nn.Sequential()

        for i, (c_in, c_out) in enumerate(zip(channels[:-1], channels[1:], strict=True)):
            layer: dict[str, nn.Module] = OrderedDict()
            padding = input_padding if i == 0 else hidden_padding
            if padding is True:
                padding = "same"
            elif padding is False:
                padding = 0

            layer["conv"] = nn.Conv2d(c_in, c_out, spatial_kernel_sizes[i], padding=padding, bias=False)
            layer["norm"] = nn.BatchNorm2d(c_out, momentum=0.1)
            layer["bias"] = Bias2DLayer(c_out)
            layer["nonlin"] = nn.ELU()
            if dropout_rate > 0.0:
                layer["dropout"] = nn.Dropout2d(p=dropout_rate)

            self.features.add_module(f"layer{i}", nn.Sequential(layer))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

    @property
    def out_channels(self) -> int:
        for module in reversed(list(self.features.modules())):
            if isinstance(module, nn.Conv2d):
                return module.out_channels
        raise RuntimeError("No Conv2d found in features")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def spatial_laplace(self) -> torch.Tensor:
        """Laplace regularization on the first layer's spatial weights."""
        # Each feature layer is a Sequential wrapping a conv-bundle with a ``.conv``;
        # ModuleList/Sequential indexing only yields ``Module`` for pyright.
        weight = cast(nn.Conv2d, self.features[0].conv).weight
        return self._input_regularizer(weight, avg=False)

    def group_sparsity(self) -> torch.Tensor:
        """Group sparsity regularization on hidden layers."""
        if len(self.features) <= 1:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        sparsities: list[torch.Tensor] = []
        for layer in list(self.features)[1:]:
            w = cast(nn.Conv2d, layer.conv).weight
            val = w.pow(2).sum([2, 3]).sqrt().sum(1) / torch.sqrt(1e-8 + w.pow(2).sum([1, 2, 3]))
            sparsities.append(val.sum())
        return torch.sum(torch.stack(sparsities))

    def regularizer(self) -> torch.Tensor:
        reg: torch.Tensor = torch.tensor(0.0, device=next(self.parameters()).device)
        if self.gamma_input != 0.0:
            reg = reg + self.gamma_input * self.spatial_laplace()
        if self.gamma_hidden != 0.0:
            reg = reg + self.gamma_hidden * self.group_sparsity()
        return reg
