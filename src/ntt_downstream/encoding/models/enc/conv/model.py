"""Convolutional Core + Gaussian Readout encoding model.

Architecture adapted from the open-retina Goldin 2022 model
(https://github.com/open-retina/open-retina, MIT License, Copyright (c) 2025
open-retina; Goldin et al., 2022):
a stacked 2D convolutional core extracts spatial features from static images,
and a per-session Gaussian mask readout predicts a single firing rate per neuron.

Currently only supports MSE loss on time-averaged responses. Temporal response
prediction (e.g. Poisson on full time bins) requires a temporal component and
is not yet implemented.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import Stacked2DCore
from .readout import MultiSessionGaussianReadout


class ConvCoreReadout(nn.Module):
    """Conv core + Gaussian readout for static image -> neural rate prediction.

    Interface mirrors ``ImageEncodingHiera``: ``forward(images, key, responses=...)``
    returns ``{"loss": ...}`` during training or ``(B, N)`` rates during inference.

    Only ``mse_rate`` loss is supported (trained on time-averaged responses).
    Temporal response prediction (e.g. Poisson on per-bin responses) requires a
    temporal component and will raise ``NotImplementedError``.
    """

    def __init__(
        self,
        session_info: dict[str, int],
        in_channels: int = 3,
        image_size: tuple[int, int] = (112, 112),
        hidden_channels: list[int] | None = None,
        spatial_kernel_sizes: list[int] | None = None,
        n_time_bins: int = 200,
        loss_mode: str = "mse_rate",
        # Core regularization
        gamma_input: float = 0.5,
        gamma_hidden: float = 0.0,
        input_padding: int | str = 0,
        hidden_padding: int | str = "same",
        core_dropout: float = 0.0,
        # Readout params
        readout_scale: bool = True,
        readout_bias: bool = True,
        readout_gaussian_mean_scale: float = 6.0,
        readout_gaussian_var_scale: float = 4.0,
        readout_positive: bool = False,
        readout_mask_l1_reg: float = 0.0,
        readout_feature_l1_reg: float = 0.0,
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [32]
        if spatial_kernel_sizes is None:
            spatial_kernel_sizes = [13]

        if loss_mode != "mse_rate":
            raise NotImplementedError(
                f"ConvCoreReadout currently only supports 'mse_rate' loss mode. "
                f"Got '{loss_mode}'. Training on temporal responses (e.g. Poisson) "
                f"requires a temporal component that is not yet implemented."
            )

        self.n_time_bins = n_time_bins
        self.loss_mode = loss_mode

        # ---- Core ----
        self.core = Stacked2DCore(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            spatial_kernel_sizes=spatial_kernel_sizes,
            gamma_input=gamma_input,
            gamma_hidden=gamma_hidden,
            input_padding=input_padding,
            hidden_padding=hidden_padding,
            dropout_rate=core_dropout,
        )

        # Determine readout input spatial size via a probe forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *image_size)
            core_out = self.core(dummy)
            _, out_c, out_h, out_w = core_out.shape

        # ---- Readout ----
        self.readout = MultiSessionGaussianReadout(
            in_channels=out_c,
            spatial_size=(out_h, out_w),
            session_info=session_info,
            gaussian_mean_scale=readout_gaussian_mean_scale,
            gaussian_var_scale=readout_gaussian_var_scale,
            positive=readout_positive,
            scale=readout_scale,
            bias=readout_bias,
            mask_l1_reg=readout_mask_l1_reg,
            feature_weights_l1_reg=readout_feature_l1_reg,
        )

    @property
    def backbone(self) -> Stacked2DCore:
        """Alias for ``core``.

        Allows ``EncodingLitModule.configure_optimizers`` to split parameter
        groups using ``self.model.backbone`` regardless of model type.
        """
        return self.core

    def forward(self, images: torch.Tensor, key: str, **kwargs) -> dict | torch.Tensor:
        """Forward pass.

        Args:
            images: ``(B, C, H, W)`` input images.
            key: Session ID string.
            responses: ``(B, T, N)`` ground truth (optional, triggers loss computation).

        Returns:
            Training (responses given): ``{"loss": Tensor}``.
            Inference: ``(B, N)`` predicted mean rates.
        """
        responses = kwargs.pop("responses", None)

        features = self.core(images)  # (B, C_out, H', W')
        pred_rate = self.readout(features, key)  # (B, N)

        if responses is not None:
            return self._compute_loss(pred_rate, responses, key)

        return pred_rate

    def _compute_loss(self, pred_rate: torch.Tensor, responses: torch.Tensor, key: str) -> dict:
        """Compute training loss with regularization.

        Args:
            pred_rate: ``(B, N)`` predicted firing rates.
            responses: ``(B, T, N)`` ground truth neural responses.
            key: Session ID for readout regularization.

        Returns:
            Dict with ``"loss"`` (total) and ``"aux/*"`` components for logging.
        """
        # Data loss
        target_mean = responses.mean(dim=1)  # (B, T, N) -> (B, N)
        mse = F.mse_loss(pred_rate, target_mean)

        # Core regularization components
        zero = torch.tensor(0.0, device=pred_rate.device)
        core_laplace = self.core.gamma_input * self.core.spatial_laplace() if self.core.gamma_input else zero
        core_sparsity = self.core.gamma_hidden * self.core.group_sparsity() if self.core.gamma_hidden else zero

        # Readout regularization components
        readout_reg = self.readout.regularizer_components(key)

        reg_total = core_laplace + core_sparsity + readout_reg["mask"] + readout_reg["features"]

        return {
            "loss": mse + reg_total,
            "aux/mse": mse,
            "aux/reg_core_laplace": core_laplace,
            "aux/reg_core_sparsity": core_sparsity,
            "aux/reg_readout_mask": readout_reg["mask"],
            "aux/reg_readout_features": readout_reg["features"],
        }
