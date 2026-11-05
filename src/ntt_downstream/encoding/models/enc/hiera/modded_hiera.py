# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0
# (https://www.apache.org/licenses/LICENSE-2.0; see https://github.com/facebookresearch/hiera).
# Modified for ntt-downstream: RMSNorm is the default norm layer and the model is adapted
# for use as an image backbone in the neural-encoding models.
# --------------------------------------------------------
#
# Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles
#
# Chaitanya Ryali, Yuan-Ting Hu, Daniel Bolya, Chen Wei, Haoqi Fan,
# Po-Yao Huang, Vaibhav Aggarwal, Arkabandhu Chowdhury, Omid Poursaeed,
# Judy Hoffman, Jitendra Malik, Yanghao Li, Christoph Feichtenhofer.
#
# Paper: https://arxiv.org/abs/2306.00989/
#
# References:
# slowfast: https://github.com/facebookresearch/SlowFast
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

import math
from functools import partial
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.drop import DropPath  # DROPPATH HAS AN INPLACE OP
from timm.layers.mlp import Mlp
from torch.nn import RMSNorm

from .base_model import FeedForward
from .utils import Reroll, Unroll, calculate_spatial_shape, conv_nd, do_masked_conv, do_pool, pretrained_model


class MaskUnitAttention(nn.Module):
    """
    Computes either Mask Unit or Global Attention. Also is able to perform q pooling.

    Note: this assumes the tokens have already been flattened and unrolled into mask units.
    See `Unroll` for more details.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        heads: int,
        q_stride: int = 1,
        window_size: int = 0,
        use_mask_unit_attn: bool = False,
    ):
        """
        Args:
        - dim, dim_out: The input and output feature dimensions.
        - heads: The number of attention heads.
        - q_stride: If greater than 1, pool q with this stride. The stride should be flattened (e.g., 2x2 = 4).
        - window_size: The current (flattened) size of a mask unit *after* pooling (if any).
        - use_mask_unit_attn: Use Mask Unit or Global Attention.
        """
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.heads = heads
        self.q_stride = q_stride

        self.head_dim = dim_out // heads
        self.scale = (self.head_dim) ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim_out)
        self.proj = nn.Linear(dim_out, dim_out)

        self.window_size = window_size
        self.use_mask_unit_attn = use_mask_unit_attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input should be of shape [Batch, Sequence, Dim]."""
        B, N, _ = x.shape  # [B, S, D]
        num_windows = (N // (self.q_stride * self.window_size)) if self.use_mask_unit_attn else 1

        qkv = self.qkv(x).reshape(B * num_windows, -1, 3, self.heads, self.head_dim)
        q, k, v = torch.unbind(qkv, dim=2)  # [B*num_windows, S, H, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        q_processed = q
        if self.q_stride > 1:
            q_reshaped = q.reshape(B * num_windows, self.heads, -1, self.q_stride, self.head_dim)
            q_processed = q_reshaped.max(dim=3).values

        # Ensure contiguous regardless of whether pooling happened
        q_processed = q_processed.contiguous()

        x = F.scaled_dot_product_attention(q_processed.contiguous(), k, v)

        x = x.view(B, num_windows, self.heads, -1, self.head_dim)
        x = x.permute(0, 3, 1, 2, 4).reshape(B, -1, self.dim_out)
        x = self.proj(x)
        return x


class HieraBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        act_layer: type[nn.Module] = nn.GELU,
        q_stride: int = 1,
        window_size: int = 0,
        use_mask_unit_attn: bool = False,
        use_gated_ffn: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out

        self.norm1 = norm_layer(dim)
        self.attn = MaskUnitAttention(dim, dim_out, heads, q_stride, window_size, use_mask_unit_attn)

        self.norm2 = norm_layer(dim_out)
        if use_gated_ffn:
            self.mlp = FeedForward(dim=dim_out, mult=mlp_ratio, multiple_of=128)
        else:
            self.mlp = Mlp(dim_out, int(dim_out * mlp_ratio), act_layer=act_layer)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention + Q Pooling
        x_norm = self.norm1(x)
        if self.dim != self.dim_out:
            x = do_pool(self.proj(x_norm), stride=self.attn.q_stride)
        x = x + self.drop_path(self.attn(x_norm))

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Patch embed that supports any number of spatial dimensions (1d, 2d, 3d)."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel: tuple[int, ...],
        stride: tuple[int, ...],
        padding: tuple[int, ...],
    ):
        super().__init__()

        # Support any number of spatial dimensions
        self.spatial_dims = len(kernel)
        self.proj = conv_nd(self.spatial_dims)(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = do_masked_conv(x, self.proj, mask)
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(2, 1)
        return x


class Hiera(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, ...] = (224, 224),
        in_chans: int = 3,
        embed_dim: int = 96,  # initial embed dim
        num_heads: int = 1,  # initial number of heads
        num_classes: int = 1000,
        stages: tuple[int, ...] = (2, 3, 16, 3),
        q_pool: int = 3,  # number of q_pool stages
        q_stride: tuple[int, ...] = (2, 2),
        mask_unit_size: tuple[int, ...] = (8, 8),  # must divide q_stride ** (#stages-1)
        # mask_unit_attn: which stages use mask unit attention?
        mask_unit_attn: tuple[bool, ...] = (True, True, False, False),
        dim_mul: float = 2.0,
        head_mul: float = 2.0,
        patch_kernel: tuple[int, ...] = (7, 7),
        patch_stride: tuple[int, ...] = (4, 4),
        patch_padding: tuple[int, ...] = (3, 3),
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        norm_layer: type[nn.Module] = RMSNorm,
        sep_pos_embed: bool = False,
        use_gated_ffn: bool = False,
    ):
        super().__init__()

        assert len(input_size) == len(patch_stride) == len(patch_kernel) == len(patch_padding), (
            "All spatial parameters must have the same dimensionality"
        )

        depth = sum(stages)
        self.patch_stride = patch_stride
        self.tokens_spatial_shape = calculate_spatial_shape(input_size, patch_stride, patch_kernel, patch_padding)

        num_tokens = math.prod(self.tokens_spatial_shape)
        flat_mu_size = math.prod(mask_unit_size)
        flat_q_stride = math.prod(q_stride)

        assert q_pool < len(stages)
        self.q_pool, self.q_stride = q_pool, q_stride
        self.mu_size, self.mask_unit_size = flat_mu_size, mask_unit_size
        self.mask_spatial_shape = [i // s for i, s in zip(self.tokens_spatial_shape, self.mask_unit_size, strict=False)]
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]

        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_kernel, patch_stride, patch_padding)

        self.sep_pos_embed = sep_pos_embed
        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(
                    1,
                    self.tokens_spatial_shape[1] * self.tokens_spatial_shape[2],
                    embed_dim,
                )
            )
            self.pos_embed_temporal = nn.Parameter(torch.zeros(1, self.tokens_spatial_shape[0], embed_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))

        # Setup roll and reroll modules
        self.unroll = Unroll(
            input_size=input_size,
            patch_stride=patch_stride,
            patch_kernel=patch_kernel,
            patch_padding=patch_padding,
            unroll_schedule=[q_stride] * len(self.stage_ends[:-1]),
        )
        self.reroll = Reroll(
            input_size=input_size,
            patch_stride=patch_stride,
            patch_kernel=patch_kernel,
            patch_padding=patch_padding,
            unroll_schedule=[q_stride] * len(self.stage_ends[:-1]),
            stage_ends=self.stage_ends,
            q_pool=q_pool,
        )
        # q_pool locations
        q_pool_blocks = [x + 1 for x in self.stage_ends[:q_pool]]
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks
        cur_stage = 0
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            # Mask unit or global attention.
            # Lag by 1 block, so that global attention,
            # applied post pooling on lower resolution
            use_mask_unit_attn = mask_unit_attn[cur_stage]

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1
                if i in q_pool_blocks:
                    flat_mu_size //= flat_q_stride

            block = HieraBlock(
                dim=embed_dim,
                dim_out=dim_out,
                heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                q_stride=(flat_q_stride if i in q_pool_blocks else 1),
                window_size=flat_mu_size,
                use_mask_unit_attn=use_mask_unit_attn,
                use_gated_ffn=use_gated_ffn,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        # Initialize everything
        if sep_pos_embed:
            nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)
        else:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(partial(self._init_weights))

    def _init_weights(self, m, init_bias=0.02):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, init_bias)
        elif isinstance(m, (nn.LayerNorm, nn.RMSNorm)):
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(cast(torch.Tensor, m.bias), init_bias)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore  # pyright: ignore[reportArgumentType] -- torch stub types the decorator arg as bool
    def no_weight_decay(self):
        if self.sep_pos_embed:
            return ["pos_embed_spatial", "pos_embed_temporal"]
        else:
            return ["pos_embed"]

    def get_random_mask(self, x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        """
        Generates a random mask, mask_ratio fraction are dropped.
        1 is *keep*, 0 is *remove*. Useful for MAE, FLIP, etc.
        """
        B = x.shape[0]
        # Tokens selected for masking at mask unit level
        num_windows = math.prod(self.mask_spatial_shape)  # num_mask_units
        len_keep = int(num_windows * (1 - mask_ratio))
        noise = torch.rand(B, num_windows, device=x.device)

        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Generate the binary mask: 1 is *keep*, 0 is *remove*
        # Note this is opposite to original MAE
        mask = torch.zeros([B, num_windows], device=x.device)
        mask[:, :len_keep] = 1
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask.bool()

    def get_pos_embed(self) -> torch.Tensor:
        if self.sep_pos_embed:
            return self.pos_embed_spatial.repeat(1, self.tokens_spatial_shape[0], 1) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.tokens_spatial_shape[1] * self.tokens_spatial_shape[2],
                dim=1,
            )
        else:
            return self.pos_embed

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_intermediates: bool = False,
        output_layers: tuple[int, ...] | int | None = None,  # stage indices to extract (by order of stage ends)
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        mask should be a boolean tensor of shape [B, #MUt*#MUy*#MUx] where #MU are the number of mask units in that dim.
        Note: 1 in mask is *keep*, 0 is *remove*; mask.sum(dim=-1) should be the same across the batch.
        """
        x = self.patch_embed(
            x,
            mask=mask.view(x.shape[0], 1, *self.mask_spatial_shape)  # B, C, *mask_spatial_shape
            if mask is not None
            else None,
        )
        x = x + self.get_pos_embed()
        x = self.unroll(x)

        # Discard masked tokens
        if mask is not None:
            x = x[mask[..., None].tile(1, self.mu_size, x.shape[2])].view(x.shape[0], -1, x.shape[-1])

        extracted: list[torch.Tensor] = []
        stage_counter = -1
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            # Collect stage outputs at the end of each stage
            if i in self.stage_ends:
                stage_counter += 1
                if output_layers is not None:
                    # Normalize to tuple for membership check
                    if isinstance(output_layers, int):
                        should_collect = stage_counter == output_layers
                    else:
                        should_collect = stage_counter in output_layers
                    if should_collect:
                        extracted.append(self.reroll(x, i, mask=mask))

        # If specific layers requested, return them (single tensor for int, list for multiple)
        if output_layers is not None:
            if isinstance(output_layers, int):
                # If the user asked for a single layer but none collected (out of range), fall back to final
                return extracted[0] if len(extracted) > 0 else self.reroll(x, i, mask=mask)
            return extracted

        # Default behavior: return final features
        return self.reroll(x, i, mask=mask)


# Image models


@pretrained_model(
    {
        "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_tiny_224.pth",
        "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_tiny_224.pth",
    },
    default="mae_in1k_ft_in1k",
)
def hiera_tiny_224(**kwdargs):
    return Hiera(embed_dim=96, num_heads=1, stages=(1, 2, 7, 2), **kwdargs)


@pretrained_model(
    {
        "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_small_224.pth",
        "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_small_224.pth",
    },
    default="mae_in1k_ft_in1k",
)
def hiera_small_224(**kwdargs):
    return Hiera(embed_dim=96, num_heads=1, stages=(1, 2, 11, 2), **kwdargs)


@pretrained_model(
    {
        "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_base_224.pth",
        "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_224.pth",
    },
    default="mae_in1k_ft_in1k",
)
def hiera_base_224(**kwdargs):
    return Hiera(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3), **kwdargs)


@pretrained_model(
    {
        "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_base_plus_224.pth",
        "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_plus_224.pth",
    },
    default="mae_in1k_ft_in1k",
)
def hiera_base_plus_224(**kwdargs):
    return Hiera(embed_dim=112, num_heads=2, stages=(2, 3, 16, 3), **kwdargs)


@pretrained_model(
    {
        "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_large_224.pth",
        "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_large_224.pth",
    },
    default="mae_in1k_ft_in1k",
)
def hiera_large_224(**kwdargs):
    return Hiera(embed_dim=144, num_heads=2, stages=(2, 6, 36, 4), **kwdargs)


@pretrained_model(
    {
        "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_huge_224.pth",
        "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_huge_224.pth",
    },
    default="mae_in1k_ft_in1k",
)
def hiera_huge_224(**kwdargs):
    return Hiera(embed_dim=256, num_heads=4, stages=(2, 6, 36, 4), **kwdargs)


# Video models


@pretrained_model(
    {
        "mae_k400_ft_k400": "https://dl.fbaipublicfiles.com/hiera/hiera_base_16x224.pth",
        "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_16x224.pth",
    },
    default="mae_k400_ft_k400",
)
def hiera_base_16x224(num_classes: int = 400, **kwdargs):
    return Hiera(
        num_classes=num_classes,  # K400 has 400 classes
        input_size=(16, 224, 224),
        q_stride=(1, 2, 2),
        mask_unit_size=(1, 8, 8),
        patch_kernel=(3, 7, 7),
        patch_stride=(2, 4, 4),
        patch_padding=(1, 3, 3),
        sep_pos_embed=True,
        **kwdargs,
    )


@pretrained_model(
    {
        "mae_k400_ft_k400": "https://dl.fbaipublicfiles.com/hiera/hiera_base_plus_16x224.pth",
        "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_plus_16x224.pth",
    },
    default="mae_k400_ft_k400",
)
def hiera_base_plus_16x224(**kwdargs):
    return hiera_base_16x224(embed_dim=112, num_heads=2, stages=(2, 3, 16, 3), **kwdargs)


@pretrained_model(
    {
        "mae_k400_ft_k400": "https://dl.fbaipublicfiles.com/hiera/hiera_large_16x224.pth",
        "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_large_16x224.pth",
    },
    default="mae_k400_ft_k400",
)
def hiera_large_16x224(**kwdargs):
    return hiera_base_16x224(embed_dim=144, num_heads=2, stages=(2, 6, 36, 4), **kwdargs)


@pretrained_model(
    {
        "mae_k400_ft_k400": "https://dl.fbaipublicfiles.com/hiera/hiera_huge_16x224.pth",
        "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_huge_16x224.pth",
    },
    default="mae_k400_ft_k400",
)
def hiera_huge_16x224(**kwdargs):
    return hiera_base_16x224(embed_dim=256, num_heads=4, stages=(2, 6, 36, 4), **kwdargs)
