"""Image encoding model using Hiera backbone with cross-attention readout.

Predicts neural spike responses from static images. Each session has its own
set of learnable neuron queries that cross-attend to backbone image features.

Supports:
- Poisson NLL loss (direct rate prediction)
- Flow matching loss (via tokenizer latent space)
- Diffusion loss (via tokenizer latent space)
- MSE / Smooth L1 loss on tokenizer latents
- Cross-entropy on discrete tokenizer tokens
- BCE loss on BSQ binary codes (via tokenizer latent space)
- compress_n tokenizer mode (temporal tiling of neuron chunks)
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any, cast

import torch
from jaxtyping import Float
from torch import nn
from torch.nn import RMSNorm
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# ---------------------------------------------------------------------------
# Utility modules
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, outdim=None):
        super().__init__()
        outdim = outdim or dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, outdim),
        )
        self.init_weights()

    def forward(self, x):
        return self.net(x)

    def init_weights(self, std=0.5, cutoff_factor: int = 3):
        # nn.Sequential.__getitem__ is typed as returning Module; both entries are Linear layers.
        fc_in = cast(nn.Linear, self.net[0])
        fc_out = cast(nn.Linear, self.net[2])
        nn.init.trunc_normal_(fc_in.weight, mean=0.0, std=std, a=-cutoff_factor * std, b=cutoff_factor * std)
        nn.init.trunc_normal_(fc_out.weight, mean=0.0, std=std, a=-cutoff_factor * std, b=cutoff_factor * std)
        fc_in.bias.data.zero_()
        fc_out.bias.data.zero_()


class FeedForward(nn.Module):
    """Two-layer gated SiLU feed-forward network."""

    def __init__(
        self,
        dim: int,
        mult: float = 1.0,
        multiple_of: int = 128,
        activation: str = "silu",
        use_projection_bias: bool = False,
        init_std: float = 0.02,
        outdim: int | None = None,
        hidden_dim: int | None = None,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = (int(multiple_of * ((dim * mult + multiple_of - 1) // multiple_of))) or hidden_dim
        outdim = outdim or dim
        assert hidden_dim is not None, "hidden_dim must be given when the computed hidden size is 0"
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=use_projection_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=use_projection_bias)
        self.down_proj = nn.Linear(hidden_dim, outdim, bias=use_projection_bias)
        self.activation = activation
        self.init_std = init_std
        self.dropout = nn.Dropout(p=p_dropout)

    def init_weights(self, layer_num: int | None = None):
        init_weights_util((self.up_proj,), self.init_std)
        scaled_std = depth_scaled_std(self.init_std, layer_num=layer_num)
        init_weights_util((self.gate_proj, self.down_proj), scaled_std)

    def activation_fn(self, x: Float[torch.Tensor, "batch seq dim"]):
        if self.activation == "silu":
            return F.silu(x)
        elif self.activation == "gelu":
            return F.gelu(x)
        else:
            raise NotImplementedError(f"Unknown activation function: '{self.activation}'")

    def forward(self, x: Float[torch.Tensor, "batch seq dim"]) -> Float[torch.Tensor, "batch seq dim"]:
        x = self.activation_fn(self.up_proj(x)) * self.gate_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return self.dropout(x)


def init_weights_util(
    modules_and_params: Iterable[nn.Module | nn.Parameter | None],
    init_std: float = 0.02,
):
    for module_or_param in modules_and_params:
        if module_or_param is None:
            continue
        elif isinstance(module_or_param, (RMSNorm, nn.LayerNorm)):
            module_or_param.reset_parameters()
        elif isinstance(module_or_param, nn.Linear):
            nn.init.trunc_normal_(module_or_param.weight, std=init_std)
            if module_or_param.bias is not None:
                module_or_param.bias.data.zero_()
        elif isinstance(module_or_param, nn.Embedding):
            nn.init.trunc_normal_(module_or_param.weight, std=init_std)
            if module_or_param.padding_idx is not None:
                module_or_param.weight.data[module_or_param.padding_idx].zero_()
        elif isinstance(module_or_param, nn.Parameter):
            nn.init.trunc_normal_(module_or_param, std=init_std)
        else:
            raise NotImplementedError(f"Unsupported module or parameter type: {type(module_or_param)}")


def depth_scaled_std(init_std: float, layer_num: int | None = None) -> float:
    scaled_std = init_std
    if layer_num is not None:
        scaled_std /= (2 * (layer_num + 1)) ** 0.5
    return scaled_std


class FiLMResidualBlock(nn.Module):
    """Pre-norm residual FFN block with optional per-token FiLM modulation.

    One block:
        h = fc1(norm(x))
        if use_film: h = (1 + γ(c)) · h + β(c)   # broadcast over batch
        h = fc2(act(h))
        return x + h

    ``γ`` and ``β`` are projected from a per-token conditioning vector and
    zero-initialized so the block starts as an identity residual.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        cond_dim: int | None = None,
        dropout: float = 0.0,
        norm_layer: type[nn.Module] = RMSNorm,
    ):
        super().__init__()
        self.norm = norm_layer(dim)
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.use_film = cond_dim is not None
        if self.use_film:
            assert cond_dim is not None
            self.film = nn.Linear(cond_dim, 2 * hidden_dim)
            # Zero-init so the first forward pass is x + fc2(act(fc1(norm(x)))).
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        h = self.fc1(self.norm(x))
        if self.use_film and cond is not None:
            gamma, beta = self.film(cond).chunk(2, dim=-1)
            h = (1.0 + gamma) * h + beta
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return x + h

    def init_weights(self, std: float | None = None, cutoff_factor: int = 3):
        std = std if std is not None else float(self.fc1.in_features**-0.5)
        for lin in (self.fc1, self.fc2):
            nn.init.trunc_normal_(lin.weight, mean=0.0, std=std, a=-cutoff_factor * std, b=cutoff_factor * std)
        # FiLM weights stay at zero so the block starts as identity.


class FiLMResidualHead(nn.Module):
    """Deep residual projection head with optional per-token FiLM modulation.

    Used as ``token_prediction_proj`` when ``deep_token_head`` is true.
    Shape: ``(B, M, dim)`` → ``(B, M, out_dim)``, where ``M`` is e.g. N neurons
    (or N·T_latent tiled queries).

    FiLM conditioning (optional): ``(1, M, dim)`` or ``(B, M, dim)`` tensor of
    per-token features. Zero-initialized, so at step 0 the head behaves like
    a plain ResMLP with random FC weights.

    Args:
        dim: Input feature dimension (= readout ``dim_q``).
        out_dim: Output feature dimension (= tokenizer ``latent_dim``).
        depth: Number of residual blocks.
        hidden_mult: Hidden-dim multiplier inside each block. Hidden = ``hidden_mult · dim``.
        use_film: Enable per-token FiLM modulation from ``cond``.
        dropout: Dropout inside each block.
        norm_layer: Normalization class (defaults to RMSNorm).
    """

    def __init__(
        self,
        dim: int,
        out_dim: int,
        depth: int = 2,
        hidden_mult: float = 2.0,
        use_film: bool = True,
        dropout: float = 0.0,
        norm_layer: type[nn.Module] = RMSNorm,
    ):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.use_film = use_film
        hidden_dim = int(hidden_mult * dim)
        cond_dim = dim if use_film else None
        self.blocks = nn.ModuleList(
            [
                FiLMResidualBlock(dim, hidden_dim, cond_dim=cond_dim, dropout=dropout, norm_layer=norm_layer)
                for _ in range(depth)
            ]
        )
        self.final_norm = norm_layer(dim)
        self.final_proj = nn.Linear(dim, out_dim, bias=False)
        self.init_weights()

    def init_weights(self, cutoff_factor: int = 3):
        std = self.dim**-0.5
        for block in self.blocks:
            cast(FiLMResidualBlock, block).init_weights(std=std, cutoff_factor=cutoff_factor)
        nn.init.trunc_normal_(self.final_proj.weight, mean=0.0, std=std, a=-cutoff_factor * std, b=cutoff_factor * std)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, cond)
        return self.final_proj(self.final_norm(x))


class ParametrizedSoftplus(nn.Module):
    """Learnable softplus: w * softplus(a * x + b).

    Default (a=1, b=0, w=1) is the standard softplus with floor at log(2) ≈ 0.693.
    """

    def __init__(self, a: float = 1.0, b: float = 0.0, w: float = 1.0, learn_a: bool = False):
        super().__init__()
        if learn_a:
            self.a = nn.Parameter(torch.tensor(a))
        else:
            self.register_buffer("a", torch.tensor(a))
        self.b = nn.Parameter(torch.tensor(b))
        self.w = nn.Parameter(torch.tensor(w))

    def forward(self, x):
        return self.w * F.softplus(self.a * x + self.b)


# ---------------------------------------------------------------------------
# Readout layers (stackable cross-attention and self-attention)
# ---------------------------------------------------------------------------


class ReadoutCrossAttentionLayer(nn.Module):
    """Cross-attention: neuron queries attend to backbone image features.

    Each layer has its own K/V projections so different layers can attend to
    different aspects of the spatial features.
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int,
        dim_head: int,
        mlp_ratio: float = 4,
        use_gated_ffn: bool = False,
        norm_layer: type[nn.Module] = RMSNorm,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_q = dim_q

        self.q_norm = norm_layer(dim_q)
        self.kv_norm = norm_layer(dim_kv)
        self.wq = nn.Linear(dim_q, dim_q, bias=False)
        self.wk = nn.Linear(dim_kv, dim_q, bias=False)
        self.wv = nn.Linear(dim_kv, dim_q, bias=False)
        self.wo = nn.Linear(dim_q, dim_q, bias=False)
        self.post_attn_norm = norm_layer(dim_q)
        if use_gated_ffn:
            self.mlp = FeedForward(dim=dim_q, mult=mlp_ratio)
        else:
            self.mlp = MLP(dim=dim_q, hidden_dim=int(dim_q * mlp_ratio))

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: (B, N, dim_q) neuron query representations
            kv: (B, S, dim_kv) backbone spatial features
        Returns:
            (B, N, dim_q) refined neuron representations
        """
        b, n, _ = q.shape

        residual = q
        q_normed = self.q_norm(q)
        kv_normed = self.kv_norm(kv)

        q_proj = self.wq(q_normed).view(b, n, self.num_heads, self.dim_head).transpose(1, 2)
        k = self.wk(kv_normed).view(b, -1, self.num_heads, self.dim_head).transpose(1, 2)
        v = self.wv(kv_normed).view(b, -1, self.num_heads, self.dim_head).transpose(1, 2)

        attn_ctx = (
            sdpa_kernel(
                [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH], set_priority=True
            )
            if q.dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        with attn_ctx:
            o = F.scaled_dot_product_attention(q_proj.contiguous(), k.contiguous(), v.contiguous())

        o = o.transpose(1, 2).contiguous().view(b, n, self.dim_q)
        o = self.wo(o) + residual

        o = self.post_attn_norm(o)
        o = self.mlp(o) + o
        return o

    def init_weights(self, cutoff_factor: int = 3):
        self.mlp.init_weights()
        std = self.dim_q**-0.5
        for lin in (self.wq, self.wk, self.wv, self.wo):
            nn.init.trunc_normal_(lin.weight, mean=0.0, std=std, a=-cutoff_factor * std, b=cutoff_factor * std)


class ReadoutSelfAttentionLayer(nn.Module):
    """Self-attention among neuron queries for population context."""

    def __init__(
        self,
        dim_q: int,
        num_heads: int,
        dim_head: int,
        mlp_ratio: float = 4,
        use_gated_ffn: bool = False,
        norm_layer: type[nn.Module] = RMSNorm,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_q = dim_q

        self.q_norm = norm_layer(dim_q)
        self.wq = nn.Linear(dim_q, dim_q, bias=False)
        self.wk = nn.Linear(dim_q, dim_q, bias=False)
        self.wv = nn.Linear(dim_q, dim_q, bias=False)
        self.wo = nn.Linear(dim_q, dim_q, bias=False)
        self.post_attn_norm = norm_layer(dim_q)
        if use_gated_ffn:
            self.mlp = FeedForward(dim=dim_q, mult=mlp_ratio)
        else:
            self.mlp = MLP(dim=dim_q, hidden_dim=int(dim_q * mlp_ratio))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: (B, N, dim_q) neuron query representations
        Returns:
            (B, N, dim_q) population-contextualized representations
        """
        b, n, _ = q.shape

        residual = q
        q_normed = self.q_norm(q)

        q_proj = self.wq(q_normed).view(b, n, self.num_heads, self.dim_head).transpose(1, 2)
        k = self.wk(q_normed).view(b, n, self.num_heads, self.dim_head).transpose(1, 2)
        v = self.wv(q_normed).view(b, n, self.num_heads, self.dim_head).transpose(1, 2)

        attn_ctx = (
            sdpa_kernel(SDPBackend.FLASH_ATTENTION) if q.dtype in (torch.float16, torch.bfloat16) else nullcontext()
        )
        with attn_ctx:
            o = F.scaled_dot_product_attention(q_proj.contiguous(), k.contiguous(), v.contiguous())

        o = o.transpose(1, 2).contiguous().view(b, n, self.dim_q)
        o = self.wo(o) + residual

        o = self.post_attn_norm(o)
        o = self.mlp(o) + o
        return o

    def init_weights(self, cutoff_factor: int = 3):
        self.mlp.init_weights()
        std = self.dim_q**-0.5
        for lin in (self.wq, self.wk, self.wv, self.wo):
            nn.init.trunc_normal_(lin.weight, mean=0.0, std=std, a=-cutoff_factor * std, b=cutoff_factor * std)


# ---------------------------------------------------------------------------
# Per-session readout
# ---------------------------------------------------------------------------


class SessionReadout(nn.Module):
    """Per-session neuron query + bias + session embedding."""

    def __init__(
        self,
        n_neurons: int,
        dim_head: int = 32,
        num_heads: int = 4,
        bias: bool = True,
        device="cuda",
        dtype=torch.float32,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.unique_ids = n_neurons
        dim_q = dim_head * num_heads
        self.in_features = dim_q
        self.query = nn.Parameter(torch.empty(1, n_neurons, dim_q, **factory_kwargs))
        self.session_embedding = nn.Parameter(torch.empty(1, 1, dim_q, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(1, n_neurons, 1, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.init_weights()

    def init_weights(self, cutoff_factor: int = 3):
        readout_std = self.in_features**-0.5
        nn.init.trunc_normal_(
            self.query, mean=0.0, std=readout_std, a=-cutoff_factor * readout_std, b=cutoff_factor * readout_std
        )
        nn.init.trunc_normal_(
            self.session_embedding,
            mean=0.0,
            std=readout_std,
            a=-cutoff_factor * readout_std,
            b=cutoff_factor * readout_std,
        )
        if self.bias is not None:
            self.bias.data.zero_()


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class ImageEncodingHiera(nn.Module):
    """Image → neural response model using Hiera backbone with cross-attention readout.

    For static images: pass (B, C, H, W) through 2D Hiera backbone,
    then cross-attend neuron queries to spatial features.

    Args:
        backbone: Hiera backbone module.
        session_info: dict mapping session_id → n_neurons. Used to create
            per-session readout heads.
        n_time_bins: Number of temporal bins in the response (e.g., 200 for 200ms at 1ms).
        dim: Backbone output feature dimension.
        dim_head: Dimension per attention head.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dim multiplier.
        learnable_softplus: Use ParametrizedSoftplus (learnable w, b) instead of nn.Softplus.
        softplus_beta: Beta for softplus activation (sharpness). Used as init `a` when learnable.
        use_session_embedding: Whether to add session embeddings to queries.
        skip_connection: Whether to use residual connections in readout.
        use_gated_ffn: Use gated SiLU FFN instead of simple MLP.
        norm_layer: Normalization layer class.
        token_prediction_dim: Output dim for tokenizer-based losses.
        diffloss: Optional DiffLoss module for diffusion loss mode.
        energy_loss: Optional EnergyLoss module for EAR energy-score loss mode.
        n_neurons_override: Override n_neurons per session (for compress_n).
        compress_n_latent_n: Latent channels for compress_n mode.
        compress_n_t_coarse: Coarse temporal steps for compress_n.
        compress_n_t_target: Target temporal steps for compress_n.
        compress_n_temporal_upsample: Upsample method ("interpolate" or "conv_transpose").
        loss_mode: Loss type ("poisson", "flow_matching", "diffusion", etc.).
        tokenizer_wrapper: NTT tokenizer wrapper for encoding/decoding.
        diffusion_batch_mul: Repeat factor for lower-variance diffusion loss.
        flow_head_width: FlowMatchingHead MLP width.
        flow_head_depth: FlowMatchingHead MLP depth.
        flow_opts: FlowOptions for rectified-flow training (time sampling,
            OT coupling, conditioning dropout for CFG); applies to all three
            flow heads (pointwise, full-grid, LDNS).
        ce_vocab_size: Vocabulary size for cross-entropy loss mode.
        ce_sample_temperature: Sampling temperature for cross-entropy.
        tokenizer_aux_loss_weight: Weight for tokenizer auxiliary loss.
    """

    def __init__(
        self,
        backbone,
        session_info: dict[str, int] | None = None,
        n_time_bins: int = 200,
        dim: int = 192,
        dim_head: int = 32,
        num_heads: int = 4,
        mlp_ratio: float = 4,
        learnable_softplus: bool = False,
        softplus_beta: float = 0.1,
        use_session_embedding: bool = False,
        use_subject_embedding: bool = False,
        subject_info: dict[str, str] | None = None,
        skip_connection: bool = True,
        use_gated_ffn: bool = False,
        norm_layer: type[nn.Module] = RMSNorm,
        backbone_dim: int | None = None,
        token_prediction_dim: int | None = None,
        # Duck-typed loss heads (expose .sample / are callable); Any avoids nn.Module.__getattr__ widening.
        diffloss: Any = None,
        energy_loss: Any = None,
        n_neurons_override: dict[str, int] | None = None,
        # compress_n parameters
        compress_n_latent_n: int | None = None,
        compress_n_t_coarse: int | None = None,
        compress_n_t_target: int | None = None,
        compress_n_temporal_upsample: str = "interpolate",
        # When set, the tokenizer is fed the full response window (e.g. 250 bins
        # for a 2.5s LDNS context) but only the leading ``compress_n_target_slice``
        # latent time-bins are used as the loss target. Lets us give length-sensitive
        # tokenizers (LDNS uses InstanceNorm1d over time) their trained operating
        # length while training the encoder to predict only the post-stim region.
        compress_n_target_slice: int | None = None,
        # LDNS sequence-aware denoiser parameters
        use_ldns_denoiser: bool = False,
        ldns_hidden_channels: int = 64,
        ldns_num_blocks: int = 4,
        ldns_kernel_size: int = 15,
        # Full-grid Perceiver-block denoiser parameters
        use_full_grid_denoiser: bool = False,
        full_grid_hidden_channels: int = 64,
        full_grid_num_blocks: int = 4,
        full_grid_num_heads: int = 4,
        full_grid_mlp_ratio: int = 2,
        full_grid_kernel_size: int = 7,
        full_grid_use_s4: bool = False,
        # Internalized loss parameters
        loss_mode: str | None = None,
        tokenizer_wrapper: Any = None,
        diffusion_batch_mul: int = 4,
        flow_head_width: int = 256,
        flow_head_depth: int = 3,
        flow_opts=None,
        ce_vocab_size: int | None = None,
        ce_sample_temperature: float = 1.0,
        # Class-imbalance mitigations for the discrete CE/BCE loss paths.
        # All default OFF: with these defaults the loss is byte-identical to
        # plain F.cross_entropy / binary_cross_entropy_with_logits. Sparse spike
        # data makes the "silent" token dominate the CE target, so the encoder
        # can collapse to the image-independent majority class (rate_pearson≈0);
        # these knobs let us counter that. focal_gamma reshapes per-position CE;
        # ce_class_weight_beta tempers the inverse-frequency class weights that
        # the lit module estimates once at setup and hands in via
        # ``set_ce_class_weight``. ``set_bce_pos_weight`` similarly receives the
        # per-bit positive-frequency reweighting for the BSQ/bce path.
        ce_focal_gamma: float = 0.0,
        ce_class_weight_beta: float = 0.5,
        # Number of independent codebooks per latent position. 1 = single-codebook
        # (transformer-vq, FSQ, BSQ, plain VQ). >1 = residual / multi-codebook VQ — the
        # ce_head emits Q*V logits and the wrapper exposes targets as
        # (B, S, T, Q) int64.
        num_codebooks: int = 1,
        tokenizer_aux_loss_weight: float = 0.0,
        response_zscore: bool = False,
        # Decoder fine-tuning
        decoder_recon_loss_weight: float = 0.0,
        decoder_recon_loss_mode: str = "poisson",
        # MSE-on-latents image-conditional loss shaping
        mse_center_weight: float = 0.0,
        cosine_sim_weight: float = 0.0,
        # compress_t temporal tiling (for tokenizers whose encoder yields t_comp>1)
        token_t_latent: int = 1,
        # Multi-layer readout
        readout_layer_types: list[str] | None = None,
        # Deep token-prediction head (false = shallow 2-layer MLP)
        deep_token_head: bool = True,
        deep_token_head_depth: int = 2,
        deep_token_head_width_mult: float = 2.0,
        deep_token_head_use_film: bool = True,
        deep_token_head_dropout: float = 0.0,
    ):
        super().__init__()

        if session_info is None:
            raise ValueError("session_info dict (session_id → n_neurons) is required")

        self.backbone = backbone
        self.loss_mode = loss_mode
        self.tokenizer_wrapper = tokenizer_wrapper
        self.response_zscore = response_zscore

        if loss_mode not in ("poisson", "mse_rate") and tokenizer_wrapper is None:
            raise ValueError(
                f"loss_mode={loss_mode!r} requires a tokenizer, but no tokenizer_wrapper was provided. "
                f"Check that your config includes a tokenizer section."
            )
        if loss_mode == "bce" and tokenizer_wrapper is not None and tokenizer_wrapper.bsq_codebook_scale is None:
            raise ValueError(
                "loss_mode='bce' requires a BSQ tokenizer with codebook_scale, "
                "but bsq_codebook_scale is None. Check that your tokenizer has a "
                "BSQ bottleneck with a codebook_scale attribute."
            )
        self.diffusion_batch_mul = diffusion_batch_mul
        self.ce_sample_temperature = ce_sample_temperature
        # Class-imbalance knobs (default OFF — see __init__ signature comment).
        self.ce_focal_gamma = float(ce_focal_gamma)
        self.ce_class_weight_beta = float(ce_class_weight_beta)
        # Reweighting tensors are estimated once at setup (frozen tokenizer →
        # fixed target distribution) and registered as buffers via the setters
        # below. They are None until then, and a None buffer means "no
        # reweighting" so the default path stays byte-identical.
        self.register_buffer("ce_class_weight", None, persistent=False)
        self.register_buffer("bce_pos_weight", None, persistent=False)
        self.num_codebooks = int(num_codebooks)
        if self.num_codebooks < 1:
            raise ValueError(f"num_codebooks must be >= 1, got {self.num_codebooks}")
        self.tokenizer_aux_loss_weight = tokenizer_aux_loss_weight
        self.decoder_recon_loss_weight = decoder_recon_loss_weight
        self.decoder_recon_loss_mode = decoder_recon_loss_mode
        self.mse_center_weight = float(mse_center_weight)
        self.cosine_sim_weight = float(cosine_sim_weight)
        self.token_t_latent = int(token_t_latent)
        self.norm = norm_layer
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim = dim
        self.dim_q = dim_head * num_heads

        # Readout layer stack
        if readout_layer_types is None:
            readout_layer_types = ["cross"]
        for lt in readout_layer_types:
            if lt not in ("cross", "self"):
                raise ValueError(f"Unknown readout layer type '{lt}'. Expected 'cross' or 'self'.")
        self.readout_layer_types = readout_layer_types
        self.readout_layers = nn.ModuleList()
        for lt in readout_layer_types:
            if lt == "cross":
                self.readout_layers.append(
                    ReadoutCrossAttentionLayer(
                        dim_q=self.dim_q,
                        dim_kv=dim,
                        num_heads=num_heads,
                        dim_head=dim_head,
                        mlp_ratio=mlp_ratio,
                        use_gated_ffn=use_gated_ffn,
                        norm_layer=norm_layer,
                    )
                )
            else:  # "self"
                self.readout_layers.append(
                    ReadoutSelfAttentionLayer(
                        dim_q=self.dim_q,
                        num_heads=num_heads,
                        dim_head=dim_head,
                        mlp_ratio=mlp_ratio,
                        use_gated_ffn=use_gated_ffn,
                        norm_layer=norm_layer,
                    )
                )

        self.neuron_proj = nn.Linear(self.dim_q, n_time_bins, bias=False)
        self.token_prediction_dim = token_prediction_dim
        self.compress_n_latent_n = compress_n_latent_n
        self.compress_n_t_coarse = compress_n_t_coarse
        self.compress_n_t_target = compress_n_t_target
        self.compress_n_temporal_upsample = compress_n_temporal_upsample
        self.compress_n_target_slice = compress_n_target_slice

        # Compress-N temporal tiling
        if compress_n_latent_n is not None and compress_n_t_coarse is not None:
            self.temporal_embeds = nn.Parameter(torch.zeros(1, compress_n_t_coarse, self.dim_q))
            nn.init.normal_(self.temporal_embeds, std=self.dim_q**-0.5)
            self.compress_n_latent_proj = nn.Linear(self.dim_q, compress_n_latent_n, bias=False)
            if (
                compress_n_temporal_upsample == "conv_transpose"
                and compress_n_t_target is not None
                and compress_n_t_target > compress_n_t_coarse
            ):
                upsample_factor = compress_n_t_target // compress_n_t_coarse
                self.compress_n_temporal_upsampler = nn.ConvTranspose1d(
                    compress_n_latent_n, compress_n_latent_n, kernel_size=upsample_factor, stride=upsample_factor
                )
            self.token_prediction_proj = None
        else:
            if token_prediction_dim is not None:
                if deep_token_head:
                    self.token_prediction_proj = FiLMResidualHead(
                        dim=self.dim_q,
                        out_dim=token_prediction_dim,
                        depth=int(deep_token_head_depth),
                        hidden_mult=float(deep_token_head_width_mult),
                        use_film=bool(deep_token_head_use_film),
                        dropout=float(deep_token_head_dropout),
                        norm_layer=norm_layer,
                    )
                else:
                    # Shallow 2-layer head, an explicit ablation option.
                    self.token_prediction_proj = nn.Sequential(
                        nn.Linear(self.dim_q, self.dim_q),
                        nn.GELU(),
                        nn.Linear(self.dim_q, token_prediction_dim),
                    )
            else:
                self.token_prediction_proj = None
        self.deep_token_head = bool(deep_token_head)
        self.deep_token_head_use_film = bool(deep_token_head_use_film)

        # Compress-T temporal tiling (distinct from compress_n path). For compress_t
        # tokenizers with t_comp>1 (e.g. 100ms tokenizer applied to a 200ms window →
        # 2 latent timesteps), neuron queries are tiled along the temporal axis with
        # learned per-timestep embeddings so the readout produces (B, N, T_latent, D).
        if self.token_t_latent > 1:
            if compress_n_latent_n is not None and compress_n_t_coarse is not None:
                raise ValueError(
                    "token_t_latent>1 (compress_t tiling) is mutually exclusive with "
                    "compress_n_t_coarse (compress_n tiling)."
                )
            self.token_t_embeds = nn.Parameter(torch.zeros(1, self.token_t_latent, self.dim_q))
            nn.init.normal_(self.token_t_embeds, std=self.dim_q**-0.5)

        if learnable_softplus:
            # Init w=1/a so that at init: w * softplus(a*x) = (1/a)*softplus(a*x) = nn.Softplus(beta=a)(x)
            self.activation = ParametrizedSoftplus(a=softplus_beta, w=1.0 / softplus_beta)
        else:
            self.activation = nn.Softplus(beta=softplus_beta)

        self.skip = skip_connection
        self.use_session_embedding = use_session_embedding
        self.use_subject_embedding = use_subject_embedding

        # Projection for backbone dim mismatch (created eagerly so it's in state_dict & optimizer)
        _backbone_dim = backbone_dim if backbone_dim is not None else dim
        self.backbone_proj: nn.Linear | None = (
            nn.Linear(_backbone_dim, dim, bias=False) if _backbone_dim != dim else None
        )

        # Per-session readouts
        self.readout = nn.ModuleDict()
        for k, n_neurons_raw in session_info.items():
            n_neurons = n_neurons_override[k] if n_neurons_override and k in n_neurons_override else n_neurons_raw
            self.readout[k] = SessionReadout(n_neurons, dim_head=dim_head, num_heads=num_heads)

        # Per-subject embeddings (shared across sessions from the same animal)
        self.session_to_subject: dict[str, str] = {}
        if use_subject_embedding:
            if subject_info is None:
                raise ValueError(
                    "subject_info dict (session_id → subject_id) is required when use_subject_embedding=True"
                )
            self.session_to_subject = dict(subject_info)
            unique_subjects = sorted(set(subject_info.values()))
            readout_std = self.dim_q**-0.5
            self.subject_embeddings = nn.ParameterDict(
                {subj: nn.Parameter(torch.zeros(1, 1, self.dim_q)) for subj in unique_subjects}
            )
            for p in self.subject_embeddings.values():
                nn.init.trunc_normal_(p, mean=0.0, std=readout_std, a=-3 * readout_std, b=3 * readout_std)

        # Effective token dimension for loss heads
        _loss_token_dim = compress_n_latent_n if compress_n_latent_n is not None else token_prediction_dim

        # Diffusion loss module
        self.diffloss = diffloss
        if self.diffloss is not None and _loss_token_dim is not None:
            self.diffusion_conditioning_norm = nn.LayerNorm(_loss_token_dim)
            if compress_n_latent_n is None:
                max_neurons = max(cast(SessionReadout, self.readout[k]).unique_ids for k in self.readout)
                self.diffusion_pos_embed = nn.Parameter(torch.zeros(1, max_neurons, _loss_token_dim))
                nn.init.normal_(self.diffusion_pos_embed, std=0.02)

        # Energy-score loss module (EAR)
        self.energy_loss = energy_loss
        if self.energy_loss is not None and _loss_token_dim is not None:
            self.energy_conditioning_norm = nn.LayerNorm(_loss_token_dim)
            if compress_n_latent_n is None:
                max_neurons = max(cast(SessionReadout, self.readout[k]).unique_ids for k in self.readout)
                self.energy_pos_embed = nn.Parameter(torch.zeros(1, max_neurons, _loss_token_dim))
                nn.init.normal_(self.energy_pos_embed, std=0.02)

        # Internalized loss modules
        self.use_ldns_denoiser = use_ldns_denoiser and compress_n_latent_n is not None
        self.use_full_grid_denoiser = bool(use_full_grid_denoiser) and _loss_token_dim is not None
        if self.use_full_grid_denoiser:
            # use_full_grid_denoiser is only True when _loss_token_dim is set (see above).
            assert _loss_token_dim is not None
            if loss_mode != "flow_matching":
                raise NotImplementedError(
                    "use_full_grid_denoiser=True currently supports only loss_mode='flow_matching'."
                )
            if self.use_ldns_denoiser:
                raise ValueError("use_full_grid_denoiser and use_ldns_denoiser are mutually exclusive.")
            from ..losses.full_grid_denoiser import FullGridFlowMatchingHead

            max_spatial = max(cast(SessionReadout, self.readout[k]).unique_ids for k in self.readout)
            max_time = (
                int(compress_n_t_target)
                if compress_n_t_target is not None
                else int(compress_n_t_coarse)
                if compress_n_t_coarse is not None
                else int(self.token_t_latent)
            )
            self.full_grid_flow_head = FullGridFlowMatchingHead(
                latent_dim=int(_loss_token_dim),
                conditioning_dim=int(_loss_token_dim),
                hidden_channels=int(full_grid_hidden_channels),
                num_blocks=int(full_grid_num_blocks),
                num_heads=int(full_grid_num_heads),
                mlp_ratio=int(full_grid_mlp_ratio),
                max_spatial=int(max_spatial),
                max_time=int(max_time),
                use_s4=bool(full_grid_use_s4),
                kernel_size=int(full_grid_kernel_size),
                flow_opts=flow_opts,
            )

        if self.use_ldns_denoiser and _loss_token_dim is not None:
            from ..losses.ldns_denoiser import LDNSDiffLoss, LDNSFlowMatchingHead

            # use_ldns_denoiser is only True when compress_n_latent_n is set (see above).
            assert compress_n_latent_n is not None

            if loss_mode == "flow_matching":
                self.ldns_flow_head = LDNSFlowMatchingHead(
                    latent_n=compress_n_latent_n,
                    hidden_channels=ldns_hidden_channels,
                    conditioning_dim=compress_n_latent_n,
                    num_blocks=ldns_num_blocks,
                    kernel_size=ldns_kernel_size,
                    flow_opts=flow_opts,
                )
            elif loss_mode == "diffusion":
                self.ldns_diffloss = LDNSDiffLoss(
                    latent_n=compress_n_latent_n,
                    hidden_channels=ldns_hidden_channels,
                    conditioning_dim=compress_n_latent_n,
                    num_blocks=ldns_num_blocks,
                    kernel_size=ldns_kernel_size,
                )
        elif loss_mode == "flow_matching" and _loss_token_dim is not None and not self.use_full_grid_denoiser:
            from ntt_downstream.encoding.models.enc.losses.flow_matching_head import FlowMatchingHead

            self.flow_head = FlowMatchingHead(
                token_dim=_loss_token_dim,
                conditioning_dim=_loss_token_dim,
                width=flow_head_width,
                depth=flow_head_depth,
                flow_opts=flow_opts,
            )

        # Energy + LDNS combination is not supported.
        if self.use_ldns_denoiser and loss_mode == "energy":
            raise NotImplementedError(
                "loss_mode='energy' with use_ldns_denoiser=True is not implemented. "
                "Run without LDNS or disable the energy head."
            )

        if loss_mode == "cross_entropy" and _loss_token_dim is not None:
            if ce_vocab_size is None:
                raise ValueError("ce_vocab_size is required when loss_mode='cross_entropy'")
            self.ce_vocab_size = int(ce_vocab_size)
            # Multi-codebook (residual VQ) tokenizers predict ``num_codebooks``
            # independent codes per latent position. Output a flat ``Q*V`` head
            # and split into ``Q`` independent ``V``-sized softmaxes at loss /
            # sampling time. For Q=1 this is a single V-way softmax per position.
            self.ce_head = nn.Linear(_loss_token_dim, self.ce_vocab_size * self.num_codebooks)

        self.init_weights()

    def set_ce_class_weight(self, weight: torch.Tensor) -> None:
        """Register the inverse-frequency CE class weight as a buffer.

        ``weight`` is a (V,) tensor passed to ``F.cross_entropy(weight=...)`` in
        the cross-entropy loss path. Estimated once at setup by the lit module
        (frozen tokenizer → fixed target token distribution). When this is None
        (the default) the CE loss is unweighted and byte-identical to plain CE.
        """
        # Buffers are registered as None in __init__; assign directly so the
        # tensor moves with .to(device) and is dtype-managed like other buffers.
        self.ce_class_weight = weight

    def set_bce_pos_weight(self, pos_weight: torch.Tensor) -> None:
        """Register the per-bit BCE positive weight as a buffer.

        ``pos_weight`` is a tensor broadcastable to the BSQ logits, passed to
        ``F.binary_cross_entropy_with_logits(pos_weight=...)``. Estimated once
        at setup by the lit module. When None (default) the BCE loss is
        byte-identical to the unweighted ``binary_cross_entropy_with_logits``.
        """
        self.bce_pos_weight = pos_weight

    def _cross_entropy_loss(self, logits: torch.Tensor, target_flat: torch.Tensor) -> torch.Tensor:
        """Cross-entropy with optional inverse-frequency class weighting + focal.

        ``logits`` is (P, V), ``target_flat`` is (P,) of class indices. With the
        default knobs (``ce_focal_gamma == 0`` and ``ce_class_weight is None``)
        this reduces EXACTLY to ``F.cross_entropy(logits, target_flat)`` — the
        no-op default path live training jobs depend on.

        - ``ce_class_weight`` (buffer, (V,)): inverse-frequency class weights,
          passed straight to ``F.cross_entropy(weight=...)``.
        - ``ce_focal_gamma`` (> 0): focal reshaping. ``p_t = exp(-ce)`` (the
          model's probability of the true class) and the per-position loss is
          ``(1 - p_t)**gamma * ce``, composing with class weighting if both set.
        """
        weight = self.ce_class_weight  # (V,) or None
        gamma = self.ce_focal_gamma

        if gamma == 0.0:
            # No-op default path AND the plain class-weighted path: a single
            # reduced cross_entropy call, byte-identical to the original when
            # weight is None.
            return F.cross_entropy(logits, target_flat, weight=weight)

        # Focal path: keep per-position CE so we can scale by (1 - p_t)**gamma.
        # When weight is set, ce_per_position is already the weighted CE; p_t is
        # derived from the UNWEIGHTED CE so it stays a true probability in [0, 1]
        # (the standard focal+alpha composition).
        ce_per_position = F.cross_entropy(logits, target_flat, weight=weight, reduction="none")
        with torch.no_grad():
            ce_unweighted = (
                ce_per_position if weight is None else F.cross_entropy(logits, target_flat, reduction="none")
            )
            focal_factor = (1.0 - torch.exp(-ce_unweighted)).pow(gamma)
        return (focal_factor * ce_per_position).mean()

    def set_response_stats(self, session_stats: dict[str, tuple[torch.Tensor, torch.Tensor]]):
        """Register per-session response mean/std as buffers for z-scored MSE loss.

        Args:
            session_stats: {session_id: (mean, std)} where each tensor has shape (N_neurons,).
        """
        for sid, (mean, std) in session_stats.items():
            # Use register_buffer so they move with the model and appear in state_dict
            self.register_buffer(f"_resp_mean_{sid}", mean)
            self.register_buffer(f"_resp_std_{sid}", std)

    def _get_response_stats(self, key: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return (mean, std) buffers for a session, or None if not set."""
        mean = getattr(self, f"_resp_mean_{key}", None)
        std = getattr(self, f"_resp_std_{key}", None)
        if mean is not None and std is not None:
            return mean, std
        return None

    def _predict_rates(self, o, key):
        """Project to firing rates: time-bin projection → bias → softplus.

        Used at eval time for poisson/mse_rate models (no tokenizer).
        Returns: (B, N, T) non-negative rate predictions.
        """
        o = self.neuron_proj(o)  # (B, N, D) -> (B, N, T)
        o = o + cast(SessionReadout, self.readout[key]).bias
        return self.activation(o)

    def _predict_tokens(self, o, film_cond: torch.Tensor | None = None):
        """Project to tokenizer latent space (compress_t).

        If ``token_t_latent == 1`` (default), returns ``(B, S=N, T=1, D)``.
        If ``token_t_latent > 1``, ``o`` arrives as ``(B, N*T_latent, dim_q)`` from
        the temporal-tiled readout, and we reshape to ``(B, S=N, T=T_latent, D)``.

        ``film_cond`` (optional): per-token conditioning for the deep FiLM head,
        shaped ``(1, M, dim_q)`` or ``(B, M, dim_q)``. Ignored by the
        shallow MLP head.
        """
        if self.token_prediction_proj is None:
            raise ValueError("return_token_prediction=True requires token_prediction_dim to be set")
        if isinstance(self.token_prediction_proj, FiLMResidualHead):
            z = self.token_prediction_proj(o, cond=film_cond)  # (B, M, D)
        else:
            z = self.token_prediction_proj(o)  # (B, M, D)
        if self.diffloss is not None:
            z = self.diffusion_conditioning_norm(z)
            n = z.shape[1]
            z = z + self.diffusion_pos_embed[:, :n, :]
        elif self.energy_loss is not None:
            z = self.energy_conditioning_norm(z)
            n = z.shape[1]
            z = z + self.energy_pos_embed[:, :n, :]
        if self.token_t_latent > 1:
            B, M, D = z.shape
            if M % self.token_t_latent != 0:
                raise ValueError(f"Readout output length {M} is not divisible by token_t_latent={self.token_t_latent}")
            N = M // self.token_t_latent
            z = z.reshape(B, N, self.token_t_latent, D)
            return {"token_prediction": z}
        # Unsqueeze T=1 for unified (B, S, T, D) format
        return {"token_prediction": z.unsqueeze(2)}

    def _is_ldns_tokenizer(self) -> bool:
        """Match the tokenizer's KIND for the LDNS-specific encode-long path."""
        if self.tokenizer_wrapper is None:
            return False
        return (self.tokenizer_wrapper.tokenizer_kind or "").lower() == "ldns"

    def _maybe_ldns_slice(self, t: torch.Tensor) -> torch.Tensor:
        """Apply the LDNS encode-long / predict-short token-time slice.

        When an LDNS tokenizer is active and ``compress_n_target_slice`` is set,
        the tokenizer is fed the full response window but only the leading
        ``compress_n_target_slice`` latent time-bins (dim 2 of the unified
        ``(B, S, T, D)`` layout) are used as the loss target. For all other
        tokenizers — or when the slice is unset — this is the identity so the
        general code path is unchanged.
        """
        if self.compress_n_target_slice is not None and self._is_ldns_tokenizer():
            return t[:, :, : self.compress_n_target_slice, :]
        return t

    def _assert_token_time_match(self, z: torch.Tensor, target: torch.Tensor) -> None:
        """Fail loud if the prediction and (sliced) target token-time disagree.

        Both ``z`` (token_prediction) and ``target`` use the unified
        ``(B, S, T, D)`` / ``(B, S, T, Q)`` layout where dim 2 is token-time.
        For the LDNS encode-long / predict-short recipe the target is sliced to
        ``compress_n_target_slice`` (K) while ``z`` is expected to already emit K
        (the readout's ``n_temporal_tokens`` / ``compress_n_t_target`` must be
        configured to K). A mismatch means a real misconfiguration: the
        cross-entropy path would raise an opaque flat-size error, but the
        continuous flow/diffusion/bce/mse paths flatten or broadcast over
        token-time and would SILENTLY corrupt the loss. We do not auto-slice
        ``z`` — masking the mismatch would hide the misconfiguration — so we
        raise here instead.

        Scoped to the LDNS-active case (slice set + LDNS tokenizer): other
        compress_n tokenizers may legitimately relate z's and the encode
        target's token-time differently, and this guard must not change their
        behavior.
        """
        if not (self.compress_n_target_slice is not None and self._is_ldns_tokenizer()):
            return
        z_t = z.shape[2]
        target_t = target.shape[2]
        if z_t != target_t:
            raise ValueError(
                "LDNS encode-long/predict-short length mismatch: prediction "
                f"token-time z.shape[2]={z_t} != sliced target token-time {target_t}. "
                f"These must agree. Set the readout's n_temporal_tokens "
                f"(compress_n_t_target, which controls z's T) equal to "
                f"compress_n_target_slice (={self.compress_n_target_slice}, the "
                "target K)."
            )

    def _predict_tokens_compress_n(self, o, num_chunks):
        """Project to tokenizer latent space (compress_n temporal-tiling).

        Returns ``(B, S=num_chunks, T=t_target, D=latent_n)`` in the unified format.
        """
        b = o.shape[0]
        t_coarse = self.compress_n_t_coarse
        latent_n = self.compress_n_latent_n

        o = o.view(b, num_chunks, t_coarse, self.dim_q)
        z = self.compress_n_latent_proj(o)  # (B, C, T_coarse, latent_n)

        t_target = self.compress_n_t_target
        if t_target is not None and t_target != t_coarse:
            z_flat = z.reshape(b * num_chunks, t_coarse, latent_n).permute(0, 2, 1)
            if self.compress_n_temporal_upsample == "interpolate":
                z_flat = F.interpolate(z_flat, size=t_target, mode="linear", align_corners=False)
            elif self.compress_n_temporal_upsample == "conv_transpose":
                z_flat = self.compress_n_temporal_upsampler(z_flat)
                z_flat = z_flat[:, :, :t_target]
            z = z_flat.permute(0, 2, 1).reshape(b, num_chunks, -1, latent_n)

        # z is already (B, S, T, D) — unified format

        if self.diffloss is not None:
            z = self.diffusion_conditioning_norm(z)
        elif self.energy_loss is not None:
            z = self.energy_conditioning_norm(z)

        return {"token_prediction": z}

    def _encode_and_attend(self, images, key):
        """Encode image through backbone and cross-attend with neuron queries.

        Passes neuron queries through the readout layer stack (cross-attention
        to image features, optionally interleaved with self-attention among neurons).

        Args:
            images: (B, C, H, W) static images
            key: session key

        Returns:
            tuple: (o, b) where o is post-readout neuron representations
                   (B, N, dim_q) and b is the batch size.
        """
        b = images.shape[0]

        feats = self.backbone(images)  # (B, H', W', D)

        d = feats.shape[-1]
        x = feats.view(b, -1, d)  # Flatten spatial: (B, H'*W', D)

        # Project if backbone dim != model dim
        if self.backbone_proj is not None:
            x = self.backbone_proj(x)

        # Prepare initial neuron queries with session/subject embeddings
        q = cast(SessionReadout, self.readout[key]).query  # (1, N, D)
        if self.use_session_embedding:
            q = q + cast(SessionReadout, self.readout[key]).session_embedding
        if self.use_subject_embedding:
            q = q + self.subject_embeddings[self.session_to_subject[key]]

        # Temporal tiling for multi-temporal readout
        if self.has_temporal_readout:
            q = q.unsqueeze(2) + self.temporal_embeds.unsqueeze(1)  # (1, C, T_coarse, D)
            q = q.reshape(1, -1, self.dim_q)
        elif self.token_t_latent > 1:
            # Compress-T tiling: expand neuron queries along T_latent with learned offsets
            q = q.unsqueeze(2) + self.token_t_embeds.unsqueeze(1)  # (1, N, T_latent, D)
            q = q.reshape(1, -1, self.dim_q)

        # Pre-readout per-token embedding — fed as FiLM conditioning to the
        # token-prediction head so each (neuron, temporal-slot) can bend the
        # shared projection. Shape (1, M, dim_q), broadcasts over batch.
        film_cond = q

        # Expand queries to batch size
        q = q.expand(b, -1, -1)

        # Pass through readout layer stack
        for layer_type, layer in zip(self.readout_layer_types, self.readout_layers, strict=True):
            if layer_type == "cross":
                q = layer(q, x)
            else:  # "self"
                q = layer(q)

        return q, b, film_cond

    def forward(self, images, key, **kwargs):
        """Forward pass — serves both training and evaluation.

        Training (responses given):
            Returns loss dict via ``_training_loss()``.
            Called by ``EncodingLitModule.training_step()``.

        Evaluation (no responses):
            For tokenizer-based models (return_token_prediction=True):
                Returns ``{"token_prediction": z}`` for downstream sampling
                via ``sample_tokens()``.
            For poisson/mse_rate models:
                Returns ``(B, N, T)`` predicted firing rates directly.
            Called by ``EncodingLitModule._predict_and_align()``.

        Args:
            images: (B, C, H, W) static images.
            key: session ID string.
            responses: (B, T, N) ground truth responses — if provided, computes loss.
            return_token_prediction: if True, return tokenizer-space latent predictions.
        """
        return_token_prediction = kwargs.pop("return_token_prediction", False)
        responses = kwargs.pop("responses", None)

        # Training: compute loss against ground-truth responses
        if self.loss_mode is not None and responses is not None:
            return self._training_loss(images, key, responses, **kwargs)

        # Evaluation: produce predictions only
        # Backbone encoding and cross-attention is shared for all model modes
        o, b, film_cond = self._encode_and_attend(images, key)

        # Then handle the different model modes
        if return_token_prediction:
            if self.has_temporal_readout:
                num_chunks = cast(SessionReadout, self.readout[key]).unique_ids
                return self._predict_tokens_compress_n(o, num_chunks)
            return self._predict_tokens(o, film_cond=film_cond)
        return self._predict_rates(o, key)

    def _training_loss(self, images, key, responses, **kwargs):
        """Compute training loss for all loss modes."""
        valid_samples = kwargs.pop("valid_samples", 0)
        o, b, film_cond = self._encode_and_attend(images, key)

        # Direct rate prediction modes (no tokenizer)
        if self.loss_mode in ("poisson", "mse_rate"):
            pred = self._predict_rates(o, key)  # (B, N, T)
            target = responses.transpose(1, 2)  # (B, T, N) -> (B, N, T)
            if self.loss_mode == "poisson":
                loss = F.poisson_nll_loss(
                    pred[:, :, valid_samples:],
                    target[:, :, valid_samples:],
                    log_input=False,
                )
            else:  # mse_rate
                p = pred[:, :, valid_samples:]
                t = target[:, :, valid_samples:]
                stats = self._get_response_stats(key) if self.response_zscore else None
                if stats is not None:
                    mean, std = stats  # (N,)
                    mean = mean.unsqueeze(0).unsqueeze(2)  # (1, N, 1)
                    std = std.unsqueeze(0).unsqueeze(2)  # (1, N, 1)
                    p = (p - mean) / std
                    t = (t - mean) / std
                loss = F.mse_loss(p, t)
            return {"loss": loss}

        # All tokenizer-based modes — both z and target use unified (B, S, T, D)
        if self.has_temporal_readout:
            num_chunks = cast(SessionReadout, self.readout[key]).unique_ids
            token_out = self._predict_tokens_compress_n(o, num_chunks)
            z = token_out["token_prediction"]  # (B, S, T, D)
        else:
            token_out = self._predict_tokens(o, film_cond=film_cond)
            z = token_out["token_prediction"]  # (B, S, T=1, D)

        D = z.shape[-1]

        # Cross-entropy: encode to discrete indices
        if self.loss_mode == "cross_entropy":
            with torch.no_grad():
                with torch.autocast(device_type="cuda", enabled=False):
                    encode_out = self.tokenizer_wrapper.encode(responses.float(), key, raw_discrete=True)
                    target_indices = encode_out["target"]  # (B, S, T, Q)
            # LDNS-specific encode-long / predict-short slice. Gated by tokenizer
            # kind so other compress_n tokenizers cannot accidentally pick this up.
            target_indices = self._maybe_ldns_slice(target_indices)
            # Couple prediction and (sliced) target token-time. If z's token-time
            # (z.shape[2]) disagrees with the sliced target K, cross_entropy below
            # raises an opaque size error and the continuous branch can broadcast
            # silently — so fail loud here naming the two config knobs that must
            # agree: ``compress_n_target_slice`` (target K) and the readout's
            # temporal-token count (n_temporal_tokens / compress_n_t_target → z's T).
            self._assert_token_time_match(z, target_indices)
            # Multi-codebook (residual VQ): the ce_head emits Q*V logits per
            # position; reshape to (B*S*T*Q, V) and let F.cross_entropy reduce
            # over (positions × codebooks). For Q=1 this reduces to the
            # plain (B*S*T, V) flat path.
            z_flat = z.reshape(-1, D)
            logits = self.ce_head(z_flat)  # (B*S*T, Q*V)
            logits = logits.reshape(-1, self.ce_vocab_size)  # (B*S*T*Q, V)
            target_flat = target_indices.reshape(-1).long()  # (B*S*T*Q,)
            loss = self._cross_entropy_loss(logits, target_flat)
            return {"loss": loss, "token_prediction": z}

        # Continuous tokenizer modes
        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=False):
                tokenizer_out = self.tokenizer_wrapper.encode(
                    responses.float(), key, skip_target_scaling=(self.loss_mode == "bce")
                )
                target = tokenizer_out["target"]  # (B, S, T, D) unified
        target = self._maybe_ldns_slice(target)
        # Same length-coupling guard as the CE branch. The continuous flow/
        # diffusion/bce/mse paths flatten or broadcast over token-time, so a
        # z/target mismatch here would NOT raise — it would silently corrupt the
        # loss. Assert before any reshape so a misconfiguration surfaces loudly.
        self._assert_token_time_match(z, target)
        target = target.to(z.dtype)

        aux_loss = tokenizer_out.get("aux_loss")

        energy_diag: dict | None = None

        if self.loss_mode in ("flow_matching", "diffusion", "energy"):
            if self.use_full_grid_denoiser:
                # Full-grid path: model the complete (S,T) latent field jointly.
                # The denoiser reuses Perceiver M-space blocks internally.
                if self.diffusion_batch_mul > 1:
                    z_grid = z.repeat(self.diffusion_batch_mul, 1, 1, 1)
                    target_grid = target.repeat(self.diffusion_batch_mul, 1, 1, 1)
                else:
                    z_grid = z
                    target_grid = target
                loss = self.full_grid_flow_head.compute_loss(target_grid, z_grid)
            elif self.has_temporal_readout and self.use_ldns_denoiser:
                # Sequence-aware path: LDNS conv1d heads expect (B*S, D, T).
                # Unified z is (B, S, T, D) — permute D before T at the boundary.
                B, S, T, _D = z.shape
                z_cond = z.mean(dim=2).reshape(B * S, _D)  # temporal mean → (B*S, D)
                target_seq = target.reshape(B * S, T, _D).permute(0, 2, 1)  # (B*S, D, T)
                if self.diffusion_batch_mul > 1:
                    z_cond = z_cond.repeat(self.diffusion_batch_mul, 1)
                    target_seq = target_seq.repeat(self.diffusion_batch_mul, 1, 1)
                if self.loss_mode == "flow_matching":
                    loss = self.ldns_flow_head.compute_loss(target_seq, z_cond)
                elif self.loss_mode == "diffusion":
                    loss = self.ldns_diffloss(target_seq, z_cond)
                else:  # energy
                    raise NotImplementedError("loss_mode='energy' with LDNS denoiser is not supported.")
            else:
                # Pointwise path — flatten (B, S, T, D) → (-1, D) for both modes
                z_flat = z.reshape(-1, D)
                target_flat = target.reshape(-1, D)

                if self.diffusion_batch_mul > 1:
                    z_flat = z_flat.repeat(self.diffusion_batch_mul, 1)
                    target_flat = target_flat.repeat(self.diffusion_batch_mul, 1)

                if self.loss_mode == "flow_matching":
                    loss = self.flow_head.compute_loss(target_flat, z_flat)
                elif self.loss_mode == "diffusion":
                    loss = self.diffloss(target_flat, z_flat)
                else:  # energy
                    loss, energy_diag = self.energy_loss(target_flat, z_flat, return_diagnostics=True)

        elif self.loss_mode == "bce":
            bce_target = (target > 0).float().to(z.dtype)
            # pos_weight is None by default → byte-identical to the unweighted
            # call. When set (bce_pos_weight=auto), it is the per-bit neg/pos
            # ratio estimated at setup, broadcast over the (..., n_bits) logits.
            pos_weight = self.bce_pos_weight
            if pos_weight is not None:
                pos_weight = pos_weight.to(z.dtype)
            loss = F.binary_cross_entropy_with_logits(z, bce_target, pos_weight=pos_weight)

        elif self.loss_mode == "mse":
            z_f = z.float()
            t_f = target.float()
            loss = F.mse_loss(z_f, t_f)
            _mse_extras: dict[str, torch.Tensor] = {"mse_plain": loss.detach()}
            if self.mse_center_weight > 0.0 or self.cosine_sim_weight > 0.0:
                # Subtract per-position batch mean to isolate image-conditional deviation.
                # Shape: (B, N, T, D) → mean over B, keep (N, T, D).
                z_c = z_f - z_f.mean(dim=0, keepdim=True)
                t_c = t_f - t_f.mean(dim=0, keepdim=True)
                if self.mse_center_weight > 0.0:
                    mse_centered = F.mse_loss(z_c, t_c)
                    loss = loss + self.mse_center_weight * mse_centered
                    _mse_extras["mse_centered"] = mse_centered.detach()
                if self.cosine_sim_weight > 0.0:
                    # Per-(B, N) vector of dim (T*D): image-conditional residual direction.
                    B_, N_, T_, D_ = z_c.shape
                    z_vec = z_c.reshape(B_, N_, T_ * D_)
                    t_vec = t_c.reshape(B_, N_, T_ * D_)
                    cos_sim = F.cosine_similarity(z_vec, t_vec, dim=-1)  # (B, N)
                    cos_loss = (1.0 - cos_sim).mean()
                    loss = loss + self.cosine_sim_weight * cos_loss
                    _mse_extras["cos_loss"] = cos_loss.detach()
                    _mse_extras["cos_sim_residual"] = cos_sim.detach().mean()
        elif self.loss_mode == "smooth_l1":
            loss = F.smooth_l1_loss(z.float(), target.float())
        elif self.loss_mode == "tokenizer_poisson":
            z_pos = F.softplus(z)
            # Skip initial time bins: unified format is (B, S, T, D)
            loss = F.poisson_nll_loss(z_pos[:, :, valid_samples:, :], target[:, :, valid_samples:, :], log_input=False)
        else:
            raise ValueError(f"Unknown loss_mode: {self.loss_mode}")

        if aux_loss is not None and self.tokenizer_aux_loss_weight > 0.0:
            loss = loss + self.tokenizer_aux_loss_weight * aux_loss

        # Decoder fine-tuning: reconstruction loss through trainable decoder
        recon_loss_tensor = None
        if (
            self.decoder_recon_loss_weight > 0.0
            and self.tokenizer_wrapper is not None
            and self.loss_mode in ("mse", "smooth_l1", "tokenizer_poisson")
        ):
            latent_loss = loss.detach()

            with torch.autocast(device_type="cuda", enabled=False):
                raw_latents = self.tokenizer_wrapper._target_to_tokens(z.float())
                response_hat = self.tokenizer_wrapper.decode_differentiable(raw_latents, session_key=key)
            resp_target = responses[:, valid_samples:, :].float()
            resp_pred = response_hat[:, valid_samples:, :]
            if self.decoder_recon_loss_mode == "poisson":
                recon_loss_tensor = F.poisson_nll_loss(resp_pred, resp_target, log_input=False)
            else:
                recon_loss_tensor = F.mse_loss(resp_pred, resp_target)
            loss = loss + self.decoder_recon_loss_weight * recon_loss_tensor

        result: dict[str, torch.Tensor] = {"loss": loss, "token_prediction": z, "target": target}
        if self.loss_mode == "energy" and energy_diag is not None:
            result["energy_diversity_abs"] = energy_diag["diversity_abs"]
            result["energy_diversity_ratio"] = energy_diag["diversity_ratio"]
        if recon_loss_tensor is not None:
            result["latent_loss"] = latent_loss
            result["recon_loss"] = recon_loss_tensor.detach()
        if self.loss_mode == "mse":
            for _k, _v in _mse_extras.items():
                result[_k] = _v

        # Latent diagnostics — D is always the last dim in unified format
        if self.loss_mode not in ("diffusion", "flow_matching", "energy"):
            with torch.no_grad():
                z_f = z.detach().float().reshape(-1, D)
                t_f = target.detach().float().reshape(-1, D)
                cos = F.cosine_similarity(z_f, t_f, dim=-1).mean()
                result["latent_cosine_sim"] = cos
                ss_res = ((z_f - t_f) ** 2).mean(dim=0)
                ss_tot = t_f.var(dim=0).clamp(min=1e-8)
                r2 = (1 - ss_res / ss_tot).mean()
                result["latent_r2"] = r2

        return result

    @property
    def has_temporal_readout(self) -> bool:
        """Whether the readout predicts multiple temporal tokens (T > 1).

        When True, the model uses temporal tiling in the readout (learnable
        temporal embeddings, temporal upsampling, per-time-step latent projection).
        This is set via the ``compress_n_t_coarse`` config key — the key name
        comes from compress_n tokenizers; the same architecture applies to any tokenizer producing T > 1.

        This property is the **canonical check** for multi-temporal vs single-temporal
        behavior throughout the base model. Prefer it over checking
        ``compress_n_t_coarse is not None`` directly.
        """
        return self.compress_n_t_coarse is not None

    @torch.no_grad()
    def sample_tokens(self, z, n_flow_steps=5, temperature=1.0, cfg_scale=1.0, flow_solver="euler", flow_shift=1.0):
        """Sample tokens from conditioning vectors.

        Input and output are in unified ``(B, S, T, D)`` format.
        """
        if self.loss_mode in ("mse", "smooth_l1"):
            return z
        if self.loss_mode == "tokenizer_poisson":
            return F.softplus(z)
        if self.loss_mode == "poisson":
            raise ValueError("sample_tokens is not applicable for poisson mode")
        if self.loss_mode == "bce":
            binary = (torch.sigmoid(z) > 0.5).float()
            codebook_scale = self.tokenizer_wrapper.bsq_codebook_scale
            return (binary * 2 - 1) * codebook_scale

        B, S, T, D = z.shape

        if self.use_full_grid_denoiser:
            if self.loss_mode != "flow_matching":
                raise ValueError(f"Full-grid denoiser does not support loss_mode={self.loss_mode}")
            return self.full_grid_flow_head.sample(
                z,
                n_steps=n_flow_steps,
                temperature=temperature,
                solver=flow_solver,
                shift=flow_shift,
                cfg_scale=cfg_scale,
            )

        # LDNS sequence-aware sampling: expects (B*S, D, T) internally
        if self.has_temporal_readout and self.use_ldns_denoiser:
            z_cond = z.mean(dim=2).reshape(B * S, D)  # temporal mean → (B*S, D)
            if self.loss_mode == "flow_matching":
                # LDNS head returns (B*S, D, T)
                sampled = self.ldns_flow_head.sample(
                    z_cond,
                    T=T,
                    n_steps=n_flow_steps,
                    temperature=temperature,
                    solver=flow_solver,
                    shift=flow_shift,
                    cfg_scale=cfg_scale,
                )
            elif self.loss_mode == "diffusion":
                sampled = self.ldns_diffloss.sample(z_cond, T=T, temperature=temperature, cfg=cfg_scale)
            else:
                raise ValueError(f"LDNS denoiser does not support loss_mode={self.loss_mode}")
            # (B*S, D, T) → (B, S, T, D)
            return sampled.reshape(B, S, D, T).permute(0, 1, 3, 2)

        # Pointwise path — flatten (B, S, T, D) → (-1, D)
        z_flat = z.reshape(-1, D)

        if self.loss_mode == "flow_matching":
            sampled_flat = self.flow_head.sample(
                z_flat,
                n_steps=n_flow_steps,
                temperature=temperature,
                solver=flow_solver,
                shift=flow_shift,
                cfg_scale=cfg_scale,
            )
        elif self.loss_mode == "diffusion":
            sampled_flat = self.diffloss.sample(z_flat, temperature=temperature, cfg=cfg_scale)
        elif self.loss_mode == "energy":
            sampled_flat = self.energy_loss.sample(z_flat, temperature=temperature, cfg=cfg_scale)
        elif self.loss_mode == "cross_entropy":
            # Logits: (B*S*T, Q*V) → (B*S*T, Q, V). Sample one index per Q-slot.
            logits = self.ce_head(z_flat)
            logits = logits.reshape(-1, self.num_codebooks, self.ce_vocab_size)
            if self.ce_sample_temperature <= 0:
                indices = logits.argmax(dim=-1)  # (B*S*T, Q)
            else:
                probs = F.softmax(logits / self.ce_sample_temperature, dim=-1)
                # multinomial requires 2D input — flatten Q into the batch dim,
                # sample one index per row, then unflatten.
                flat_probs = probs.reshape(-1, self.ce_vocab_size)
                idx_flat = torch.multinomial(flat_probs, 1).squeeze(-1)
                indices = idx_flat.reshape(-1, self.num_codebooks)
            return indices.reshape(B, S, T, self.num_codebooks)
        else:
            raise ValueError(f"sample_tokens not supported for loss_mode={self.loss_mode}")

        return sampled_flat.reshape(B, S, T, D)

    def init_weights(self, cutoff_factor: int = 3):
        for layer in self.readout_layers:
            cast(ReadoutCrossAttentionLayer | ReadoutSelfAttentionLayer, layer).init_weights(
                cutoff_factor=cutoff_factor
            )
