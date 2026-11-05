"""Auditory encoding model: dilated CausalDWSep backbone + branched temporal context + cross-attention readout.

Architecture:
    Spectrogram (B, T, F)
    -> CausalDWSepBlock x N (dilated TCN) -> (B, T, rank)
    -> Truncate to T_trunc = (T // M) * M
    -> Branch A: mean-pool -> (B, W, rank) -> CausalTemporalSelfAttention -> repeat_interleave(M) -> (B, T_trunc, rank)
    -> Branch B: identity (B, T_trunc, rank)
    -> Fuse: A + B -> (B, T_trunc, rank) -> reshape -> (B, W, M, rank)
    -> Per-window CrossAttentionReadout (M KV tokens) -> (B, W, N, d_out) or (B, W, N, T_tok, d_out)
    -> Loss (rate-based or tokenizer-based)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from ntt_downstream.encoding.models.enc.losses.diffloss import DiffLoss
    from ntt_downstream.encoding.models.enc.losses.energy_loss import EnergyLoss
    from ntt_downstream.encoding.models.enc.losses.flow_matching_head import FlowMatchingHead


class CausalDWSepBlock(nn.Module):
    """Depthwise-separable causal conv block with dilation.

    Spectral weighting (first block only) -> causal dilated depthwise conv
    -> threshold ReLU -> pointwise conv (cross-channel mixing) -> dropout.

    Args:
        in_channels: Input feature dimension.
        rank: Output feature dimension.
        kernel_size: Temporal conv kernel size.
        dilation: Dilation factor for temporal conv.
        dropout: Dropout probability.
    """

    def __init__(self, in_channels: int, rank: int, kernel_size: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.spectral = nn.Linear(in_channels, rank, bias=False) if in_channels != rank else None
        self.temporal = nn.Conv1d(
            rank, rank, kernel_size=kernel_size, padding=0, dilation=dilation, groups=rank, bias=False
        )
        self.pad = (kernel_size - 1) * dilation
        self.shift = nn.Parameter(torch.randn(rank) * 0.01 + 0.05)
        self.pointwise = nn.Conv1d(rank, rank, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: (B, T, in_channels). Returns: (B, T, rank)."""
        if self.spectral is not None:
            x = self.spectral(x)
        x = x.transpose(1, 2)  # (B, rank, T)
        x = F.pad(x, (self.pad, 0))  # causal padding
        x = self.temporal(x)  # depthwise conv
        x = x.transpose(1, 2)  # (B, T, rank)
        x = F.relu(x + self.shift)  # threshold ReLU
        x = x.transpose(1, 2)  # (B, rank, T)
        x = self.pointwise(x)  # cross-channel mixing
        x = x.transpose(1, 2)  # (B, T, rank)
        return self.dropout(x)


_CPU = torch.device("cpu")


def _build_rope_cache(max_seq_len: int, head_dim: int, device: torch.device = _CPU) -> torch.Tensor:
    """Precompute RoPE cos/sin cache: (max_seq_len, head_dim/2, 2)."""
    assert head_dim % 2 == 0, f"head_dim must be even for RoPE, got {head_dim}"
    theta = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(positions, theta)  # (max_seq_len, head_dim/2)
    return torch.stack([freqs.cos(), freqs.sin()], dim=-1)  # (max_seq_len, head_dim/2, 2)


def _apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding. x: (B, H, W, d_h). rope_cache: (>=W, d_h/2, 2)."""
    W = x.shape[2]
    cache = rope_cache[:W]  # (W, d_h/2, 2)
    x_pairs = x.unflatten(-1, (-1, 2))  # (B, H, W, d_h/2, 2)
    cos = cache[None, None, :, :, 0]  # (1, 1, W, d_h/2)
    sin = cache[None, None, :, :, 1]  # (1, 1, W, d_h/2)
    x0 = x_pairs[..., 0]
    x1 = x_pairs[..., 1]
    out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
    return out.flatten(-2)  # (B, H, W, d_h)


class CausalTemporalSelfAttention(nn.Module):
    """Causal self-attention over window tokens with RoPE.

    Args:
        dim: Token dimension.
        num_heads: Number of attention heads.
        n_layers: Number of stacked self-attention layers.
        mlp_ratio: MLP hidden dim multiplier.
        dropout: Dropout probability.
        rope_max_seq_len: Max sequence length for RoPE cache.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        n_layers: int = 2,
        mlp_ratio: int = 2,
        dropout: float = 0.0,
        rope_max_seq_len: int = 512,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"conv_rank ({dim}) must be divisible by num_heads ({num_heads}). "
                f"Try conv_rank={dim - dim % num_heads} or conv_rank={dim + num_heads - dim % num_heads}."
            )
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self._rope_max_seq_len = rope_max_seq_len

        self._rope_cache: torch.Tensor  # buffer; declared so pyright doesn't widen to Tensor | Module
        self.register_buffer("_rope_cache", _build_rope_cache(rope_max_seq_len, head_dim), persistent=False)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "norm1": nn.RMSNorm(dim),
                        "attn_qkv": nn.Linear(dim, 3 * dim, bias=False),
                        "attn_out": nn.Linear(dim, dim, bias=False),
                        "attn_drop": nn.Dropout(dropout),
                        "norm2": nn.RMSNorm(dim),
                        "mlp": nn.Sequential(
                            nn.Linear(dim, dim * mlp_ratio),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(dim * mlp_ratio, dim),
                        ),
                    }
                )
            )

    def _get_rope_cache(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if seq_len <= self._rope_max_seq_len:
            return self._rope_cache
        return _build_rope_cache(seq_len, self.head_dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: (B, W, dim). Returns: (B, W, dim)."""
        B, W, D = x.shape
        rope_cache = self._get_rope_cache(W, x.device)

        for layer_module in self.layers:
            layer = cast(nn.ModuleDict, layer_module)
            x_norm = layer["norm1"](x)
            qkv = layer["attn_qkv"](x_norm).reshape(B, W, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)  # (B, H, W, d_h)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            q = _apply_rope(q, rope_cache)
            k = _apply_rope(k, rope_cache)

            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn_out = layer["attn_drop"](attn_out)
            attn_out = attn_out.transpose(1, 2).reshape(B, W, D)
            attn_out = layer["attn_out"](attn_out)
            x = x + attn_out
            x = x + layer["mlp"](layer["norm2"](x))

        return x


class CrossAttentionReadout(nn.Module):
    """Per-window cross-attention: neuron queries attend to M window bin tokens.

    Rate mode: N queries -> (B, W, N, d_out) where d_out = M (bins per window).
    Tokenizer mode: N*T_tok queries -> (B, W, N, T_tok, d_out) where d_out = latent_dim.

    Args:
        n_units: Number of neurons for this session.
        dim: Window bin token dimension (backbone rank).
        dim_q: Query/attention hidden dimension.
        num_heads: Number of attention heads.
        d_out: Output dimension per neuron per window (rate mode: M, tokenizer: D).
        n_tokens_per_neuron: T_tok for tokenizer mode. None = rate mode.
        dropout: Dropout probability.
        deep_token_head: Use a deep FiLM residual head for the output projection
            instead of a single Linear. Matches the Hiera model's
            ``token_prediction_proj`` upgrade; specializes the shared projection
            per-query (per-neuron) via FiLM modulated by the learned query
            vector. Defaults to True.
        deep_token_head_depth: Residual blocks in the deep head.
        deep_token_head_width_mult: Hidden width multiplier inside each block.
        deep_token_head_use_film: Enable FiLM modulation (vs plain ResMLP).
    """

    def __init__(
        self,
        n_units: int,
        dim: int,
        dim_q: int = 64,
        num_heads: int = 4,
        d_out: int = 10,
        n_tokens_per_neuron: int | None = None,
        dropout: float = 0.0,
        deep_token_head: bool = True,
        deep_token_head_depth: int = 2,
        deep_token_head_width_mult: float = 2.0,
        deep_token_head_use_film: bool = True,
    ):
        super().__init__()
        self.n_units = n_units
        self.dim_q = dim_q
        self.d_out = d_out
        self.num_heads = num_heads
        self.n_tokens_per_neuron = n_tokens_per_neuron
        self.deep_token_head = bool(deep_token_head)

        readout_std = dim_q**-0.5
        self.queries = nn.Parameter(torch.empty(1, n_units, dim_q))
        nn.init.trunc_normal_(self.queries, mean=0.0, std=readout_std, a=-3 * readout_std, b=3 * readout_std)

        # Token position embeddings for tokenizer mode
        if n_tokens_per_neuron is not None:
            self.token_pos_emb = nn.Parameter(torch.empty(1, 1, n_tokens_per_neuron, dim_q))
            nn.init.trunc_normal_(self.token_pos_emb, mean=0.0, std=readout_std, a=-3 * readout_std, b=3 * readout_std)

        self.wq = nn.Linear(dim_q, dim_q, bias=False)
        self.wk = nn.Linear(dim, dim_q, bias=False)
        self.wv = nn.Linear(dim, dim_q, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.norm = nn.RMSNorm(dim_q)
        self.mlp = nn.Sequential(
            nn.Linear(dim_q, dim_q * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_q * 2, dim_q),
        )
        self.norm2 = nn.RMSNorm(dim_q)
        if self.deep_token_head:
            from ntt_downstream.encoding.models.enc.hiera.base_model import FiLMResidualHead

            self.out_proj = FiLMResidualHead(
                dim=dim_q,
                out_dim=d_out,
                depth=int(deep_token_head_depth),
                hidden_mult=float(deep_token_head_width_mult),
                use_film=bool(deep_token_head_use_film),
                dropout=dropout,
            )
        else:
            self.out_proj = nn.Linear(dim_q, d_out)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, window_features: torch.Tensor, query_bias: torch.Tensor | None = None) -> torch.Tensor:
        """Cross-attention readout over M bins per window.

        Args:
            window_features: (B, W, M, dim) fine-grained backbone features.
            query_bias: Optional (1, 1, dim_q) additive bias (e.g. subject embedding).

        Returns:
            Rate mode: (B, W, N, d_out).
            Tokenizer mode: (B, W, N, T_tok, d_out).
        """
        B, W, M, _ = window_features.shape
        N = self.n_units
        T_tok = self.n_tokens_per_neuron
        head_dim = self.dim_q // self.num_heads

        # Build queries
        queries = self.queries  # (1, N, dim_q)
        if query_bias is not None:
            queries = queries + query_bias

        if T_tok is not None:
            # Tokenizer mode: expand to N*T_tok queries
            q_base = queries.unsqueeze(2) + self.token_pos_emb  # (1, N, T_tok, dim_q)
            q_base = q_base.reshape(1, N * T_tok, self.dim_q)
            n_queries = N * T_tok
        else:
            q_base = queries
            n_queries = N

        q = self.wq(q_base.expand(B * W, -1, -1))  # (B*W, n_queries, dim_q)
        kv = window_features.reshape(B * W, M, -1)
        k = self.wk(kv)  # (B*W, M, dim_q)
        v = self.wv(kv)  # (B*W, M, dim_q)

        q = q.view(B * W, n_queries, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(B * W, M, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(B * W, M, self.num_heads, head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = self.attn_drop(attn_out)
        attn_out = attn_out.transpose(1, 2).reshape(B * W, n_queries, self.dim_q)

        # Residual uses raw queries (before Wq projection)
        raw_q_expanded = q_base.expand(B * W, -1, -1)
        x = self.norm(attn_out + raw_q_expanded)
        x = x + self.mlp(self.norm2(x))

        # Output projection — deep FiLM head gets the pre-attention query
        # (per-neuron identity + optional token-position embedding) as FiLM
        # conditioning so the shared projection can specialize per-query.
        if self.deep_token_head:
            out = self.out_drop(self.out_proj(x, cond=q_base))  # (B*W, n_queries, d_out)
        else:
            out = self.out_drop(self.out_proj(x))  # (B*W, n_queries, d_out)

        if T_tok is not None:
            return out.view(B, W, N, T_tok, self.d_out)
        return out.view(B, W, N, self.d_out)


class AudioEncodingModel(nn.Module):
    """Auditory encoding: dilated CausalDWSep TCN -> branched temporal context -> per-window cross-attention.

    Args:
        session_info: {session_id: n_neurons} mapping.
        n_freq_channels: Spectrogram frequency channels (32).
        conv_rank: CausalDWSepBlock output dimension.
        conv_kernel_size: Temporal conv kernel size.
        conv_dilations: Dilation per conv layer; list length determines layer count.
        window_size: Bins per window (M).
        dim_q: Cross-attention query dimension.
        num_heads: Attention heads.
        n_self_attn_layers: Depth of causal temporal self-attention.
        mlp_ratio: MLP hidden dim multiplier for self-attention.
        loss_mode: Loss mode. Rate modes: "poisson", "mse".
            Tokenizer modes: "mse_latent", "smooth_l1_latent", "flow_matching",
            "diffusion", "cross_entropy", "bce".
        use_subject_embedding: Add per-subject learnable embeddings.
        subject_info: {session_id: subject_id} mapping (required if use_subject_embedding).
        learnable_softplus: Use learnable softplus parameters.
        softplus_beta: Softplus beta.
        token_prediction_dim: When set, readout predicts latent tokens of this
            dimension instead of rate bins. Enables tokenizer loss modes.
        n_tokens_per_neuron: Number of latent tokens per neuron for tokenizer mode.
        dropout: Dropout probability.
        rope_max_seq_len: Max sequence length for RoPE cache.
    """

    RATE_LOSS_MODES = {"poisson", "mse"}
    TOKENIZER_LOSS_MODES = {
        "mse_latent",
        "smooth_l1_latent",
        "flow_matching",
        "diffusion",
        "energy",
        "cross_entropy",
        "bce",
    }

    def __init__(
        self,
        session_info: dict[str, int],
        n_freq_channels: int = 32,
        conv_rank: int = 48,
        conv_kernel_size: int = 7,
        conv_dilations: list[int] | None = None,
        window_size: int = 10,
        dim_q: int = 64,
        num_heads: int = 4,
        n_self_attn_layers: int = 2,
        mlp_ratio: int = 2,
        loss_mode: str = "poisson",
        use_subject_embedding: bool = False,
        subject_info: dict[str, str] | None = None,
        learnable_softplus: bool = False,
        softplus_beta: float = 1.0,
        token_prediction_dim: int | None = None,
        n_tokens_per_neuron: int | None = None,
        dropout: float = 0.0,
        rope_max_seq_len: int = 512,
        # Tokenizer-loss configuration
        diffusion_cfg: dict | None = None,
        flow_cfg: dict | None = None,
        energy_cfg: dict | None = None,
        ce_vocab_size: int | None = None,
        diffusion_batch_mul: int = 1,
        tokenizer_aux_loss_weight: float = 0.0,
        # Deep token-prediction head (applies to the readout's output projection).
        deep_token_head: bool = True,
        deep_token_head_depth: int = 2,
        deep_token_head_width_mult: float = 2.0,
        deep_token_head_use_film: bool = True,
    ):
        super().__init__()
        self.window_size = window_size
        self.loss_mode = loss_mode
        self.conv_rank = conv_rank
        self.dim_q = dim_q
        self.use_subject_embedding = use_subject_embedding
        self._is_rate_mode = token_prediction_dim is None
        self.diffusion_batch_mul = int(diffusion_batch_mul)
        self.tokenizer_aux_loss_weight = float(tokenizer_aux_loss_weight)
        self.ce_vocab_size = ce_vocab_size

        if conv_dilations is None:
            conv_dilations = [1, 2, 4, 8]

        blocks = []
        in_ch = n_freq_channels
        for d in conv_dilations:
            blocks.append(CausalDWSepBlock(in_ch, conv_rank, conv_kernel_size, dilation=d, dropout=dropout))
            in_ch = conv_rank
        self.conv_blocks = nn.ModuleList(blocks)

        self.temporal_attn = CausalTemporalSelfAttention(
            dim=conv_rank,
            num_heads=num_heads,
            n_layers=n_self_attn_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            rope_max_seq_len=rope_max_seq_len,
        )

        if self._is_rate_mode:
            d_out = window_size
            n_tok = None
        else:
            d_out = token_prediction_dim
            n_tok = n_tokens_per_neuron

        self.readout = nn.ModuleDict()
        for sid, n_units in session_info.items():
            self.readout[sid] = CrossAttentionReadout(
                n_units=n_units,
                dim=conv_rank,
                dim_q=dim_q,
                num_heads=num_heads,
                # None only if token_prediction_dim was left unset in a non-rate mode;
                # cast keeps runtime behaviour identical.
                d_out=cast(int, d_out),
                n_tokens_per_neuron=n_tok,
                dropout=dropout,
                deep_token_head=bool(deep_token_head),
                deep_token_head_depth=int(deep_token_head_depth),
                deep_token_head_width_mult=float(deep_token_head_width_mult),
                deep_token_head_use_film=bool(deep_token_head_use_film),
            )

        # Per-subject embeddings (shared across sessions from the same animal)
        self.session_to_subject: dict[str, str] = {}
        if use_subject_embedding:
            if subject_info is None:
                raise ValueError(
                    "subject_info dict (session_id → subject_id) is required when use_subject_embedding=True"
                )
            self.session_to_subject = dict(subject_info)
            unique_subjects = sorted(set(subject_info.values()))
            readout_std = dim_q**-0.5
            self.subject_embeddings = nn.ParameterDict(
                {subj: nn.Parameter(torch.zeros(1, 1, dim_q)) for subj in unique_subjects}
            )
            for p in self.subject_embeddings.values():
                nn.init.trunc_normal_(p, mean=0.0, std=readout_std, a=-3 * readout_std, b=3 * readout_std)

        # Learned gate for branch fusion (keeps self-attention from drowning out conv features)
        self.context_gate = nn.Parameter(torch.tensor(0.1))

        if loss_mode == "poisson":
            if learnable_softplus:
                from ntt_downstream.encoding.models.enc.hiera.base_model import ParametrizedSoftplus

                self.activation = ParametrizedSoftplus(a=softplus_beta, w=1.0 / softplus_beta)
            else:
                self.activation = nn.Softplus(beta=softplus_beta)

            # Initialize readout out_proj bias so initial predictions ≈ mean firing rate.
            # softplus(bias, beta) ≈ target_mean. For beta=1, bias=-2 gives softplus(-2)≈0.127.
            # Deep FiLM head has a bias-free final projection; for that path we
            # skip the bias init — training-step-0 predictions will start near
            # zero and the model learns the offset quickly.
            for readout_head in self.readout.values():
                out_proj = readout_head.out_proj
                if isinstance(out_proj, nn.Linear) and out_proj.bias is not None:
                    nn.init.constant_(out_proj.bias, -2.0)
        else:
            self.activation = None

        # Tokenizer-loss heads — built per loss_mode.
        # For generative modes (flow/diffusion/energy) and CE, token_prediction_dim
        # must be set so we know the latent dimensionality D.
        if loss_mode in ("flow_matching", "diffusion", "energy", "cross_entropy"):
            if token_prediction_dim is None:
                raise ValueError(
                    f"loss_mode={loss_mode!r} requires token_prediction_dim to be set "
                    "(auto-derived from tokenizer.latent_dim in configure_model)"
                )

        D = token_prediction_dim  # may be None for rate / mse_latent / bce modes

        self.flow_head: FlowMatchingHead | None = None
        self.diffloss: DiffLoss | None = None
        self.energy_loss: EnergyLoss | None = None
        self.ce_head: nn.Linear | None = None

        if loss_mode == "flow_matching":
            from ntt_downstream.encoding.models.enc.losses.flow_matching_head import FlowMatchingHead
            from ntt_downstream.encoding.models.enc.losses.flow_utils import FlowOptions

            fcfg = dict(flow_cfg or {})
            flow_opts = FlowOptions(
                time_sampling=str(fcfg.get("time_sampling", "uniform")),
                logit_normal_mean=float(fcfg.get("logit_normal_mean", 0.0)),
                logit_normal_std=float(fcfg.get("logit_normal_std", 1.0)),
                ot_coupling=bool(fcfg.get("ot_coupling", False)),
                cond_dropout=float(fcfg.get("cond_dropout", 0.0)),
            )
            assert D is not None  # guaranteed by the token_prediction_dim check above
            self.flow_head = FlowMatchingHead(
                token_dim=D,
                conditioning_dim=D,
                width=int(fcfg.get("width", 256)),
                depth=int(fcfg.get("depth", 3)),
                flow_opts=flow_opts,
            )
        elif loss_mode == "diffusion":
            from ntt_downstream.encoding.models.enc.losses.diffloss import DiffLoss

            dcfg = dict(diffusion_cfg or {})
            assert D is not None  # guaranteed by the token_prediction_dim check above
            self.diffloss = DiffLoss(
                target_channels=D,
                z_channels=D,
                depth=int(dcfg.get("depth", 3)),
                width=int(dcfg.get("width", 256)),
                num_sampling_steps=str(dcfg.get("num_sampling_steps", "100")),
                grad_checkpointing=bool(dcfg.get("grad_checkpointing", False)),
            )
        elif loss_mode == "energy":
            from ntt_downstream.encoding.models.enc.losses.energy_loss import EnergyLoss

            ecfg = dict(energy_cfg or {})
            assert D is not None  # guaranteed by the token_prediction_dim check above
            self.energy_loss = EnergyLoss(
                target_channels=D,
                z_channels=D,
                width=int(ecfg.get("width", 256)),
                depth=int(ecfg.get("depth", 3)),
                noise_channels=int(ecfg.get("noise_channels", 64)),
                alpha=float(ecfg.get("alpha", 1.5)),
                train_temperature=float(ecfg.get("train_temperature", 1.0)),
                grad_checkpointing=bool(ecfg.get("grad_checkpointing", False)),
            )
        elif loss_mode == "cross_entropy":
            if ce_vocab_size is None:
                raise ValueError(
                    "loss_mode='cross_entropy' requires ce_vocab_size (size of the "
                    "discrete tokenizer codebook). Set model.ce_vocab_size in the config."
                )
            assert D is not None  # guaranteed by the token_prediction_dim check above
            self.ce_head = nn.Linear(D, int(ce_vocab_size))

        # Set externally by LitModule for tokenizer loss modes.
        self.tokenizer_wrapper: Any = None

    def _encode(self, spectrograms: torch.Tensor) -> torch.Tensor:
        """Run backbone + branched context.

        Args: spectrograms: (B, T, F).
        Returns: (B, W, M, rank) — M fine-grained bins per window, enriched with self-attn context.
        """
        x = spectrograms
        for block in self.conv_blocks:
            x = block(x)

        M = self.window_size
        T = x.shape[1]
        T_trunc = (T // M) * M
        x = x[:, :T_trunc, :]  # (B, T_trunc, rank)

        # Branch A: mean-pool -> self-attention -> broadcast
        x_pooled = x.view(x.shape[0], T_trunc // M, M, -1).mean(dim=2)  # (B, W, rank)
        x_context = self.temporal_attn(x_pooled)  # (B, W, rank)
        x_broadcast = x_context.repeat_interleave(M, dim=1)  # (B, T_trunc, rank)

        # Fuse: fine-grained + gated global context
        x_fused = x + self.context_gate * x_broadcast  # (B, T_trunc, rank)

        return x_fused.view(x.shape[0], T_trunc // M, M, -1)  # (B, W, M, rank)

    def forward(
        self,
        spectrograms: torch.Tensor,
        session_key: str,
        responses: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        tokenizer_targets: dict | None = None,
        **kwargs,
    ) -> torch.Tensor | dict:
        """Forward pass.

        Training (responses provided): returns {"loss": scalar, ...}.
        Inference (no responses): returns predicted tensor.
            - Rate mode: (B, T_trunc, N) predicted rates.
            - Tokenizer mode: (B, W, N, T_tok, D) raw latent predictions.
        """
        B = spectrograms.shape[0]
        T_in = spectrograms.shape[1]
        M = self.window_size
        T_trunc = (T_in // M) * M

        window_features = self._encode(spectrograms)  # (B, W, M, rank)
        readout = cast(CrossAttentionReadout, self.readout[session_key])

        # Build query bias from subject embedding only
        query_bias = None
        if self.use_subject_embedding:
            query_bias = self.subject_embeddings[self.session_to_subject[session_key]]

        out = readout(window_features, query_bias=query_bias)
        N = readout.n_units

        if self._is_rate_mode:
            pred = self._readout_to_rates(out, B, T_trunc, N)
            if responses is None:
                return pred
            return self._rate_loss(pred, responses, lengths, T_trunc)
        else:
            # Tokenizer mode: out is (B, W, N, T_tok, D) or (B, W, N, D).
            if responses is None:
                return out
            return self._tokenizer_loss(out, responses, session_key, lengths, tokenizer_targets=tokenizer_targets)

    # ------------------------------------------------------------------
    # Rate-mode helpers
    # ------------------------------------------------------------------

    def _readout_to_rates(self, out: torch.Tensor, B: int, T_trunc: int, N: int) -> torch.Tensor:
        """Reshape readout (B, W, N, M) -> (B, T, N) and apply activation."""
        pred = out.permute(0, 1, 3, 2).reshape(B, T_trunc, N)
        if self.activation is not None:
            pred = self.activation(pred)
        return pred

    def _build_mask(self, lengths: torch.Tensor | None, T_trunc: int, pred: torch.Tensor) -> torch.Tensor | None:
        """Build a (B, T, N) boolean mask from per-sample lengths."""
        if lengths is None:
            return None
        t_indices = torch.arange(T_trunc, device=lengths.device).unsqueeze(0)
        mask = t_indices < lengths.unsqueeze(1).clamp(max=T_trunc)
        return mask.unsqueeze(2).expand_as(pred)

    def _rate_loss(
        self,
        pred: torch.Tensor,
        responses: torch.Tensor,
        lengths: torch.Tensor | None,
        T_trunc: int,
    ) -> dict:
        """Compute rate-prediction loss (Poisson or MSE)."""
        target = responses[:, :T_trunc, :]
        mask = self._build_mask(lengths, T_trunc, pred)

        if self.loss_mode == "poisson":
            loss = self._poisson_loss(pred, target, mask)
        elif self.loss_mode == "mse":
            loss = self._mse_loss(pred, target, mask)
        else:
            raise ValueError(f"Unsupported rate loss_mode: {self.loss_mode}")

        return {"loss": loss}

    def _poisson_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        eps = 1e-8
        elementwise = pred - target * torch.log(pred + eps)
        if mask is not None:
            return (elementwise * mask).sum() / mask.sum().clamp(min=1)
        return elementwise.mean()

    def _mse_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        elementwise = (pred - target) ** 2
        if mask is not None:
            return (elementwise * mask).sum() / mask.sum().clamp(min=1)
        return elementwise.mean()

    # ------------------------------------------------------------------
    # Tokenizer-mode helpers
    # ------------------------------------------------------------------

    def _tokenizer_loss(
        self,
        z: torch.Tensor,
        responses: torch.Tensor,
        session_key: str,
        lengths: torch.Tensor | None,
        tokenizer_targets: dict | None = None,
    ) -> dict:
        """Compute loss for tokenizer-based modes.

        Args:
            z: (B, W, N, T_tok, D) or (B, W, N, D) model latent predictions from readout.
            responses: (B, T, N) ground-truth neural responses.
            session_key: Session identifier.
            lengths: (B,) valid lengths or None (currently unused).
        """
        if self.tokenizer_wrapper is None and tokenizer_targets is None:
            raise RuntimeError("tokenizer_wrapper must be set for tokenizer loss modes")

        if tokenizer_targets is None:
            assert self.tokenizer_wrapper is not None  # ensured by the RuntimeError above
            with torch.no_grad():
                tok_out = self.tokenizer_wrapper.encode_windowed(
                    responses.float(),
                    session_key,
                    raw_discrete=(self.loss_mode == "cross_entropy"),
                    skip_target_scaling=(self.loss_mode == "bce"),
                )
                target = tok_out["target"]  # (B, W*T_tok, S, D), or (B, W*T_tok, S, 1) long for CE
        else:
            tok_out = dict(tokenizer_targets)
            target = tok_out["target"]

        T_tok = tok_out["tokens_per_window"]
        W = tok_out["n_windows"]

        # Reshape predictions to match target
        if z.ndim == 5:
            # (B, W, N, T_tok, D) -> (B, W*T_tok, N, D)
            B_z, W_z, N_z, Tt, D_z = z.shape
            z = z.permute(0, 1, 3, 2, 4).reshape(B_z, W_z * Tt, N_z, D_z)
        elif T_tok > 1:
            # (B, W, N, D) with one prediction per window: repeat across the T_tok tokens
            z = z.unsqueeze(2).expand(-1, -1, T_tok, -1, -1)  # (B, W, T_tok, S, D)
            z = z.reshape(z.shape[0], W * T_tok, z.shape[3], z.shape[4])  # (B, W*T_tok, S, D)

        z_flat = z.reshape(-1, z.shape[-1])
        target_flat = target.reshape(-1, target.shape[-1])

        aux_loss = tok_out.get("aux_loss") if isinstance(tok_out, dict) else None
        energy_diag: dict | None = None
        result: dict = {}

        if self.loss_mode == "mse_latent":
            # fp32 cast: under AMP autocast the readout output is bf16, which
            # can overflow / lose precision on large latents. The visual-side
            # head in hiera/base_model.py does the same.
            loss = F.mse_loss(z_flat.float(), target_flat.float())
        elif self.loss_mode == "smooth_l1_latent":
            loss = F.smooth_l1_loss(z_flat.float(), target_flat.float())
        elif self.loss_mode == "bce":
            bce_target = (target_flat > 0).float().to(z_flat.dtype)
            loss = F.binary_cross_entropy_with_logits(z_flat, bce_target)
        elif self.loss_mode in ("flow_matching", "diffusion", "energy"):
            z_head = z_flat
            target_head = target_flat.to(z_flat.dtype)
            if self.diffusion_batch_mul > 1:
                z_head = z_head.repeat(self.diffusion_batch_mul, 1)
                target_head = target_head.repeat(self.diffusion_batch_mul, 1)
            if self.loss_mode == "flow_matching":
                assert self.flow_head is not None
                loss = self.flow_head.compute_loss(target_head, z_head)
            elif self.loss_mode == "diffusion":
                assert self.diffloss is not None
                loss = self.diffloss(target_head, z_head)
            else:  # energy
                assert self.energy_loss is not None
                loss, energy_diag = self.energy_loss(target_head, z_head, return_diagnostics=True)
        elif self.loss_mode == "cross_entropy":
            # target is long indices with trailing dim 1 in (B, W*T_tok, S, 1)
            target_indices = target.reshape(-1).long()
            assert self.ce_head is not None
            logits = self.ce_head(z_flat)
            loss = F.cross_entropy(logits, target_indices)
        else:
            raise ValueError(f"Tokenizer loss mode '{self.loss_mode}' not yet implemented")

        # Aux-loss pass-through
        if aux_loss is not None and self.tokenizer_aux_loss_weight > 0.0:
            loss = loss + self.tokenizer_aux_loss_weight * aux_loss

        result["loss"] = loss
        if energy_diag is not None:
            result["energy_diversity_abs"] = energy_diag["diversity_abs"]
            result["energy_diversity_ratio"] = energy_diag["diversity_ratio"]

        # Diagnostics for non-generative, continuous-target modes — visible to MLflow.
        # Generative modes predict velocity/noise/samples, not the clean latent, so
        # direct cos-sim / R^2 on (z, target) is not meaningful.
        # Cross-entropy has a discrete target, same reasoning.
        if self.loss_mode in ("mse_latent", "smooth_l1_latent", "bce"):
            with torch.no_grad():
                z_f = z_flat.detach().float()
                t_f = target_flat.detach().float()
                if t_f.shape == z_f.shape:
                    cos = F.cosine_similarity(z_f, t_f, dim=-1).mean()
                    result["latent_cosine_sim"] = cos
                    ss_res = ((z_f - t_f) ** 2).mean(dim=0)
                    ss_tot = t_f.var(dim=0).clamp(min=1e-8)
                    r2 = (1 - ss_res / ss_tot).mean()
                    result["latent_r2"] = r2

        return result

    @torch.no_grad()
    def sample_tokens(
        self,
        z: torch.Tensor,
        n_flow_steps: int = 5,
        temperature: float = 1.0,
        cfg_scale: float = 1.0,
        flow_solver: str = "euler",
        flow_shift: float = 1.0,
    ) -> torch.Tensor:
        """Sample tokens from per-window conditioning latents.

        Args:
            z: ``(B, W, N, T_tok, D)`` or ``(B, W, N, D)`` conditioning latents
               emitted by the readout in tokenizer mode.

        Returns:
            Continuous modes: tokens of the same shape as ``z``.
            Cross-entropy: discrete indices of shape ``(B, W, N, T_tok)`` or
            ``(B, W, N)`` depending on whether ``T_tok > 1``.
        """
        has_t_tok = z.ndim == 5
        if has_t_tok:
            B, W, N, T_tok, D = z.shape
        else:
            B, W, N, D = z.shape
            T_tok = 1

        if self.loss_mode in ("mse_latent", "smooth_l1_latent"):
            return z

        if self.loss_mode == "bce":
            codebook_scale = getattr(self.tokenizer_wrapper, "bsq_codebook_scale", 1.0)
            binary = (torch.sigmoid(z) > 0.5).float()
            return (binary * 2 - 1) * codebook_scale

        z_flat = z.reshape(-1, D)

        if self.loss_mode == "flow_matching":
            assert self.flow_head is not None
            sampled_flat = self.flow_head.sample(
                z_flat,
                n_steps=n_flow_steps,
                temperature=temperature,
                solver=flow_solver,
                shift=flow_shift,
                cfg_scale=cfg_scale,
            )
        elif self.loss_mode == "diffusion":
            assert self.diffloss is not None
            sampled_flat = self.diffloss.sample(z_flat, temperature=temperature, cfg=cfg_scale)
        elif self.loss_mode == "energy":
            assert self.energy_loss is not None
            sampled_flat = self.energy_loss.sample(z_flat, temperature=temperature, cfg=cfg_scale)
        elif self.loss_mode == "cross_entropy":
            assert self.ce_head is not None
            logits = self.ce_head(z_flat)
            indices = logits.argmax(dim=-1)
            if has_t_tok:
                return indices.reshape(B, W, N, T_tok)
            return indices.reshape(B, W, N)
        else:
            raise ValueError(f"sample_tokens not supported for loss_mode={self.loss_mode!r}")

        if has_t_tok:
            return sampled_flat.reshape(B, W, N, T_tok, D)
        return sampled_flat.reshape(B, W, N, D)
