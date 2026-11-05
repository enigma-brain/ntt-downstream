"""Masked Autoregressive (MAR) Neural Encoder for static images.

Bidirectional transformer jointly attends over image features and neural tokens,
then a flow matching head generates masked tokens conditioned on the transformer's
hidden states. This enables inter-neuron conditioning during generation.

The masked-autoregressive scheme (truncated-normal mask ratio, random unmasking
order, per-token diffusion batch multiplier) follows MAR (https://github.com/LTH14/mar,
MIT License, Copyright (c) 2024 Tianhong Li; Li et al., 2024); the transformer is a
plain bidirectional encoder, and the token head is DiffLoss, FlowMatchingHead or
EnergyLoss. Modified for ntt-downstream: neural tokens are conditioned on image features.
"""

from __future__ import annotations

import math
from typing import Any, cast

import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..losses.diffloss import DiffLoss
from ..losses.flow_matching_head import FlowMatchingHead  # noqa: F401

# ---------------------------------------------------------------------------
# Bidirectional Transformer
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Pre-norm bidirectional transformer block with GELU MLP."""

    def __init__(self, dim, num_heads, mlp_ratio: float = 4, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=attn_dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(proj_dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + h
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        return x


class MARDecoder(nn.Module):
    """Stack of bidirectional TransformerBlocks + final LayerNorm."""

    def __init__(self, dim, depth=3, num_heads=8, mlp_ratio: float = 4, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(dim, num_heads, mlp_ratio, attn_dropout=attn_dropout, proj_dropout=proj_dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Per-session readout
# ---------------------------------------------------------------------------


class MARSessionReadout(nn.Module):
    """Per-session embeddings for the MAR model.

    Stores per-session spatial embeddings (one per neuron or per chunk)
    and a session-level embedding broadcast to all tokens.

    The ``n_spatial`` attribute is the number of spatial units: N neurons
    for compress_t tokenizers, or num_chunks for compress_n.
    """

    def __init__(self, n_spatial: int, mar_dim: int):
        super().__init__()
        self.n_spatial = n_spatial
        # neuron_embed is part of the checkpoint state-dict keys.
        self.neuron_embed = nn.Parameter(torch.randn(1, n_spatial, mar_dim) * 0.02)
        self.session_embed = nn.Parameter(torch.randn(1, 1, mar_dim) * 0.02)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class ImageEncodingHieraMAR(nn.Module):
    """Masked Autoregressive model for image → neural response prediction.

    Uses a bidirectional transformer over concatenated image + neural tokens,
    with flow matching to generate masked neural tokens.

    Args:
        backbone: Hiera backbone module.
        tokenizer_wrapper: NTT tokenizer wrapper.
        session_info: dict mapping session_id → n_neurons.
        dim: Backbone output feature dimension.
        mar_dim: MAR transformer hidden dimension.
        mar_depth: Number of MAR transformer layers.
        mar_num_heads: Number of MAR attention heads.
        loss_mode: "flow_matching", "diffusion", "poisson", or "cross_entropy".
    """

    def __init__(
        self,
        backbone,
        tokenizer_wrapper: Any = None,
        session_info: dict[str, int] | None = None,
        dim: int = 192,
        backbone_dim: int | None = None,
        mar_dim: int = 256,
        mar_depth: int = 3,
        mar_num_heads: int = 8,
        mar_mlp_ratio: float = 4,
        mar_attn_dropout: float = 0.0,
        mar_proj_dropout: float = 0.0,
        flow_head_width: int = 256,
        flow_head_depth: int = 3,
        flow_opts=None,
        # Energy loss (EAR) — arXiv:2505.07812
        energy_head_width: int = 256,
        energy_head_depth: int = 3,
        energy_noise_channels: int = 64,
        energy_alpha: float = 1.5,
        energy_train_temperature: float = 1.0,
        energy_grad_checkpointing: bool = False,
        # Masking schedule
        mask_ratio_min: float = 0.7,
        mask_ratio_std: float = 0.25,
        # Diffusion pos embed
        use_diffusion_pos_embed: bool = True,
        # Buffer tokens
        num_buffer_tokens: int = 8,
        # Cosine unmasking
        unmask_schedule: str = "cosine",
        # Diffusion batch multiplier
        diffusion_batch_mul: int = 4,
        # Loss mode
        loss_mode: str = "flow_matching",
        num_sampling_steps: str = "100",
        # Cross-entropy options
        ce_vocab_size: int | None = None,
        ce_sample_temperature: float = 1.0,
        # Subject embedding
        use_subject_embedding: bool = False,
        subject_info: dict[str, str] | None = None,
        # Unified spatial-temporal token grid
        n_neurons_override: dict[str, int] | None = None,
        n_temporal_tokens: int = 1,
    ):
        super().__init__()

        if session_info is None:
            raise ValueError("session_info dict (session_id → n_neurons) is required")

        self.backbone = backbone
        self.loss_mode = loss_mode
        self.tokenizer_wrapper = tokenizer_wrapper
        self.n_temporal_tokens = n_temporal_tokens

        if loss_mode == "poisson":
            # Raw firing-rate tokens: one scalar rate per neuron/time bin.
            self.token_dim = 1
        elif loss_mode == "cross_entropy":
            if tokenizer_wrapper is None:
                raise ValueError("loss_mode='cross_entropy' requires a tokenizer_wrapper")
            self.token_dim = 1
            self.ce_vocab_size = ce_vocab_size or self._get_vocab_size()
            self.ce_sample_temperature = ce_sample_temperature
        else:
            if tokenizer_wrapper is None:
                raise ValueError(f"loss_mode={loss_mode!r} requires a tokenizer_wrapper")
            self.token_dim = self._get_token_dim()

        self.dim = dim
        self.mar_dim = mar_dim

        # Truncated normal masking ratio generator
        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - 1.0) / mask_ratio_std, 0, loc=1.0, scale=mask_ratio_std
        )
        self.unmask_schedule = unmask_schedule
        self.diffusion_batch_mul = diffusion_batch_mul

        # Projections
        self.video_proj = nn.Linear(dim, mar_dim)
        if loss_mode == "cross_entropy":
            self.neural_embed = nn.Embedding(self.ce_vocab_size, mar_dim)
        else:
            self.neural_proj = nn.Linear(self.token_dim, mar_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, mar_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Buffer tokens
        self.num_buffer_tokens = num_buffer_tokens
        if num_buffer_tokens > 0:
            self.buffer_tokens = nn.Parameter(torch.zeros(1, num_buffer_tokens, mar_dim))
            nn.init.normal_(self.buffer_tokens, std=0.02)

        # MAR decoder
        self.mar_decoder = MARDecoder(
            dim=mar_dim,
            depth=mar_depth,
            num_heads=mar_num_heads,
            mlp_ratio=mar_mlp_ratio,
            attn_dropout=mar_attn_dropout,
            proj_dropout=mar_proj_dropout,
        )

        # Shared temporal embedding (when n_temporal_tokens > 1)
        if n_temporal_tokens > 1:
            self.time_embed = nn.Parameter(torch.zeros(1, n_temporal_tokens, mar_dim))
            nn.init.normal_(self.time_embed, std=0.02)
        else:
            self.time_embed = None

        # Per-session spatial units: N neurons (compress_t) or num_chunks (compress_n)
        spatial_info: dict[str, int] = {}
        for k, n_neurons in session_info.items():
            spatial_info[k] = n_neurons_override[k] if n_neurons_override and k in n_neurons_override else n_neurons
        self._spatial_info = spatial_info

        # Diffusion positional embedding — sized to max(S * T) across sessions
        self.use_diffusion_pos_embed = use_diffusion_pos_embed
        if use_diffusion_pos_embed:
            max_seq = max(spatial_info.values()) * n_temporal_tokens
            self.diffusion_pos_embed = nn.Parameter(torch.zeros(1, max_seq, mar_dim))
            nn.init.normal_(self.diffusion_pos_embed, std=0.02)

        # Loss head
        if loss_mode == "flow_matching":
            self.flow_head = FlowMatchingHead(
                token_dim=self.token_dim,
                conditioning_dim=mar_dim,
                width=flow_head_width,
                depth=flow_head_depth,
                flow_opts=flow_opts,
            )
        elif loss_mode == "diffusion":
            self.diffloss = DiffLoss(
                target_channels=self.token_dim,
                z_channels=mar_dim,
                width=flow_head_width,
                depth=flow_head_depth,
                num_sampling_steps=str(num_sampling_steps),
            )
        elif loss_mode == "energy":
            from ..losses.energy_loss import EnergyLoss

            self.energy_loss = EnergyLoss(
                target_channels=self.token_dim,
                z_channels=mar_dim,
                width=energy_head_width,
                depth=energy_head_depth,
                noise_channels=energy_noise_channels,
                alpha=energy_alpha,
                train_temperature=energy_train_temperature,
                grad_checkpointing=energy_grad_checkpointing,
            )
        elif loss_mode == "poisson":
            self.poisson_head = nn.Linear(mar_dim, self.token_dim)
        elif loss_mode in ("mse", "smooth_l1"):
            self.mse_head = nn.Sequential(
                nn.Linear(mar_dim, mar_dim),
                nn.GELU(),
                nn.Linear(mar_dim, self.token_dim),
            )
        elif loss_mode == "cross_entropy":
            self.ce_head = nn.Linear(mar_dim, self.ce_vocab_size)
        else:
            raise ValueError(f"Unknown loss_mode: {loss_mode}")

        # Projection for backbone dim mismatch (created eagerly so it's in state_dict & optimizer)
        _backbone_dim = backbone_dim if backbone_dim is not None else dim
        self.backbone_proj: nn.Linear | None = (
            nn.Linear(_backbone_dim, dim, bias=False) if _backbone_dim != dim else None
        )

        # Per-session readouts (spatial_embed + session_embed)
        self.readout = nn.ModuleDict()
        for k, n_spatial in spatial_info.items():
            self.readout[k] = MARSessionReadout(n_spatial=n_spatial, mar_dim=mar_dim)

        # Per-subject embeddings (shared across sessions from the same animal)
        self.use_subject_embedding = use_subject_embedding
        self.session_to_subject: dict[str, str] = {}
        if use_subject_embedding:
            if subject_info is None:
                raise ValueError(
                    "subject_info dict (session_id → subject_id) is required when use_subject_embedding=True"
                )
            self.session_to_subject = dict(subject_info)
            unique_subjects = sorted(set(subject_info.values()))
            self.subject_embeddings = nn.ParameterDict(
                {subj: nn.Parameter(torch.randn(1, 1, mar_dim) * 0.02) for subj in unique_subjects}
            )

    def _get_embeds(self, readout, key):
        """Build per-token embeddings for the ``(S * T)`` token sequence.

        Factored as ``spatial_embed[s] + time_embed[t] + session_embed``.
        When ``n_temporal_tokens == 1`` (compress_t with t_comp=1), this
        reduces to the original ``neuron_embed + session_embed``.

        Returns:
            ``(1, S * T, mar_dim)`` embedding tensor.
        """
        S = readout.n_spatial
        T = self.n_temporal_tokens
        spatial = readout.neuron_embed  # (1, S, mar_dim)

        if T > 1 and self.time_embed is not None:
            # Tile: (1, S, 1, D) + (1, 1, T, D) → (1, S, T, D) → (1, S*T, D)
            embeds = spatial.unsqueeze(2) + self.time_embed.unsqueeze(1)  # (1, S, T, D)
            embeds = embeds.reshape(1, S * T, self.mar_dim)
        else:
            embeds = spatial  # (1, S, mar_dim) — T==1 so S*T == S

        embeds = embeds + readout.session_embed
        if self.use_subject_embedding:
            subj_id = self.session_to_subject[key]
            embeds = embeds + self.subject_embeddings[subj_id]
        return embeds

    def _get_token_dim(self):
        spec = getattr(self.tokenizer_wrapper.tokenizer, "spec", None)
        if spec is None:
            raise ValueError("Tokenizer must have a spec attribute to determine token_dim")
        latent_dim = getattr(spec, "latent_dim", None) or getattr(spec, "z_dim", None)
        if latent_dim is None:
            raise ValueError("Tokenizer spec must have 'latent_dim' or 'z_dim'")
        return int(latent_dim)

    def _get_vocab_size(self):
        tok = self.tokenizer_wrapper.tokenizer
        if hasattr(tok, "get_vocab_size"):
            return int(tok.get_vocab_size())
        if self.tokenizer_wrapper.discrete_target_vocab_size is not None:
            return int(self.tokenizer_wrapper.discrete_target_vocab_size)
        raise ValueError("Cannot determine vocab size. Set ce_vocab_size explicitly.")

    def _prepend_buffer(self, x):
        if self.num_buffer_tokens <= 0:
            return x
        B = x.shape[0]
        return torch.cat([self.buffer_tokens.expand(B, -1, -1), x], dim=1)

    def _strip_buffer(self, x):
        if self.num_buffer_tokens <= 0:
            return x
        return x[:, self.num_buffer_tokens :, :]

    def encode_image(self, images, key):
        """Encode static images into feature tokens.

        Args:
            images: (B, C, H, W) static images
            key: session key
        Returns:
            image_tokens (B, S, mar_dim)
        """
        b = images.shape[0]

        x = self.backbone(images)  # (B, H', W', D)

        d = x.shape[-1]
        x = x.view(b, -1, d)

        # Project if backbone dim != model dim
        if self.backbone_proj is not None:
            x = self.backbone_proj(x)

        image_tokens = self.video_proj(x)  # (B, S, mar_dim)
        return image_tokens

    def _encode_responses(self, responses, key, image_dtype):
        """Encode ground-truth responses into the flat token sequence ``(B, N_tok, D)``.

        For both compress_t and compress_n, produces a unified flat layout via
        :meth:`TokenizerWrapper.encode` → reshape ``(B, S, T, D)`` → ``(B, S*T, D)``.

        Returns:
            ``(gt_tokens, N_tokens)`` where ``gt_tokens`` is ``(B, N_tok, D)``
            (float for continuous, long for cross_entropy) and ``N_tok = S * T``.
        """
        B = responses.shape[0]

        if self.loss_mode == "poisson":
            if responses.ndim != 3:
                raise ValueError(f"Poisson MAR expects responses with shape (B, T, N), got {tuple(responses.shape)}")
            if responses.shape[1] != self.n_temporal_tokens:
                raise ValueError(
                    "Poisson MAR response length does not match n_temporal_tokens: "
                    f"responses has T={responses.shape[1]}, model has T={self.n_temporal_tokens}"
                )
            gt_tokens = responses.permute(0, 2, 1).unsqueeze(-1).to(image_dtype)  # (B, N, T, 1)
            return gt_tokens.reshape(B, -1, 1), gt_tokens.shape[1] * gt_tokens.shape[2]

        if self.loss_mode == "cross_entropy":
            with torch.no_grad():
                with torch.autocast(device_type="cuda", enabled=False):
                    enc = self.tokenizer_wrapper.encode(responses.float(), key, raw_discrete=True)
                    target_4d = enc["target"]  # (B, S, T, ...)
            gt_tokens = target_4d.reshape(B, -1).long()  # (B, S*T)
            return gt_tokens, gt_tokens.shape[1]

        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=False):
                enc = self.tokenizer_wrapper.encode(responses.float(), key)
                target_4d = enc["target"]  # (B, S, T, D)
        # Flatten spatial-temporal grid: (B, S, T, D) → (B, S*T, D)
        gt_tokens = target_4d.reshape(B, -1, target_4d.shape[-1]).to(image_dtype)
        return gt_tokens, gt_tokens.shape[1]

    def forward(self, images, key, responses=None, **kwargs):
        """Training forward pass.

        Tokens are laid out as a flat ``(S * T)`` sequence where S is the
        number of spatial units (neurons or chunks) and T is the number of
        temporal tokens.  Masking is uniform over all ``S * T`` positions.

        Args:
            images: (B, C, H, W) static images
            key: session key
            responses: (B, T_data, N) ground truth responses — required for training
        Returns:
            dict with "loss" scalar
        """
        if responses is None:
            raise ValueError("responses required for training forward pass")

        B = images.shape[0]
        readout = cast(MARSessionReadout, self.readout[key])

        # 1. Encode image
        image_tokens = self.encode_image(images, key)

        # 2. Encode responses to flat token sequence (B, N_tok, D)
        gt_tokens, N_tokens = self._encode_responses(responses, key, image_tokens.dtype)

        # 3. Build per-token embeddings (1, N_tok, mar_dim)
        embeds = self._get_embeds(readout, key)  # (1, S*T, mar_dim)

        # 4. Random masking — truncated normal biased toward high mask ratios
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked = int(max(1, round(N_tokens * mask_rate)))
        mask = torch.zeros(B, N_tokens, dtype=torch.bool, device=images.device)
        for i in range(B):
            perm = torch.randperm(N_tokens, device=images.device)[:num_masked]
            mask[i, perm] = True

        if self.loss_mode == "cross_entropy":
            neural_projected = self.neural_embed(gt_tokens)
        else:
            neural_projected = self.neural_proj(gt_tokens)

        neural_tokens = torch.where(
            mask.unsqueeze(-1),
            self.mask_token.expand(B, N_tokens, -1) + embeds[:, :N_tokens, :],
            neural_projected + embeds[:, :N_tokens, :],
        )

        # 5. Concatenate [buffer] + image + neural tokens → MAR decoder
        combined = torch.cat([image_tokens, neural_tokens], dim=1)
        combined = self._prepend_buffer(combined)
        decoded = self.mar_decoder(combined)
        decoded = self._strip_buffer(decoded)

        # 6. Extract conditioning for masked positions
        S_img = image_tokens.shape[1]
        neural_hidden = decoded[:, S_img:, :]

        if self.use_diffusion_pos_embed:
            neural_hidden = neural_hidden + self.diffusion_pos_embed[:, :N_tokens, :]

        masked_conditioning = neural_hidden[mask]

        if self.loss_mode == "cross_entropy":
            masked_targets_ce = gt_tokens[mask]
        else:
            masked_targets = gt_tokens[mask]

        if self.diffusion_batch_mul > 1 and self.loss_mode in ("flow_matching", "diffusion", "energy"):
            masked_conditioning = masked_conditioning.repeat(self.diffusion_batch_mul, 1)
            masked_targets = masked_targets.repeat(self.diffusion_batch_mul, 1)

        # 7. Compute loss
        energy_diag: dict | None = None
        if self.loss_mode == "flow_matching":
            loss = self.flow_head.compute_loss(masked_targets, masked_conditioning)
        elif self.loss_mode == "diffusion":
            loss = self.diffloss(masked_targets, masked_conditioning)
        elif self.loss_mode == "energy":
            loss, energy_diag = self.energy_loss(masked_targets, masked_conditioning, return_diagnostics=True)
        elif self.loss_mode == "mse":
            predicted = self.mse_head(masked_conditioning)
            loss = F.mse_loss(predicted.float(), masked_targets.float())
        elif self.loss_mode == "smooth_l1":
            predicted = self.mse_head(masked_conditioning)
            loss = F.smooth_l1_loss(predicted.float(), masked_targets.float())
        elif self.loss_mode == "poisson":
            predicted_log_rate = self.poisson_head(masked_conditioning)
            loss = F.poisson_nll_loss(predicted_log_rate, masked_targets, log_input=True)
        elif self.loss_mode == "cross_entropy":
            logits = self.ce_head(masked_conditioning)
            loss = F.cross_entropy(logits, masked_targets_ce)

        result = {"loss": loss}
        if self.loss_mode == "energy" and energy_diag is not None:
            result["energy_diversity_abs"] = energy_diag["diversity_abs"]
            result["energy_diversity_ratio"] = energy_diag["diversity_ratio"]

        # Latent diagnostics for direct prediction modes (MSE/smooth_l1).
        if self.loss_mode in ("mse", "smooth_l1"):
            with torch.no_grad():
                pred_f = predicted.detach().float()
                tgt_f = masked_targets[: pred_f.shape[0]].detach().float()
                cos = F.cosine_similarity(pred_f, tgt_f, dim=-1).mean()
                result["latent_cosine_sim"] = cos
                ss_res = ((pred_f - tgt_f) ** 2).mean(dim=0)
                ss_tot = tgt_f.var(dim=0).clamp(min=1e-8)
                r2 = (1 - ss_res / ss_tot).mean()
                result["latent_r2"] = r2

        return result

    def _unflatten_and_decode(self, flat_tokens, key):
        """Convert flat ``(B, S*T, D)`` generated tokens to ``(B, T_recon, N)`` predictions.

        Reshapes the flat sequence back to ``(B, S, T, D)``, unscales from
        target space to raw tokenizer latent space, and decodes via
        ``decode_differentiable``.
        """
        S = cast(MARSessionReadout, self.readout[key]).n_spatial
        T = self.n_temporal_tokens
        tokens_4d = flat_tokens.float().reshape(flat_tokens.shape[0], S, T, -1)
        raw = self.tokenizer_wrapper._target_to_tokens(tokens_4d)
        return self.tokenizer_wrapper.decode_differentiable(raw, session_key=key)

    @torch.no_grad()
    def generate(
        self,
        images,
        key,
        n_mar_steps=8,
        n_flow_steps=5,
        temperature=1.0,
        flow_solver="euler",
        flow_shift=1.0,
        cfg_scale=1.0,
        **kwargs,
    ):
        """Iterative unmasking inference.

        Generates tokens over the unified ``(S * T)`` sequence, then decodes
        back to ``(B, T_recon, N)`` neural responses.

        Args:
            images: (B, C, H, W) static images
            key: session key
            n_mar_steps: number of MAR unmasking steps
            n_flow_steps: number of ODE integration steps per MAR step
            temperature: scale for initial noise
            flow_solver: "euler" or "heun"
            flow_shift: inference time-grid shift (1.0 = uniform); flow only
            cfg_scale: classifier-free guidance scale (1.0 = off; requires
                training with flow_cond_dropout > 0); flow only
        Returns:
            predictions (B, T_recon, N) — decoded through tokenizer
        """
        B = images.shape[0]
        readout = cast(MARSessionReadout, self.readout[key])
        S = readout.n_spatial
        T = self.n_temporal_tokens
        N_tokens = S * T

        image_tokens = self.encode_image(images, key)
        S_img = image_tokens.shape[1]

        embeds = self._get_embeds(readout, key)  # (1, N_tokens, mar_dim)

        is_ce = self.loss_mode == "cross_entropy"

        if is_ce:
            generated_indices = torch.zeros(B, N_tokens, device=images.device, dtype=torch.long)
        else:
            generated_tokens = torch.zeros(B, N_tokens, self.token_dim, device=images.device, dtype=image_tokens.dtype)
        revealed = torch.zeros(B, N_tokens, dtype=torch.bool, device=images.device)

        for step in range(n_mar_steps):
            if is_ce:
                neural_projected = self.neural_embed(generated_indices)
            else:
                neural_projected = self.neural_proj(generated_tokens)
            neural_tokens = torch.where(
                (~revealed).unsqueeze(-1),
                self.mask_token.expand(B, N_tokens, -1) + embeds[:, :N_tokens, :],
                neural_projected + embeds[:, :N_tokens, :],
            )

            combined = torch.cat([image_tokens, neural_tokens], dim=1)
            combined = self._prepend_buffer(combined)
            decoded = self.mar_decoder(combined)
            decoded = self._strip_buffer(decoded)
            neural_hidden = decoded[:, S_img:, :]

            if self.use_diffusion_pos_embed:
                neural_hidden = neural_hidden + self.diffusion_pos_embed[:, :N_tokens, :]

            still_masked = ~revealed
            if not still_masked.any():
                break

            masked_cond = neural_hidden[still_masked]
            if is_ce:
                logits = self.ce_head(masked_cond)
                if self.ce_sample_temperature <= 0:
                    new_indices = logits.argmax(dim=-1)
                else:
                    probs = F.softmax(logits / self.ce_sample_temperature, dim=-1)
                    new_indices = torch.multinomial(probs, 1).squeeze(-1)
                generated_indices[still_masked] = new_indices
                max_probs = F.softmax(logits, dim=-1).max(dim=-1).values
            elif self.loss_mode == "flow_matching":
                new_tokens = self.flow_head.sample(
                    masked_cond,
                    n_steps=n_flow_steps,
                    temperature=temperature,
                    solver=flow_solver,
                    shift=flow_shift,
                    cfg_scale=cfg_scale,
                )
            elif self.loss_mode == "diffusion":
                new_tokens = self.diffloss.sample(masked_cond, temperature=temperature)
            elif self.loss_mode == "energy":
                new_tokens = self.energy_loss.sample(masked_cond, temperature=temperature)
            elif self.loss_mode in ("mse", "smooth_l1"):
                new_tokens = self.mse_head(masked_cond)
            elif self.loss_mode == "poisson":
                new_tokens = self.poisson_head(masked_cond).exp()

            if not is_ce:
                generated_tokens[still_masked] = new_tokens.to(generated_tokens.dtype)

            # Confidence scoring
            confidence = torch.full((B, N_tokens), float("-inf"), device=images.device)
            if is_ce:
                confidence[still_masked] = max_probs.float()
            else:
                confidence[still_masked] = -new_tokens.float().norm(dim=-1)

            n_currently_masked = still_masked.sum(dim=1)

            if self.unmask_schedule == "cosine":
                mask_ratio = math.cos(math.pi / 2.0 * (step + 1) / n_mar_steps)
                target_masked = max(0, int(math.floor(N_tokens * mask_ratio)))
                n_to_reveal = (n_currently_masked - target_masked).clamp(min=1)
            else:
                remaining_steps = max(1, n_mar_steps - step)
                n_to_reveal = torch.ceil(n_currently_masked.float() / remaining_steps).long()
                n_to_reveal = n_to_reveal.clamp(min=1)

            n_to_reveal = torch.minimum(n_to_reveal, n_currently_masked)

            max_k = int(n_to_reveal.max().item())
            if max_k > 0:
                _, topk_indices = confidence.topk(max_k, dim=1)
                arange_k = torch.arange(max_k, device=images.device).unsqueeze(0)
                valid_mask = arange_k < n_to_reveal.unsqueeze(1)
                reveal_update = torch.zeros_like(revealed)
                reveal_update.scatter_(1, topk_indices, valid_mask)
                revealed = revealed | reveal_update

        # Decode to (B, T_recon, N) predictions
        if self.loss_mode == "poisson":
            predictions = generated_tokens.reshape(B, S, T, -1)
            # Poisson: tokens are raw rates, not latent space — transpose back
            predictions = predictions.squeeze(-1).permute(0, 2, 1)  # (B, T, S=N)
        else:
            with torch.autocast(device_type="cuda", enabled=False):
                if is_ce:
                    # CE generation produces flat discrete indices; the wrapper's
                    # differentiable decode path accepts them in unified 4D layout.
                    indices_4d = generated_indices.reshape(B, S, T).unsqueeze(-1)
                    predictions = self.tokenizer_wrapper.decode_differentiable(indices_4d, session_key=key)
                else:
                    predictions = self._unflatten_and_decode(generated_tokens, key)

        return predictions
