"""POYO subclass whose forward accepts pre-computed input embeddings.

Adapted from torch_brain (https://github.com/neuro-galaxy/torch_brain),
``torch_brain/models/poyo.py``. Apache-2.0, Copyright 2025 NeuroGalaxy.
Modified for ntt-downstream: adds the ``input_embeddings`` bypass so ntt
tokenizer output can replace POYO's unit/token-type input embedding.
"""

from collections.abc import Iterable
from typing import cast

import torch
from torch import Tensor, nn
from torchtyping import TensorType

from ntt_downstream.decoding.backends.poyo.poyo import POYO


class POYOWithEmbeddingBypass(POYO):
    """POYO subclass whose forward can accept pre-computed input embeddings.

    When `input_embeddings` is None, behaves identically to `POYO.forward`.
    When provided, it replaces `unit_emb(input_unit_index) + token_type_emb(
    input_token_type)` as the input token content. Everything downstream
    (encoder cross-attention, latents, decoder, readout) is unchanged.
    """

    def forward(
        self,
        *,
        # input sequence
        input_unit_index: TensorType["batch", "n_in", int] | None = None,
        input_timestamps: TensorType["batch", "n_in", float],
        input_token_type: TensorType["batch", "n_in", int] | None = None,
        input_mask: TensorType["batch", "n_in", bool] | None = None,
        input_embeddings: TensorType["batch", "n_in", "dim", float] | None = None,
        # latent sequence
        latent_index: TensorType["batch", "n_latent", int],
        latent_timestamps: TensorType["batch", "n_latent", float],
        # output sequence
        output_session_index: TensorType["batch", "n_out", int],
        output_timestamps: TensorType["batch", "n_out", float],
        output_mask: TensorType["batch", "n_out", bool] | None = None,
        unpack_output: bool = False,
    ) -> TensorType["batch", "n_out", "dim_out", float] | list[TensorType[..., "dim_out", float]]:
        """Forward pass; bypass unit/token-type embedding lookup when input_embeddings is given."""

        # ---- input token content --------------------------------------------------
        if input_embeddings is None:
            # Standard POYO input path — unit vocab must be initialized.
            if self.unit_emb.is_lazy():
                raise ValueError(
                    "Unit vocabulary has not been initialized, please use `model.unit_emb.initialize_vocab(unit_ids)`"
                )
            if input_unit_index is None or input_token_type is None:
                raise ValueError("input_unit_index and input_token_type are required when input_embeddings is None.")
            inputs = self.unit_emb(input_unit_index) + self.token_type_emb(input_token_type)
        else:
            # Bypass path — validate shape/dtype loudly.
            if not isinstance(input_embeddings, Tensor):
                raise TypeError(f"input_embeddings must be a torch.Tensor, got {type(input_embeddings).__name__}.")
            if input_embeddings.dim() != 3:
                raise ValueError(f"input_embeddings must have shape [B, N, D]; got {tuple(input_embeddings.shape)}.")
            if input_embeddings.shape[-1] != self.dim:
                raise ValueError(
                    f"input_embeddings last dim ({input_embeddings.shape[-1]}) does not match model dim ({self.dim})."
                )
            if not torch.is_floating_point(input_embeddings):
                raise TypeError(
                    f"input_embeddings must be a floating-point tensor; got dtype {input_embeddings.dtype}."
                )
            b, n = input_embeddings.shape[:2]
            if input_timestamps.shape != (b, n):
                raise ValueError(
                    f"input_timestamps shape {tuple(input_timestamps.shape)} "
                    f"incompatible with input_embeddings [B,N]=({b},{n})."
                )
            inputs = input_embeddings

        if self.session_emb.is_lazy():
            raise ValueError(
                "Session vocabulary has not been initialized, please use "
                "`model.session_emb.initialize_vocab(session_ids)`"
            )

        input_timestamp_emb = self.rotary_emb(input_timestamps)

        # latents
        latents = self.latent_emb(latent_index)
        latent_timestamp_emb = self.rotary_emb(latent_timestamps)

        # outputs
        output_queries = self.session_emb(output_session_index)
        output_timestamp_emb = self.rotary_emb(output_timestamps)

        # encode
        latents = latents + self.enc_atn(
            latents,
            inputs,
            latent_timestamp_emb,
            input_timestamp_emb,
            input_mask,
        )
        latents = latents + self.enc_ffn(latents)

        # process
        for self_attn, self_ff in cast(Iterable[nn.Sequential], self.proc_layers):
            latents = latents + self.dropout(self_attn(latents, latent_timestamp_emb))
            latents = latents + self.dropout(self_ff(latents))

        # decode
        output_queries = output_queries + self.dec_atn(
            output_queries,
            latents,
            output_timestamp_emb,
            latent_timestamp_emb,
        )
        output_latents = output_queries + self.dec_ffn(output_queries)
        output = self.readout(output_latents)

        if unpack_output:
            if output_mask is None:
                raise ValueError("output_mask is required when unpack_output=True.")
            output = [output[b][output_mask[b]] for b in range(output.size(0))]

        return output


class POYOWithEmbeddingBypass_wo_crossattention(POYO):
    """POYO variant with the encoder cross-attention removed.

    Input tokens (from `unit_emb + token_type_emb`, or from `input_embeddings`)
    flow directly into the self-attention `proc_layers` — there is no learnable
    latent grid and no `enc_atn` compression. The decoder cross-attention then
    queries the processed input tokens (with `input_mask` as the context mask).

    `latent_index` / `latent_timestamps` are accepted for tokenizer API
    compatibility but unused. `enc_atn`, `enc_ffn`, `latent_emb` are still
    constructed by the parent `__init__` but not invoked here, so DDP needs
    `find_unused_parameters=True` (already required for POYO multi-GPU).
    """

    def __init__(self, *, use_input_pre_norm: bool = True, **kwargs):
        super().__init__(**kwargs)
        # Pre-norm on the residual stream before proc_layers. Frozen-tokenizer
        # `input_embeddings` can have a very different scale from `unit_emb`
        # (init_scale=0.02); RotarySelfAttention's internal LN normalizes the
        # attention path but not the residual itself.
        self.input_pre_norm = nn.LayerNorm(self.dim) if use_input_pre_norm else nn.Identity()

    def forward(
        self,
        *,
        # input sequence
        input_unit_index: TensorType["batch", "n_in", int] | None = None,
        input_timestamps: TensorType["batch", "n_in", float],
        input_token_type: TensorType["batch", "n_in", int] | None = None,
        input_mask: TensorType["batch", "n_in", bool] | None = None,
        input_embeddings: TensorType["batch", "n_in", "dim", float] | None = None,
        # latent sequence (unused — kept for API compatibility)
        latent_index: TensorType["batch", "n_latent", int] | None = None,
        latent_timestamps: TensorType["batch", "n_latent", float] | None = None,
        # output sequence
        output_session_index: TensorType["batch", "n_out", int],
        output_timestamps: TensorType["batch", "n_out", float],
        output_mask: TensorType["batch", "n_out", bool] | None = None,
        unpack_output: bool = False,
    ) -> TensorType["batch", "n_out", "dim_out", float] | list[TensorType[..., "dim_out", float]]:
        """Forward without encoder cross-attention."""

        # ---- input token content --------------------------------------------------
        if input_embeddings is None:
            if self.unit_emb.is_lazy():
                raise ValueError(
                    "Unit vocabulary has not been initialized, please use `model.unit_emb.initialize_vocab(unit_ids)`"
                )
            if input_unit_index is None or input_token_type is None:
                raise ValueError("input_unit_index and input_token_type are required when input_embeddings is None.")
            inputs = self.unit_emb(input_unit_index) + self.token_type_emb(input_token_type)
        else:
            if not isinstance(input_embeddings, Tensor):
                raise TypeError(f"input_embeddings must be a torch.Tensor, got {type(input_embeddings).__name__}.")
            if input_embeddings.dim() != 3:
                raise ValueError(f"input_embeddings must have shape [B, N, D]; got {tuple(input_embeddings.shape)}.")
            if input_embeddings.shape[-1] != self.dim:
                raise ValueError(
                    f"input_embeddings last dim ({input_embeddings.shape[-1]}) does not match model dim ({self.dim})."
                )
            if not torch.is_floating_point(input_embeddings):
                raise TypeError(
                    f"input_embeddings must be a floating-point tensor; got dtype {input_embeddings.dtype}."
                )
            b, n = input_embeddings.shape[:2]
            if input_timestamps.shape != (b, n):
                raise ValueError(
                    f"input_timestamps shape {tuple(input_timestamps.shape)} "
                    f"incompatible with input_embeddings [B,N]=({b},{n})."
                )
            inputs = input_embeddings

        if self.session_emb.is_lazy():
            raise ValueError(
                "Session vocabulary has not been initialized, please use "
                "`model.session_emb.initialize_vocab(session_ids)`"
            )

        input_timestamp_emb = self.rotary_emb(input_timestamps)

        # No encoder cross-attention: input tokens *are* the processor input.
        latents = self.input_pre_norm(inputs)
        latent_timestamp_emb = input_timestamp_emb

        # outputs
        output_queries = self.session_emb(output_session_index)
        output_timestamp_emb = self.rotary_emb(output_timestamps)

        # process — self-attn over input tokens; mask padded positions.
        for self_attn, self_ff in cast(Iterable[nn.Sequential], self.proc_layers):
            latents = latents + self.dropout(self_attn(latents, latent_timestamp_emb, input_mask))
            latents = latents + self.dropout(self_ff(latents))

        # decode — output queries cross-attend to processed input tokens.
        output_queries = output_queries + self.dec_atn(
            output_queries,
            latents,
            output_timestamp_emb,
            latent_timestamp_emb,
            input_mask,
        )
        output_latents = output_queries + self.dec_ffn(output_queries)
        output = self.readout(output_latents)

        if unpack_output:
            if output_mask is None:
                raise ValueError("output_mask is required when unpack_output=True.")
            output = [output[b][output_mask[b]] for b in range(output.size(0))]

        return output
