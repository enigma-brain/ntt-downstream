"""Tokenizer output container shared by all tokenizers and the adapter."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TokenizerOutput:
    """Output of any tokenizer, fed to the adapter.

    tokens: float [B, N, D] (continuous) OR int64 [B, N, Q] (codebook indices)
    timestamps: [B, N] -- per-token center timestamps in seconds
    unit_ids: Optional[B, N] -- per-token unit index (None for population tokenizers)
    mask: [B, N] -- True for valid tokens, False for padding
    combines_units: bool -- True if tokenizer produces population tokens (no per-unit dim)
    token_bin_size_s: Optional[float] -- token time resolution in seconds (None for variable-length)
    vocab_sizes: Optional[tuple[int, ...]] -- (K_0, ..., K_{Q-1}) iff tokens is int (one entry per codebook)
    """

    tokens: torch.Tensor
    timestamps: torch.Tensor
    unit_ids: torch.Tensor | None
    mask: torch.Tensor
    combines_units: bool
    token_bin_size_s: float | None
    vocab_sizes: tuple[int, ...] | None = None
