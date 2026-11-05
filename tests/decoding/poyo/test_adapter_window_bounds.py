"""Sentinel timestamp sanity check.

The adapter's `window_start` / `window_end` define where sentinels are
placed and the contract for which timestamps the tokenizer is allowed
to emit. Today both equal POYO's `tokenize()` window
([0, sequence_length]); the runtime check guards against silent drift
if anyone moves to absolute timestamps later.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from ntt_downstream.decoding.backends.poyo.adapter import adapt
from ntt_downstream.decoding.backends.poyo.tokenizer_output import TokenizerOutput


def _tok(B=1, N=4, D=8, ts=None, combines_units=False, with_unit_ids=True):
    if ts is None:
        ts = torch.linspace(0.0, 1.0, N).unsqueeze(0).expand(B, -1).contiguous()
    return TokenizerOutput(
        tokens=torch.zeros(B, N, D),
        timestamps=ts,
        unit_ids=torch.tensor([[0, 1, 2, 0]][:B]) if with_unit_ids else None,
        mask=torch.ones(B, N, dtype=torch.bool),
        combines_units=combines_units,
        token_bin_size_s=None,
    )


def _emb(D=8, n=10):
    return nn.Embedding(n, D)


def test_in_bounds_passes():
    tok = _tok(ts=torch.tensor([[0.0, 0.25, 0.5, 1.0]]))
    out = adapt(
        tok,
        _emb(8),
        _emb(8, 4),
        "none",
        sentinels=True,
        target_dim=8,
        window_start=0.0,
        window_end=1.0,
    )
    # 3 unique units → 3 sentinels prepended.
    assert out.embeddings.shape[1] == 3 + 4


def test_out_of_bounds_above_raises():
    tok = _tok(ts=torch.tensor([[0.0, 0.25, 0.5, 1.5]]))  # 1.5 > window_end
    with pytest.raises(ValueError, match="outside adapter window"):
        adapt(
            tok,
            _emb(8),
            _emb(8, 4),
            "none",
            sentinels=True,
            target_dim=8,
            window_start=0.0,
            window_end=1.0,
        )


def test_out_of_bounds_below_raises():
    tok = _tok(ts=torch.tensor([[-0.5, 0.25, 0.5, 1.0]]))  # -0.5 < window_start
    with pytest.raises(ValueError, match="outside adapter window"):
        adapt(
            tok,
            _emb(8),
            _emb(8, 4),
            "none",
            sentinels=True,
            target_dim=8,
            window_start=0.0,
            window_end=1.0,
        )


def test_check_skipped_when_sentinels_disabled():
    """The check only matters when sentinels are emitted; skip otherwise."""
    tok = _tok(ts=torch.tensor([[0.0, 0.25, 0.5, 1.5]]))  # would fail above
    # No sentinels → no check fires; this should not raise.
    out = adapt(
        tok,
        _emb(8),
        _emb(8, 4),
        "none",
        sentinels=False,
        target_dim=8,
        window_start=0.0,
        window_end=1.0,
    )
    assert out.embeddings.shape[1] == 4


def test_check_skipped_when_all_padded():
    """All-False mask → no valid timestamps → no bounds to check."""
    tok = TokenizerOutput(
        tokens=torch.zeros(1, 4, 8),
        timestamps=torch.tensor([[100.0, 200.0, 300.0, 400.0]]),  # garbage
        unit_ids=torch.tensor([[0, 1, 2, 0]]),
        mask=torch.zeros(1, 4, dtype=torch.bool),
        combines_units=False,
        token_bin_size_s=None,
    )
    out = adapt(
        tok,
        _emb(8),
        _emb(8, 4),
        "none",
        sentinels=True,
        target_dim=8,
        window_start=0.0,
        window_end=1.0,
    )
    # No valid units → no sentinels prepended; sequence length unchanged.
    assert out.embeddings.shape[1] == 4
