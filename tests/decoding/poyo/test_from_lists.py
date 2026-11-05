"""Tests for ``NTTSpikeTimesTokenizer._from_lists`` (the BinnedSpikesInput path).

``_from_lists`` pads per-sample spike lists to ``max(n, 1)``
and delegates to ``_from_padded_spike_times``. The wrapper's ``ntt`` mode feeds
``SpikeTimesInput``, so the ``BinnedSpikesInput`` path needs its own checks of
the padding and dtype coercion. These tests exercise differing spike counts,
a zero-spike sample (the ``max(n, 1)`` clamp), and the bool-mask coercion.
"""

from __future__ import annotations

import torch
from ntt.core.components.types import BinnedSpikesInput

from ntt_downstream.decoding.backends.poyo.ntt_spike_times_tokenizer import (
    NTTSpikeTimesTokenizer,
)


def _binned_batch() -> BinnedSpikesInput:
    """3 samples (T=2, U=2): 3 spikes, 1 spike, 0 spikes."""
    B, T, U = 3, 2, 2
    spike_matrix = torch.zeros(B, T, U, dtype=torch.int32)
    spike_matrix[0] = torch.tensor([[1, 0], [1, 1]], dtype=torch.int32)  # 3 spikes
    spike_matrix[1] = torch.tensor([[1, 0], [0, 0]], dtype=torch.int32)  # 1 spike
    # sample 2 stays all-zero (0 spikes)
    return BinnedSpikesInput(
        spike_matrix=spike_matrix,
        pad_and_gap_mask=torch.ones(B, T, U, dtype=torch.int32),
        sampling_rate=1000.0,
        unit_ids=torch.arange(U).expand(B, -1).contiguous(),
        unit_ids_pad_mask=torch.ones(B, U, dtype=torch.int32),
        duration_sec=torch.full((B,), T / 1000.0),
        session_id=("s0", "s1", "s2"),
        subject_id=("u", "u", "u"),
        species=("sp", "sp", "sp"),
        absolute_start=torch.zeros(B, dtype=torch.float64),
        absolute_end=torch.full((B,), T / 1000.0, dtype=torch.float64),
    )


def test_from_lists_via_binned_input_shapes_and_dtypes():
    D = 8
    tok = NTTSpikeTimesTokenizer(token_dim=D, token_value=0.5)
    out = tok(_binned_batch())

    # n_max = max(3, 1, 0, 1) = 3 across the 3 samples.
    assert out.tokens.shape == (3, 3, D)
    assert out.tokens.dtype == torch.float32
    assert torch.all(out.tokens == 0.5)  # constant content fill

    # Mask must be bool (delegation builds a long padded_mask then .bool()s it).
    assert out.mask.dtype == torch.bool
    assert out.timestamps.dtype == torch.float32
    assert out.unit_ids.dtype == torch.long

    # Per-sample valid counts equal the total spike counts in the matrix.
    assert int(out.mask[0].sum()) == 3
    assert int(out.mask[1].sum()) == 1
    assert int(out.mask[2].sum()) == 0  # zero-spike sample: clamped to n_max, all pad

    assert out.combines_units is False
    assert out.token_bin_size_s is None


def test_from_lists_all_empty_batch_clamps_to_one():
    """Every sample empty → n_max clamps to 1 (no zero-width tensor)."""
    B, T, U = 2, 2, 2
    batch = BinnedSpikesInput(
        spike_matrix=torch.zeros(B, T, U, dtype=torch.int32),
        pad_and_gap_mask=torch.ones(B, T, U, dtype=torch.int32),
        sampling_rate=1000.0,
        unit_ids=torch.arange(U).expand(B, -1).contiguous(),
        unit_ids_pad_mask=torch.ones(B, U, dtype=torch.int32),
        duration_sec=torch.full((B,), T / 1000.0),
        session_id=("s0", "s1"),
        subject_id=("u", "u"),
        species=("sp", "sp"),
        absolute_start=torch.zeros(B, dtype=torch.float64),
        absolute_end=torch.full((B,), T / 1000.0, dtype=torch.float64),
    )
    out = NTTSpikeTimesTokenizer(token_dim=4)(batch)
    assert out.tokens.shape == (2, 1, 4)
    assert not out.mask.any()  # all padding
