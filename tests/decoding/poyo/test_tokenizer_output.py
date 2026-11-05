"""Tests for the TokenizerOutput dataclass."""

import torch

from ntt_downstream.decoding.backends.poyo.tokenizer_output import TokenizerOutput


def test_vocab_sizes_default_is_none():
    out = TokenizerOutput(
        tokens=torch.zeros(2, 5, 8),
        timestamps=torch.zeros(2, 5),
        unit_ids=None,
        mask=torch.ones(2, 5, dtype=torch.bool),
        combines_units=True,
        token_bin_size_s=0.1,
    )
    assert out.vocab_sizes is None


def test_vocab_sizes_set_explicitly():
    out = TokenizerOutput(
        tokens=torch.zeros(2, 5, 4, dtype=torch.int64),
        timestamps=torch.zeros(2, 5),
        unit_ids=None,
        mask=torch.ones(2, 5, dtype=torch.bool),
        combines_units=True,
        token_bin_size_s=0.1,
        vocab_sizes=(1024, 1024, 1024, 1024),
    )
    assert out.vocab_sizes == (1024, 1024, 1024, 1024)
