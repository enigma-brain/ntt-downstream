"""Selection of the full-window POYO→NTT handoff for session-keyed tokenizers.

The active-unit handoff (``V2TrainWrapper._ntt_frozen_tokenize``) passes
POYO-global unit indices as ``unit_ids``. A session-keyed tokenizer (the
Perceiver looks each unit up in a per-session NuCLR table by *local* id)
clamps those ids instead of raising, so decoding trains on the wrong
embeddings without any error. ``full_window_units: auto`` must therefore pick
the full-window wrapper for such tokenizers, and an explicit ``false`` must
fail loudly.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from ntt_downstream.decoding.backends.poyo.frozen_ntt_tokenizer import FrozenNTTTokenizer
from ntt_downstream.decoding.train import _resolve_full_window_units


def _cfg(full_window_units=None, *, set_key: bool = True):
    tok = {"mode": "ntt_frozen"}
    if set_key:
        tok["full_window_units"] = full_window_units
    return OmegaConf.create({"tokenizer": tok})


def _tok(session_keyed: bool):
    return SimpleNamespace(is_session_keyed=session_keyed)


@pytest.mark.parametrize("value", ["auto", None])
def test_auto_selects_full_window_for_session_keyed_tokenizer(value):
    assert _resolve_full_window_units(_cfg(value), "ntt_frozen", _tok(True)) is True


@pytest.mark.parametrize("value", ["auto", None])
def test_auto_keeps_active_unit_path_for_other_tokenizers(value):
    assert _resolve_full_window_units(_cfg(value), "ntt_frozen", _tok(False)) is False


def test_missing_key_behaves_like_auto():
    cfg = _cfg(set_key=False)
    assert _resolve_full_window_units(cfg, "ntt_frozen", _tok(True)) is True
    assert _resolve_full_window_units(cfg, "ntt_frozen", _tok(False)) is False


def test_auto_without_ntt_tokenizer_is_off():
    assert _resolve_full_window_units(_cfg("auto"), "fake", None) is False


def test_explicit_false_on_session_keyed_tokenizer_raises():
    with pytest.raises(ValueError, match="session-keyed"):
        _resolve_full_window_units(_cfg(False), "ntt_frozen", _tok(True))


def test_explicit_values_are_honoured_otherwise():
    assert _resolve_full_window_units(_cfg(True), "ntt_frozen", _tok(False)) is True
    assert _resolve_full_window_units(_cfg(False), "ntt_frozen", _tok(False)) is False


def test_explicit_true_requires_ntt_frozen():
    with pytest.raises(ValueError, match="ntt_frozen"):
        _resolve_full_window_units(_cfg(True), "ntt", _tok(False))


def test_invalid_value_raises():
    with pytest.raises(ValueError, match="full_window_units"):
        _resolve_full_window_units(_cfg("yes"), "ntt_frozen", _tok(False))


# ---------------------------------------------------------------------------
# FrozenNTTTokenizer.is_session_keyed on real tokenizers
# ---------------------------------------------------------------------------


def test_poisson_tokenizer_is_not_session_keyed():
    tok = FrozenNTTTokenizer.from_poisson(rate_bin_size_ms=100.0, input_bin_size_ms=10.0)
    assert tok.is_session_keyed is False


def test_perceiver_tokenizer_is_session_keyed(tmp_path: Path):
    perceiver_mod = pytest.importorskip("ntt.contrib.tokenizer.perceiver.perceiver_tokenizer")
    path = tmp_path / "embs.pt"
    torch.save(
        {
            "embs": torch.randn(4, 8),
            "unit_ids": np.asarray([f"s/{u}" for u in range(4)]),
            "session_ids": np.asarray(["s"] * 4),
        },
        path,
    )
    spec = perceiver_mod.PerceiverTokenizerSpec(
        hidden_dim=16,
        latent_tokens=2,
        latent_dim=4,
        d_embed=8,
        num_heads=4,
        mlp_ratio=1,
        num_encoder_N_blocks=1,
        num_encoder_M_blocks=1,
        num_decoder_blocks=1,
        use_s4=False,
        grad_checkpointing=False,
        window_length_seconds=0.08,
        bin_size_ms=10.0,
        embeddings_path=str(path),
    )
    tok = FrozenNTTTokenizer(perceiver_mod.PerceiverTokenizer(spec))
    assert tok.is_session_keyed is True
