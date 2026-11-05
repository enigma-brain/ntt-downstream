"""Tests for FrozenNTTTokenizer's ``has_time_dim=False`` (time-pooled) tokenizers.

Two latent shapes flow through ``_encode_with_timestamps_4d``:

  * 3D ``[B, U, D]`` — the PCA baseline in ntt releases that return it 3D. The
    wrapper splices a singleton time axis (``T'=1``) and rewrites the
    ``[B, N]`` neuron-axis timestamps into ``[B, 1]`` window centers. Releases
    whose PCA ``encode`` already returns ``[B, 1, U, D]`` take the 4D path below;
    the tests check both give the same shapes and window-center timestamps.
  * 4D ``[B, 1, 1, D]`` — ``simple_vae`` (also ``has_time_dim=False`` but emits
    a singleton time axis directly). These are already canonical and are passed
    through unchanged.

The simple_vae case guards the ``has_time_dim=False`` branch: a strict
``assert latents.ndim == 3`` there would reject simple_vae's 4D output, make
``output_dim`` return ``-1`` and make the forward pass raise ``AssertionError``.

All CPU-only — the PCA tokenizer encodes correctly even unfitted (zeros of the
right shape) and simple_vae runs untrained.
"""

from __future__ import annotations

import pytest
import torch

from ntt_downstream.decoding.backends.poyo.frozen_ntt_tokenizer import FrozenNTTTokenizer
from ntt_downstream.decoding.backends.poyo.tokenizer_output import TokenizerOutput


def _pca_frozen(n_neurons: int = 8, n_components: int = 4) -> FrozenNTTTokenizer:
    from ntt.contrib.tokenizer.pca_baseline import (
        PCABaselineTokenizer,
        PCABaselineTokenizerSpec,
    )

    spec = PCABaselineTokenizerSpec(
        n_neurons=n_neurons,
        window_length_seconds=1.0,
        bin_size_ms=10.0,
        n_components=n_components,
    )
    return FrozenNTTTokenizer(PCABaselineTokenizer(spec), freeze=True)


def _simple_vae_frozen(latent_dim: int = 8) -> FrozenNTTTokenizer:
    from ntt.contrib.tokenizer.simple_vae import (
        SimpleVAETokenizer,
        SimpleVAETokenizerSpec,
    )

    spec = SimpleVAETokenizerSpec(
        n_neurons=1,
        window_length_seconds=1,
        bin_size_ms=1.0,
        latent_dim=latent_dim,
        hidden_dims=(64, 32),
    )
    return FrozenNTTTokenizer(SimpleVAETokenizer(spec), freeze=True)


# --- PCA baseline: 3D -> spliced 4D + window-center timestamps ---


def test_pca_spec_is_timeless():
    """Guard the premise: PCA spec is has_time_dim=False and encode is either
    3D ``[B, U, D]`` or canonical 4D ``[B, 1, U, D]`` (ntt releases differ);
    the wrapper handles both."""
    from ntt.core.model.tokenizer.validation import make_dummy_input

    frozen = _pca_frozen()
    assert frozen.tokenizer.spec.has_time_dim is False
    raw = frozen.tokenizer.encode(make_dummy_input(frozen.tokenizer.spec))
    assert raw.ndim == 3 or (raw.ndim == 4 and raw.shape[1] == 1)


def test_pca_encode_with_timestamps_4d_splices_singleton_time_axis():
    from ntt.core.model.tokenizer.validation import make_dummy_input

    frozen = _pca_frozen(n_neurons=8, n_components=4)
    dummy = make_dummy_input(frozen.tokenizer.spec)

    latents, ts = frozen._encode_with_timestamps_4d(dummy)
    # [B, T'=1, U, D], whether ntt's encode returned 3D (spliced) or 4D.
    assert latents.ndim == 4
    assert latents.shape[1] == 1
    assert latents.shape[2] == 8  # U preserved
    assert latents.shape[3] == 4  # D = n_components

    # Timestamps must be the [B, 1] window centers, not the upstream [B, N]
    # neuron-axis garbage. Window center = (absolute_start + absolute_end) / 2.
    tt = ts["token_timestamps"]
    assert tt.shape == (dummy.absolute_start.shape[0], 1)
    expected = (dummy.absolute_start.to(torch.float64) + dummy.absolute_end.to(torch.float64)) / 2
    assert torch.allclose(tt.squeeze(-1), expected)


def test_pca_forward_output_dim_and_constant_window_timestamps():
    from ntt.core.model.tokenizer.validation import make_dummy_input

    frozen = _pca_frozen(n_neurons=8, n_components=4)
    out = frozen(make_dummy_input(frozen.tokenizer.spec))

    assert isinstance(out, TokenizerOutput)
    # output_dim probe must see the latent feature dim, not the -1 failure code.
    assert frozen.output_dim == 4
    # T'=1, U=8 -> N = 8 content tokens.
    assert out.tokens.ndim == 3
    assert out.tokens.shape[1] == 8
    assert out.timestamps.shape == out.tokens.shape[:2]
    # Because T'=1, every unit token in a sample shares the single window-center
    # time — the broken upstream path produced *distinct* per-neuron times.
    for b in range(out.tokens.shape[0]):
        valid = out.timestamps[b][out.mask[b]]
        assert valid.numel() > 0
        assert torch.allclose(valid, valid[0].expand_as(valid))


# --- simple_vae: already-4D singleton time axis passes through ---


def test_simple_vae_spec_is_timeless_but_emits_4d():
    """Guard the premise that distinguishes simple_vae from PCA: timeless
    spec, but a 4D encode with a singleton time axis."""
    from ntt.core.model.tokenizer.validation import make_dummy_input

    torch.manual_seed(0)
    frozen = _simple_vae_frozen(latent_dim=8)
    assert frozen.tokenizer.spec.has_time_dim is False
    raw = frozen.tokenizer.encode(make_dummy_input(frozen.tokenizer.spec))
    assert raw.ndim == 4 and raw.shape[1] == 1  # [B, 1, 1, D]


def test_simple_vae_encode_with_timestamps_4d_passes_through():
    from ntt.core.model.tokenizer.validation import make_dummy_input

    torch.manual_seed(0)
    frozen = _simple_vae_frozen(latent_dim=8)
    dummy = make_dummy_input(frozen.tokenizer.spec)

    # Must NOT raise (a strict `assert ndim == 3` would).
    latents, _ = frozen._encode_with_timestamps_4d(dummy)
    assert latents.ndim == 4
    assert latents.shape[1] == 1  # singleton time axis
    assert latents.shape[-1] == 8


def test_simple_vae_output_dim_is_latent_dim_not_minus_one():
    """output_dim of a timeless 4D tokenizer is the latent dim, never -1."""
    torch.manual_seed(0)
    frozen = _simple_vae_frozen(latent_dim=8)
    assert frozen.output_dim == 8


def test_simple_vae_forward_runs():
    from ntt.core.model.tokenizer.validation import make_dummy_input

    torch.manual_seed(0)
    frozen = _simple_vae_frozen(latent_dim=8)
    out = frozen(make_dummy_input(frozen.tokenizer.spec))
    assert isinstance(out, TokenizerOutput)
    # T'=1, U=1 -> a single population token per sample.
    assert out.tokens.ndim == 3
    assert out.tokens.shape[1] == 1
    assert out.tokens.shape[-1] == 8


def test_timeless_4d_rejects_multi_time_axis():
    """A has_time_dim=False tokenizer emitting T' > 1 violates the singleton
    timestamp contract and must fail loud."""
    torch.manual_seed(0)
    frozen = _simple_vae_frozen(latent_dim=8)

    # Patch the wrapped tokenizer to emit a [B, 2, 1, D] latent (T'=2).
    real_encode = frozen.tokenizer.encode_with_timestamps

    def _bad_encode(batch):
        latents, ts = real_encode(batch)
        return latents.repeat(1, 2, 1, 1), ts  # T' = 2

    frozen.tokenizer.encode_with_timestamps = _bad_encode
    from ntt.core.model.tokenizer.validation import make_dummy_input

    with pytest.raises(AssertionError, match=r"singleton time axis"):
        frozen._encode_with_timestamps_4d(make_dummy_input(frozen.tokenizer.spec))
