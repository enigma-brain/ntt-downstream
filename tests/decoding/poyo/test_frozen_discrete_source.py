"""Tests for FrozenNTTTokenizer's discrete ``indices_source`` auto-routing.

The runtime ``"bottleneck"`` path mirrors WanNeural.encode's prelude
(``_extract_data`` + ``_pre_encode``). A tokenizer whose bottleneck exposes
``get_indices`` but lacks that prelude (e.g. the wavelet-* FSQ / VQPT tokenizers) must NOT be
routed to ``"bottleneck"`` under ``discrete_indices_source="auto"`` — it emits
indices via ``forward().tokens`` instead, and would raise ``AttributeError`` on
the first batch. The ``"bottleneck"`` choice is gated on the prelude actually
existing, otherwise ``"forward"`` is picked.

"cosmos" in the names below is the ntt module name of the wavelet-* tokenizers.
Those cases use a small real CPU tokenizer; the WanNeural-vs-wavelet heuristic
itself is also unit-tested directly with lightweight stubs.
"""

from __future__ import annotations

import types

import pytest
import torch

from ntt_downstream.decoding.backends.poyo.frozen_ntt_tokenizer import FrozenNTTTokenizer
from ntt_downstream.decoding.backends.poyo.tokenizer_output import TokenizerOutput


def _cosmos_fsq_frozen(discrete_indices_source: str = "auto") -> FrozenNTTTokenizer:
    """A small CPU-buildable wavelet-* FSQ tokenizer in embedding mode.

    Same routing class as the residual-FSQ/VQPT checkpoints:
    the bottleneck has ``get_indices`` but the tokenizer has no
    ``_extract_data``/``_pre_encode``.
    """
    from ntt.contrib.tokenizer.cosmos.cosmos_2d import CosmosTokenizer2D, CosmosTokenizer2DSpec

    torch.manual_seed(0)
    spec = CosmosTokenizer2DSpec(
        n_neurons=8,
        window_length_seconds=1.0,
        bin_size_ms=50.0,
        bottleneck_type="fsq",
        channels=8,
        channels_mult=(1, 2),
        z_channels=8,
        neuron_compression=2,
        temporal_compression=4,
        patch_h=1,
        patch_w=2,
        num_res_blocks=1,
        fsq_levels=(8, 5, 5),
    )
    return FrozenNTTTokenizer(
        CosmosTokenizer2D(spec),
        freeze=True,
        discrete_input_mode="embedding",
        discrete_indices_source=discrete_indices_source,
    )


def test_cosmos_premise_get_indices_but_no_wanneural_prelude():
    """Guard the premise: wavelet-* has a get_indices bottleneck, no _extract_data/
    _pre_encode, and a float encode() (so the int-encode probe falls through)."""
    from ntt.core.model.tokenizer.validation import make_dummy_input

    frozen = _cosmos_fsq_frozen()
    tok = frozen.tokenizer
    assert hasattr(tok.bottleneck, "get_indices")
    assert not hasattr(tok, "_extract_data")
    assert not hasattr(tok, "_pre_encode")
    assert torch.is_floating_point(tok.encode(make_dummy_input(tok.spec)))


def test_cosmos_auto_routes_to_forward_not_bottleneck():
    """auto resolves to "forward" for the wavelet-* tokenizers.

    They expose a get_indices bottleneck but no _extract_data/_pre_encode
    prelude, so "bottleneck" would fail on the first batch."""
    frozen = _cosmos_fsq_frozen()
    assert frozen._indices_source_resolved == "forward"


def test_cosmos_embedding_forward_runs_and_emits_int_tokens():
    """forward() runs end-to-end and emits int64
    tokens with the FSQ vocab (prod(levels) = 8*5*5 = 200)."""
    from ntt.core.model.tokenizer.validation import make_dummy_input

    frozen = _cosmos_fsq_frozen()
    out = frozen(make_dummy_input(frozen.tokenizer.spec))
    assert isinstance(out, TokenizerOutput)
    assert out.tokens.dtype == torch.int64
    assert out.tokens.ndim == 3  # [B, N, Q]
    assert frozen._vocab_size == 200
    assert out.vocab_sizes == (200,) * frozen._num_codebooks


def test_cosmos_residual_fsq_multi_codebook_aligns_q():
    """Residual-FSQ (VQPT) with Q>1: num_codebooks, the token trailing axis, and
    len(vocab_sizes) must all agree (guards the _infer_num_codebooks Q path)."""
    from ntt.contrib.tokenizer.cosmos.cosmos_2d import CosmosTokenizer2D, CosmosTokenizer2DSpec
    from ntt.core.model.tokenizer.validation import make_dummy_input

    torch.manual_seed(0)
    spec = CosmosTokenizer2DSpec(
        n_neurons=8,
        window_length_seconds=1.0,
        bin_size_ms=50.0,
        bottleneck_type="vqpt",
        vqpt_type="residual_fsq",
        vqpt_kwargs={"levels": [8, 5, 5], "num_quantizers": 3},
        channels=8,
        channels_mult=(1, 2),
        z_channels=8,
        neuron_compression=2,
        temporal_compression=4,
        patch_h=1,
        patch_w=2,
        num_res_blocks=1,
    )
    frozen = FrozenNTTTokenizer(CosmosTokenizer2D(spec), freeze=True, discrete_input_mode="embedding")
    assert frozen._indices_source_resolved == "forward"
    assert frozen._num_codebooks == 3
    assert frozen._vocab_size == 200  # prod(8, 5, 5)
    out = frozen(make_dummy_input(spec))
    assert out.tokens.dtype == torch.int64
    assert out.tokens.shape[-1] == frozen._num_codebooks  # Q trailing axis
    assert out.vocab_sizes == (200,) * 3


def test_cosmos_explicit_bottleneck_source_is_honored_not_redirected():
    """An explicit discrete_indices_source pin must bypass auto entirely (even
    if it's the "wrong" one for this tokenizer) — only auto resolves."""
    frozen = _cosmos_fsq_frozen(discrete_indices_source="bottleneck")
    assert frozen._indices_source_resolved == "bottleneck"


# --- heuristic unit tests: WanNeural-prelude presence decides bottleneck vs forward ---


def _probe(*, has_prelude: bool, discrete_indices_source: str = "auto"):
    """Drive _probe_from_structural_hint on a bare instance with stub spec/bn.

    Avoids building a heavy real tokenizer: we only exercise the auto-routing
    branch, which reads self.discrete_indices_source + self.tokenizer and infers
    vocab/codebooks from the spec.
    """
    inst = object.__new__(FrozenNTTTokenizer)
    inst.discrete_indices_source = discrete_indices_source
    tok_attrs = {"spec": types.SimpleNamespace(latent_type="discrete", codebook_size=64)}
    if has_prelude:
        tok_attrs["_extract_data"] = lambda batch: (batch, None)
        tok_attrs["_pre_encode"] = lambda data: data
    inst.tokenizer = types.SimpleNamespace(**tok_attrs)
    spec = types.SimpleNamespace(latent_type="discrete", codebook_size=64)
    bn = types.SimpleNamespace(get_indices=lambda h: h)  # has get_indices, no num_quantizers
    return inst._probe_from_structural_hint(spec, bn)


def test_auto_picks_bottleneck_when_wanneural_prelude_present():
    probe = _probe(has_prelude=True)
    assert probe.indices_source == "bottleneck"


def test_auto_picks_forward_when_prelude_absent():
    probe = _probe(has_prelude=False)
    assert probe.indices_source == "forward"


def test_explicit_source_bypasses_heuristic():
    # Explicit pin wins even when the prelude is absent (would otherwise -> forward).
    probe = _probe(has_prelude=False, discrete_indices_source="bottleneck")
    assert probe.indices_source == "bottleneck"
