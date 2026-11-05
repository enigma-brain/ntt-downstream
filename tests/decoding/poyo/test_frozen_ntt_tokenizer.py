"""Tests for FrozenNTTTokenizer wrapper."""

from __future__ import annotations

import os

import pytest
import torch

from ntt_downstream.decoding.backends.poyo.frozen_ntt_tokenizer import FrozenNTTTokenizer
from ntt_downstream.decoding.backends.poyo.tokenizer_output import TokenizerOutput

# Set RESIDUAL_LFQ_URI to point at a residual-LFQ checkpoint — either a
# local .ckpt path or an MLflow artifact URI (`mlflow-artifacts:` / `mlflow:`).
# Tests that need this URI skip when it's empty.
RESIDUAL_LFQ_URI = os.environ.get("RESIDUAL_LFQ_URI", "")


# -- Poisson baseline tests (no checkpoint needed) --


def test_poisson_baseline_population():
    """Poisson baseline produces valid population TokenizerOutput."""
    from ntt.core.components.types import BinnedSpikesInput
    from ntt.core.model.tokenizer.validation import make_dummy_input

    frozen = FrozenNTTTokenizer.from_poisson(
        rate_bin_size_ms=50.0,
        input_bin_size_ms=1.0,
        window_length_seconds=1.0,
    )

    # Poisson baseline is per-unit (has_unit_dim=True, compress_n=False)
    assert frozen.combines_units is False
    assert frozen.input_bin_size_ms == 1.0
    assert frozen.token_bin_size_s is not None

    # Build a dummy input via ntt's validation utility
    dummy = make_dummy_input(frozen.tokenizer.spec)
    assert isinstance(dummy, BinnedSpikesInput)

    out = frozen(dummy)
    assert isinstance(out, TokenizerOutput)

    B = dummy.spike_matrix.shape[0]
    assert out.tokens.ndim == 3  # [B, N, D] where N = T' * U
    assert out.tokens.shape[0] == B
    assert out.timestamps.shape == out.tokens.shape[:2]
    assert out.mask.shape == out.tokens.shape[:2]
    assert out.unit_ids is not None  # per-unit tokenizer
    assert out.unit_ids.shape == out.tokens.shape[:2]
    assert out.combines_units is False
    assert out.token_bin_size_s is not None


def test_poisson_baseline_all_params_frozen():
    """All parameters in the frozen tokenizer have requires_grad=False."""
    frozen = FrozenNTTTokenizer.from_poisson(
        rate_bin_size_ms=50.0,
        input_bin_size_ms=1.0,
        window_length_seconds=1.0,
    )
    for name, p in frozen.named_parameters():
        assert not p.requires_grad, f"Parameter {name} has requires_grad=True"


def test_poisson_baseline_output_dim():
    """output_dim property returns correct dimension."""
    frozen = FrozenNTTTokenizer.from_poisson(
        rate_bin_size_ms=50.0,
        input_bin_size_ms=1.0,
        window_length_seconds=1.0,
    )
    dim = frozen.output_dim
    assert isinstance(dim, int)
    assert dim > 0


def test_wrong_input_type_raises():
    """Passing wrong type to forward() raises TypeError."""
    frozen = FrozenNTTTokenizer.from_poisson(
        rate_bin_size_ms=50.0,
        input_bin_size_ms=1.0,
        window_length_seconds=1.0,
    )
    with pytest.raises(TypeError, match="Expected BinnedSpikesInput"):
        frozen(torch.zeros(2, 10, 5))


def test_wrong_tokenizer_type_raises():
    """Passing non-BaseTokenizer to __init__ raises TypeError."""
    with pytest.raises(TypeError, match="Expected BaseTokenizer"):
        FrozenNTTTokenizer(torch.nn.Linear(10, 10))


# -- Helper-function tests (unit-level, no checkpoint) --


def test_unflatten_tn_indices_3d_input():
    """3-D input (B, T*N) reshapes to (B, T, N, 1)."""
    from ntt_downstream.decoding.backends.poyo.frozen_ntt_tokenizer import (
        _unflatten_tn_indices,
    )

    flat = torch.arange(2 * 3 * 4, dtype=torch.int64).reshape(2, 3 * 4)
    out = _unflatten_tn_indices(flat, T=3, N=4)
    assert out.shape == (2, 3, 4, 1)
    assert out.dtype == torch.int64
    # Layout follows the wanneural _to_seq convention: t-major, n-minor.
    # flat[b, t*4 + n] should map to out[b, t, n, 0].
    for b in range(2):
        for t in range(3):
            for n in range(4):
                assert out[b, t, n, 0].item() == b * 12 + t * 4 + n


def test_unflatten_tn_indices_4d_input():
    """4-D input (B, T*N, Q) reshapes to (B, T, N, Q)."""
    from ntt_downstream.decoding.backends.poyo.frozen_ntt_tokenizer import (
        _unflatten_tn_indices,
    )

    flat = torch.arange(2 * 3 * 4 * 5, dtype=torch.int64).reshape(2, 3 * 4, 5)
    out = _unflatten_tn_indices(flat, T=3, N=4)
    assert out.shape == (2, 3, 4, 5)
    assert out.dtype == torch.int64


def test_unflatten_tn_indices_length_mismatch_raises():
    """Mismatch between sequence length and T*N raises."""
    from ntt_downstream.decoding.backends.poyo.frozen_ntt_tokenizer import (
        _unflatten_tn_indices,
    )

    flat = torch.arange(2 * 11, dtype=torch.int64).reshape(2, 11)
    with pytest.raises(ValueError, match=r"length"):
        _unflatten_tn_indices(flat, T=3, N=4)  # 3*4=12, not 11


# -- Discrete-mode probe tests (Poisson baseline = non-discrete) --


def test_poisson_baseline_is_not_discrete():
    """Poisson baseline is not a discrete tokenizer."""
    frozen = FrozenNTTTokenizer.from_poisson(rate_bin_size_ms=100.0, input_bin_size_ms=1.0)
    assert frozen.is_discrete is False
    assert frozen.discrete_input_mode == "z_q"


def test_poisson_baseline_embedding_mode_raises():
    """Requesting embedding mode on a non-discrete tokenizer raises at init."""
    with pytest.raises(ValueError, match=r"discrete"):
        FrozenNTTTokenizer.from_poisson(
            rate_bin_size_ms=100.0,
            input_bin_size_ms=1.0,
            discrete_input_mode="embedding",
        )


def test_poisson_baseline_num_codebooks_raises():
    """num_codebooks is undefined for non-discrete tokenizers."""
    frozen = FrozenNTTTokenizer.from_poisson(rate_bin_size_ms=100.0, input_bin_size_ms=1.0)
    with pytest.raises(ValueError, match=r"discrete"):
        _ = frozen.num_codebooks


@pytest.mark.skipif(
    not RESIDUAL_LFQ_URI,
    reason="RESIDUAL_LFQ_URI env var required for discrete checkpoint tests",
)
class TestDiscreteProbe:
    """Probe metadata against the residual-LFQ WanNeural checkpoint."""

    def test_probe_finds_residual_lfq_metadata(self):
        frozen = FrozenNTTTokenizer.from_checkpoint(RESIDUAL_LFQ_URI, device="cpu", discrete_input_mode="embedding")
        assert frozen.is_discrete is True
        assert frozen.num_codebooks == 4
        assert frozen.vocab_size == 1024
        assert frozen.discrete_input_mode == "embedding"

    def test_probe_zq_mode_default(self):
        frozen = FrozenNTTTokenizer.from_checkpoint(RESIDUAL_LFQ_URI, device="cpu")
        assert frozen.is_discrete is True
        assert frozen.discrete_input_mode == "z_q"


@pytest.mark.skipif(
    not RESIDUAL_LFQ_URI,
    reason="RESIDUAL_LFQ_URI env var required for residual-LFQ checkpoint tests",
)
class TestEncodeHelpers:
    """Verify the two encode-with-timestamps helpers against residual-LFQ."""

    def test_encode_indices_returns_int64_with_q_axis(self):
        from ntt.core.model.tokenizer.validation import make_dummy_input

        frozen = FrozenNTTTokenizer.from_checkpoint(RESIDUAL_LFQ_URI, device="cpu", discrete_input_mode="embedding")
        dummy = make_dummy_input(frozen.tokenizer.spec)
        indices, ts_meta = frozen._encode_indices_with_timestamps(dummy)
        assert indices.dtype == torch.int64
        assert indices.ndim == 4  # [B, T', U, Q]
        assert indices.shape[-1] == 4
        assert "token_timestamps" in ts_meta

    def test_encode_zq_returns_float(self):
        from ntt.core.model.tokenizer.validation import make_dummy_input

        frozen = FrozenNTTTokenizer.from_checkpoint(RESIDUAL_LFQ_URI, device="cpu")
        dummy = make_dummy_input(frozen.tokenizer.spec)
        latents, ts_meta = frozen._encode_zq_with_timestamps(dummy)
        assert torch.is_floating_point(latents)
        assert latents.ndim == 4  # [B, T', U, D]
        assert latents.shape[-1] == 16  # z_dim
        assert "token_timestamps" in ts_meta


@pytest.mark.skipif(
    not RESIDUAL_LFQ_URI,
    reason="RESIDUAL_LFQ_URI env var required for residual-LFQ checkpoint tests",
)
class TestForwardDualMode:
    def test_forward_zq_mode_returns_float(self):
        from ntt.core.model.tokenizer.validation import make_dummy_input

        frozen = FrozenNTTTokenizer.from_checkpoint(RESIDUAL_LFQ_URI, device="cpu")
        out = frozen(make_dummy_input(frozen.tokenizer.spec))
        assert torch.is_floating_point(out.tokens)
        assert out.tokens.ndim == 3  # [B, N, D]
        assert out.tokens.shape[-1] == 16  # z_dim
        assert out.vocab_sizes is None

    def test_forward_embedding_mode_returns_int64(self):
        from ntt.core.model.tokenizer.validation import make_dummy_input

        frozen = FrozenNTTTokenizer.from_checkpoint(RESIDUAL_LFQ_URI, device="cpu", discrete_input_mode="embedding")
        out = frozen(make_dummy_input(frozen.tokenizer.spec))
        assert out.tokens.dtype == torch.int64
        assert out.tokens.ndim == 3  # [B, N, Q]
        assert out.tokens.shape[-1] == 4  # num_codebooks
        assert out.vocab_sizes == (1024, 1024, 1024, 1024)
        assert out.timestamps.shape == out.tokens.shape[:2]
        assert out.mask.shape == out.tokens.shape[:2]

    def test_forward_zq_and_embedding_have_same_n_per_sample(self):
        """N (sequence length) must agree between modes for the same input."""
        from ntt.core.model.tokenizer.validation import make_dummy_input

        z = FrozenNTTTokenizer.from_checkpoint(RESIDUAL_LFQ_URI, device="cpu")
        e = FrozenNTTTokenizer.from_checkpoint(RESIDUAL_LFQ_URI, device="cpu", discrete_input_mode="embedding")
        dummy = make_dummy_input(z.tokenizer.spec)
        out_z = z(dummy)
        out_e = e(dummy)
        assert out_z.tokens.shape[:2] == out_e.tokens.shape[:2]
        if out_z.unit_ids is None:
            assert out_e.unit_ids is None
        else:
            assert torch.equal(out_z.unit_ids, out_e.unit_ids)
        assert torch.equal(out_z.mask, out_e.mask)


# -- Checkpoint-based tests (require real checkpoint files) --

# Override with NTT_TOKENIZER_CKPTS to point at a directory holding the
# `my_tokenizer/model.ckpt` checkpoint (any ntt tokenizer checkpoint) used by
# the from-checkpoint tests below. They skip if the file is missing.
CKPT_DIR = os.environ.get("NTT_TOKENIZER_CKPTS", "./tokenizer_ckpts/")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Checkpoint tests need GPU for practical speed",
)
class TestFromCheckpoint:
    """Tests that load real ntt checkpoints."""

    @pytest.fixture(
        params=[
            f"{CKPT_DIR}/my_tokenizer/model.ckpt",
        ]
    )
    def ckpt_path(self, request):
        import os

        path = request.param
        if not os.path.exists(path):
            pytest.skip(f"Checkpoint not found: {path}")
        return path

    def test_load_and_forward(self, ckpt_path):
        """Load checkpoint, run forward, verify output shapes."""
        from ntt.core.components.types import BinnedSpikesInput
        from ntt.core.model.tokenizer.validation import make_dummy_input

        frozen = FrozenNTTTokenizer.from_checkpoint(ckpt_path, device="cpu")

        # Verify frozen
        for name, p in frozen.named_parameters():
            assert not p.requires_grad, f"Parameter {name} has requires_grad=True"

        # Run with dummy input
        dummy = make_dummy_input(frozen.tokenizer.spec)
        assert isinstance(dummy, BinnedSpikesInput)

        out = frozen(dummy)
        assert isinstance(out, TokenizerOutput)

        B = dummy.spike_matrix.shape[0]
        assert out.tokens.ndim == 3
        assert out.tokens.shape[0] == B
        assert out.timestamps.shape[0] == B
        assert out.mask.shape[0] == B

        # Check consistency of combines_units and unit_ids
        if out.combines_units:
            assert out.unit_ids is None
        else:
            assert out.unit_ids is not None
            assert out.unit_ids.shape == out.tokens.shape[:2]

    def test_output_dim_matches_tokens(self, ckpt_path):
        """output_dim matches actual token dimension from forward()."""
        from ntt.core.model.tokenizer.validation import make_dummy_input

        frozen = FrozenNTTTokenizer.from_checkpoint(ckpt_path, device="cpu")
        dummy = make_dummy_input(frozen.tokenizer.spec)

        out = frozen(dummy)
        assert out.tokens.shape[-1] == frozen.output_dim


# -- output_dim probe behavior (no checkpoint needed) --


def test_output_dim_is_cached():
    """output_dim is computed once and cached across accesses."""
    frozen = FrozenNTTTokenizer.from_poisson(
        rate_bin_size_ms=50.0,
        input_bin_size_ms=1.0,
        window_length_seconds=1.0,
    )
    # Capture the underlying tokenizer.encode_with_timestamps and count calls.
    # cached_property stores the result in instance __dict__ on first access.
    real_eswt = frozen.tokenizer.encode_with_timestamps
    call_count = {"n": 0}

    def counting_eswt(*args, **kwargs):
        call_count["n"] += 1
        return real_eswt(*args, **kwargs)

    frozen.tokenizer.encode_with_timestamps = counting_eswt

    # Poisson baseline has no z_dim/embed_dim → goes through the encode probe
    # on first access. Subsequent accesses must hit the cache.
    d1 = frozen.output_dim
    d2 = frozen.output_dim
    d3 = frozen.output_dim
    assert d1 == d2 == d3
    assert d1 > 0
    assert call_count["n"] <= 1, f"output_dim should probe at most once; got {call_count['n']} probes"


def test_output_dim_returns_minus_one_on_probe_failure(caplog):
    """A raising encode_with_timestamps yields -1 plus a logged warning."""
    import logging

    frozen = FrozenNTTTokenizer.from_poisson(
        rate_bin_size_ms=50.0,
        input_bin_size_ms=1.0,
        window_length_seconds=1.0,
    )

    def boom(*_args, **_kwargs):
        raise RuntimeError("synthetic probe failure for test")

    frozen.tokenizer.encode_with_timestamps = boom

    # Poisson lacks z_dim/embed_dim, so the static fast-path doesn't fire and
    # we exercise the probe → except → -1 path.
    with caplog.at_level(logging.WARNING):
        d = frozen.output_dim
    assert d == -1
    assert any("probe failed" in rec.message.lower() for rec in caplog.records), (
        f"Expected a 'probe failed' warning in caplog; got {[r.message for r in caplog.records]}"
    )


# -- _flatten_4d combines_units branch with abstract latent slots (U > 1) --


def test_flatten_4d_combines_units_with_abstract_latent_slots():
    """combines_units=True + U>1 (Perceiver-style abstract latent slots).

    Verifies the abstract-latent branch in `_flatten_4d`: when the U axis
    indexes abstract latent slots (not session units), tokens flatten across
    U, the population mask broadcasts across slots, and per-slot proxy
    unit_ids (per-session slot proxies: ``batch.unit_ids[:, :U]`` tiled across T') are
    emitted so each (session, latent_slot) pair gets a learnable entry in
    the unit_emb table.
    """
    from ntt.core.components.types import BinnedSpikesInput

    # Build a frozen wrapper around any tokenizer; we'll call _flatten_4d
    # directly and force the combines_units branch by toggling the flag.
    frozen = FrozenNTTTokenizer.from_poisson(
        rate_bin_size_ms=50.0,
        input_bin_size_ms=1.0,
        window_length_seconds=1.0,
    )
    # Force the abstract-latent path even though Poisson is per-unit.
    frozen._combines_units = True

    B, T_input, U_input = 2, 100, 5
    T_out, U_out, D = 5, 4, 8  # U_out > 1 is the new case

    spike_matrix = torch.zeros((B, T_input, U_input), dtype=torch.int32)
    pad_and_gap_mask = torch.ones((B, T_input, U_input), dtype=torch.int32)
    # Mark the second half of batch 1's time window as gap-padded so the
    # collapsed-population mask is non-trivial (some tokens invalid).
    pad_and_gap_mask[1, T_input // 2 :, :] = 0

    batch = BinnedSpikesInput(
        spike_matrix=spike_matrix,
        pad_and_gap_mask=pad_and_gap_mask,
        sampling_rate=1000.0,
        unit_ids=torch.arange(U_input, dtype=torch.int32).unsqueeze(0).expand(B, -1).contiguous(),
        unit_ids_pad_mask=torch.ones((B, U_input), dtype=torch.int32),
        duration_sec=torch.tensor([0.1, 0.1]),
        session_id=("s0", "s1"),
        subject_id=("u", "u"),
        species=("sp", "sp"),
        absolute_start=torch.zeros((B,), dtype=torch.float64),
        absolute_end=torch.full((B,), 0.1, dtype=torch.float64),
    )

    latents = torch.randn(B, T_out, U_out, D)
    ts_metadata = {"token_timestamps": torch.linspace(0, 0.1, T_out).expand(B, T_out)}

    tokens, timestamps, unit_ids, mask = frozen._flatten_4d(latents, batch, ts_metadata)

    N = T_out * U_out
    assert tokens.shape == (B, N, D)
    assert timestamps.shape == (B, N)
    assert mask.shape == (B, N)
    assert mask.dtype == torch.bool

    # combines_units=True (per-session slot proxies) emits per-slot proxy unit_ids: slot k
    # → batch.unit_ids[:, k], tiled across T_out. Shape [B, N].
    assert unit_ids is not None
    assert unit_ids.shape == (B, N)
    expected_proxy = batch.unit_ids[:, :U_out].unsqueeze(1).expand(B, T_out, U_out).reshape(B, N)
    assert torch.equal(unit_ids, expected_proxy)

    # Mask should be constant across the U axis within each (B, T_out) slice
    # because we broadcast a [B, T_out, 1] population mask.
    mask_view = mask.reshape(B, T_out, U_out)
    for b in range(B):
        for t in range(T_out):
            row = mask_view[b, t]
            assert row.all() or not row.any(), f"mask row [b={b}, t={t}] is non-uniform across U: {row.tolist()}"

    # Tokens reshape t-major: tokens[b, t*U + u, :] == latents[b, t, u, :].
    for b in range(B):
        for t in range(T_out):
            for u in range(U_out):
                assert torch.equal(tokens[b, t * U_out + u], latents[b, t, u]), (
                    f"token order mismatch at [b={b}, t={t}, u={u}]"
                )
