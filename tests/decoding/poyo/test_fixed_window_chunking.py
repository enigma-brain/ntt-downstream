import pytest
import torch
from ntt.core.components.types import BinnedSpikesInput

from ntt_downstream.decoding.backends.poyo.frozen_ntt_tokenizer import FrozenNTTTokenizer
from ntt_downstream.decoding.backends.poyo.tokenizer_output import TokenizerOutput


def _to(tokens, ts, uids, mask, *, combines=False, bin_s=0.1, vocab=None):
    return TokenizerOutput(
        tokens=tokens,
        timestamps=ts,
        unit_ids=uids,
        mask=mask,
        combines_units=combines,
        token_bin_size_s=bin_s,
        vocab_sizes=vocab,
    )


def test_merge_continuous_concats_along_token_axis():
    B, D = 2, 4
    o0 = _to(
        torch.zeros(B, 3, D),
        torch.full((B, 3), 0.05),
        torch.zeros(B, 3, dtype=torch.long),
        torch.ones(B, 3, dtype=torch.bool),
    )
    o1 = _to(
        torch.ones(B, 3, D),
        torch.full((B, 3), 0.15),
        torch.ones(B, 3, dtype=torch.long),
        torch.ones(B, 3, dtype=torch.bool),
    )
    merged = FrozenNTTTokenizer._merge_chunk_outputs([o0, o1])
    assert merged.tokens.shape == (B, 6, D)
    assert torch.equal(merged.timestamps[:, :3], o0.timestamps)
    assert torch.equal(merged.timestamps[:, 3:], o1.timestamps)
    assert torch.equal(merged.unit_ids[:, 3:], o1.unit_ids)
    assert merged.vocab_sizes is None


def test_merge_discrete_concats_int_tokens():
    B, Q = 2, 4
    o0 = _to(
        torch.zeros(B, 1, Q, dtype=torch.long),
        torch.full((B, 1), 0.05),
        None,
        torch.ones(B, 1, dtype=torch.bool),
        combines=True,
        vocab=(1024,) * Q,
    )
    o1 = _to(
        torch.ones(B, 1, Q, dtype=torch.long),
        torch.full((B, 1), 0.15),
        None,
        torch.ones(B, 1, dtype=torch.bool),
        combines=True,
        vocab=(1024,) * Q,
    )
    merged = FrozenNTTTokenizer._merge_chunk_outputs([o0, o1])
    assert merged.tokens.shape == (B, 2, Q)
    assert merged.tokens.dtype == torch.long
    assert merged.unit_ids is None
    assert merged.vocab_sizes == (1024,) * Q


def test_merge_single_output_is_identity():
    B, D = 1, 4
    o0 = _to(
        torch.zeros(B, 3, D), torch.zeros(B, 3), torch.zeros(B, 3, dtype=torch.long), torch.ones(B, 3, dtype=torch.bool)
    )
    assert FrozenNTTTokenizer._merge_chunk_outputs([o0]) is o0


def test_merge_rejects_mismatched_invariants():
    B, D = 1, 4
    o0 = _to(
        torch.zeros(B, 1, D), torch.zeros(B, 1), None, torch.ones(B, 1, dtype=torch.bool), combines=True, bin_s=0.1
    )
    o1 = _to(
        torch.zeros(B, 1, D), torch.zeros(B, 1), None, torch.ones(B, 1, dtype=torch.bool), combines=True, bin_s=0.2
    )
    with pytest.raises(ValueError):
        FrozenNTTTokenizer._merge_chunk_outputs([o0, o1])


class _StubFixedSpec:
    variable_t = False
    has_time_dim = False
    window_length_bins = 10  # native window = 10 input bins
    bin_size_ms = 10.0
    bin_size_seconds = 0.01
    effective_window_length_bins = 10  # read by ntt's _effective_window_length_bins


class _StubFixedTokenizer:
    """Emits one token per window per unit; token value = mean spike count."""

    def __init__(self):
        self.spec = _StubFixedSpec()

    def encode_with_timestamps(self, batch):
        # batch.spike_matrix: [B, T=10, U]; emit [B, U, D=1] (3D, has_time_dim=False)
        z = batch.spike_matrix.float().mean(dim=1, keepdim=False).unsqueeze(-1)  # [B, U, 1]
        return z, {}  # ts rewritten downstream from absolute_start/end


# sr=100 Hz makes the stub self-consistent: bin_size_ms=10.0 ⇒ 100 bins/s, so a
# 10-bin native window is 0.1 s and the three chunk centers land at the brief's
# expected 0.05 / 0.15 / 0.25 s. (sr and bin_size_ms must agree for the
# window-center timestamps _slice_batch_time computes to be meaningful.)
def _make_full_window_binned(B=1, T=30, U=2, sr=100.0):
    sm = torch.arange(B * T * U, dtype=torch.int32).reshape(B, T, U)
    return BinnedSpikesInput(
        spike_matrix=sm,
        pad_and_gap_mask=torch.ones(B, T, U, dtype=torch.int32),
        sampling_rate=sr,
        unit_ids=torch.arange(U).unsqueeze(0).expand(B, U).contiguous(),
        unit_ids_pad_mask=torch.ones(B, U, dtype=torch.long),
        duration_sec=torch.full((B,), T / sr, dtype=torch.float64),
        session_id=("s",) * B,
        subject_id=("x",) * B,
        species=("x",) * B,
        absolute_start=torch.zeros(B, dtype=torch.float64),
        absolute_end=torch.full((B,), T / sr, dtype=torch.float64),
    )


def _make_frozen(tokenizer):
    tok = FrozenNTTTokenizer.__new__(FrozenNTTTokenizer)
    # Minimal attribute surface used by the single-window path for this stub.
    # Mirrors a real __init__: _freeze and _requires_neuron_padding are read by
    # the z_q single-window path (encode ctx + neuron-pad guard); set them to
    # the defaults a real construction would yield for this stub spec.
    tok.tokenizer = tokenizer
    tok.discrete_input_mode = "z_q"
    tok.fixed_window_seconds = None
    tok._freeze = True
    tok._requires_neuron_padding = False
    tok._combines_units = False
    tok._has_unit_dim = True
    tok._token_bin_size_s = 0.01
    tok.slot_global_ids = torch.empty(0, dtype=torch.long)
    return tok


def test_forward_chunks_fixed_window_input():
    tok = _make_frozen(_StubFixedTokenizer())
    binned = _make_full_window_binned(B=1, T=30, U=2)  # 30 bins / 10-bin window = 3 chunks
    out = tok.forward(binned)
    # has_time_dim=False, per-unit: each chunk → U tokens; 3 chunks → 3*U
    assert out.tokens.shape[1] == 3 * 2
    # timestamps strictly increasing across chunk boundaries (window centers 0.05, 0.15, 0.25)
    centers = out.timestamps[0].unique().sort().values
    assert torch.allclose(centers, torch.tensor([0.05, 0.15, 0.25], dtype=centers.dtype), atol=1e-6)


def test_forward_non_multiple_window_raises():
    """The runtime backstop fires when the window isn't a multiple of the chunk.

    With ``window_length_bins=10`` and a 25-bin window, the dispatch gate
    (``requires_fixed_input_window(spec) and t_in > chunk_width``) is satisfied
    (25 > 10) so ``forward`` routes into ``_forward_chunked``, where
    ``25 % 10 != 0`` must raise ``ValueError`` rather than silently dropping the
    trailing partial chunk. This check is an explicit ``raise`` (it survives ``python -O``);
    setup validation normally catches this, but the backstop must
    still hold if it's bypassed.
    """
    tok = _make_frozen(_StubFixedTokenizer())
    binned = _make_full_window_binned(B=1, T=25, U=2)  # 25 bins, not a multiple of 10
    with pytest.raises(ValueError, match="not an integer multiple"):
        tok.forward(binned)


def test_chunk_width_uses_effective_window_bins():
    """chunk_width_bins must follow ntt's effective-window source, not window_length_bins.

    The setup validator (assert_tokenizer_accepts_eval_window) and ntt's
    canonical chunker both reason in effective_window_length_bins; reading
    window_length_bins here would desync runtime chunking from validation.
    """

    class _SpecEffDiffersFromWindow:
        effective_window_length_bins = 7  # ntt's canonical chunk width
        window_length_bins = 10  # a different value than the effective one
        bin_size_ms = 10.0

    tok = _make_frozen(_StubFixedTokenizer())
    tok.tokenizer.spec = _SpecEffDiffersFromWindow()
    tok.fixed_window_seconds = None
    assert tok.chunk_width_bins() == 7


def test_forward_chunked_matches_sequential_reference():
    """Batched chunk-encode equals the per-chunk sequential encode (oracle).

    Pins ``_forward_chunked`` to a sequential reference built directly from
    ``_forward_single_window`` + ``_merge_chunk_outputs``, so the batched
    single-pass rewrite must produce identical tokens/timestamps/mask/unit_ids.
    The stub tokenizer is purely per-sample (mean), so a batched encode is
    bit-identical to encoding chunk-by-chunk.
    """
    from ntt_downstream.decoding.backends.poyo.frozen_ntt_tokenizer import (
        _slice_batch_time,
    )

    tok = _make_frozen(_StubFixedTokenizer())
    binned = _make_full_window_binned(B=2, T=30, U=2)  # 3 chunks of 10 bins
    chunk_width = 10
    n_chunks = 3
    sr = float(binned.sampling_rate)

    ref_parts = [
        tok._forward_single_window(
            _slice_batch_time(binned, t_start=k * chunk_width, t_end=(k + 1) * chunk_width, sampling_rate=sr)
        )
        for k in range(n_chunks)
    ]
    ref = FrozenNTTTokenizer._merge_chunk_outputs(ref_parts)

    out = tok.forward(binned)

    assert out.tokens.shape == ref.tokens.shape
    assert torch.allclose(out.tokens, ref.tokens, atol=1e-6)
    assert torch.equal(out.timestamps, ref.timestamps)
    assert torch.equal(out.mask, ref.mask)
    if ref.unit_ids is None:
        assert out.unit_ids is None
    else:
        assert torch.equal(out.unit_ids, ref.unit_ids)


# --- deterministic full-window raster for fixed-window specs -----------------


def _fake_poyo_batch_with_spikes_in(span, bin_ms, *, n_units=2, spikes_per_unit=3):
    """Build a minimal POYO ``model_inputs`` batch with spikes confined to ``span``.

    Mirrors the fields read at the top of ``_ntt_frozen_tokenize``:
    ``input_token_type`` (0 = spike), ``input_unit_index``, ``input_timestamps``
    (seconds), and ``input_mask``. Spikes are placed strictly inside
    ``span = [t0, t1]`` so a variable raster would span only ``t1 - t0`` rather
    than the full ``[0, sequence_length]`` window.
    """
    t0, t1 = span
    ts_list = []
    uid_list = []
    for u in range(n_units):
        # Evenly spaced timestamps inside the sub-interval (open at the upper end).
        for k in range(spikes_per_unit):
            frac = (k + 0.5) / spikes_per_unit
            ts_list.append(t0 + frac * (t1 - t0))
            uid_list.append(u)
    N = len(ts_list)
    input_timestamps = torch.tensor(ts_list, dtype=torch.float32).unsqueeze(0)  # [1, N]
    input_unit_index = torch.tensor(uid_list, dtype=torch.long).unsqueeze(0)  # [1, N]
    input_token_type = torch.zeros(1, N, dtype=torch.long)  # all spikes
    input_mask = torch.ones(1, N, dtype=torch.bool)
    return {
        "model_inputs": {
            "input_token_type": input_token_type,
            "input_unit_index": input_unit_index,
            "input_timestamps": input_timestamps,
            "input_mask": input_mask,
        }
    }


class _FixedRasterSpec:
    variable_t = False
    bin_size_ms = 10.0
    window_length_bins = 10
    has_time_dim = False


class _RasterCaptor:
    """Stands in for the frozen tokenizer; exposes ``tokenizer.spec`` + bin size."""

    input_bin_size_ms = 10.0

    def __init__(self):
        self.tokenizer = type("T", (), {"spec": _FixedRasterSpec()})()


def _bin_spikes_wrapper(seq_len=1.0):
    from ntt_downstream.decoding.backends.poyo import train_wrapper as TW

    wrapper = TW.V2TrainWrapper.__new__(TW.V2TrainWrapper)
    wrapper.ntt_tokenizer = _RasterCaptor()
    wrapper.model = type("M", (), {"sequence_length": seq_len})()
    return wrapper


def test_default_wrapper_full_window_raster_for_fixed_tokenizer():
    """Fixed-window specs get a deterministic ``[0, sequence_length]`` raster.

    Spikes live only in ``[0.4, 0.6]`` s of a 1.0 s window; a variable raster
    would emit ~20 bins. The fixed-window branch must bin the whole window
    (``round(1.0 / 0.01) = 100`` bins) with ``absolute_start == 0``.
    """
    wrapper = _bin_spikes_wrapper(seq_len=1.0)
    batch = _fake_poyo_batch_with_spikes_in([0.4, 0.6], bin_ms=10.0)
    binned = wrapper._bin_spikes(batch, fixed_window=True)
    assert binned.spike_matrix.shape[1] == 100  # 1.0 s / 0.01 s
    assert float(binned.absolute_start[0]) == 0.0
    assert float(binned.absolute_end[0]) == pytest.approx(1.0)


def test_default_wrapper_variable_raster_unchanged_for_variable_tokenizer():
    """Variable specs keep the spike-derived raster.

    The same [0.4, 0.6] s spikes under the variable path must span far fewer
    than 100 bins and anchor ``absolute_start`` at the first spike, not 0.
    """
    wrapper = _bin_spikes_wrapper(seq_len=1.0)
    batch = _fake_poyo_batch_with_spikes_in([0.4, 0.6], bin_ms=10.0)
    binned = wrapper._bin_spikes(batch, fixed_window=False)
    assert binned.spike_matrix.shape[1] < 100
    assert float(binned.absolute_start[0]) > 0.0


# --- setup-time fail-loud validation + fixed_window_seconds override ---------


def test_incompatible_window_fails_loud():
    """Document our reliance on ntt's spec contract for the non-override path.

    ``_instantiate_ntt_tokenizer`` defers to this helper when no
    ``fixed_window_seconds`` override is set, so this asserts the helper
    behaves as we rely on: rejects a window that is not a positive integer
    multiple of the tokenizer's native window, accepts one that is.
    """
    from ntt.evaluation.utils.window_contract import (
        IncompatibleEvalWindowError,
        assert_tokenizer_accepts_eval_window,
    )

    class _Spec:
        variable_t = False
        KIND = "stub_fixed"
        effective_window_length_seconds = 0.3

    with pytest.raises(IncompatibleEvalWindowError):
        assert_tokenizer_accepts_eval_window(_Spec(), 1.0)  # 1.0 not a multiple of 0.3
    assert_tokenizer_accepts_eval_window(_Spec(), 0.9)  # 0.9 == 3 * 0.3, OK


# These tests call the REAL setup-time validator from ``train.py`` (no copy),
# driving it with tiny stub specs so no checkpoint or Hydra config is needed.
from ntt_downstream.decoding.train import _validate_tokenizer_eval_window


class _VariableSpec:
    """variable_t=True spec with an awkward token quantum (e.g. Perceiver).

    ``requires_fixed_input_window`` returns ``not variable_t`` → False here, so
    the helper must NOT subject it to ntt's quantum-multiple contract.
    """

    variable_t = True
    KIND = "stub_variable"
    token_bin_size_ms = 33.33  # 1000 / 33.33 ≈ 30.003 — not an integer multiple
    bin_size_seconds = 0.01  # used only by the override path's bin-space check


class _FixedSpec:
    """variable_t=False spec with a 0.3 s native window (rigid-fixed)."""

    variable_t = False
    KIND = "stub_fixed"
    effective_window_length_seconds = 0.3


def test_variable_t_awkward_quantum_does_not_raise():
    """variable_t=True tokenizers are not constrained by the setup validation.

    Perceiver (33.33 ms) and transformer-vq (non-100 ms quanta) run unchunked.
    At the universal ``sequence_length=1.0`` s the quantum-multiple
    contract would reject them, so the helper must skip it for variable_t specs.
    """
    _validate_tokenizer_eval_window(_VariableSpec(), 1.0, None)  # no raise


def test_fixed_window_non_multiple_raises():
    """Fixed-window specs go through ntt's contract: non-multiple fails loud."""
    from ntt.evaluation.utils.window_contract import IncompatibleEvalWindowError

    with pytest.raises(IncompatibleEvalWindowError):
        _validate_tokenizer_eval_window(_FixedSpec(), 1.0, None)  # 1.0 not a multiple of 0.3


def test_override_window_divisibility_check():
    """The override path raises on a non-multiple and accepts an exact multiple.

    With an override set, the helper validates against the user-declared window
    directly (independent of the spec), so a tiny variable_t stub suffices.
    """
    spec = _VariableSpec()
    with pytest.raises(ValueError):
        _validate_tokenizer_eval_window(spec, 1.0, 0.3)  # 1.0 not a multiple of 0.3
    with pytest.raises(ValueError):
        _validate_tokenizer_eval_window(spec, 0.0, 0.3)  # k < 1
    # Exact integer multiples must pass.
    _validate_tokenizer_eval_window(spec, 0.9, 0.3)
    _validate_tokenizer_eval_window(spec, 1.0, 0.5)


def test_override_bin_space_divisibility():
    """Override that divides in seconds but not in bins fails loud at setup.

    The runtime chunks a round(seq/bin)-bin raster into round(w/bin)-bin
    chunks; independent rounding can break bin-space divisibility even when
    seconds divide exactly. bin=7 ms, w=0.1 s (→14 bins), seq=0.3 s (3*0.1 s,
    seconds-OK; raster round(0.3/0.007)=43 bins, 43 % 14 != 0).
    """

    class _BinSpec:
        variable_t = False
        KIND = "stub_bin"
        bin_size_seconds = 0.007

    with pytest.raises(ValueError):
        _validate_tokenizer_eval_window(_BinSpec(), 0.3, 0.1)


# --- end-to-end integration on a REAL fixed-window tokenizer -----------------
#
# The unit tests above drive the chunking machinery with stub tokenizers. This
# slow test closes the loop with a real ntt fixed-window tokenizer
# (``SimpleVAETokenizer``, ``variable_t=False``, ``has_time_dim=False``) wrapped
# in the production ``FrozenNTTTokenizer`` and run through its real ``forward``
# dispatch. We build the tokenizer purely from its spec (no checkpoint —
# random init is fine for a no-crash + shape/timestamp invariant test) with a
# native window SHORTER than the POYO window, so the chunking branch must fire.


def _build_one_second_binned(B=2, sr=1000.0, n_units=1, seed=0):
    """A deterministic ``[0, 1.0]`` s full-window ``BinnedSpikesInput``.

    ``sr = 1000 Hz`` ⇒ 1000 input bins for a 1.0 s window, which is an exact
    multiple of the tokenizer's 100-bin (0.1 s) native window. The matrix is
    seeded so the test is reproducible; values are irrelevant (no numerical
    assertions), only shapes + window bounds matter.
    """
    T = round(1.0 * sr)  # 1000 bins
    g = torch.Generator().manual_seed(seed)
    spike_matrix = (torch.rand(B, T, n_units, generator=g) < 0.1).to(torch.float32)
    return BinnedSpikesInput(
        spike_matrix=spike_matrix,
        pad_and_gap_mask=torch.ones(B, T, n_units, dtype=torch.int32),
        sampling_rate=sr,
        unit_ids=torch.arange(n_units).unsqueeze(0).expand(B, n_units).contiguous(),
        unit_ids_pad_mask=torch.ones(B, n_units, dtype=torch.long),
        duration_sec=torch.full((B,), T / sr, dtype=torch.float64),
        session_id=("s",) * B,
        subject_id=("x",) * B,
        species=("x",) * B,
        absolute_start=torch.zeros(B, dtype=torch.float64),
        absolute_end=torch.full((B,), T / sr, dtype=torch.float64),
    )


@pytest.mark.slow
def test_fixed_window_tokenizer_runs_on_one_second_window():
    """A real fixed-window tokenizer runs on a 1 s POYO window without crashing.

    Builds a config-only ``SimpleVAETokenizer`` (no checkpoint) whose native
    input window is 0.1 s (100 input bins at ``bin_size_ms=1.0``), wraps it in
    ``FrozenNTTTokenizer``, and feeds a deterministic 1.0 s ``BinnedSpikesInput``
    straight into ``forward``. With a 1.0 s window and a 0.1 s native window the
    chunking branch slices into ``n_chunks = 1.0 / 0.1 = 10`` native chunks.

    ``SimpleVAE`` is ``has_time_dim=False`` + population (``compress_n``/no unit
    dim), so each chunk emits exactly one token ⇒ ``per_chunk_tokens = 1`` and
    the merged sequence has ``n_chunks * per_chunk_tokens = 10`` tokens. We
    assert no exception, that token/timestamp shapes match that invariant, and
    that every token timestamp lands inside ``[0, 1.0]`` (the chunk centers
    ``0.05, 0.15, ..., 0.95``).
    """
    from ntt.contrib.tokenizer.simple_vae import (
        SimpleVAETokenizer,
        SimpleVAETokenizerSpec,
    )
    from ntt.evaluation.utils.window_chunking import requires_fixed_input_window

    native_window_s = 0.1
    spec = SimpleVAETokenizerSpec(
        window_length_seconds=native_window_s, bin_size_ms=1.0
    )  # n_neurons fixed at 1 for SimpleVAE
    # Sanity-check the fixture actually exercises the chunking path: a
    # variable_t=True spec here would silently take the single-window branch.
    assert requires_fixed_input_window(spec)
    assert spec.window_length_bins == 100  # 0.1 s / 0.001 s

    tokenizer = SimpleVAETokenizer(spec)
    tokenizer.eval()
    frozen = FrozenNTTTokenizer(tokenizer=tokenizer)

    seq_len_s = 1.0
    binned = _build_one_second_binned(B=2, sr=1000.0)
    assert binned.spike_matrix.shape[1] == round(seq_len_s * 1000.0)  # full 1 s raster

    out = frozen.forward(binned)  # must not raise

    n_chunks = round(seq_len_s / native_window_s)  # 10
    per_chunk_tokens = 1  # population, has_time_dim=False ⇒ one token per chunk
    assert out.tokens.shape[1] == n_chunks * per_chunk_tokens
    assert out.timestamps.shape == out.tokens.shape[:2]

    # Every token timestamp sits strictly inside the deployed window.
    assert float(out.timestamps.min()) >= 0.0
    assert float(out.timestamps.max()) <= seq_len_s
    # The 10 chunk centers, stitched in order, are the unique timestamps.
    centers = out.timestamps[0].unique().sort().values
    expected = torch.arange(n_chunks, dtype=centers.dtype) * native_window_s + native_window_s / 2
    assert torch.allclose(centers, expected, atol=1e-5)


@pytest.mark.slow
def test_full_window_wrapper_chunks_fixed_window_tokenizer():
    """FullWindowUnitV2TrainWrapper runs a fixed-window tokenizer via chunking.

    FrozenNTTTokenizer.forward routes fixed-window tokenizers through
    _forward_chunked even from the full-window wrapper. Build a real SimpleVAE
    (0.1 s native window = 100 input bins) and feed a 1 s POYO window through the wrapper: it must chunk into 10 native
    windows and stitch, with the raster sized as n_chunks * chunk_width (the
    full-window analog of chunk_width_bins being the single source of truth).

    (A ceil-gap — n_bins not a multiple of chunk_width — is unreachable with a
    well-formed spec: it needs a non-integer native_window/bin, which makes the
    tokenizer's own architecture inconsistent with its spec, so we test the
    realistic clean-bin path instead.)
    """
    from ntt.contrib.tokenizer.simple_vae import (
        SimpleVAETokenizer,
        SimpleVAETokenizerSpec,
    )
    from ntt.evaluation.utils.window_chunking import requires_fixed_input_window

    from ntt_downstream.decoding.backends.poyo.full_window_train_wrapper import (
        FullWindowUnitV2TrainWrapper,
    )

    spec = SimpleVAETokenizerSpec(window_length_seconds=0.1, bin_size_ms=1.0)
    assert requires_fixed_input_window(spec)
    tokenizer = SimpleVAETokenizer(spec)
    tokenizer.eval()
    frozen = FrozenNTTTokenizer(tokenizer=tokenizer)
    chunk_width = frozen.chunk_width_bins()
    assert chunk_width == 100  # 0.1 s / 0.001 s

    wrapper = FullWindowUnitV2TrainWrapper.__new__(FullWindowUnitV2TrainWrapper)
    torch.nn.Module.__init__(wrapper)  # set up _modules so the nn.Module assign works
    wrapper.ntt_tokenizer = frozen
    wrapper.model = type("M", (), {"sequence_length": 1.0})()
    wrapper.cfg = None  # _zero_unit_padding_target returns None when cfg.tokenizer absent

    # POYO batch: one unit, spikes inside [0, 1.0] s, all spike tokens
    # (token_type 0) so _global_units_from_start_sentinels uses its
    # unique-units fallback.
    ts = torch.tensor([[0.05, 0.25, 0.55, 0.95]], dtype=torch.float32)
    batch = {
        "model_inputs": {
            "input_token_type": torch.zeros(1, 4, dtype=torch.long),
            "input_unit_index": torch.zeros(1, 4, dtype=torch.long),
            "input_timestamps": ts,
            "input_mask": torch.ones(1, 4, dtype=torch.bool),
        }
    }

    out = wrapper._ntt_frozen_tokenize(batch)  # must not raise; chunks 1 s into 10× 0.1 s
    n_chunks = round(1.0 / (chunk_width * 0.001))  # 10
    assert out.tokens.shape[1] == n_chunks
    assert float(out.timestamps.min()) >= 0.0
    assert float(out.timestamps.max()) <= 1.0
