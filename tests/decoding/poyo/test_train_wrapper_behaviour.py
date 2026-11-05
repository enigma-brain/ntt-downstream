"""Behaviour tests for the POYO train wrapper and the frozen-tokenizer padding.

None of
these tests require an MLflow checkpoint or GPU — everything runs against
synthetic inputs or the Poisson baseline.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from ntt.core.components.types import BinnedSpikesInput, SpikeTimesInput
from omegaconf import OmegaConf
from torch import nn

from ntt_downstream.decoding.backends.poyo.frozen_ntt_tokenizer import FrozenNTTTokenizer
from ntt_downstream.decoding.backends.poyo.tokenizer_output import TokenizerOutput
from ntt_downstream.decoding.backends.poyo.train_wrapper import V2TrainWrapper

# ----------------------------------------------------------------------
# Shared fixtures: a minimal POYO-like model + Lightning-free wrapper.
# ----------------------------------------------------------------------


def _minimal_cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "log_every_n_steps": 1,
            "gpus": 1,
            "num_nodes": 1,
            "batch_size": 4,
            "optim": {
                "base_lr": 1e-6,
                "weight_decay": 0.0,
                "warmup_pct": 0.1,
                "div_factor": 25,
                "final_div_factor": 100,
            },
        }
    )


class _DummyPOYO(nn.Module):
    """Minimal POYO stand-in exposing only the attributes V2TrainWrapper reads."""

    def __init__(self, dim: int = 64, sequence_length: float = 1.0) -> None:
        super().__init__()
        self.dim = dim
        self.sequence_length = sequence_length
        self.unit_emb = nn.Embedding(64, dim)
        self.token_type_emb = nn.Embedding(8, dim)
        self.session_emb = nn.Embedding(8, dim)


class _DummyModalitySpec:
    def loss_fn(self, *args, **kwargs):
        return torch.tensor(0.0)


def _wrapper_fake_mode() -> V2TrainWrapper:
    return V2TrainWrapper(
        cfg=_minimal_cfg(),
        model=_DummyPOYO(dim=64),
        modality_spec=_DummyModalitySpec(),
        tokenizer_mode="fake",
        unit_embedding_mode="none",
        sentinels=False,
    )


def _attach_mock_trainer(wrapper: V2TrainWrapper, log_dir: Path) -> None:
    """Stub out trainer/logger so _log_run_info can write run_info.json."""
    fake_trainer = MagicMock()
    fake_trainer.is_global_zero = True
    fake_trainer.log_dir = str(log_dir)
    fake_trainer.default_root_dir = str(log_dir)
    fake_trainer.logger = MagicMock()
    fake_trainer.logger.log_hyperparams = MagicMock()
    wrapper._trainer = fake_trainer  # bypass the LightningModule property setter


# ----------------------------------------------------------------------
# token_dims_per_sec must equal tokens_per_second * token_dim
# ----------------------------------------------------------------------


def _make_discrete_tok_out(B=2, N=10, Q=4, K=1024) -> TokenizerOutput:
    return TokenizerOutput(
        tokens=torch.randint(0, K, (B, N, Q), dtype=torch.int64),
        timestamps=torch.linspace(0.0, 1.0, N).unsqueeze(0).expand(B, -1).contiguous(),
        unit_ids=torch.randint(0, 32, (B, N), dtype=torch.long),
        mask=torch.ones(B, N, dtype=torch.bool),
        combines_units=False,
        token_bin_size_s=0.1,  # fixed-rate path → uses model.sequence_length
        vocab_sizes=(K,) * Q,
    )


def _make_float_tok_out(B=2, N=10, D=16) -> TokenizerOutput:
    return TokenizerOutput(
        tokens=torch.randn(B, N, D),
        timestamps=torch.linspace(0.0, 1.0, N).unsqueeze(0).expand(B, -1).contiguous(),
        unit_ids=torch.randint(0, 32, (B, N), dtype=torch.long),
        mask=torch.ones(B, N, dtype=torch.bool),
        combines_units=False,
        token_bin_size_s=0.1,
    )


def test_run_info_token_dims_per_sec_consistent_for_discrete(tmp_path: Path):
    """For int64 token outputs, token_dims_per_sec must equal
    tokens_per_second * token_dim, where token_dim is model.dim
    (the width after the code embedding), not the codebook count Q."""
    wrapper = _wrapper_fake_mode()
    Q, K = 4, 1024
    tok_out = _make_discrete_tok_out(Q=Q, K=K)
    wrapper._fake_tokenize = lambda batch: tok_out
    _attach_mock_trainer(wrapper, tmp_path)

    wrapper._log_run_info({"session_id": ["sess0"]})

    info = json.loads((tmp_path / "run_info.json").read_text())
    assert info["token_dim"] == wrapper.model.dim
    assert info["token_dims_per_sec"] == pytest.approx(info["tokens_per_second"] * info["token_dim"])
    # Discrete-mode metadata is preserved.
    assert info["discrete_input_mode"] == "embedding"
    assert info["num_codebooks"] == Q
    assert info["vocab_size"] == K


def test_run_info_token_dims_per_sec_consistent_for_float(tmp_path: Path):
    """Float branch: token_dim should be the trailing tensor axis (D),
    not model.dim. Sanity check that the refactor didn't break this."""
    wrapper = _wrapper_fake_mode()
    D = 16
    tok_out = _make_float_tok_out(D=D)
    wrapper._fake_tokenize = lambda batch: tok_out
    _attach_mock_trainer(wrapper, tmp_path)

    wrapper._log_run_info({"session_id": ["sess0"]})

    info = json.loads((tmp_path / "run_info.json").read_text())
    assert info["token_dim"] == D
    assert info["token_dims_per_sec"] == pytest.approx(info["tokens_per_second"] * info["token_dim"])
    assert "num_codebooks" not in info  # not a discrete tokenizer


# ----------------------------------------------------------------------
# run-info probe must NOT fire from on_fit_start
# ----------------------------------------------------------------------


def test_run_info_deferred_to_first_batch(tmp_path: Path):
    """on_fit_start does not probe; the probe must fire on
    on_train_batch_start with batch_idx==0 and be idempotent thereafter."""
    wrapper = _wrapper_fake_mode()
    tok_out = _make_float_tok_out()
    call_count = {"n": 0}

    def counting_fake_tokenize(batch):
        call_count["n"] += 1
        return tok_out

    wrapper._fake_tokenize = counting_fake_tokenize
    _attach_mock_trainer(wrapper, tmp_path)

    # Pre-condition: latch is unset.
    assert wrapper._run_info_logged is False

    # First batch fires the probe.
    wrapper.on_train_batch_start({"session_id": ["sess0"]}, batch_idx=0)
    assert wrapper._run_info_logged is True
    assert call_count["n"] == 1
    assert (tmp_path / "run_info.json").exists()

    # Subsequent batches must NOT re-probe.
    wrapper.on_train_batch_start({"session_id": ["sess0"]}, batch_idx=1)
    wrapper.on_train_batch_start({"session_id": ["sess0"]}, batch_idx=2)
    assert call_count["n"] == 1


def test_run_info_eval_only_path_via_validation_hook(tmp_path: Path):
    """eval_only runs never enter training, so on_validation_batch_start
    is the only path that can write run_info.json."""
    wrapper = _wrapper_fake_mode()
    tok_out = _make_float_tok_out()
    wrapper._fake_tokenize = lambda batch: tok_out
    _attach_mock_trainer(wrapper, tmp_path)

    wrapper.on_validation_batch_start({"session_id": ["sess0"]}, batch_idx=0)
    assert wrapper._run_info_logged is True
    assert (tmp_path / "run_info.json").exists()


def test_on_fit_start_does_not_touch_train_dataloader():
    """on_fit_start must not override LightningModule's default — any
    override would iter() the train_loader and
    perturb the sampler RNG."""
    # If a subclass defines on_fit_start, it appears in __dict__. The
    # base LightningModule's no-op default is what we want; structural
    # check is sufficient.
    assert "on_fit_start" not in V2TrainWrapper.__dict__, (
        "on_fit_start was reintroduced; it must not consume from the "
        "train dataloader (the resulting iter() perturbs sampler RNG "
        "and burns persistent_workers prefetch). Move probing to "
        "on_train_batch_start instead."
    )


# ----------------------------------------------------------------------
# _ntt_frozen_tokenize must anchor token timestamps to t_min
# ----------------------------------------------------------------------


def _build_sparse_poyo_batch(
    *,
    spike_window_lo: float = 0.4,
    spike_window_hi: float = 0.6,
    n_spikes: int = 60,
) -> dict:
    """Build a single-sample POYO batch dict with spikes confined to
    [spike_window_lo, spike_window_hi] of a 1.0s window."""
    ts = torch.linspace(spike_window_lo, spike_window_hi, n_spikes)
    uids = torch.zeros(n_spikes, dtype=torch.long)  # all from unit 0
    mi = {
        "input_token_type": torch.zeros(1, n_spikes, dtype=torch.long),  # 0=DEFAULT
        "input_unit_index": uids.unsqueeze(0),
        "input_timestamps": ts.unsqueeze(0),
        "input_mask": torch.ones(1, n_spikes, dtype=torch.bool),
    }
    return {"model_inputs": mi, "session_id": ["sess0"]}


def test_ntt_frozen_tokenize_preserves_original_timestamps_for_sparse_window():
    """A sparse-spike window (spikes only in [0.4, 0.6] of a 1.0s window)
    must yield token timestamps in [0.4, 0.6 + token_bin], NOT in
    [0, ~0.2] (which is what anchoring at abs_start=0 would produce)."""
    wrapper = V2TrainWrapper(
        cfg=_minimal_cfg(),
        model=_DummyPOYO(dim=64, sequence_length=1.0),
        modality_spec=_DummyModalitySpec(),
        tokenizer_mode="ntt_frozen",
        unit_embedding_mode="none",
        sentinels=False,
        ntt_tokenizer=FrozenNTTTokenizer.from_poisson(
            rate_bin_size_ms=50.0,
            input_bin_size_ms=1.0,
            window_length_seconds=1.0,
        ),
    )
    batch = _build_sparse_poyo_batch(spike_window_lo=0.4, spike_window_hi=0.6, n_spikes=60)
    out = wrapper._ntt_frozen_tokenize(batch)

    valid_ts = out.timestamps[out.mask]
    assert valid_ts.numel() > 0
    t_lo, t_hi = float(valid_ts.min().item()), float(valid_ts.max().item())
    # Token centers must lie within the original spike window (with one
    # token-bin of slack on each side for the (i+0.5)*bin centering).
    token_bin = 0.05
    assert t_lo >= 0.4 - token_bin / 2 - 1e-3, (
        f"Earliest token timestamp {t_lo:.4f} is too early; expected ≥ ~0.4. "
        f"Timestamps collapsed towards 0 instead of anchoring at t_min."
    )
    assert t_hi <= 0.6 + token_bin + 1e-3, f"Latest token timestamp {t_hi:.4f} is too late; expected ≤ ~0.65."


def test_ntt_frozen_tokenize_stays_in_window_at_boundary():
    """Sparse spikes near window_end (single spike at t=0.99 in a 1.0s
    window) must NOT emit token timestamps past sequence_length — the
    adapter's bounds guard would crash on `sentinels=True` runs.

    Anchoring at t_min combined with _pad_time_if_needed's
    matrix-extension to align with token_bin overshoots window_end by
    up to token_bin/2 if uncorrected. The wrapper must clamp.
    """
    seq_len = 1.0
    wrapper = V2TrainWrapper(
        cfg=_minimal_cfg(),
        model=_DummyPOYO(dim=64, sequence_length=seq_len),
        modality_spec=_DummyModalitySpec(),
        tokenizer_mode="ntt_frozen",
        unit_embedding_mode="none",
        sentinels=False,
        ntt_tokenizer=FrozenNTTTokenizer.from_poisson(
            rate_bin_size_ms=50.0,
            input_bin_size_ms=1.0,
            window_length_seconds=seq_len,
        ),
    )
    # Single spike very near window end; makes t_max ≈ window_end and
    # forces _pad_time_if_needed to extend the matrix past window_end.
    mi = {
        "input_token_type": torch.zeros(1, 1, dtype=torch.long),
        "input_unit_index": torch.zeros(1, 1, dtype=torch.long),
        "input_timestamps": torch.tensor([[0.99]]),
        "input_mask": torch.ones(1, 1, dtype=torch.bool),
    }
    out = wrapper._ntt_frozen_tokenize({"model_inputs": mi, "session_id": ["s"]})
    valid_ts = out.timestamps[out.mask]
    assert valid_ts.numel() > 0
    t_max = float(valid_ts.max().item())
    # Allow a 1e-3 numerical slack matching the adapter's tolerance.
    assert t_max <= seq_len + 1e-3, (
        f"Boundary token timestamp {t_max:.4f} exceeds sequence_length "
        f"{seq_len}; adapter bounds guard would crash with sentinels=True."
    )


# ----------------------------------------------------------------------
# _pad_neurons_if_needed must duplicate-pad narrow samples
# ----------------------------------------------------------------------


def _make_compress_n_frozen_tokenizer(n_expected: int) -> FrozenNTTTokenizer:
    """Build a Poisson-baseline FrozenNTTTokenizer and force its
    compress_n / fixed-N path so _pad_neurons_if_needed actually fires.
    We patch the cached spec-derived flags directly since Poisson is
    natively per-unit (variable N)."""
    frozen = FrozenNTTTokenizer.from_poisson(
        rate_bin_size_ms=50.0,
        input_bin_size_ms=1.0,
        window_length_seconds=1.0,
    )
    frozen._requires_neuron_padding = True
    frozen._expected_n_neurons = n_expected
    return frozen


def test_pad_neurons_mixed_batch_no_zero_gaps():
    """Mixed-session batch: sample 0 has 50 valid neurons, sample 1 has 128.
    The batch width U_current=128 already meets n_expected, but sample 0's
    columns 50..127 are scatter_add zeros and must not be left as such:
    sample 0's columns 50..127 must contain duplicates of its own real
    columns (positive entries from a sample we deliberately fill non-zero),
    not zeros."""
    n_expected = 128
    frozen = _make_compress_n_frozen_tokenizer(n_expected=n_expected)

    B, T = 2, 100
    U_current = 128

    spike_matrix = torch.zeros(B, T, U_current, dtype=torch.int32)
    pad_mask = torch.zeros(B, T, U_current, dtype=torch.int32)
    unit_ids = torch.zeros(B, U_current, dtype=torch.long)
    unit_ids_pad_mask = torch.zeros(B, U_current, dtype=torch.int32)

    # Sample 0: 50 valid neurons, real spikes set to a known non-zero pattern
    # so we can detect when a column was duplicated vs. left as zero.
    spike_matrix[0, :, :50] = torch.randint(1, 5, (T, 50), dtype=torch.int32)
    pad_mask[0, :, :50] = 1
    unit_ids[0, :50] = torch.arange(1000, 1050)  # distinct uids for sample 0
    unit_ids_pad_mask[0, :50] = 1

    # Sample 1: 128 valid neurons (the wide one driving U_current)
    spike_matrix[1] = torch.randint(1, 5, (T, U_current), dtype=torch.int32)
    pad_mask[1] = 1
    unit_ids[1] = torch.arange(2000, 2000 + U_current)
    unit_ids_pad_mask[1] = 1

    batch = BinnedSpikesInput(
        spike_matrix=spike_matrix,
        pad_and_gap_mask=pad_mask,
        sampling_rate=1000.0,
        unit_ids=unit_ids,
        unit_ids_pad_mask=unit_ids_pad_mask,
        duration_sec=torch.full((B,), 0.1),
        session_id=("s0", "s1"),
        subject_id=("u", "u"),
        species=("sp", "sp"),
        absolute_start=torch.zeros(B, dtype=torch.float64),
        absolute_end=torch.full((B,), 0.1, dtype=torch.float64),
    )

    out = frozen._pad_neurons_if_needed(batch)

    # Sample 0's columns 50..127 must be filled with duplicates from its
    # own columns 0..49, not zeros.
    sample0_filler_cols = out.spike_matrix[0, :, 50:128]
    sample0_real_cols = out.spike_matrix[0, :, :50]
    assert sample0_filler_cols.sum() > 0, (
        "Sample 0's filler columns 50..127 are all zeros. They should be duplicates of the sample's own real columns."
    )
    # Each filler column must equal at least one of the real columns
    # (since fillers are sampled with replacement from real columns).
    for c in range(50, 128):
        col = out.spike_matrix[0, :, c]
        match = any(torch.equal(col, sample0_real_cols[:, k]) for k in range(50))
        assert match, f"Sample 0's filler column {c} is not a duplicate of any real column"

    # Sample 0's filler columns must be marked valid in unit_ids_pad_mask
    # so the tokenizer doesn't mask them out.
    assert (out.unit_ids_pad_mask[0, 50:128] == 1).all()

    # Sample 1 was already wide enough: real columns must be untouched.
    assert torch.equal(out.spike_matrix[1, :, :U_current], spike_matrix[1])


def test_pad_neurons_zero_work_fast_path():
    """When every sample already has exactly n_expected real columns,
    the function should be a no-op (no rebuild)."""
    n_expected = 128
    frozen = _make_compress_n_frozen_tokenizer(n_expected=n_expected)

    B, T = 2, 100
    spike_matrix = torch.randint(1, 5, (B, T, n_expected), dtype=torch.int32)
    pad_mask = torch.ones(B, T, n_expected, dtype=torch.int32)
    unit_ids = torch.arange(n_expected).expand(B, -1).contiguous()
    unit_ids_pad_mask = torch.ones(B, n_expected, dtype=torch.int32)

    batch = BinnedSpikesInput(
        spike_matrix=spike_matrix,
        pad_and_gap_mask=pad_mask,
        sampling_rate=1000.0,
        unit_ids=unit_ids,
        unit_ids_pad_mask=unit_ids_pad_mask,
        duration_sec=torch.full((B,), 0.1),
        session_id=("s", "s"),
        subject_id=("u", "u"),
        species=("sp", "sp"),
        absolute_start=torch.zeros(B, dtype=torch.float64),
        absolute_end=torch.full((B,), 0.1, dtype=torch.float64),
    )
    out = frozen._pad_neurons_if_needed(batch)
    # The fast-path returns the same object unchanged.
    assert out is batch


def test_pad_neurons_filler_indices_stable_across_batch_position():
    """Filler indices for a given sample must depend only on that
    sample's unit ID set, not on its position-in-batch — the same
    sample at index 0 of one batch and index 1 of another must get
    identical filler columns. Seeding with a constant per
    call would incorrectly key the fillers on draw order."""
    n_expected = 128
    frozen = _make_compress_n_frozen_tokenizer(n_expected=n_expected)
    T = 50

    def _make_sample(seed_uids: torch.Tensor, fill_value: int) -> dict:
        n_actual = seed_uids.shape[0]
        # Fill columns with sample-specific values so we can detect which
        # sample's columns ended up where in the duplicated output.
        spike = torch.full((T, n_actual), fill_value, dtype=torch.int32)
        return {
            "spike": spike,
            "uids": seed_uids,
            "n_actual": n_actual,
        }

    def _make_batch(samples: list[dict], pad_to_U: int) -> BinnedSpikesInput:
        B = len(samples)
        spike_matrix = torch.zeros(B, T, pad_to_U, dtype=torch.int32)
        pad_mask = torch.zeros(B, T, pad_to_U, dtype=torch.int32)
        unit_ids = torch.zeros(B, pad_to_U, dtype=torch.long)
        unit_ids_pad_mask = torch.zeros(B, pad_to_U, dtype=torch.int32)
        for b, s in enumerate(samples):
            n = s["n_actual"]
            spike_matrix[b, :, :n] = s["spike"]
            pad_mask[b, :, :n] = 1
            unit_ids[b, :n] = s["uids"]
            unit_ids_pad_mask[b, :n] = 1
        return BinnedSpikesInput(
            spike_matrix=spike_matrix,
            pad_and_gap_mask=pad_mask,
            sampling_rate=1000.0,
            unit_ids=unit_ids,
            unit_ids_pad_mask=unit_ids_pad_mask,
            duration_sec=torch.full((B,), 0.05),
            session_id=tuple(f"s{b}" for b in range(B)),
            subject_id=("u",) * B,
            species=("sp",) * B,
            absolute_start=torch.zeros(B, dtype=torch.float64),
            absolute_end=torch.full((B,), 0.05, dtype=torch.float64),
        )

    sample_a = _make_sample(torch.arange(1000, 1050), fill_value=7)
    sample_b = _make_sample(torch.arange(2000, 2050), fill_value=99)

    # Two batches, A in different positions.
    batch_AB = _make_batch([sample_a, sample_b], pad_to_U=50)
    batch_BA = _make_batch([sample_b, sample_a], pad_to_U=50)

    out_AB = frozen._pad_neurons_if_needed(batch_AB)
    out_BA = frozen._pad_neurons_if_needed(batch_BA)

    # Sample A's full row must match across the two batches regardless of
    # its position-in-batch (filler seed depends only on uids).
    assert torch.equal(out_AB.spike_matrix[0], out_BA.spike_matrix[1]), (
        "Sample A's spike_matrix differs across batch positions; the filler seed leaked batch order."
    )
    assert torch.equal(out_AB.unit_ids[0], out_BA.unit_ids[1])
    assert torch.equal(out_AB.spike_matrix[1], out_BA.spike_matrix[0])
    assert torch.equal(out_AB.unit_ids[1], out_BA.unit_ids[0])


# ----------------------------------------------------------------------
# _ntt_tokenize must build per-sample unit_ids
# ----------------------------------------------------------------------


def test_ntt_tokenize_unit_ids_are_per_sample():
    """Two samples with disjoint unit sets must produce per-sample
    unique-uid lists, not a global cross-batch broadcast."""
    captured = {}

    class _CaptureTokenizer(nn.Module):
        token_dim = 64

        def forward(self, spike_input: SpikeTimesInput) -> TokenizerOutput:
            captured["spike_input"] = spike_input
            B = spike_input.spike_unit_index.shape[0]
            # Return an inert TokenizerOutput so the wrapper doesn't crash.
            return TokenizerOutput(
                tokens=torch.zeros(B, 1, self.token_dim),
                timestamps=torch.zeros(B, 1),
                unit_ids=torch.zeros(B, 1, dtype=torch.long),
                mask=torch.zeros(B, 1, dtype=torch.bool),
                combines_units=False,
                token_bin_size_s=None,
            )

    wrapper = V2TrainWrapper(
        cfg=_minimal_cfg(),
        model=_DummyPOYO(dim=64),
        modality_spec=_DummyModalitySpec(),
        tokenizer_mode="ntt",
        unit_embedding_mode="none",
        sentinels=False,
        ntt_tokenizer=_CaptureTokenizer(),
    )

    # Sample 0: units {1, 2, 3} at three timestamps; Sample 1: units {4, 5}.
    mi = {
        "input_token_type": torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ),
        "input_unit_index": torch.tensor(
            [
                [1, 2, 3, 1, 2],
                [4, 5, 4, 4, 5],
            ]
        ),
        "input_timestamps": torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.1, 0.2, 0.3, 0.4, 0.5],
            ]
        ),
        "input_mask": torch.ones(2, 5, dtype=torch.bool),
    }
    batch = {"model_inputs": mi}
    wrapper._ntt_tokenize(batch)

    spike_input = captured["spike_input"]
    uids = spike_input.unit_ids
    uid_mask = spike_input.unit_ids_pad_mask
    assert uids.shape[0] == 2

    # Sample 0's valid uids must be exactly {1, 2, 3}.
    sample0_valid = set(uids[0][uid_mask[0].bool()].tolist())
    assert sample0_valid == {1, 2, 3}, (
        f"Sample 0 unit_ids should be {{1,2,3}} (per-sample), got {sample0_valid}. "
        f"unit_ids must be per-sample, not the batch-global set {{1,2,3,4,5}}."
    )

    # Sample 1's valid uids must be exactly {4, 5}.
    sample1_valid = set(uids[1][uid_mask[1].bool()].tolist())
    assert sample1_valid == {4, 5}, f"Sample 1 unit_ids should be {{4,5}} (per-sample), got {sample1_valid}."
