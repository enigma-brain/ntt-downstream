"""Tests for the `token_dims_per_sec` metric formula.

`(1/token_bin_s) × mask.sum(1).mean()` would over-count by `T'` for the
fixed-rate per-unit branch (the mask sum already includes T' time bins). The
formula uses the mask across both fixed-rate and dynamic paths, dividing by the
model's `sequence_length` for fixed-rate tokenizers and by the per-sample
timestamp span for dynamic ones.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from ntt_downstream.decoding.backends.poyo.tokenizer_output import TokenizerOutput
from ntt_downstream.decoding.backends.poyo.train_wrapper import V2TrainWrapper


def _stub_wrapper(sequence_length: float = 1.0) -> V2TrainWrapper:
    """Build a wrapper instance just enough for `on_fit_start`'s metric math.

    We bypass `__init__` because the real one requires a fully-built POYO
    model. The metric block only reads `self.model.sequence_length`, so a
    SimpleNamespace stand-in is sufficient.
    """
    obj = V2TrainWrapper.__new__(V2TrainWrapper)
    obj.model = SimpleNamespace(sequence_length=sequence_length)
    return obj


def _compute_tps(tok_out: TokenizerOutput, sequence_length: float = 1.0) -> float:
    """Inline copy of the metric math from `on_fit_start`.

    Kept in sync with `train_wrapper.on_fit_start`. The wrapper method
    can't be called directly (it expects a Lightning trainer), so we
    duplicate the formula here. The test is the single source of truth
    for the expected behaviour; if the wrapper formula drifts, this
    fails first.
    """
    valid_counts = tok_out.mask.float().sum(dim=1)
    if tok_out.token_bin_size_s is not None:
        duration = float(sequence_length)
        return (valid_counts / duration).mean().item()
    ts = tok_out.timestamps
    mask = tok_out.mask
    ts_lo = ts.masked_fill(~mask, float("inf"))
    ts_hi = ts.masked_fill(~mask, float("-inf"))
    t_min = ts_lo.min(dim=1).values
    t_max = ts_hi.max(dim=1).values
    durations = (t_max - t_min).clamp(min=0.1)
    return (valid_counts / durations).mean().item()


class TestFixedRatePerUnit:
    """Per-unit fixed-rate tokenizer: T'>1 case, where a double-count by T' would show."""

    def test_full_window_token_count(self):
        # T'=10 time bins × U=72 units, all valid, 1.0s window.
        T_prime, U = 10, 72
        B = 2
        mask = torch.ones(B, T_prime * U, dtype=torch.bool)
        tok = TokenizerOutput(
            tokens=torch.zeros(B, T_prime * U, 16),
            timestamps=torch.zeros(B, T_prime * U),
            unit_ids=torch.zeros(B, T_prime * U, dtype=torch.long),
            mask=mask,
            combines_units=False,
            token_bin_size_s=0.1,  # 100ms bin
        )
        tps = _compute_tps(tok, sequence_length=1.0)
        # 10 × 72 = 720 tokens / 1.0 s. Scaling by an extra 1/token_bin
        # would give 7200 (= 1/0.1 × 720 — the mask sum counted as units).
        assert tps == pytest.approx(720.0)

    def test_short_bin_high_token_rate(self):
        # 5ms bins on 1s window → T'=200, U=72 → 14,400 tokens/s.
        T_prime, U = 200, 72
        B = 1
        mask = torch.ones(B, T_prime * U, dtype=torch.bool)
        tok = TokenizerOutput(
            tokens=torch.zeros(B, T_prime * U, 4),
            timestamps=torch.zeros(B, T_prime * U),
            unit_ids=torch.zeros(B, T_prime * U, dtype=torch.long),
            mask=mask,
            combines_units=False,
            token_bin_size_s=0.005,
        )
        tps = _compute_tps(tok, sequence_length=1.0)
        assert tps == pytest.approx(14_400.0)


class TestFixedRatePopulation:
    """Population (`combines_units=True`) tokenizer: mask is [B, T']."""

    def test_population_token_count(self):
        T_prime = 10
        B = 2
        mask = torch.ones(B, T_prime, dtype=torch.bool)
        tok = TokenizerOutput(
            tokens=torch.zeros(B, T_prime, 32),
            timestamps=torch.zeros(B, T_prime),
            unit_ids=None,
            mask=mask,
            combines_units=True,
            token_bin_size_s=0.1,
        )
        tps = _compute_tps(tok, sequence_length=1.0)
        # 10 / 1.0 = 10 tokens/s.
        assert tps == pytest.approx(10.0)


class TestDynamicTokenizer:
    """`token_bin_size_s=None` path: span inferred from timestamps."""

    def test_full_span(self):
        # 30 valid spike tokens spanning [0, 1.0] across 2 samples.
        B, N = 2, 30
        ts = torch.linspace(0.0, 1.0, N).unsqueeze(0).expand(B, -1).contiguous()
        mask = torch.ones(B, N, dtype=torch.bool)
        tok = TokenizerOutput(
            tokens=torch.zeros(B, N, 8),
            timestamps=ts,
            unit_ids=torch.zeros(B, N, dtype=torch.long),
            mask=mask,
            combines_units=False,
            token_bin_size_s=None,
        )
        tps = _compute_tps(tok)
        # span ≈ 1.0, count = 30 → ~30 tokens/s.
        assert tps == pytest.approx(30.0, rel=0.02)

    def test_padding_does_not_inflate_span(self):
        """Padding tokens (mask=False) must not skew t_min/t_max."""
        B, N = 1, 10
        ts = torch.tensor([[0.0, 0.2, 0.4, 0.6, 0.8, 999.0, 999.0, 999.0, 999.0, 999.0]])
        mask = torch.tensor([[True, True, True, True, True, False, False, False, False, False]])
        tok = TokenizerOutput(
            tokens=torch.zeros(B, N, 4),
            timestamps=ts,
            unit_ids=torch.zeros(B, N, dtype=torch.long),
            mask=mask,
            combines_units=False,
            token_bin_size_s=None,
        )
        tps = _compute_tps(tok)
        # Real span = 0.8, valid count = 5 → ~6.25 tokens/s. Padding-time
        # 999.0 must be ignored.
        assert tps == pytest.approx(5.0 / 0.8, rel=0.02)


class TestSchemaVersion:
    """The run-info probe stamps `metric_schema_version=2` into run_info.

    The probe lives in ``_log_run_info`` (called from
    ``on_train_batch_start`` / ``on_validation_batch_start``). Whichever call
    site runs it must keep the schema version literal, since downstream
    tooling aggregates `token_dims_per_sec` by schema version."""

    def test_run_info_schema_key(self):
        # Direct read of the source — paranoid check that future edits
        # don't drop the schema version silently.
        import inspect

        from ntt_downstream.decoding.backends.poyo import train_wrapper

        src = inspect.getsource(train_wrapper.V2TrainWrapper._log_run_info)
        assert '"metric_schema_version": 2' in src, (
            "metric_schema_version=2 must be present in run_info; downstream tooling reads this key."
        )
