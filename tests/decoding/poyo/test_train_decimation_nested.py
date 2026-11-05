"""Lock in the nesting + clamping invariants for train_decimation.

Subsets for different `cfg.trials` budgets must be nested (n=16 is a subset of
n=32), otherwise calibration curves are non-monotonic on the data axis (going
N → 2N could *drop* trials). A requested count above the eligible-pool size on
small sessions is clamped, and the gap is recorded so a plot can pick it up.

These tests freeze:

  1. `decimate_train_interval` returns nested subsets across `cfg.trials`
     for a fixed seed (trials@N ⊂ trials@2N ⊂ ... ⊂ trials@N_eligible).
  2. Different seeds give different (but each-self-nested) orderings.
  3. `summarize_decimation` reports `clamped=True` when the eligibility
     filter dropped requested → kept, with the right per-rid counts.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from temporaldata import Interval

from ntt_downstream.decoding.backends.poyo.train_decimation import (
    TrainDecimationConfig,
    decimate_train_interval,
    summarize_decimation,
)


def _make_data(starts: np.ndarray, ends: np.ndarray, is_valid: np.ndarray) -> Any:
    """Minimal stand-in for `temporaldata.Data` used by decimate_train_interval.

    Returns `Any` so pyright doesn't complain at the seven call sites that
    pass it where `data: Data` is annotated. The function only touches
    `data.trials.{start, end, is_valid}` — SimpleNamespace satisfies the
    runtime duck-typed contract without pulling in the full
    Data/IrregularTimeSeries machinery.
    """
    return SimpleNamespace(
        trials=SimpleNamespace(
            start=starts.astype(np.float64),
            end=ends.astype(np.float64),
            is_valid=is_valid.astype(bool),
        )
    )


def _trial_set(interval: Interval) -> set[tuple[float, float]]:
    """A trial is identified by its (start, end) pair — order-independent."""
    starts = np.asarray(interval.start, dtype=np.float64)
    ends = np.asarray(interval.end, dtype=np.float64)
    return {(float(s), float(e)) for s, e in zip(starts, ends, strict=True)}


class TestNestedSubsets:
    """Subsets must be prefix-stable across budgets for a fixed seed."""

    def _setup(self, n: int = 100):
        # 100 back-to-back 1-second valid trials in a single baseline span.
        starts = np.arange(n, dtype=np.float64)
        ends = starts + 0.9
        is_valid = np.ones(n, dtype=bool)
        baseline = Interval(
            start=np.array([0.0]),
            end=np.array([float(n) + 1.0]),
        )
        return _make_data(starts, ends, is_valid), baseline

    @pytest.mark.parametrize("seed", [0, 42, 12345])
    def test_doubling_is_superset(self, seed):
        data, baseline = self._setup(n=64)
        small = decimate_train_interval(data, baseline, TrainDecimationConfig(trials=8, seed=seed))
        big = decimate_train_interval(data, baseline, TrainDecimationConfig(trials=16, seed=seed))
        assert _trial_set(small).issubset(_trial_set(big)), (
            f"seed={seed}: trials@8 must be a subset of trials@16, "
            f"got {len(_trial_set(small) - _trial_set(big))} dropped trials"
        )

    def test_full_chain_nested(self):
        # Lock in the nesting chain: 4 ⊂ 8 ⊂ 16 ⊂ 32 ⊂ 64.
        data, baseline = self._setup(n=80)
        budgets = [4, 8, 16, 32, 64]
        sets = [
            _trial_set(decimate_train_interval(data, baseline, TrainDecimationConfig(trials=n, seed=42)))
            for n in budgets
        ]
        for prev, nxt, prev_n, nxt_n in zip(sets[:-1], sets[1:], budgets[:-1], budgets[1:], strict=True):
            assert prev.issubset(nxt), f"trials@{prev_n} must be a subset of trials@{nxt_n}; dropped: {prev - nxt}"
            assert len(nxt - prev) == nxt_n - prev_n, (
                f"trials@{nxt_n} should add exactly {nxt_n - prev_n} new "
                f"trials over trials@{prev_n}, got {len(nxt - prev)}"
            )

    def test_different_seeds_disagree(self):
        # Sanity: nesting holds *within* a seed but not across seeds.
        data, baseline = self._setup(n=64)
        s_a = _trial_set(decimate_train_interval(data, baseline, TrainDecimationConfig(trials=16, seed=1)))
        s_b = _trial_set(decimate_train_interval(data, baseline, TrainDecimationConfig(trials=16, seed=2)))
        assert s_a != s_b, "different seeds should produce different subsets"


class TestClampingSummary:
    """summarize_decimation surfaces the requested-vs-kept gap."""

    def test_no_clamp(self):
        cfg = TrainDecimationConfig(trials=4, seed=0)
        cache = {
            "rid_a": Interval(
                start=np.array([0.0, 1.0, 2.0, 3.0]),
                end=np.array([0.5, 1.5, 2.5, 3.5]),
            ),
        }
        summary = summarize_decimation(cache, cfg)
        assert summary == {
            "requested_trials": 4,
            "kept_trials_total": 4,
            "kept_trials_by_recording": {"rid_a": 4},
            "seed": 0,
            "clamped": False,
        }

    def test_clamp_signaled(self):
        # Requested 64 but only 54 eligible — the failure mode of a small
        # session whose eligible pool is below the budget.
        cfg = TrainDecimationConfig(trials=64, seed=42)
        starts = np.arange(54, dtype=np.float64)
        cache = {
            "perich_miller_population_2018/t_20130819_center_out_reaching": Interval(
                start=starts,
                end=starts + 0.5,
            ),
        }
        summary = summarize_decimation(cache, cfg)
        assert summary["requested_trials"] == 64
        assert summary["kept_trials_total"] == 54
        assert summary["clamped"] is True

    def test_disabled_returns_empty(self):
        # `cfg=None` (decimation off) → no JSON block emitted.
        assert summarize_decimation({}, None) == {}
        assert summarize_decimation({"rid": Interval(start=np.array([0.0]), end=np.array([1.0]))}, None) == {}
