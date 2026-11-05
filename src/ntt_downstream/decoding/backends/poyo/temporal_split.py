"""Runtime time-aware split override for trial-based brainsets.

The brainsets preprocessing pipelines bake `train_domain` / `valid_domain` /
`test_domain` into HDF5 by random per-trial assignment (Perich-Miller and
Churchland-Shenoy use `sklearn.train_test_split` with seed 42). This module
re-derives those domains at dataset-construction time from the loaded
`Data.trials` object, picking trials in temporal order instead, while
preserving trial structure (whole trials, never sliced). Train is anchored to
the start of the trial sequence and valid+test to the end; if the ratios sum
to <1.0 the middle is discarded (lets you shrink train while keeping the eval
trials fixed across runs).

The HDF5 files are not modified. The override is gated by a per-brainset
allowlist so e.g. Odoherty (already temporal-block based) can be left
untouched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from temporaldata import Data, Interval

logger = logging.getLogger(__name__)


_VALID_MODES = ("temporal", "random")


@dataclass(frozen=True)
class TemporalSplitConfig:
    ratios: tuple[float, float, float]  # (train, valid, test); sum ≤ 1.0
    brainsets: frozenset[str]  # allowlist of brainset names
    dilation: float = 1.0  # seconds; matches pipeline default
    mode: str = "temporal"  # "temporal" (chronological) or "random"
    seed: int = 42  # only used when mode == "random"

    @staticmethod
    def from_cfg(node) -> TemporalSplitConfig | None:
        """Build from an OmegaConf node, or return None when disabled/missing."""
        if node is None:
            return None
        if not bool(getattr(node, "enabled", False)):
            return None
        ratios = tuple(float(r) for r in node.ratios)
        if len(ratios) != 3:
            raise ValueError(f"temporal_split.ratios must have 3 entries (train, valid, test); got {ratios}")
        if any(r < 0 for r in ratios):
            raise ValueError(f"temporal_split.ratios must be non-negative; got {ratios}")
        s = sum(ratios)
        if not (0.0 < s <= 1.001):
            raise ValueError(f"temporal_split.ratios must sum to ≤1.0 (any leftover is discarded); got {s}")
        brainsets = frozenset(str(b) for b in (node.brainsets or []))
        if not brainsets:
            raise ValueError("temporal_split.enabled=true requires a non-empty brainsets allowlist")
        dilation = float(getattr(node, "dilation", 1.0))
        mode = str(getattr(node, "mode", "temporal"))
        if mode not in _VALID_MODES:
            raise ValueError(f"temporal_split.mode must be one of {_VALID_MODES}; got {mode!r}")
        seed = int(getattr(node, "seed", 42))
        return TemporalSplitConfig(
            ratios=ratios,
            brainsets=brainsets,
            dilation=dilation,
            mode=mode,
            seed=seed,
        )


def compute_temporal_domains(data: Data, cfg: TemporalSplitConfig) -> dict[str, Interval]:
    """Return `{train, valid, test}` Interval dict for one recording.

    Trials are placed in an order array and laid out as:
        [ train (n_train) | gap (discarded) | valid (n_valid) | test (n_test) ]
    Train is pinned to the start of the order array; valid+test are pinned
    to the end. When `cfg.ratios` sum to <1.0 the middle gap is discarded —
    valid+test stay anchored at the *end* of the order, so as `n_train`
    shrinks across a ratio sweep the same valid/test trials stay held out
    (only the gap grows). When ratios sum to ~1.0 the gap is empty and the
    layout reduces to a contiguous train→valid→test.

    Order array depends on `cfg.mode`:
      * "temporal": trials sorted ascending by start time (chronological;
        eval is the tail of the recording — leakage-safe).
      * "random": trials shuffled by `np.random.default_rng(cfg.seed)`.
        Eval trials are randomly scattered through the recording, so the
        train domain is carved by per-eval-trial dilation buffers; with a
        fixed `cfg.seed` the assignment is reproducible across runs and —
        because eval trials are anchored to the tail of the permutation —
        also stable across train-ratio sweeps.

    `train_domain` is the full session domain minus the dilated union of
    (gap ∪ valid ∪ test), matching the upstream pipeline's `cfg.dilation`-second
    buffer so train windows can't bleed into eval (or discarded) trials.
    """
    trials = data.trials  # pyright: ignore[reportAttributeAccessIssue]  # temporaldata attaches fields dynamically
    domain = data.domain
    # Every split, including the no-valid-trials fallback, returns an Interval.
    assert domain is not None, "recording has no domain"
    is_valid = np.asarray(trials.is_valid).astype(bool)
    starts_all = np.asarray(trials.start, dtype=np.float64)
    ends_all = np.asarray(trials.end, dtype=np.float64)
    starts = starts_all[is_valid]
    ends = ends_all[is_valid]

    n = int(starts.shape[0])
    empty = Interval(start=np.zeros(0, dtype=np.float64), end=np.zeros(0, dtype=np.float64))
    if n == 0:
        return {"train": domain, "valid": empty, "test": empty}

    if cfg.mode == "random":
        # Seeded permutation independent of trial start times. Eval trials
        # land at the *end* of `order` (same slicing as temporal mode), so
        # holding `cfg.seed`, `n_valid`, and `n_test` fixed keeps the eval
        # trial set identical across train-ratio sweeps.
        order = np.random.default_rng(cfg.seed).permutation(n)
    else:
        order = np.argsort(starts, kind="stable")

    # Independent rounding per split, then clamp so train+valid+test ≤ n.
    # Order of clamping (train, then test, then valid) preserves train and the
    # tail-anchored test in the small-n edge case.
    #
    # NB: clamping `n_valid` against `n - n_train - n_test` (not against
    # `n - n_train` alone) is what guarantees the slices `[n_train+n_gap :
    # n_train+n_gap+n_valid]` and `[n - n_test : n]` are disjoint after
    # rounding. Adversarial ratios like (0.7, 0.15, 0.15) on n=10 round up
    # `n_test` to 2 and would otherwise leave only 1 slot for valid; the
    # clamp shrinks valid to fit, never test.
    n_train = max(0, min(int(round(cfg.ratios[0] * n)), n))
    n_test = max(0, min(int(round(cfg.ratios[2] * n)), n - n_train))
    n_valid = max(0, min(int(round(cfg.ratios[1] * n)), n - n_train - n_test))
    n_gap = n - n_train - n_valid - n_test
    # Defensive: layout invariants the slicing below assumes.
    assert n_gap >= 0, f"n_gap={n_gap} < 0 (n={n}, ratios={cfg.ratios})"
    assert n_train + n_gap + n_valid + n_test == n, (
        f"split sizes don't sum to n: {n_train}+{n_gap}+{n_valid}+{n_test} != {n}"
    )

    def _slice(lo: int, hi: int) -> Interval:
        # `order[lo:hi]` selects which trials land in this split. Re-sort by
        # trial start so the resulting Interval is monotonic — required by
        # temporaldata's set-ops (`|`, `difference`). In temporal mode `order`
        # is already start-sorted so this is a no-op.
        if hi <= lo:
            return Interval(
                start=np.zeros(0, dtype=np.float64),
                end=np.zeros(0, dtype=np.float64),
            )
        idx = order[lo:hi]
        idx = idx[np.argsort(starts[idx], kind="stable")]
        return Interval(start=starts[idx], end=ends[idx])

    gap_iv = _slice(n_train, n_train + n_gap)
    valid_iv = _slice(n_train + n_gap, n_train + n_gap + n_valid)
    test_iv = _slice(n - n_test, n)

    excluded_union = gap_iv | valid_iv | test_iv
    if len(excluded_union) > 0:
        excluded_union = excluded_union.dilate(cfg.dilation)
        train_iv = domain.difference(excluded_union)
    else:
        train_iv = domain

    return {"train": train_iv, "valid": valid_iv, "test": test_iv}


def log_split_summary(
    recording_id: str,
    domains: dict[str, Interval],
    n_trials: int,
    cfg: TemporalSplitConfig | None = None,
) -> None:
    """Emit one INFO line per recording with split sizes and time bounds."""

    def _bounds(iv: Interval) -> str:
        if len(iv) == 0:
            return "empty"
        return f"[{float(np.min(iv.start)):.2f}s, {float(np.max(iv.end)):.2f}s]"  # pyright: ignore[reportAttributeAccessIssue]  # Interval stub omits start/end

    train = domains["train"]
    valid = domains["valid"]
    test = domains["test"]
    mode_tag = ""
    if cfg is not None:
        mode_tag = f" mode={cfg.mode}"
        if cfg.mode == "random":
            mode_tag += f" seed={cfg.seed}"
    logger.info(
        f"temporal_split[{recording_id}]{mode_tag} n_trials={n_trials} "
        f"train(n={len(train)}, {_bounds(train)}) "
        f"valid(n={len(valid)}, {_bounds(valid)}) "
        f"test(n={len(test)}, {_bounds(test)})"
    )


def applies_to(brainset: str, cfg: TemporalSplitConfig | None) -> bool:
    return cfg is not None and brainset in cfg.brainsets


def get_temporal_sampling_intervals(
    dataset,
    split: str,
    cfg: TemporalSplitConfig,
    cache: dict[str, dict[str, Interval]],
) -> dict[str, Interval]:
    """Compute temporal-split sampling intervals for `dataset.recording_ids`.

    Caches the per-recording `{train, valid, test}` Interval dict on first
    access (logs a one-line summary) and returns the requested split for
    every recording.
    """
    out: dict[str, Interval] = {}
    for rid in dataset.recording_ids:
        if rid not in cache:
            data = dataset.get_recording(rid)
            domains = compute_temporal_domains(data, cfg)
            n_trials = int(np.asarray(data.trials.is_valid).astype(bool).sum())
            cache[rid] = domains
            log_split_summary(rid, domains, n_trials, cfg)
        out[rid] = cache[rid][split]
    return out
