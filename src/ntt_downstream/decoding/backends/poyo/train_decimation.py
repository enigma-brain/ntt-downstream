"""Seeded random decimation of the train split to a fixed trial count.

Composes on top of whichever split source the dataset already exposes
(HDF5-baked random splits, or the runtime `temporal_split` override): when
`train_decimation.trials=N` is set, the train sampling intervals returned
by `get_sampling_intervals("train")` are replaced by the time spans of `N`
seeded-random valid trials drawn from the baseline train domain. Valid /
test domains pass through unchanged.

Used by the calibration sweeps to vary the train data budget (e.g. compare
`n_trials ∈ {4, 8, 16, …}`) while keeping the baked random valid/test
splits intact. Works regardless of whether `temporal_split.enabled` is
true or false — the decimation is applied on top of whichever train
intervals the dataset returns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from temporaldata import Data, Interval

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainDecimationConfig:
    trials: int
    seed: int = 42
    # Optional brainset allowlist; mirrors `temporal_split.brainsets`.
    # `None` = no allowlist, decimation applies to every brainset (the
    # single-session default — there's only one brainset, no list to
    # write down). When set, only the listed brainsets get decimated;
    # all others pass through with their full baseline train domain.
    brainsets: frozenset[str] | None = None

    @staticmethod
    def from_cfg(node) -> TrainDecimationConfig | None:
        """Build from an OmegaConf node, or return None when disabled/missing.

        Disabled means `node is None` or `node.trials is None`. Negative
        trial counts raise; zero is allowed (yields an empty train domain,
        intended for the eval-only leg of the calibration sweep).
        """
        if node is None:
            return None
        n_raw = getattr(node, "trials", None)
        if n_raw is None:
            return None
        n = int(n_raw)
        if n < 0:
            raise ValueError(f"train_decimation.trials must be ≥0; got {n}")
        seed = int(getattr(node, "seed", 42))
        brainsets_raw = getattr(node, "brainsets", None)
        # OmegaConf renders an unset list as `None` and an empty list as
        # `[]`; treat both as "no allowlist" so users can clear the list
        # without dropping the key entirely.
        if brainsets_raw:
            brainsets: frozenset[str] | None = frozenset(str(b) for b in brainsets_raw)
        else:
            brainsets = None
        return TrainDecimationConfig(trials=n, seed=seed, brainsets=brainsets)


def applies_to(brainset: str, cfg: TrainDecimationConfig | None) -> bool:
    """Whether `cfg` should decimate the given brainset.

    Mirrors `temporal_split.applies_to`. `cfg.brainsets is None` means
    "no allowlist" → decimate every brainset. A non-empty allowlist
    restricts decimation to its members only.
    """
    if cfg is None:
        return False
    if cfg.brainsets is None:
        return True
    return brainset in cfg.brainsets


def decimate_train_interval(
    data: Data,
    baseline_train: Interval,
    cfg: TrainDecimationConfig,
    min_duration: float = 0.0,
) -> Interval:
    """Return an Interval covering `cfg.trials` seeded-random valid trials
    drawn from those whose time span lies fully inside `baseline_train`.

    The selection is by trial *index* (after filtering for `is_valid=True`),
    not by time, so the same `cfg.seed` picks the same trial set regardless
    of the surrounding split layout. The returned Interval is sorted by
    start time so temporaldata's set-ops (`|`, `difference`) keep working.

    Subsets are *nested across budgets* for a fixed `cfg.seed`: the trials
    chosen for `cfg.trials = N` are a subset of those chosen for any
    `cfg.trials = M ≥ N`. This is what makes the calibration curve
    interpretable — going from N to 2N adds N trials rather than swapping
    the random sample. Implemented by drawing one full seeded permutation
    of the eligible pool and slicing the first `n_pick` entries; using
    `rng.choice(..., replace=False)` would *not* give nesting because
    independent draws of different sizes yield statistically independent
    subsets even with the same seed.

    A trial is eligible iff a single baseline interval covers both its
    start and its end — `searchsorted` gives the rightmost interval whose
    start ≤ trial.start, then we verify the same interval also reaches
    trial.end. Trials that the baked split has already excluded from train
    (i.e. valid/test trials) live inside a different baseline interval (or
    in a gap) and fall through this check.

    `min_duration` (seconds, default 0.0) further restricts eligibility to
    trials whose own span is ≥ that threshold. Pass the trainer's
    window_length here so decimation runs *after* the same short-interval
    filter the sampler applies, avoiding the failure mode where N picks
    contain trials too short to yield any sampler windows.

    If `cfg.trials` exceeds the eligible count, all eligible trials are
    kept and a warning is emitted; the caller is expected to clamp upstream
    (the calibration scripts already filter on `floor(0.7 × n_session)`).
    The actual kept count is `len(returned_interval)` and is also recorded
    by `apply_train_decimation` for surfacing to `run_result.json`, so
    plotters can distinguish requested-vs-kept on clamped budgets.
    """
    trials = data.trials  # pyright: ignore[reportAttributeAccessIssue]  # temporaldata attaches fields dynamically
    is_valid = np.asarray(trials.is_valid).astype(bool)
    starts = np.asarray(trials.start, dtype=np.float64)[is_valid]
    ends = np.asarray(trials.end, dtype=np.float64)[is_valid]
    n_total = int(starts.shape[0])

    empty = Interval(
        start=np.zeros(0, dtype=np.float64),
        end=np.zeros(0, dtype=np.float64),
    )
    if n_total == 0 or cfg.trials == 0:
        return empty

    # Interval stub omits start/end.
    bs = np.asarray(baseline_train.start, dtype=np.float64)  # pyright: ignore[reportAttributeAccessIssue]
    be = np.asarray(baseline_train.end, dtype=np.float64)  # pyright: ignore[reportAttributeAccessIssue]
    if bs.shape[0] == 0:
        return empty

    # Defensive sort — temporaldata Intervals are conventionally monotonic
    # but we don't want a wrong searchsorted result if a parent ever returns
    # an out-of-order Interval.
    if bs.shape[0] > 1 and not np.all(bs[:-1] <= bs[1:]):
        order = np.argsort(bs, kind="stable")
        bs = bs[order]
        be = be[order]

    idx = np.searchsorted(bs, starts, side="right") - 1
    idx = np.clip(idx, 0, bs.shape[0] - 1)
    eligible = (bs[idx] <= starts) & (be[idx] >= ends)
    if min_duration > 0.0:
        eligible &= (ends - starts) >= min_duration
    eligible_idx = np.where(eligible)[0]
    n_eligible = int(eligible_idx.shape[0])
    if n_eligible == 0:
        return empty

    n_pick = min(cfg.trials, n_eligible)
    if n_pick < cfg.trials:
        logger.warning(
            "train_decimation: requested trials=%d but only %d eligible "
            "(baseline train has %d valid trials fully contained); "
            "keeping all eligible trials",
            cfg.trials,
            n_eligible,
            n_eligible,
        )

    # One seeded permutation of the eligible pool; take the first n_pick.
    # This is what makes subsets *nested across budgets* — for a fixed seed,
    # the trials chosen at n_pick=N are a prefix of those at any 2N. Using
    # rng.choice(..., replace=False) here would silently produce different
    # (non-nested) subsets per size, breaking calibration-curve monotonicity.
    rng = np.random.default_rng(cfg.seed)
    permuted = rng.permutation(eligible_idx)
    chosen = np.sort(permuted[:n_pick])
    return Interval(start=starts[chosen], end=ends[chosen])


def apply_train_decimation(
    intervals_by_rid: dict[str, Interval],
    dataset,
    cfg: TrainDecimationConfig,
    cache: dict[str, Interval],
    min_duration: float = 0.0,
) -> dict[str, Interval]:
    """Decimate the train split per recording, caching the result.

    `intervals_by_rid` is the baseline train dict (output of the parent
    `get_sampling_intervals("train")`). For each recording we load the
    `temporaldata.Data` once via `dataset.get_recording(rid)`, call
    `decimate_train_interval`, and cache the resulting Interval keyed by
    rid. Subsequent calls with the same `cfg` reuse the cached entry.

    `min_duration` is forwarded to `decimate_train_interval` so trials
    shorter than the trainer's window_length are excluded from the
    eligible pool *before* the seeded random pick (avoids picking N
    trials that the sampler would silently drop).
    """
    out: dict[str, Interval] = {}
    for rid, baseline in intervals_by_rid.items():
        cached = cache.get(rid)
        if cached is None:
            data = dataset.get_recording(rid)
            cached = decimate_train_interval(data, baseline, cfg, min_duration)
            cache[rid] = cached
            kept = len(cached)
            # Surface the requested-vs-kept gap loudly when it occurs:
            # the eligibility filter (trial fully inside baseline_train ∧
            # duration ≥ min_duration) can clamp `requested` down to fewer
            # trials, which silently makes neighbouring calibration budgets
            # collapse onto the same data. The gap is also persisted via
            # `summarize_decimation` so `run_result.json` records it for
            # downstream plotters.
            if kept < cfg.trials:
                logger.warning(
                    f"train_decimation[{rid}] requested_trials={cfg.trials} "
                    f"clamped to kept_trials={kept} (eligible pool exhausted "
                    f"after baseline_train + min_duration={min_duration:.3f}s "
                    f"filter); calibration x-axis should plot kept_trials, "
                    f"not requested_trials, for this point."
                )
            logger.info(
                f"train_decimation[{rid}] seed={cfg.seed} "
                f"requested_trials={cfg.trials} "
                f"min_duration={min_duration:.3f}s → kept {kept} trial(s)"
            )
        out[rid] = cached
    return out


def collect_decimation_summary(dataset: Any) -> dict[str, Any]:
    """Walk a train dataset and aggregate decimation summaries across children.

    Handles both flows:
      - Single-session: a `MultiSessionCODataset` that exposes
        `train_decimation_summary()` directly.
      - Multi-brainset: a `NestedSpikingDataset` whose `.datasets` mapping
        holds per-brainset child instances; each child's
        `_train_decimation_cache` and `_TRAIN_DECIMATION` are merged into a
        single summary so the run-level JSON has one block, not N.

    Returns ``{}`` when no decimation ran in any child.
    """
    if hasattr(dataset, "train_decimation_summary"):
        return dataset.train_decimation_summary()
    children = getattr(dataset, "datasets", None)
    if not children:
        return {}
    children_iter = children.values() if isinstance(children, dict) else children
    by_rid: dict[str, int] = {}
    cfg: TrainDecimationConfig | None = None
    for child in children_iter:
        child_cache = getattr(child, "_train_decimation_cache", None)
        child_cfg = getattr(child, "_TRAIN_DECIMATION", None)
        if not child_cache or child_cfg is None:
            continue
        cfg = child_cfg
        for rid, interval in child_cache.items():
            by_rid[rid] = int(len(interval))
    if cfg is None or not by_rid:
        return {}
    requested = int(cfg.trials)
    return {
        "requested_trials": requested,
        "kept_trials_total": int(sum(by_rid.values())),
        "kept_trials_by_recording": by_rid,
        "seed": int(cfg.seed),
        "clamped": any(v < requested for v in by_rid.values()),
    }


def summarize_decimation(
    cache: dict[str, Interval],
    cfg: TrainDecimationConfig | None,
) -> dict[str, Any]:
    """Build a JSON-serializable summary of decimation outcomes from a cache.

    Returns a dict shaped for direct embedding in `run_result.json`::

        {
          "requested_trials": 64,
          "kept_trials_total": 54,
          "kept_trials_by_recording": {"<rid>": 54, ...},
          "seed": 42,
          "clamped": true,            # any rid where kept < requested
        }

    Returns ``{}`` when `cfg is None` or `cache` is empty (i.e. decimation
    didn't run or nothing materialised yet — the calibration sweeps always
    materialise via the sampler before this is called, so an empty result
    is itself a signal). Plotters should plot ``kept_trials`` on the
    calibration x-axis (or de-dupe rows on it) when ``clamped`` is true,
    otherwise neighbouring requested-budgets collapse onto the same data.
    """
    if cfg is None or not cache:
        return {}
    by_rid = {rid: int(len(interval)) for rid, interval in cache.items()}
    kept_total = int(sum(by_rid.values()))
    requested = int(cfg.trials)
    return {
        "requested_trials": requested,
        "kept_trials_total": kept_total,
        "kept_trials_by_recording": by_rid,
        "seed": int(cfg.seed),
        "clamped": any(v < requested for v in by_rid.values()),
    }
