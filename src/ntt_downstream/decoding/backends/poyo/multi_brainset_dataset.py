"""Multi-brainset center-out reaching dataset.

Composes per-brainset `brainsets.datasets.*` instances into a single
`NestedSpikingDataset` whose recording IDs follow the canonical
`{brainset}/{session}` convention used by the ntt selection YAMLs.

Consumes a torch_brain selection list (the resolved output of
`${ntt_split:...}`) and instantiates one child dataset per brainset
entry. Optional filters: restrict to a subset of brainsets, and drop
non-center-out sessions (CO-only).

Each brainset stamps its own `data.config["readout"]` via a per-class
`get_recording_hook`. The configs share `readout_id="cursor_velocity_2d"`
but differ in `normalize_std` and movement-phase weights — Perich uses
the CO_READOUT_CONFIG for `_center_out_reaching` sessions and
RT_READOUT_CONFIG for `_random_target_reaching` sessions (selected per
session at hook time); Churchland and Odoherty use brainset-specific
configs mirrored from `torch_brain/examples/poyo/datasets/poyo_1.py`.

The Perich CO/RT configs are imported from
`ntt_downstream.decoding.backends.poyo.datasets` so the single-session and
multi-session paths share a single source of truth.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import deepcopy
from typing import Any

import torchmetrics
from brainsets.datasets import (
    ChurchlandShenoyNeural2012,
    OdohertySabesNonhuman2017,
    PerichMillerPopulation2018,
)
from temporaldata import Data
from torch_brain.dataset import NestedSpikingDataset

from ntt_downstream.decoding.backends.poyo.datasets import (
    CO_READOUT_CONFIG,
    RT_READOUT_CONFIG,
)
from ntt_downstream.decoding.backends.poyo.temporal_split import (
    TemporalSplitConfig,
    get_temporal_sampling_intervals,
)
from ntt_downstream.decoding.backends.poyo.temporal_split import (
    applies_to as _ts_applies_to,
)
from ntt_downstream.decoding.backends.poyo.train_decimation import (
    TrainDecimationConfig,
    apply_train_decimation,
)
from ntt_downstream.decoding.backends.poyo.train_decimation import (
    applies_to as _decim_applies_to,
)

CHURCHLAND_READOUT_CONFIG: dict[str, Any] = {
    "readout": {
        "readout_id": "cursor_velocity_2d",
        "normalize_mean": 0.0,
        "normalize_std": 800.0,
        "weights": {
            "movement_phases.hold_period": 0.1,
            "movement_phases.reach_period": 5.0,
            "movement_phases.return_period": 1.0,
            "cursor_outlier_segments": 1.0,
        },
        "metrics": [{"metric": torchmetrics.R2Score()}],
    }
}

ODOHERTY_READOUT_CONFIG: dict[str, Any] = {
    "readout": {
        "readout_id": "cursor_velocity_2d",
        "normalize_mean": 0.0,
        "normalize_std": 200.0,
        "metrics": [{"metric": torchmetrics.R2Score()}],
    }
}


BRAINSET_REGISTRY: dict[str, type] = {
    "perich_miller_population_2018": PerichMillerPopulation2018,
    "odoherty_sabes_nonhuman_2017": OdohertySabesNonhuman2017,
    "churchland_shenoy_neural_2012": ChurchlandShenoyNeural2012,
}


def _perich_readout_selector(session_id: str) -> dict[str, Any]:
    if session_id.endswith("center_out_reaching"):
        return CO_READOUT_CONFIG
    if session_id.endswith("random_target_reaching"):
        return RT_READOUT_CONFIG
    raise ValueError(f"Unrecognized perich_miller session suffix for readout config: {session_id!r}")


# Selector signature: (session_id) -> readout config dict. Brainsets with a
# single config wrap their dict in a constant lambda; perich branches on the
# session-id suffix so CO and RT sessions get the right movement-phase weights.
READOUT_CONFIG_REGISTRY: dict[str, Callable[[str], dict[str, Any]]] = {
    "perich_miller_population_2018": _perich_readout_selector,
    "odoherty_sabes_nonhuman_2017": lambda _sid: ODOHERTY_READOUT_CONFIG,
    "churchland_shenoy_neural_2012": lambda _sid: CHURCHLAND_READOUT_CONFIG,
}


def _with_readout(
    base_cls: type,
    readout_selector: Callable[[str], dict[str, Any]],
    *,
    temporal_split: TemporalSplitConfig | None = None,
    train_decimation: TrainDecimationConfig | None = None,
    train_window_length: float | None = None,
) -> type:
    """Return a subclass of `base_cls` whose `get_recording_hook` stamps the
    readout config returned by `readout_selector(data.session.id)`.

    The hook must live on the per-brainset class because `NestedDataset.__getitem__`
    dispatches straight to the child dataset's `__getitem__` and never invokes a
    wrapper-level hook.

    When `temporal_split` is provided, `get_sampling_intervals(split=...)` is
    overridden to return time-aware trial splits computed from the loaded
    `Data.trials` object (per-recording, cached). The HDF5 files are not
    modified; the baked-in random splits are simply ignored.

    When `train_decimation` is provided, the train sampling intervals
    (whether from `temporal_split` or the HDF5-baked default) are further
    subsampled to the time spans of N seeded-random valid trials.
    """

    class _Stamped(base_cls):  # type: ignore[misc, valid-type]
        _READOUT_SELECTOR = staticmethod(readout_selector)
        _TEMPORAL_SPLIT = temporal_split
        _TRAIN_DECIMATION = train_decimation
        _TRAIN_WINDOW_LENGTH = train_window_length

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._temporal_split_cache: dict[str, dict[str, Any]] = {}
            self._train_decimation_cache: dict[str, Any] = {}

        def get_recording_hook(self, data: Data) -> None:
            data.config = deepcopy(self._READOUT_SELECTOR(data.session.id))  # pyright: ignore[reportAttributeAccessIssue]  # temporaldata attaches fields dynamically
            super().get_recording_hook(data)

        def get_sampling_intervals(self, split=None, *args, **kwargs):
            cfg = self._TEMPORAL_SPLIT
            # split=None means "full session domain" — let the base class
            # handle it (used by tier-2/3 cross-session evals).
            if cfg is None or split not in ("train", "valid", "test"):
                baseline = super().get_sampling_intervals(split, *args, **kwargs)
            else:
                baseline = get_temporal_sampling_intervals(self, split, cfg, self._temporal_split_cache)
            if self._TRAIN_DECIMATION is not None and split == "train" and isinstance(baseline, dict):
                return apply_train_decimation(
                    baseline,
                    self,
                    self._TRAIN_DECIMATION,
                    self._train_decimation_cache,
                    min_duration=float(self._TRAIN_WINDOW_LENGTH or 0.0),
                )
            return baseline

    _Stamped.__name__ = f"_Readout_{base_cls.__name__}"
    _Stamped.__qualname__ = _Stamped.__name__
    return _Stamped


def _filter_co_only(brainset: str, sessions: list[str]) -> list[str]:
    """Drop sessions known to be non-CO. Currently strips `_random_target_reaching`."""

    if brainset == "perich_miller_population_2018":
        return [s for s in sessions if not s.endswith("_random_target_reaching")]
    return list(sessions)


def build_multi_brainset_co_dataset(
    root: str,
    selection: Iterable[dict[str, Any]],
    transform: Callable | None = None,
    restrict_brainsets: list[str] | None = None,
    co_only: bool = False,
    temporal_split: TemporalSplitConfig | None = None,
    train_decimation: TrainDecimationConfig | None = None,
    train_window_length: float | None = None,
) -> NestedSpikingDataset:
    """Compose a `NestedSpikingDataset` from a torch_brain selection list.

    Args:
        root: Path to the processed-brainsets directory (parent of
            `<brainset>/<session>.h5`).
        selection: Resolved output of `${ntt_split:...}` — a list with at
            least one element shaped `{"selection": [{"brainset", "sessions"}, ...]}`.
        transform: Optional sample-level transform (e.g. POYO's
            `model.tokenize`).
        restrict_brainsets: When set, drop any selection entries whose
            brainset is not in this list.
        co_only: When True, drop non-center-out sessions per
            `_filter_co_only`.
        temporal_split: When set, override the HDF5-baked random trial splits
            with time-ordered trial splits for any brainset listed in
            `temporal_split.brainsets`. Other brainsets keep their default
            `get_sampling_intervals` behaviour.
        train_decimation: When set, decimate the train sampling intervals
            to N seeded-random trials (independent of `temporal_split`).
            Restricted to brainsets in `train_decimation.brainsets` when
            that allowlist is non-None; unrestricted (every child dataset)
            when it's None. Valid/test passes through unchanged.

    Returns:
        Wrapper dataset whose `recording_ids` are `{brainset}/{session}`.
    """

    allow = set(restrict_brainsets) if restrict_brainsets else None
    sessions_by_brainset: dict[str, list[str]] = {}
    for entry in selection:
        for sub in entry.get("selection", []):
            brainset = str(sub["brainset"])
            if allow is not None and brainset not in allow:
                continue
            if brainset not in BRAINSET_REGISTRY:
                raise ValueError(f"Unknown brainset {brainset!r}; add it to BRAINSET_REGISTRY.")
            sessions = [str(s) for s in (sub.get("sessions") or [])]
            if co_only:
                sessions = _filter_co_only(brainset, sessions)
            if not sessions:
                continue
            sessions_by_brainset.setdefault(brainset, []).extend(sessions)

    if not sessions_by_brainset:
        raise ValueError(
            f"Selection produced no sessions after filtering "
            f"(restrict_brainsets={restrict_brainsets}, co_only={co_only})."
        )

    children: dict[str, Any] = {}
    for brainset, sessions in sessions_by_brainset.items():
        ts_for_child = temporal_split if _ts_applies_to(brainset, temporal_split) else None
        # Mirror the per-brainset allowlist gating used for temporal_split.
        # Without this gate, multi-brainset runs that set
        # `train_decimation.trials=N` silently decimate every brainset (e.g.
        # Churchland / Odoherty) even when the calibration script is
        # Perich-only, which would change the eval baselines of every
        # multi-brainset calibration run.
        decim_for_child = train_decimation if _decim_applies_to(brainset, train_decimation) else None
        cls = _with_readout(
            BRAINSET_REGISTRY[brainset],
            READOUT_CONFIG_REGISTRY[brainset],
            temporal_split=ts_for_child,
            train_decimation=decim_for_child,
            train_window_length=train_window_length,
        )
        children[brainset] = cls(
            root=root,
            recording_ids=sorted(set(sessions)),
        )

    return NestedSpikingDataset(datasets=children, transform=transform)
