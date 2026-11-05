"""Perich-Miller Monkey-C datasets (CO + RT) for downstream decoding."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torchmetrics
from brainsets.datasets import PerichMillerPopulation2018
from temporaldata import Data

from ntt_downstream.decoding.backends.poyo.temporal_split import (
    TemporalSplitConfig,
    get_temporal_sampling_intervals,
)
from ntt_downstream.decoding.backends.poyo.train_decimation import (
    TrainDecimationConfig,
    apply_train_decimation,
    summarize_decimation,
)

# Per-recording readout configs, mirroring upstream
# `torch_brain.examples.poyo.datasets.poyo_mp.PoyoMPDataset`:
# CO sessions (suffix `_center_out_reaching`) get phase weights tuned for
# reach-period evaluation; RT sessions (suffix `_random_target_reaching`)
# get a flatter weighting evaluated on the random-period interval.
CO_READOUT_CONFIG = {
    "readout": {
        "readout_id": "cursor_velocity_2d",
        "normalize_mean": 0.0,
        "normalize_std": 20.0,
        "weights": {
            "movement_phases.random_period": 1.0,
            "movement_phases.hold_period": 0.1,
            "movement_phases.reach_period": 5.0,
            "movement_phases.return_period": 1.0,
            "movement_phases.invalid": 0.1,
            "cursor_outlier_segments": 0.0,
        },
        "metrics": [{"metric": torchmetrics.R2Score()}],
        "eval_interval": "movement_phases.reach_period",
    }
}


RT_READOUT_CONFIG = {
    "readout": {
        "readout_id": "cursor_velocity_2d",
        "normalize_mean": 0.0,
        "normalize_std": 20.0,
        "weights": {
            "movement_phases.random_period": 1.0,
            "movement_phases.hold_period": 0.1,
            "cursor_outlier_segments": 0.0,
        },
        "metrics": [{"metric": torchmetrics.R2Score()}],
        "eval_interval": "movement_phases.random_period",
    }
}


_SUPPORTED_SUFFIXES = ("_center_out_reaching", "_random_target_reaching")


def _readout_config_for(recording_id: str) -> dict:
    """Return a deepcopy of the readout config for ``recording_id``.

    Dispatch by recording-id suffix, matching upstream POYO's
    ``PoyoMPDataset.get_recording_hook`` semantics. Raises ``ValueError`` if
    the suffix is not one of the supported Perich-Miller task suffixes —
    failing loudly so unsupported sessions don't silently fall through with
    a default config.
    """
    if recording_id.endswith("_center_out_reaching"):
        return deepcopy(CO_READOUT_CONFIG)
    if recording_id.endswith("_random_target_reaching"):
        return deepcopy(RT_READOUT_CONFIG)
    raise ValueError(f"Unsupported recording id suffix for {recording_id!r}; expected one of {_SUPPORTED_SUFFIXES}.")


class SingleSessionCODataset(PerichMillerPopulation2018):
    """Single Monkey-C session (CO or RT).

    Despite "CO" in the name, this class accepts both
    center-out (``_center_out_reaching``) and random-target
    (``_random_target_reaching``) recordings; the appropriate readout
    config is attached per-sample by ``get_recording_hook``. The name
    is kept because ``ntt_downstream.decoding.train`` imports it.
    """

    def __init__(
        self,
        root,
        recording_id: str,
        transform=None,
        temporal_split: TemporalSplitConfig | None = None,
        train_decimation: TrainDecimationConfig | None = None,
        train_window_length: float | None = None,
        **kwargs,
    ):
        if not recording_id.endswith(_SUPPORTED_SUFFIXES):
            raise ValueError(
                f"Unsupported recording id suffix for {recording_id!r}; expected one of {_SUPPORTED_SUFFIXES}."
            )
        super().__init__(
            root,
            recording_ids=[recording_id],
            transform=transform,
            **kwargs,
        )
        self._temporal_split_cfg = temporal_split
        self._temporal_split_cache: dict[str, dict[str, Any]] = {}
        self._train_decimation_cfg = train_decimation
        self._train_decimation_cache: dict[str, Any] = {}
        # Forwarded to apply_train_decimation as `min_duration` so
        # decimation runs *after* the sampler's short-interval filter.
        self._train_window_length = train_window_length

    def get_recording_hook(self, data: Data):
        data.config = _readout_config_for(data.session.id)  # pyright: ignore[reportAttributeAccessIssue]  # temporaldata attaches fields dynamically
        super().get_recording_hook(data)

    def get_sampling_intervals(self, split=None, *args, **kwargs):
        cfg = self._temporal_split_cfg
        if cfg is None or split not in ("train", "valid", "test"):
            baseline = super().get_sampling_intervals(split, *args, **kwargs)
        else:
            baseline = get_temporal_sampling_intervals(self, split, cfg, self._temporal_split_cache)
        # Decimation only applies to the train split — valid / test pass
        # through unchanged so the calibration sweeps keep evaluating on
        # the same held-out trials regardless of how many train trials
        # were kept.
        if self._train_decimation_cfg is not None and split == "train" and isinstance(baseline, dict):
            return apply_train_decimation(
                baseline,
                self,
                self._train_decimation_cfg,
                self._train_decimation_cache,
                min_duration=float(self._train_window_length or 0.0),
            )
        return baseline

    def train_decimation_summary(self) -> dict[str, Any]:
        """Per-rid (requested, kept) snapshot for `run_result.json`.

        Empty dict when decimation is disabled or hasn't materialised yet.
        Materialisation happens during sampler construction (which calls
        `get_sampling_intervals("train")`); `train.py` reads this after
        the train sampler is built so the cache is always populated.
        """
        return summarize_decimation(self._train_decimation_cache, self._train_decimation_cfg)


class MultiSessionCODataset(PerichMillerPopulation2018):
    """Multiple Monkey-C sessions (CO and/or RT).

    Despite "CO" in the name, this class accepts a mix of
    center-out (``_center_out_reaching``) and random-target
    (``_random_target_reaching``) recordings; per-session readout configs
    are routed by suffix in ``get_recording_hook``. The name is kept because
    ``ntt_downstream.decoding.train`` imports it.
    """

    def __init__(self, root, recording_ids: list[str], transform=None, **kwargs):
        bad = [rid for rid in recording_ids if not rid.endswith(_SUPPORTED_SUFFIXES)]
        if bad:
            raise ValueError(
                f"Unsupported recording id suffix(es): {bad}; expected each to end with one of {_SUPPORTED_SUFFIXES}."
            )
        super().__init__(
            root,
            recording_ids=recording_ids,
            transform=transform,
            **kwargs,
        )

    def get_recording_hook(self, data: Data):
        data.config = _readout_config_for(data.session.id)  # pyright: ignore[reportAttributeAccessIssue]  # temporaldata attaches fields dynamically
        super().get_recording_hook(data)
