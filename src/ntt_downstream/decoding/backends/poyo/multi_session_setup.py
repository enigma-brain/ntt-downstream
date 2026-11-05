"""Multi-session + tiered-eval setup helpers.

Encapsulates the build-time wiring for a POYO training run against an
ntt-style selection list: the train dataset, per-tier eval datasets,
samplers, dataloaders, and a tier-aware metric aggregator that turns
per-session R²s logged by `DecodingStitchEvaluator` into per-tier
averages.

Partly adapted from the POYO training example in torch_brain
(https://github.com/neuro-galaxy/torch_brain), Apache-2.0, Copyright 2025
NeuroGalaxy. Modified for ntt-downstream: multi-session training with
tier-aware evaluation and DDP-safe samplers.
"""

from __future__ import annotations

import logging
import os
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader


def dist_rank_world() -> tuple[int, int]:
    """Return (rank, world_size) at sampler-construction time.

    Lightning initializes the process group inside `trainer.fit`, so
    `torch.distributed.is_initialized()` is False here. Read env vars
    instead — Lightning's subprocess launcher sets RANK/WORLD_SIZE, and
    SLURM `srun` sets SLURM_PROCID/SLURM_NTASKS. Falls back to (0, 1) for
    single-process runs.
    """
    rank = os.environ.get("RANK")
    world = os.environ.get("WORLD_SIZE")
    if rank is None or world is None:
        rank = os.environ.get("SLURM_PROCID", "0")
        world = os.environ.get("SLURM_NTASKS", "1")
    return int(rank), int(world)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn that re-seeds NumPy and Python RNGs.

    PyTorch auto-seeds its own RNG per worker but leaves NumPy/random
    inheriting parent state, which breaks reproducibility for transforms
    that call `np.random.*` or `random.*` (e.g. UnitDropout). Derive both
    from `torch.initial_seed()` so each worker is deterministic given a
    fixed base seed.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


from torch_brain.data import collate
from torch_brain.data.sampler import (
    DistributedStitchingFixedWindowSampler,
    RandomFixedWindowSampler,
)

from ntt_downstream.decoding.backends.poyo.multi_brainset_dataset import (
    build_multi_brainset_co_dataset,
)
from ntt_downstream.decoding.backends.poyo.stitch_evaluator import (
    DecodingStitchEvaluatorWithOptionalValTable,
)
from ntt_downstream.decoding.backends.poyo.temporal_split import TemporalSplitConfig
from ntt_downstream.decoding.backends.poyo.train_decimation import TrainDecimationConfig

logger = logging.getLogger(__name__)


class SafeDecodingStitchEvaluator(DecodingStitchEvaluatorWithOptionalValTable):
    """`DecodingStitchEvaluator` that skips per-session metric compute when
    the session received fewer than 2 stitched samples this epoch.

    R²Score requires ≥2 samples; without this guard, a single under-sampled
    val session crashes the whole validation epoch (and therefore the run).
    Skipped sessions are reset and excluded from the epoch's
    `average_<prefix>_metric`.

    Also coerces session_ids to plain `str`. The nested-dataset namespacing
    yields `numpy.str_` keys, which mix with the parent's plain-`str`
    `'average_<prefix>_metric'` key inside the per-epoch metrics
    DataFrame and cause `wandb.Table` to raise a type-mismatch error.
    `numpy.str_` is a subclass of `str`, so coercion is hash-equal and the
    cache (populated from batch session_ids) still resolves correctly.

    Note: this filter is per-rank. Eval is therefore expected to run on a
    single rank (`num_replicas=1, rank=0`) — divergent `kept` sets across
    DDP ranks would deadlock the parent loop's `metric_fn.compute()` in
    NCCL.
    """

    _MIN_SAMPLES = 2

    def __init__(self, *, session_ids, **kwargs):
        super().__init__(session_ids=[str(s) for s in session_ids], **kwargs)

    def on_validation_epoch_end(self, trainer, pl_module, prefix: str = "val"):
        full_metrics = self.metrics
        kept: dict[str, Any] = {}
        skipped: list[str] = []
        for sid, metric_fn in full_metrics.items():
            cache = self.cache.get(sid, {"pred": []})
            preds = cache.get("pred") or []
            n = sum(int(p.shape[0]) for p in preds) if preds else 0
            if n >= self._MIN_SAMPLES:
                kept[sid] = metric_fn
            else:
                skipped.append(sid)
                metric_fn.reset()
        try:
            self.metrics = kept
            super().on_validation_epoch_end(trainer, pl_module, prefix=prefix)
        finally:
            self.metrics = full_metrics
        if skipped and trainer.is_global_zero and not getattr(self, "quiet", False):
            head = ", ".join(skipped[:5])
            more = "" if len(skipped) <= 5 else f" (+{len(skipped) - 5} more)"
            logger.info(f"Stitch eval skipped {len(skipped)} sessions with <{self._MIN_SAMPLES} samples: {head}{more}")

    def on_test_epoch_end(self, trainer, pl_module, prefix: str = "test"):
        return self.on_validation_epoch_end(trainer, pl_module, prefix=prefix)


@dataclass
class TierSpec:
    """One eval pool: name, dataset, sampler-split, and recording_ids."""

    name: str
    dataset: Any
    split: str | None
    recording_ids: list[str] = field(default_factory=list)


@dataclass
class MultiSessionBundle:
    """Everything `ntt_downstream.decoding.train` needs after dataset/loader construction."""

    train_ds: Any
    train_loader: DataLoader
    val_tiers: list[TierSpec]
    test_tiers: list[TierSpec]
    val_loaders: list[DataLoader]
    test_loaders: list[DataLoader]
    all_unit_ids: list[str]
    all_session_ids: list[str]


def build_multi_session_bundle(
    cfg: Any,
    *,
    sequence_length: float,
    train_transform=None,
    eval_transform=None,
) -> MultiSessionBundle:
    """Build train + per-tier eval datasets, samplers, and dataloaders.

    Reads `cfg.dataset`, `cfg.restrict_brainsets`, `cfg.co_only`,
    `cfg.tiers.*`, `cfg.batch_size`, `cfg.num_workers`, `cfg.seed`,
    `cfg.data_root`. Returns a bundle with everything wired up plus the
    union of unit/session IDs across train+tiers (used to initialise
    `InfiniteVocabEmbedding` vocabs).

    The dataset's `transform` is set to `train_transform` for the train
    split and `eval_transform` for every tier — caller is responsible for
    handing over `Compose([UnitDropout, model.tokenize])` and
    `Compose([model.tokenize])` respectively (matching the
    single-session flow).
    """

    temporal_split = TemporalSplitConfig.from_cfg(getattr(cfg, "temporal_split", None))
    if temporal_split is not None:
        logger.info(
            f"temporal_split active: ratios={temporal_split.ratios} "
            f"brainsets={sorted(temporal_split.brainsets)} "
            f"dilation={temporal_split.dilation}s"
        )
    train_decimation = TrainDecimationConfig.from_cfg(getattr(cfg, "train_decimation", None))
    if train_decimation is not None:
        logger.info(f"train_decimation active: trials={train_decimation.trials} seed={train_decimation.seed}")

    train_ds = build_multi_brainset_co_dataset(
        root=cfg.data_root,
        selection=cfg.dataset,
        transform=train_transform,
        restrict_brainsets=list(cfg.restrict_brainsets) if cfg.restrict_brainsets else None,
        co_only=cfg.co_only,
        temporal_split=temporal_split,
        train_decimation=train_decimation,
        # Pass sequence_length so decimation runs after the sampler's
        # short-interval filter (trials < window_length excluded from the
        # eligible pool before the seeded random pick).
        train_window_length=float(sequence_length),
    )
    logger.info(
        f"Train dataset: {len(train_ds.recording_ids)} recordings "
        f"(restrict_brainsets={list(cfg.restrict_brainsets)!r}, co_only={cfg.co_only})"
    )

    # Eval-tier datasets never decimate — valid/test must reflect the
    # whole held-out pool regardless of train budget. `tiers.enabled=false`
    # short-circuits to a train-only run so calibration / debug runs can
    # skip the tier wiring without nulling every selection by hand.
    tiers_cfg = getattr(cfg, "tiers", None)
    tiers_enabled = bool(getattr(tiers_cfg, "enabled", False)) if tiers_cfg is not None else False
    if tiers_enabled:
        val_tiers = _build_tier_specs(
            cfg,
            train_transform=eval_transform,
            tiers=[
                ("tier1_val", cfg.tiers.tier1_val),
                ("tier2_val", cfg.tiers.tier2_val),
                ("tier3_val", cfg.tiers.tier3_val),
            ],
            temporal_split=temporal_split,
        )
        test_tiers = _build_tier_specs(
            cfg,
            train_transform=eval_transform,
            tiers=[("tier2_test", cfg.tiers.tier2_test), ("tier3_test", cfg.tiers.tier3_test)],
            temporal_split=temporal_split,
        )
    else:
        val_tiers = []
        test_tiers = []
        logger.info("tiers.enabled=false: skipping tier eval datasets")

    # Train sampler / loader — RandomFixedWindowSampler over train_ds.
    # Under DDP we don't shard the random sampler — each rank draws its
    # own random windows from the full pool. Per-rank seed offset ensures
    # ranks see different mini-batches; effective batch becomes
    # world_size * cfg.batch_size.
    rank, _world = dist_rank_world()
    train_sampler = RandomFixedWindowSampler(
        sampling_intervals=train_ds.get_sampling_intervals(split="train"),
        window_length=sequence_length,
        generator=torch.Generator().manual_seed(cfg.seed + rank),
    )
    train_loader = DataLoader(
        train_ds,
        sampler=train_sampler,
        collate_fn=collate,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        # drop_last=False so low-budget calibration runs (small
        # train_decimation.trials, where total windows < batch_size) still
        # produce ≥1 batch instead of yielding an empty loader.
        drop_last=False,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(cfg.seed + rank),
    )

    val_loaders = [_build_eval_loader(t, cfg, sequence_length=sequence_length) for t in val_tiers]
    test_loaders = [_build_eval_loader(t, cfg, sequence_length=sequence_length) for t in test_tiers]

    # Union of unit / session IDs across train + every tier — caller uses
    # this to initialise the InfiniteVocabEmbedding vocabs once before
    # trainer.fit. POYO's tokenize() relies on the full vocab being
    # registered up front.
    all_units: set[str] = set(train_ds.get_unit_ids())
    all_sessions: set[str] = set(train_ds.recording_ids)
    for tier in (*val_tiers, *test_tiers):
        all_units.update(tier.dataset.get_unit_ids())
        all_sessions.update(tier.recording_ids)

    return MultiSessionBundle(
        train_ds=train_ds,
        train_loader=train_loader,
        val_tiers=val_tiers,
        test_tiers=test_tiers,
        val_loaders=val_loaders,
        test_loaders=test_loaders,
        all_unit_ids=sorted(all_units),
        all_session_ids=sorted(all_sessions),
    )


def _build_tier_specs(
    cfg: Any,
    *,
    train_transform,
    tiers: list[tuple[str, Any]],
    temporal_split: TemporalSplitConfig | None = None,
) -> list[TierSpec]:
    specs: list[TierSpec] = []
    for name, entry in tiers:
        if entry is None or entry.selection is None:
            continue
        ds = build_multi_brainset_co_dataset(
            root=cfg.data_root,
            selection=entry.selection,
            transform=train_transform,
            restrict_brainsets=list(cfg.restrict_brainsets) if cfg.restrict_brainsets else None,
            co_only=cfg.co_only,
            temporal_split=temporal_split,
        )
        specs.append(
            TierSpec(
                name=name,
                dataset=ds,
                split=entry.split,
                recording_ids=list(ds.recording_ids),
            )
        )
        logger.info(f"Tier {name}: {len(ds.recording_ids)} recordings (split={entry.split!r})")
    return specs


def _build_eval_loader(
    tier: TierSpec,
    cfg: Any,
    *,
    sequence_length: float,
) -> DataLoader:
    sampling_intervals = tier.dataset.get_sampling_intervals(split=tier.split)
    # Eval runs the full pool on every rank (num_replicas=1, rank=0). See the
    # matching comment in ntt_downstream/train.py — sharding eval across ranks
    # deadlocks SafeDecodingStitchEvaluator's per-rank filter at the parent's
    # `metric_fn.compute()` all-reduce.
    sampler = DistributedStitchingFixedWindowSampler(
        sampling_intervals=sampling_intervals,
        window_length=sequence_length,
        step=sequence_length / 2,
        batch_size=cfg.batch_size,
        num_replicas=1,
        rank=0,
    )
    return DataLoader(
        tier.dataset,
        sampler=sampler,
        collate_fn=collate,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(cfg.seed),
    )


_BRAINSET_SHORT_NAMES = {
    "odoherty_sabes_nonhuman_2017": "odoherty",
    "churchland_shenoy_neural_2012": "churchland",
    "perich_miller_population_2018": "perich_miller",
}


def _classify_recording(rid: str) -> tuple[str, str | None]:
    """Return (brainset_short, perich_task_subtype_or_None) for a recording_id.

    Recording IDs are `{brainset}/{session}`. Perich-Miller sessions encode
    task type in the suffix (`_center_out_reaching` / `_random_target_reaching`)
    and get an extra stratification level.
    """
    brainset = rid.split("/", 1)[0]
    short = _BRAINSET_SHORT_NAMES.get(brainset, brainset)
    if brainset == "perich_miller_population_2018":
        if rid.endswith("_center_out_reaching"):
            return short, "co"
        if rid.endswith("_random_target_reaching"):
            return short, "rt"
    return short, None


class TierAggregator(L.Callback):
    """Compute per-tier R² averages from `DecodingStitchEvaluator` outputs.

    Reads per-session R²s already logged by `DecodingStitchEvaluator`
    (keys = session_id strings) at the end of each validation/test epoch
    and emits, namespaced under `<prefix>/` (`val/` or `test/`):
      - `<prefix>/<tier_name>_r2`                       — mean over the tier
      - `<prefix>/<tier_name>_r2_<brainset_short>`      — mean over sessions
        of one brainset within the tier (e.g. `val/tier1_val_r2_odoherty`)
      - `<prefix>/<tier_name>_r2_perich_miller_{co,rt}` — perich-miller
        extra split by reaching task type

    Must be registered *after* the stitch evaluator so the per-session
    metrics are present in `trainer.callback_metrics` when this fires.
    """

    def __init__(self, val_tiers: Iterable[TierSpec], test_tiers: Iterable[TierSpec]):
        self._val_groups = {t.name: list(t.recording_ids) for t in val_tiers}
        self._test_groups = {t.name: list(t.recording_ids) for t in test_tiers}

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._log_tier_averages(trainer, pl_module, self._val_groups, prefix="val")

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._log_tier_averages(trainer, pl_module, self._test_groups, prefix="test")

    def _log_tier_averages(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        groups: dict[str, list[str]],
        prefix: str,
    ) -> None:
        metrics = trainer.callback_metrics
        for tier_name, recording_ids in groups.items():
            values: list[float] = []
            sub_values: dict[str, list[float]] = {}
            for rid in recording_ids:
                v = metrics.get(rid)
                if v is None:
                    continue
                fv = float(v)
                values.append(fv)
                short, sub = _classify_recording(rid)
                sub_key = short if sub is None else f"{short}_{sub}"
                sub_values.setdefault(sub_key, []).append(fv)
            if not values:
                continue
            mean_v = sum(values) / len(values)
            log_key = f"{prefix}/{tier_name}_r2"
            pl_module.log(log_key, mean_v, sync_dist=False, prog_bar=False, logger=True)
            logger.info(f"{log_key} = {mean_v:.4f} (over {len(values)} sessions)")
            for sub_key, vals in sub_values.items():
                sub_mean = sum(vals) / len(vals)
                sub_log_key = f"{prefix}/{tier_name}_r2_{sub_key}"
                pl_module.log(sub_log_key, sub_mean, sync_dist=False, prog_bar=False, logger=True)
                logger.info(f"{sub_log_key} = {sub_mean:.4f} (over {len(vals)} sessions)")
