"""POYO training script — ntt tokenizer bypass pipeline.

Usage:
    python -m ntt_downstream.decoding.train tokenizer=fake
    python -m ntt_downstream.decoding.train tokenizer=spike_times
    python -m ntt_downstream.decoding.train tokenizer=poisson
    python -m ntt_downstream.decoding.train tokenizer=frozen tokenizer.ntt_checkpoint=/path/to.ckpt

The fake tokenizer reproduces the original POYO computation through the bypass path:
    tokenizer=fake, unit_embedding_mode=add, sentinels=true
"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import hydra
import lightning as L
import rootutils
import torch

# Lightning loads checkpoints with `weights_only=True` in
# `trainer.test(ckpt_path="best"|"last")` and auto-resume (not the explicit
# `weights_only=False` calls we control), which rejects the numpy scalars and
# dtype objects embedded in our checkpoints, typically in torchmetrics state.
# The scalar type lives at `numpy._core.multiarray.scalar`, which PyTorch's
# stock allowlist does not cover, and the dtype path needs `numpy.dtype` and
# `numpy.dtypes.StrDType` (a string dtype shows up in the unit_identification
# freeze-mode metric state). Allowlist all three; these are concrete numpy
# types, not a relaxation of the weights-only contract beyond what is needed.
try:
    import numpy as _np  # noqa: F401
    import numpy._core.multiarray as _np_core_multiarray  # noqa: F401

    torch.serialization.add_safe_globals(
        [
            _np_core_multiarray.scalar,
            _np.dtype,
            _np.dtypes.StrDType,
        ]
    )
except (ImportError, AttributeError):
    # numpy <2.0 doesn't have `_core` or `dtypes.StrDType`; the
    # `numpy.core...` path is in PyTorch's stock allowlist and weights_only
    # loading already works on that branch.
    pass
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.loggers import CSVLogger, MLFlowLogger, WandbLogger
from ntt.evaluation.utils.window_chunking import requires_fixed_input_window
from ntt.evaluation.utils.window_contract import assert_tokenizer_accepts_eval_window
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch_brain.data import collate
from torch_brain.data.sampler import (
    DistributedStitchingFixedWindowSampler,
    RandomFixedWindowSampler,
)
from torch_brain.registry import MODALITY_REGISTRY
from torch_brain.transforms import Compose, UnitDropout
from torch_brain.utils import callbacks as tbrain_callbacks
from torch_brain.utils import seed_everything

from ntt_downstream.common.results.callback import MetricSnapshotCallback
from ntt_downstream.common.results.writer import check_config_resolves
from ntt_downstream.decoding.backends.poyo.datasets import SingleSessionCODataset
from ntt_downstream.decoding.backends.poyo.frozen_ntt_tokenizer import FrozenNTTTokenizer
from ntt_downstream.decoding.backends.poyo.full_window_train_wrapper import (
    FullWindowUnitV2TrainWrapper,
)
from ntt_downstream.decoding.backends.poyo.model import (  # noqa: F401  (registered for Hydra _target_)
    POYOWithEmbeddingBypass,
    POYOWithEmbeddingBypass_wo_crossattention,
)
from ntt_downstream.decoding.backends.poyo.multi_session_setup import (
    SafeDecodingStitchEvaluator,
    TierAggregator,
    build_multi_session_bundle,
    dist_rank_world,
    seed_worker,
)
from ntt_downstream.decoding.backends.poyo.ntt_spike_times_tokenizer import NTTSpikeTimesTokenizer
from ntt_downstream.decoding.backends.poyo.ntt_splits import register_ntt_split_resolver
from ntt_downstream.decoding.backends.poyo.results_storage import build_run_result, save_run_result
from ntt_downstream.decoding.backends.poyo.temporal_split import TemporalSplitConfig
from ntt_downstream.decoding.backends.poyo.train_decimation import TrainDecimationConfig, collect_decimation_summary
from ntt_downstream.decoding.backends.poyo.train_wrapper import V2TrainWrapper

# Anchor on the repo-root `.project-root` indicator so `@hydra.main` can locate
# the shared `configs/decoding` tree regardless of the process CWD (mirrors the
# encode entrypoint). Resolved at import time, before the decorator below runs.
_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

torch.set_float32_matmul_precision("medium")
logger = logging.getLogger(__name__)

register_ntt_split_resolver()


def _force_flash_attention() -> None:
    """Route every torch_brain attention call through FlashAttention.

    torch_brain's `RotaryCrossAttention` / `RotarySelfAttention` dispatch to
    xformers on CUDA when available, else to `rotary_attn_pytorch_func`
    (which calls `F.scaled_dot_product_attention`). PyTorch SDPA's flash
    kernel rejects any `attn_mask` argument, so we replace the pytorch
    branch with a flash_attn varlen wrapper that converts the bool key-
    padding mask into cu_seqlens — letting flash handle masked cross-
    attention without SDPA's mask restriction.
    """
    import torch_brain.nn.rotary_attention as _ra

    from ntt_downstream.decoding.backends.poyo.flash_attn_varlen import (
        rotary_attn_flash_func,
    )

    if _ra.xops is not None:
        logger.info("force_flash_attn: disabling xformers in torch_brain")
        _ra.xops = None

    _ra.rotary_attn_pytorch_func = rotary_attn_flash_func
    logger.info("force_flash_attn: monkey-patched rotary_attn_pytorch_func → flash_attn varlen wrapper")


def _build_single_session_datasets(
    cfg: DictConfig, train_window_s: float
) -> tuple[SingleSessionCODataset, SingleSessionCODataset]:
    """Build the per-session train + eval datasets for the single-session path.

    ``eval_ds`` is a shallow copy so the temporal-split / decimation caches and
    the underlying HDF5 handle are shared by reference; only the per-loader
    ``transform`` differs (set later). Two independent ``SingleSessionCODataset``
    objects would re-derive both caches and double the HDF5 open cost — and
    would be a correctness foot-gun if any RNG inside those caches ever became
    non-deterministic across re-runs.
    """
    # Single-session path is always Perich-Miller; apply the temporal
    # override unconditionally when enabled (the brainsets allowlist
    # only matters in the multi-brainset path).
    single_session_ts = TemporalSplitConfig.from_cfg(cfg.get("temporal_split", None))
    if single_session_ts is not None:
        seed_tag = f" seed={single_session_ts.seed}" if single_session_ts.mode == "random" else ""
        logger.info(
            f"temporal_split active (single-session): "
            f"mode={single_session_ts.mode}{seed_tag} "
            f"ratios={single_session_ts.ratios} "
            f"dilation={single_session_ts.dilation}s"
        )
    single_session_decim = TrainDecimationConfig.from_cfg(cfg.get("train_decimation", None))
    if single_session_decim is not None:
        logger.info(
            f"train_decimation active (single-session): "
            f"trials={single_session_decim.trials} "
            f"seed={single_session_decim.seed}"
        )
    # Pass model.sequence_length so train_decimation runs after the sampler's
    # short-interval filter (trials < window_length get excluded from the
    # eligible pool before the seeded random pick).
    train_ds = SingleSessionCODataset(
        root=cfg.data_root,
        recording_id=cfg.session,
        temporal_split=single_session_ts,
        train_decimation=single_session_decim,
        train_window_length=train_window_s,
    )
    eval_ds = copy.copy(train_ds)
    assert eval_ds._temporal_split_cache is train_ds._temporal_split_cache
    assert eval_ds._train_decimation_cache is train_ds._train_decimation_cache
    return train_ds, eval_ds


def _build_transforms(cfg: DictConfig, model: POYOWithEmbeddingBypass) -> tuple[Compose, Compose]:
    """Build train + eval token transforms.

    Train optionally prepends ``UnitDropout`` (POYO's default triangular
    distribution). Both pipelines end with ``model.tokenize``, which produces
    the (input_token_type, input_unit_index, input_timestamps, input_mask)
    fields the wrapper's tokenize methods consume.
    """
    train_transforms: list = []
    if cfg.unit_dropout:
        # POYO's default UnitDropout params (triangular distribution).
        # ``seed=cfg.seed`` ties the dropout RNG to the run-level seed so the
        # rejection sampler is reproducible. The np.random.permutation call
        # inside __call__ is reseeded per-worker via seed_worker.
        train_transforms.append(
            UnitDropout(
                min_units=30,
                mode_units=100,
                max_units=300,
                peak=4,
                seed=cfg.seed,
            )
        )
        logger.info("UnitDropout enabled: min=30, mode=100, max=300, peak=4")
    train_transforms.append(model.tokenize)
    return Compose(train_transforms), Compose([model.tokenize])


# Tokenizers ``main`` can build; only ``FrozenNTTTokenizer`` exposes the
# combines_units / slot-proxy API used by the vocab helpers below.
NTTTokenizer = FrozenNTTTokenizer | NTTSpikeTimesTokenizer


def _trainer_callbacks(trainer: L.Trainer) -> list[Callback]:
    """Return ``trainer.callbacks``.

    Lightning assigns ``callbacks`` in a connector mixin rather than declaring
    it on ``Trainer``, so the attribute is invisible to static analysis.
    """
    return trainer.callbacks  # pyright: ignore[reportAttributeAccessIssue]  # set by the callback connector


def _validate_tokenizer_eval_window(spec, sequence_length: float, fixed_window_seconds: float | None) -> None:
    """Fail loud at setup if the tokenizer cannot accept the POYO window.

    - ``fixed_window_seconds`` set (override for checkpoints whose spec
      mis-declares its window): require ``sequence_length`` to be a positive integer multiple of it, in both
      seconds AND bins (the runtime chunker rounds seconds→bins on each side
      independently, so seconds-divisibility alone is not sufficient).
      ``ValueError`` otherwise.
    - else, only for FIXED-window tokenizers
      (``requires_fixed_input_window(spec)``): defer to ntt's
      ``assert_tokenizer_accepts_eval_window``.
    - ``variable_t=True`` tokenizers are NOT constrained here — the runtime feeds them the full window.

    Gating the spec-contract check on ``requires_fixed_input_window`` matters:
    ``assert_tokenizer_accepts_eval_window`` additionally requires a
    ``variable_t=True`` spec's eval window to be a multiple of its
    ``token_bin_size_ms`` quantum, which would reject
    tokenizers at the universal ``sequence_length=1.0`` s — Perceiver
    (33.33 ms quantum → 30.003 bins) and transformer-vq with non-100 ms patch quanta.
    Those run unchunked, so this validation must stay a no-op for them.
    """
    if fixed_window_seconds is not None:
        w = float(fixed_window_seconds)
        k = round(sequence_length / w)
        if k < 1 or abs(sequence_length - k * w) > 1e-6:
            raise ValueError(
                f"model.sequence_length={sequence_length}s must be a positive integer "
                f"multiple of tokenizer.fixed_window_seconds={w}s."
            )
        # The runtime chunker works in BINS: it slices a round(seq/bin)-bin
        # raster into round(w/bin)-bin chunks. Seconds-space divisibility does
        # not guarantee bin-space divisibility once each side is rounded
        # independently, so re-check in bins to fail loud at setup rather than
        # raising in _forward_chunked on the first batch.
        bin_s = float(spec.bin_size_seconds)
        seq_bins = round(sequence_length / bin_s)
        w_bins = round(w / bin_s)
        if w_bins < 1 or seq_bins % w_bins != 0:
            raise ValueError(
                f"tokenizer.fixed_window_seconds={w}s at bin size {bin_s}s gives a "
                f"chunk width of {w_bins} bins, which does not divide the "
                f"{seq_bins}-bin window raster (sequence_length={sequence_length}s); "
                f"choose a compatible sequence_length / bin size."
            )
        return
    if requires_fixed_input_window(spec):
        assert_tokenizer_accepts_eval_window(spec, sequence_length)


def _resolve_full_window_units(cfg: DictConfig, tokenizer_mode: str, ntt_tokenizer: NTTTokenizer | None) -> bool:
    """Decide whether to use the full-window POYO→NTT handoff.

    ``tokenizer.full_window_units`` is ``auto`` (default; also when unset or
    null), ``true`` or ``false``. ``auto`` turns it on exactly for
    session-keyed tokenizers (``FrozenNTTTokenizer.is_session_keyed``, e.g.
    the Perceiver) and leaves every other tokenizer on the
    active-unit path (the default handoff).

    An explicit ``false`` on a session-keyed tokenizer raises: the active-unit
    handoff passes POYO-global unit indices, which the Perceiver clamps into
    its session's embedding range instead of rejecting, so the run would
    train on the wrong unit embeddings with no error.
    """
    raw = cfg.tokenizer.get("full_window_units", "auto")
    session_keyed = bool(getattr(ntt_tokenizer, "is_session_keyed", False))
    if raw is None or raw == "auto":
        return session_keyed
    if not isinstance(raw, bool):
        raise ValueError(f"tokenizer.full_window_units must be auto, true or false; got {raw!r}.")
    if raw and tokenizer_mode != "ntt_frozen":
        raise ValueError("tokenizer.full_window_units=true is only supported for tokenizer.mode=ntt_frozen.")
    if not raw and session_keyed:
        raise ValueError(
            "tokenizer.full_window_units=false with a session-keyed tokenizer "
            "(per-session unit embeddings, e.g. Perceiver): the active-unit "
            "handoff feeds POYO-global unit ids, which the tokenizer would "
            "silently clamp onto the wrong embeddings. Use auto or true."
        )
    return raw


def _instantiate_ntt_tokenizer(cfg: DictConfig) -> NTTTokenizer | None:
    """Build the optional NTT tokenizer selected by ``cfg.tokenizer``.

    Returns ``None`` for ``tokenizer.mode == "fake"`` (POYO's native path).
    """
    tokenizer_mode = cfg.tokenizer.mode
    if tokenizer_mode not in ("ntt", "ntt_frozen"):
        return None

    ntt_tokenizer_type = cfg.tokenizer.ntt_tokenizer
    assert ntt_tokenizer_type is not None, (
        f"tokenizer.ntt_tokenizer must be specified when tokenizer.mode={tokenizer_mode}"
    )
    freeze_ntt_tokenizer = cfg.tokenizer.get("freeze", True)
    if ntt_tokenizer_type == "spike_times":
        return NTTSpikeTimesTokenizer(token_dim=cfg.model.dim)
    if ntt_tokenizer_type == "frozen":
        ntt_checkpoint = cfg.tokenizer.ntt_checkpoint
        assert ntt_checkpoint is not None, (
            "tokenizer.ntt_checkpoint must be specified when tokenizer.ntt_tokenizer=frozen"
        )
        logger.info(f"Loading frozen ntt tokenizer from {ntt_checkpoint}")
        fixed_window_seconds = cfg.tokenizer.get("fixed_window_seconds", None)
        tok = FrozenNTTTokenizer.from_checkpoint(
            ntt_checkpoint,
            freeze=freeze_ntt_tokenizer,
            discrete_input_mode=cfg.tokenizer.get("discrete_input_mode", "z_q"),
            discrete_indices_source=cfg.tokenizer.get("discrete_indices_source", "auto"),
            fixed_window_seconds=fixed_window_seconds,
        )
        # Fail loud at setup: the POYO sequence window must be a positive
        # integer multiple of the tokenizer's input window, otherwise chunking
        # would silently drop or pad samples. The helper gates the spec-contract
        # check on fixed-window tokenizers so it stays a no-op for variable_t
        # tokenizers (which the runtime feeds the full window unchunked).
        _validate_tokenizer_eval_window(
            tok.tokenizer.spec,
            float(cfg.model.sequence_length),
            fixed_window_seconds,
        )
        logger.info(
            f"Frozen tokenizer: combines_units={tok.combines_units}, "
            f"input_bin_size_ms={tok.input_bin_size_ms}, "
            f"token_bin_size_s={tok.token_bin_size_s}, "
            f"output_dim={tok.output_dim}, "
            f"is_discrete={tok.is_discrete}, "
            f"discrete_input_mode={tok.discrete_input_mode}"
        )
        return tok
    if ntt_tokenizer_type == "poisson":
        # Poisson baseline: config-based, no checkpoint needed. Uses 1ms input
        # bins (standard for brainsets spike-binned data).
        tok = FrozenNTTTokenizer.from_poisson(
            rate_bin_size_ms=cfg.tokenizer.poisson_rate_bin_ms,
            input_bin_size_ms=1.0,
            window_length_seconds=cfg.model.sequence_length,
            freeze=freeze_ntt_tokenizer,
            discrete_input_mode=cfg.tokenizer.get("discrete_input_mode", "z_q"),
            discrete_indices_source=cfg.tokenizer.get("discrete_indices_source", "auto"),
        )
        logger.info(
            f"Poisson baseline tokenizer: rate_bin_ms={cfg.tokenizer.poisson_rate_bin_ms}, "
            f"combines_units={tok.combines_units}, "
            f"token_bin_size_s={tok.token_bin_size_s}, "
            f"output_dim={tok.output_dim}"
        )
        return tok
    raise ValueError(f"Unknown tokenizer.ntt_tokenizer: {ntt_tokenizer_type}")


def _load_init_ckpt(
    *,
    wrapper: L.LightningModule,
    cfg: DictConfig,
    model: POYOWithEmbeddingBypass,
    ntt_tokenizer: NTTTokenizer | None,
    unit_ids: list,
    session_ids: list,
    init_ckpt: str,
) -> None:
    """Load weights from ``init_ckpt`` and extend vocabs for the new session.

    Three phases:
      1. ``load_state_dict(strict=False)`` populates the wrapper. The IVE
         pre-hooks materialise unit/session embeddings from the ckpt's saved
         vocab.
      2. ``extend_vocab`` appends new units/sessions at the tail of the
         existing vocab — already-present IDs are filtered out so
         ``exist_ok=False`` is safe.
      3. Optional warm-start of slot-proxy unit_embs for combines_units
         tokenizers — see the inline rationale at the warm-start block.

    Must run before any dataloader iterates the datasets, because POYO's
    ``tokenize`` needs ``unit_emb`` / ``session_emb`` vocabs to resolve
    recording-local indices to global ones.
    """
    ckpt = torch.load(init_ckpt, map_location="cpu", weights_only=False)
    ckpt_state_dict = ckpt["state_dict"]
    # ``strict=False``: the ckpt may have been written with a different
    # ``input_proj`` dim (e.g. base = no projection, FT = projection) or a
    # different ``ntt_tokenizer`` graph; we tolerate those mismatches.
    load_res = wrapper.load_state_dict(ckpt_state_dict, strict=False)
    logger.info(
        f"init_ckpt loaded ({init_ckpt}): missing={len(load_res.missing_keys)} "
        f"unexpected={len(load_res.unexpected_keys)}"
    )

    # Capture base-vocab unit_id order BEFORE extending — needed by the
    # combines_units warm-start path below to identify which unit_ids served
    # as slot 0..M-1 proxies during base training.
    # InfiniteVocabEmbedding reserves index 0 for the "NA" sentinel
    # (unknown/padding); real unit_ids start at index 1, and only real
    # unit_ids ever appeared in ``batch.unit_ids[:, :M]`` at base time.
    # ``vocab`` is None only while the embedding is still lazy; the ckpt load
    # above materialises it.
    assert model.unit_emb.vocab is not None, "unit_emb vocab must be materialised by the init_ckpt load"
    assert model.session_emb.vocab is not None, "session_emb vocab must be materialised by the init_ckpt load"
    base_vocab_unit_ids = [k for k in model.unit_emb.vocab.keys() if k != "NA"]

    new_units = [u for u in unit_ids if u not in model.unit_emb.vocab]
    if new_units:
        model.unit_emb.extend_vocab(new_units, exist_ok=False)
        logger.info(f"Extended unit vocab with {len(new_units)} new unit(s)")
    new_sessions = [s for s in session_ids if s not in model.session_emb.vocab]
    if new_sessions:
        model.session_emb.extend_vocab(new_sessions, exist_ok=False)
        logger.info(f"Extended session vocab with {len(new_sessions)} new session(s)")

    # Warm-start slot-proxy unit_embs for combines_units tokenizers.
    #
    # combines_units=True (e.g. Perceiver) maps its M abstract latent slots →
    # first-M input unit_ids of each session as proxies, so each (session,
    # slot_k) gets its own row in unit_emb. With cold init, unit_identification
    # FT lands at ~0 (no prior on the session's slot bias). Copy the base
    # session's learned slot-k row into the new session's slot-k row before
    # FT — slot k carries the SAME architectural meaning across sessions for
    # these tokenizers, so this is a real prior (unlike spike_times and per-unit
    # tokenizers where unit_id k is a different neuron per session).
    warm_start = bool(cfg.tokenizer.get("warm_start_unit_emb_proxy", False))
    if not (warm_start and isinstance(ntt_tokenizer, FrozenNTTTokenizer) and ntt_tokenizer.combines_units):
        return

    M = ntt_tokenizer.num_latent_slots
    if M is None:
        logger.info(
            "warm_start_unit_emb_proxy=true but num_latent_slots is None; "
            "skipping warm-start (population/U=1 tokenizer)."
        )
        return

    base_slot_ids = base_vocab_unit_ids[:M]
    new_slot_ids = list(unit_ids)[:M]
    if len(base_slot_ids) < M or len(new_slot_ids) < M:
        raise ValueError(
            f"warm-start needs ≥{M} unit_ids on both sides (base={len(base_slot_ids)}, new={len(new_slot_ids)})."
        )
    with torch.no_grad():
        for src_id, dst_id in zip(base_slot_ids, new_slot_ids, strict=True):
            src_idx = model.unit_emb.vocab[src_id]
            dst_idx = model.unit_emb.vocab[dst_id]
            if src_idx == dst_idx:
                continue
            model.unit_emb.weight.data[dst_idx] = model.unit_emb.weight.data[src_idx].clone()
    logger.info(
        f"Warm-started {M} slot-proxy unit_embs from base session (e.g. {base_slot_ids[0]} → {new_slot_ids[0]})."
    )


def _register_global_slot_proxies(
    cfg: DictConfig, model: POYOWithEmbeddingBypass, ntt_tokenizer: NTTTokenizer | None
) -> None:
    """Register universal ``_slot_k`` tokens in ``unit_emb``.

    For combines_units=True tokenizers (Perceiver), slot k means the same
    architectural latent slot across all sessions. This registers M global
    ``_slot_0`` .. ``_slot_{M-1}`` tokens in unit_emb (shared, NOT per-session)
    and stamps the corresponding global vocab indices onto
    ``ntt_tokenizer.slot_global_ids``. ``_flatten_4d``'s combines_units branch
    then emits these as the slot unit_ids in place of ``batch.unit_ids[:, :M]``.

    Must run AFTER vocab is materialised (init_ckpt load + extend_vocab) and
    BEFORE the first forward pass.
    """
    use_global_slot_proxies = bool(cfg.tokenizer.get("use_global_slot_proxies", False))
    if not (use_global_slot_proxies and isinstance(ntt_tokenizer, FrozenNTTTokenizer) and ntt_tokenizer.combines_units):
        return

    M = ntt_tokenizer.num_latent_slots
    if M is None:
        raise ValueError(
            "use_global_slot_proxies=true but tokenizer reports num_latent_slots=None (population/U=1 case)."
        )
    slot_token_strings = [f"_slot_{k}" for k in range(M)]
    assert model.unit_emb.vocab is not None, "unit_emb vocab must be materialised before registering slot proxies"
    missing = [s for s in slot_token_strings if s not in model.unit_emb.vocab]
    if missing:
        model.unit_emb.extend_vocab(missing, exist_ok=False)
        logger.info(f"Registered {len(missing)} universal slot token(s) in unit_emb vocab.")
    slot_indices = torch.tensor(
        [model.unit_emb.vocab[s] for s in slot_token_strings],
        dtype=torch.long,
    )
    ntt_tokenizer.slot_global_ids = slot_indices
    logger.info(f"Global slot proxies: ntt_tokenizer.slot_global_ids = {slot_indices.tolist()}")


def _apply_freeze_mode(cfg: DictConfig, wrapper: L.LightningModule) -> None:
    """Apply ``cfg.freeze_mode`` to the wrapper's params.

    Currently supports two modes:
      * ``"none"`` (default): no-op.
      * ``"unit_identification"``: only unit + session embeddings are
        trainable; the rest of POYO (encoder, latent processing, decoder,
        readout, input_proj) is frozen. When ``ft_train_input_proj=true`` the
        wrapper's ``input_proj`` is additionally unfrozen — motivation
        (combines_units tokenizers): the base-trained body learns to weight
        ``unit_emb`` near init scale and rely on slot content delivered
        through ``input_proj``. Updating only ``unit_emb`` at FT trains a
        parameter the body has learned to ignore; allowing ``input_proj`` to
        adapt gives the FT a session-shared lever the body actually reads.
        Param count is tiny (~5k for D_tok=8 → 64 → 64).
    """
    freeze_mode = cfg.get("freeze_mode") or "none"
    if freeze_mode == "none":
        return
    if freeze_mode != "unit_identification":
        raise ValueError(f"Unknown freeze_mode={freeze_mode!r}; expected 'none' or 'unit_identification'.")

    train_input_proj_at_ft = bool(cfg.get("ft_train_input_proj", False))
    n_trainable = 0
    n_frozen = 0
    for n, p in wrapper.named_parameters():
        is_emb = "model.unit_emb" in n or "model.session_emb" in n
        is_input_proj = train_input_proj_at_ft and n.startswith("input_proj.")
        if is_emb or is_input_proj:
            p.requires_grad = True
            n_trainable += p.numel()
        else:
            p.requires_grad = False
            n_frozen += p.numel()
    logger.info(
        f"freeze_mode=unit_identification "
        f"(ft_train_input_proj={train_input_proj_at_ft}): "
        f"trainable={n_trainable:,} frozen={n_frozen:,}"
    )


def _build_single_session_loaders(
    *,
    cfg: DictConfig,
    model: POYOWithEmbeddingBypass,
    train_ds: SingleSessionCODataset,
    eval_ds: SingleSessionCODataset,
    eval_only: bool,
) -> tuple[DataLoader | None, list[DataLoader], list[DataLoader]]:
    """Build (train_loader, val_loaders, test_loaders) for the single-session path.

    Eval runs the full pool on every rank (num_replicas=1, rank=0). Sharding
    eval across ranks deadlocks the parent stitch evaluator's
    ``metric_fn.compute()`` because per-rank session sets diverge — the
    ``SafeDecodingStitchEvaluator`` filter is per-rank. Duplicate eval work is
    cheap relative to training; revisit if eval becomes a bottleneck.
    """
    rank, _world = dist_rank_world()
    if eval_only:
        # n_trials=0 (plain eval) leg: train interval may be empty (or trivially small) under
        # the temporal-split layout; skip the train loader entirely since
        # trainer.fit() won't run.
        train_loader: DataLoader | None = None
    else:
        train_sampler = RandomFixedWindowSampler(
            sampling_intervals=train_ds.get_sampling_intervals("train"),
            window_length=model.sequence_length,
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

    def _stitch_loader(split: str) -> DataLoader:
        sampler = DistributedStitchingFixedWindowSampler(
            sampling_intervals=eval_ds.get_sampling_intervals(split),
            window_length=model.sequence_length,
            step=model.sequence_length / 2,
            batch_size=cfg.batch_size,
            num_replicas=1,
            rank=0,
        )
        return DataLoader(
            eval_ds,
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

    return train_loader, [_stitch_loader("valid")], [_stitch_loader("test")]


def _build_loggers(cfg: DictConfig, tokenizer_mode: str) -> list:
    """Build Lightning loggers (CSV always; wandb / MLflow optional)."""
    csv_logger = CSVLogger(save_dir=cfg.log_dir, name="csv_logs")
    loggers: list = [csv_logger]

    if cfg.wandb.enabled:
        # tags / group / notes are forwarded to wandb.init() by WandbLogger's
        # **kwargs passthrough. Skip empty/None so we don't override wandb's
        # defaults for unset fields.
        wandb_extra: dict = {}
        wandb_tags = cfg.wandb.get("tags") or []
        if wandb_tags:
            wandb_extra["tags"] = list(wandb_tags)
        if cfg.wandb.get("group"):
            wandb_extra["group"] = cfg.wandb.group
        if cfg.wandb.get("notes"):
            wandb_extra["notes"] = cfg.wandb.notes
        loggers.append(
            WandbLogger(
                save_dir=cfg.log_dir,
                name=cfg.wandb.name or f"poyo-{tokenizer_mode}-{cfg.session}",
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                log_model=cfg.wandb.log_model,
                **wandb_extra,
            )
        )

    if cfg.get("mlflow") is not None and cfg.mlflow.enabled:
        # Reads MLFLOW_TRACKING_URI / MLFLOW_TRACKING_USERNAME / _PASSWORD from
        # the environment. Without a URI, MLflow falls back to a local ./mlruns
        # file store, which current MLflow refuses with an error that never
        # mentions the URI, so fail here with one that does.
        if not os.environ.get("MLFLOW_TRACKING_URI", "").strip():
            raise RuntimeError(
                "mlflow.enabled=true but MLFLOW_TRACKING_URI is unset. Set "
                "MLFLOW_TRACKING_URI, or pass mlflow.enabled=false to run without MLflow."
            )
        #
        # Orchestrator scripts (ntt_downstream.decoding.utils.mlflow_parent) create a
        # parent run upfront and pass its id via cfg.mlflow.parent_run_id;
        # attaching the `mlflow.parentRunId` tag is what MLflow uses to render
        # children nested under the parent in the UI. cfg.mlflow.tags is a
        # free-form dict the orchestrator can use to stamp sweep_id /
        # tokenizer_name / etc. for filter-driven views.
        mlflow_tags: dict[str, str] = {}
        cfg_tags = cfg.mlflow.get("tags") or {}
        if cfg_tags:
            mlflow_tags.update({str(k): str(v) for k, v in cfg_tags.items()})
        parent_run_id = cfg.mlflow.get("parent_run_id")
        if parent_run_id:
            mlflow_tags["mlflow.parentRunId"] = str(parent_run_id)
        loggers.append(
            MLFlowLogger(
                experiment_name=cfg.mlflow.experiment_name,
                run_name=cfg.mlflow.run_name or f"poyo-{tokenizer_mode}-{cfg.session}",
                tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
                log_model=cfg.mlflow.log_model,
                tags=mlflow_tags or None,
            )
        )

    return loggers


@dataclass(frozen=True)
class _TrainerSchedule:
    """Trainer-cadence settings derived from cfg + world_size.

    Either ``max_steps`` is positive (step-driven) or ``max_epochs`` is
    positive (epoch-driven). The unused side is set to Lightning's sentinel
    (``max_steps=-1`` / ``max_epochs=-1``).
    """

    use_max_steps: bool
    max_steps: int
    max_epochs: int
    val_check_interval: int | None
    check_val_every_n_epoch: int | None
    ckpt_every_n_train_steps: int | None
    ckpt_every_n_epochs: int | None


def _resolve_trainer_schedule(cfg: DictConfig, world_size: int) -> _TrainerSchedule:
    """Resolve epoch- vs step-driven trainer cadence.

    ``cfg.epochs`` / ``cfg.eval_epochs`` (or ``cfg.max_steps`` /
    ``cfg.eval_every_n_steps`` in step mode) are the 1-GPU-equivalent counts
    at effective batch ``cfg.batch_size * world_size``. The train sampler
    (``RandomFixedWindowSampler``) does not shard under DDP — each rank
    iterates the full pool — so each Lightning epoch in W-GPU mode does W× as
    many optimizer steps as a 1-GPU bs=cfg.batch_size run. Divide by W to
    match total optimizer steps and OneCycleLR ramp; eval / checkpoint
    cadence is scaled the same way so they fire at the same 1-GPU-equivalent
    points.

    Step-based override (``cfg.max_steps`` set) drives the trainer by
    optimizer-step count, which makes runs comparable across data ratios
    without per-ratio epoch scaling. ``max_epochs=-1`` makes ``max_steps`` the
    sole stop condition; ``train_wrapper.on_fit_start`` forces
    ``fit_loop.setup_data()`` if the train_dataloader hasn't been populated
    yet (Lightning's ``estimated_stepping_batches`` short-circuits the setup
    call when ``max_epochs=-1``, but we need the loader for on_fit_start
    token-stat probing).
    """
    use_max_steps = cfg.get("max_steps") is not None
    if use_max_steps:
        max_steps = max(1, int(cfg.max_steps) // world_size)
        val_check_interval = max(1, int(cfg.eval_every_n_steps) // world_size)
        ckpt_every_n_train_steps = max(1, int(cfg.checkpoint_every_n_steps) // world_size)
        logger.info(
            f"Step-based training: max_steps={max_steps} "
            f"val_check_interval={val_check_interval} "
            f"ckpt_every_n_train_steps={ckpt_every_n_train_steps} "
            f"(world_size={world_size})"
        )
        return _TrainerSchedule(
            use_max_steps=True,
            max_steps=max_steps,
            max_epochs=-1,
            val_check_interval=val_check_interval,
            check_val_every_n_epoch=None,
            ckpt_every_n_train_steps=ckpt_every_n_train_steps,
            ckpt_every_n_epochs=None,
        )

    max_epochs = max(1, cfg.epochs // world_size)
    check_val_every_n_epoch = max(1, cfg.eval_epochs // world_size)
    return _TrainerSchedule(
        use_max_steps=False,
        max_steps=-1,  # Lightning's "no step cap" sentinel
        max_epochs=max_epochs,
        val_check_interval=None,
        check_val_every_n_epoch=check_val_every_n_epoch,
        ckpt_every_n_train_steps=None,
        ckpt_every_n_epochs=check_val_every_n_epoch,
    )


def _build_ckpt_callback(schedule: _TrainerSchedule, has_validation: bool) -> ModelCheckpoint:
    """Build the ModelCheckpoint callback matching the trainer schedule.

    Without validation we still want ``save_last`` so post-fit ``test()`` has
    a ckpt to load; we just can't monitor a metric.
    """
    ckpt_kwargs: dict = {
        "save_last": True,
        "save_on_train_epoch_end": not schedule.use_max_steps,
    }
    if schedule.use_max_steps:
        ckpt_kwargs["every_n_train_steps"] = schedule.ckpt_every_n_train_steps
    else:
        ckpt_kwargs["every_n_epochs"] = schedule.ckpt_every_n_epochs

    if has_validation:
        return ModelCheckpoint(monitor="average_val_metric", mode="max", **ckpt_kwargs)
    return ModelCheckpoint(monitor=None, save_top_k=0, **ckpt_kwargs)


def _build_trainer(
    cfg: DictConfig,
    *,
    callbacks: list,
    loggers: list,
    schedule: _TrainerSchedule,
) -> L.Trainer:
    """Assemble the Lightning Trainer from cfg + resolved schedule."""
    trainer_kwargs: dict = {
        "logger": loggers,
        "default_root_dir": cfg.log_dir,
        "max_epochs": schedule.max_epochs,
        "max_steps": schedule.max_steps,
        "log_every_n_steps": cfg.log_every_n_steps,
        "callbacks": callbacks,
        "precision": cfg.precision,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": cfg.gpus,
        "num_nodes": cfg.num_nodes,
        "strategy": cfg.strategy,
        # Custom samplers (RandomFixedWindowSampler /
        # DistributedStitchingFixedWindowSampler) handle distribution
        # themselves — don't let Lightning wrap them with DistributedSampler.
        "use_distributed_sampler": False,
        "num_sanity_val_steps": 0,
        "deterministic": False,
    }
    if schedule.use_max_steps:
        # Lightning rejects ``val_check_interval > steps_per_epoch`` unless
        # ``check_val_every_n_epoch=None`` — see fit_loop.setup_data
        # (lightning/pytorch/loops/fit_loop.py:296). Setting it to None makes
        # Lightning interpret val_check_interval as a global step cadence
        # rather than within-epoch. Required when ts_train is small enough
        # that an epoch is shorter than val_check_interval (which is the
        # whole point of step mode).
        trainer_kwargs["val_check_interval"] = schedule.val_check_interval
        trainer_kwargs["check_val_every_n_epoch"] = None
    else:
        trainer_kwargs["check_val_every_n_epoch"] = schedule.check_val_every_n_epoch
    return L.Trainer(**trainer_kwargs)


def _maybe_loaders(loaders: list[DataLoader]):
    """Lightning expects a single loader, a list of loaders, or None.

    An empty list crashes with IndexError; collapse a singleton list to the
    bare loader so logs / callback dispatch use a flat (non-list) path.
    """
    if not loaders:
        return None
    return loaders if len(loaders) > 1 else loaders[0]


@hydra.main(
    version_base="1.3",
    config_path=f"{_ROOT}/configs/decoding",
    config_name="train",
)
def main(cfg: DictConfig) -> None:
    # The results record stores the fully resolved config. Resolve it now so a
    # broken interpolation stops the run here rather than after training.
    check_config_resolves(cfg, task="decoding")

    seed_everything(cfg.seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True

    if cfg.get("force_flash_attn", False):
        _force_flash_attention()

    multi_session = cfg.get("dataset") is not None

    # --- Dataset ---
    # train_ds / eval_ds are set on both paths: single-session builds them
    # immediately; multi-session leaves them None here and rebinds train_ds
    # from ``bundle.train_ds`` below (eval_ds is never used in that branch —
    # bundle owns the eval loaders).
    train_ds = None
    eval_ds = None
    if multi_session:
        # Multi-session path needs the model first (for ``model.tokenize``),
        # so we defer dataset construction. Use a one-recording probe to
        # discover the readout spec.
        from ntt_downstream.decoding.backends.poyo.multi_brainset_dataset import (
            build_multi_brainset_co_dataset,
        )

        probe_ds = build_multi_brainset_co_dataset(
            root=cfg.data_root,
            selection=cfg.dataset,
            restrict_brainsets=list(cfg.restrict_brainsets) if cfg.restrict_brainsets else None,
            co_only=cfg.co_only,
        )
        example_rec = probe_ds.get_recording(probe_ds.recording_ids[0])
    else:
        train_ds, eval_ds = _build_single_session_datasets(cfg, train_window_s=float(cfg.model.sequence_length))
        example_rec = train_ds.get_recording(train_ds.recording_ids[0])

    # `config` is attached to the recording dynamically by get_recording_hook.
    readout_id = example_rec.config["readout"]["readout_id"]  # pyright: ignore[reportAttributeAccessIssue]
    readout_spec = MODALITY_REGISTRY[readout_id]

    # --- Model ---
    # ``cfg.model._target_`` selects the variant (e.g. POYOWithEmbeddingBypass
    # or POYOWithEmbeddingBypass_wo_crossattention). Other keys are model
    # hyperparameters forwarded to the chosen class's __init__.
    model = hydra.utils.instantiate(cfg.model, readout_spec=readout_spec)

    train_transform, eval_transform = _build_transforms(cfg, model)

    # Initialize vocabs + wire up transforms.
    if multi_session:
        bundle = build_multi_session_bundle(
            cfg,
            sequence_length=model.sequence_length,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )
        train_ds = bundle.train_ds
        unit_ids = bundle.all_unit_ids
        session_ids = bundle.all_session_ids
        train_loader = bundle.train_loader
        val_loaders = bundle.val_loaders
        test_loaders = bundle.test_loaders
        val_tiers = bundle.val_tiers
        test_tiers = bundle.test_tiers
    else:
        # Single-session path: ``_build_single_session_datasets`` populated both.
        assert train_ds is not None and eval_ds is not None
        unit_ids = train_ds.get_unit_ids()
        session_ids = train_ds.recording_ids
        train_ds.transform = train_transform
        eval_ds.transform = eval_transform
        val_tiers = []
        test_tiers = []

    init_ckpt = cfg.get("init_ckpt") or None
    eval_only = bool(cfg.get("eval_only", False))
    if eval_only and init_ckpt is None:
        logger.warning(
            "eval_only=true but init_ckpt is unset; evaluating a freshly "
            "initialized random model. Set init_ckpt=/path/to/checkpoint.ckpt "
            "for calibration/plain-eval runs."
        )
    if init_ckpt is None:
        model.unit_emb.initialize_vocab(unit_ids)
        model.session_emb.initialize_vocab(session_ids)
    else:
        # Defer vocab initialization until after the wrapper exists — we'll
        # load the ckpt into the wrapper (which materialises the IVEs from
        # the ckpt's saved vocab via the load-state-dict pre-hook) and then
        # call extend_vocab() to add the current session's new units. This
        # keeps trained embeddings byte-identical and only randomises the
        # entries that are actually new.
        logger.info(f"Deferring vocab init: will load from init_ckpt={init_ckpt}")

    # --- NTT tokenizer (if needed) ---
    ntt_tokenizer = _instantiate_ntt_tokenizer(cfg)
    tokenizer_mode = cfg.tokenizer.mode

    # --- Lightning wrapper ---
    use_full_window_units = _resolve_full_window_units(cfg, tokenizer_mode, ntt_tokenizer)
    wrapper_cls = FullWindowUnitV2TrainWrapper if use_full_window_units else V2TrainWrapper
    if use_full_window_units:
        logger.info(
            "Using FullWindowUnitV2TrainWrapper: frozen tokenizer sees the full POYO window and all session units."
        )

    wrapper = wrapper_cls(
        cfg=cfg,
        model=model,
        modality_spec=readout_spec,
        tokenizer_mode=tokenizer_mode,
        unit_embedding_mode=cfg.tokenizer.unit_embedding_mode,
        sentinels=cfg.tokenizer.sentinels,
        ntt_tokenizer=ntt_tokenizer,
    )

    # --- Init from checkpoint (calibration / fine-tune) ---
    if init_ckpt is not None:
        _load_init_ckpt(
            wrapper=wrapper,
            cfg=cfg,
            model=model,
            ntt_tokenizer=ntt_tokenizer,
            unit_ids=unit_ids,
            session_ids=session_ids,
            init_ckpt=init_ckpt,
        )

    _register_global_slot_proxies(cfg, model, ntt_tokenizer)
    _apply_freeze_mode(cfg, wrapper)

    # --- DataLoaders ---
    if not multi_session:
        assert train_ds is not None and eval_ds is not None
        train_loader, val_loaders, test_loaders = _build_single_session_loaders(
            cfg=cfg,
            model=model,
            train_ds=train_ds,
            eval_ds=eval_ds,
            eval_only=eval_only,
        )

    # --- Callbacks ---
    stitch_evaluator = SafeDecodingStitchEvaluator(
        session_ids=session_ids,
        modality_spec=readout_spec,
    )

    loggers = _build_loggers(cfg, tokenizer_mode)

    world_size = int(cfg.gpus) * int(cfg.num_nodes)
    schedule = _resolve_trainer_schedule(cfg, world_size)

    has_validation = bool(val_loaders)
    if not has_validation:
        logger.warning(
            "No validation loaders configured; disabling monitored "
            "checkpointing and using the last checkpoint for any post-fit test."
        )

    callbacks = [
        stitch_evaluator,
        ModelSummary(max_depth=2),
        _build_ckpt_callback(schedule, has_validation),
        LearningRateMonitor(logging_interval="step"),
        tbrain_callbacks.MemInfo(),
        tbrain_callbacks.EpochTimeLogger(),
    ]
    if multi_session and (val_tiers or test_tiers):
        # Registered after DecodingStitchEvaluator so the per-session R²s are
        # present in trainer.callback_metrics by the time this fires.
        callbacks.append(TierAggregator(val_tiers=val_tiers, test_tiers=test_tiers))

    # results record: MetricSnapshotCallback must be registered on the trainer so
    # the post-fit `_write_decoding_results` can recover best/last/test snapshots.
    callbacks.append(
        MetricSnapshotCallback(
            monitor="average_val_metric" if has_validation else None,
            mode="max",
        )
    )

    trainer = _build_trainer(cfg, callbacks=callbacks, loggers=loggers, schedule=schedule)

    if multi_session:
        scope = (
            f"sessions={len(session_ids)} | brainsets="
            f"{list(cfg.restrict_brainsets) if cfg.restrict_brainsets else 'all'}"
        )
    else:
        scope = f"session={cfg.session}"
    logger.info(
        f"POYO | {scope} | mode={tokenizer_mode} | "
        f"unit_emb={cfg.tokenizer.unit_embedding_mode} | sentinels={cfg.tokenizer.sentinels}"
    )
    logger.info(f"Training: {len(unit_ids)} units across {len(session_ids)} sessions")

    if eval_only:
        # n_trials=0 (plain eval) leg of the calibration sweep: evaluate the loaded model on
        # the new session without any optimisation. We still run
        # trainer.test() at the end so the test-set metric ends up in
        # callback_metrics for ``_save_results`` to pick up.
        logger.info("eval_only=true → running validate() + test() without fit()")
        val_arg = _maybe_loaders(val_loaders)
        if val_arg is not None:
            trainer.validate(wrapper, val_arg)
        test_arg = _maybe_loaders(test_loaders)
        if test_arg is not None:
            trainer.test(wrapper, test_arg)
    else:
        trainer.fit(wrapper, train_loader, _maybe_loaders(val_loaders))
        test_arg = _maybe_loaders(test_loaders)
        if test_arg is not None:
            trainer.test(wrapper, test_arg, ckpt_path="best" if has_validation else "last")

    _persist_results(trainer, cfg, train_ds=train_ds, session_ids=session_ids)


def _persist_results(trainer, cfg: DictConfig, *, train_ds=None, session_ids: list[str] | None = None) -> None:
    """Write the run's results (rank 0 only): the results record, then ``run_result.json``
    (the local per-run summary used by sweep resume and the calibration CSV).

    Rank zero only: every rank would otherwise write the same per-run files
    concurrently. The results record is the contract the cohort record and
    downstream readers consume, so failing to write it fails the run: a cell
    that exits 0 without its record looks successful but silently drops out of
    the cohort. ``run_result.json`` is then not written either, because
    calibrate's ``--missing-only`` resume treats it as "cell done" and would
    never rerun the cell.
    """
    if not trainer.is_global_zero:
        return
    try:
        path = _write_decoding_results(trainer, cfg, session_ids=session_ids)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"failed to write the results record: {exc!r}") from exc
    if path is None:
        # write_results logs a write failure as a warning and returns None.
        raise RuntimeError("failed to write the results record; see the warning above")
    _save_results(trainer, cfg, train_ds=train_ds)


def _save_results(trainer: L.Trainer, cfg: DictConfig, *, train_ds=None) -> None:
    """Collect metrics from trainer and save structured results."""
    metrics = trainer.callback_metrics

    test_r2 = metrics.get("average_test_metric")
    if test_r2 is not None:
        test_r2 = float(test_r2)

    # ``average_val_metric`` in callback_metrics is the *most recent* logged
    # value (Lightning replaces the entry every validation epoch), which is
    # what the calibration CSV wants ("last validation metric"). Keep it under
    # ``val_last_r2``.
    val_last_r2 = metrics.get("average_val_metric")
    if val_last_r2 is not None:
        val_last_r2 = float(val_last_r2)
    val_best_r2 = val_last_r2

    # ModelCheckpoint, when monitoring "average_val_metric", tracks the best
    # score across the whole run; that's what we report as ``val_best_r2``.
    # With eval_only=True there's no checkpoint callback firing, so this loop
    # is a no-op and val_best_r2 falls back to the last (== only) validation
    # metric.
    for cb in _trainer_callbacks(trainer):
        if isinstance(cb, ModelCheckpoint) and cb.best_model_score is not None:
            val_best_r2 = float(cb.best_model_score)
            break

    if test_r2 is None and val_last_r2 is None:
        logger.warning("No test or val metric found in callback_metrics — skipping results save.")
        return
    if test_r2 is None:
        # eval_only / fit-only modes that don't run a test loader.
        test_r2 = float("nan")

    # Get model param count from run_info.json if available, else recount.
    assert trainer.model is not None, "trainer has no model attached"
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)

    # Read token_dims_per_sec from run_info.json if saved during on_fit_start.
    token_dims_per_sec = None
    metric_schema_version = None
    log_dir = trainer.log_dir or trainer.default_root_dir or str(cfg.log_dir)
    run_info_path = Path(log_dir) / "run_info.json"
    if run_info_path.exists():
        import json as _json

        with open(run_info_path) as f:
            run_info = _json.load(f)
        token_dims_per_sec = run_info.get("token_dims_per_sec")
        metric_schema_version = run_info.get("metric_schema_version")

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_dict, dict)

    result = build_run_result(
        cfg=cast(dict[str, Any], cfg_dict),
        test_r2=test_r2,
        val_best_r2=val_best_r2,
        model_params=trainable_params,
        token_dims_per_sec=token_dims_per_sec,
        log_dir=str(cfg.log_dir),
    )
    # Tack on the calibration-CSV-relevant fields. ``build_run_result``
    # returns a plain dict and ``save_run_result`` writes it as-is.
    result["val_last_r2"] = val_last_r2
    result["init_ckpt"] = cfg.get("init_ckpt")
    result["freeze_mode"] = cfg.get("freeze_mode") or "none"
    result["eval_only"] = bool(cfg.get("eval_only", False))
    # Propagate the metric schema version so cross-run plots can tell which
    # definition of ``token_dims_per_sec`` a run used (2 = mask-driven).
    result["metric_schema_version"] = metric_schema_version

    # Decimation summary: requested vs kept trial counts. Empty dict when
    # train_decimation isn't active. Persisting it here lets calibration
    # plotters disambiguate budgets that silently collapsed onto the same
    # data (eligibility filter clamping requested → kept). Top-level
    # ``requested_trials`` / ``kept_trials`` are what the calibration CSV reads.
    decim_summary = collect_decimation_summary(train_ds) if train_ds is not None else {}
    if decim_summary:
        result["train_decimation"] = decim_summary
        result["requested_trials"] = decim_summary["requested_trials"]
        result["kept_trials"] = decim_summary["kept_trials_total"]
        result["train_decimation_clamped"] = decim_summary["clamped"]
        if decim_summary["clamped"]:
            logger.warning(
                "train_decimation: requested_trials=%d clamped to "
                "kept_trials=%d (per-rid: %s) — calibration plots should "
                "use kept_trials on the x-axis for this run.",
                decim_summary["requested_trials"],
                decim_summary["kept_trials_total"],
                decim_summary["kept_trials_by_recording"],
            )

    save_run_result(result, log_dir=log_dir)
    logger.info(f"Run result: test_r2={test_r2:.4f}, val_best_r2={val_best_r2}, val_last_r2={val_last_r2}")


def _write_decoding_results(
    trainer: L.Trainer, cfg: DictConfig, *, session_ids: list[str] | None = None
) -> Path | None:
    """Build and write the run's ``downstream_decoding_results.json`` (rank 0).

    Blocks: ``val_best`` / ``val_last`` (periodic-val snapshots) and ``test``
    (the val-best checkpoint on the test split, when a test phase ran). Written
    to the per-run Lightning version dir, next to ``run_result.json``: that is
    the tree the benchmark sweep walks to build its cohort record and to upload.

    ``session_ids`` is the canonical recording-id list from ``main()``; when not
    provided it is discovered from the stitch evaluator's ``metrics`` dict.
    """
    from lightning.pytorch.callbacks import ModelCheckpoint
    from omegaconf import OmegaConf

    from ntt_downstream.common.results.provenance import git_sha, hostname, now_iso, ntt_provenance
    from ntt_downstream.common.results.schema import ResultsDocument, RunMetadata
    from ntt_downstream.common.results.writer import write_results
    from ntt_downstream.decoding.backends.poyo.results_parser import parse_decoding_block

    snapshot_cb = next((cb for cb in _trainer_callbacks(trainer) if isinstance(cb, MetricSnapshotCallback)), None)
    if snapshot_cb is None:
        logger.warning("MetricSnapshotCallback not found; skipping the results record.")
        return

    # Fall back to discovering session ids from the stitch evaluator's `.metrics`
    # dict (keyed by recording id) only if the caller didn't pass them. Match the
    # evaluator by type so an unrelated callback exposing a `.metrics` attribute
    # can't shadow it.
    if session_ids is None:
        for cb in _trainer_callbacks(trainer):
            if isinstance(cb, SafeDecodingStitchEvaluator):
                ev_metrics = getattr(cb, "metrics", None)
                if isinstance(ev_metrics, dict) and ev_metrics:
                    session_ids = [str(k) for k in ev_metrics]
                    break
        if session_ids is None:
            logger.warning(
                "results record: no canonical session ids passed and none discovered "
                "from the stitch evaluator; falling back to inferring them from the "
                "logged metric keys."
            )

    last_snap = snapshot_cb.last_val or {}
    best_snap = snapshot_cb.best_val or last_snap
    test_snap = snapshot_cb.test or {}

    metrics = {
        "val_best": parse_decoding_block(best_snap, session_ids=session_ids, prefix="val"),
        "val_last": parse_decoding_block(last_snap, session_ids=session_ids, prefix="val"),
    }
    if test_snap:
        metrics["test"] = parse_decoding_block(test_snap, session_ids=session_ids, prefix="test")

    checkpoint: dict = {}
    for cb in _trainer_callbacks(trainer):
        if isinstance(cb, ModelCheckpoint) and cb.monitor == "average_val_metric":
            score = cb.best_model_score
            checkpoint = {
                "best_path": cb.best_model_path or None,
                "best_score": float(score) if score is not None else None,
                "best_epoch": snapshot_cb.best_epoch,
                "best_step": snapshot_cb.best_step,
                "last_path": getattr(cb, "last_model_path", "") or None,
                "monitor": cb.monitor,
                "mode": cb.mode,
            }
            break

    mlflow_logger = next(
        (lg for lg in (trainer.loggers or []) if isinstance(lg, MLFlowLogger)),
        None,
    )

    log_dir = trainer.log_dir or trainer.default_root_dir or str(cfg.log_dir)
    sessions = sorted(session_ids) if session_ids else []
    metadata = RunMetadata(
        git_sha=git_sha(),
        timestamp=now_iso(),
        hostname=hostname(),
        run_name=getattr(mlflow_logger, "run_name", None) if mlflow_logger else None,
        mlflow_run_id=getattr(mlflow_logger, "run_id", None) if mlflow_logger else None,
        config=cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
        checkpoint=checkpoint,
        sessions=sessions,
        n_sessions=len(sessions),
        tokenizer_checkpoint=(cfg.get("tokenizer") or {}).get("ntt_checkpoint"),
        **cast(dict[str, Any], ntt_provenance()),
    )
    doc = ResultsDocument(task="decoding", kind="run", metadata=metadata, metrics=metrics)
    return write_results(doc, log_dir, loggers=trainer.loggers)


if __name__ == "__main__":
    main()
