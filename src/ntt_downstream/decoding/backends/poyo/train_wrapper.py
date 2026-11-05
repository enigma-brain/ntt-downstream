"""Lightning training wrapper for POYO with ntt tokenizer bypass.

Adapted from the POYO training example in torch_brain
(https://github.com/neuro-galaxy/torch_brain, ``examples/poyo/train.py``),
Apache-2.0, Copyright 2025 NeuroGalaxy.
Modified for ntt-downstream: routes tokenization through the embedding-bypass
path so ntt tokenizers can replace POYO's per-spike input tokens.

Tokenization modes:
- "fake": POYO's tokenize() runs normally (sentinels included).
  Content = token_type_emb(token_type), unit_ids = input_unit_index.
  Adapter mode=add produces unit_emb + token_type_emb = the original POYO
  input computation exactly.
- "ntt": real tokenizers. Raw spikes → ntt tokenizer (GPU) → adapter.
  Sentinels added by adapter.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import lightning as L
import torch
import torch.nn as nn
from ntt.core.components.types import BinnedSpikesInput, SpikeTimesInput
from ntt.evaluation.utils.window_chunking import requires_fixed_input_window
from omegaconf import DictConfig, OmegaConf
from torch_brain.optim import SparseLamb
from torch_brain.registry import ModalitySpec
from torch_brain.utils.callbacks import DataForDecodingStitchEvaluator

from ntt_downstream.decoding.backends.poyo.adapter import adapt
from ntt_downstream.decoding.backends.poyo.poyo import POYO
from ntt_downstream.decoding.backends.poyo.tokenizer_output import TokenizerOutput

if TYPE_CHECKING:
    from ntt_downstream.decoding.backends.poyo.frozen_ntt_tokenizer import FrozenNTTTokenizer

logger = logging.getLogger(__name__)


def _build_input_proj(in_dim: int, out_dim: int) -> nn.Module:
    """Build the tokenizer→model input projection.

    1-hidden-layer MLP with GELU, matching POYO's FFN convention. Hidden
    width is set to out_dim (cheap; ~2x the params of a plain Linear).
    """
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.GELU(),
        nn.Linear(out_dim, out_dim),
    )


def _session_id_field(batch: dict, batch_size: int) -> tuple[str, ...]:
    """Build the ``BinnedSpikesInput.session_id`` tuple from a POYO batch.

    Tokenizers with session-keyed lookups (e.g. Perceiver's per-session
    embedding table) need the actual id; session-agnostic ones ignore it.
    Falls back to ``"dummy"`` per sample when the batch carries no
    ``session_id`` (synthetic test inputs).
    """
    sids = batch.get("session_id")
    if sids is None:
        return ("dummy",) * batch_size
    return tuple(str(s) for s in sids)


def _clamp_output_timestamps(out: TokenizerOutput, window_end: float) -> TokenizerOutput:
    """Clamp tokenizer-emitted timestamps into ``[0, window_end]``.

    The (i+0.5)*token_bin centering inside ntt's ``BaseTokenizer`` can
    place the last token center within token_bin/2 of ``window_end``
    plus epsilon when ``_pad_time_if_needed`` rounds the binning matrix
    up. Clamping keeps the adapter's sentinel-bounds guard satisfied at
    the cost of moving a boundary token by ≤ token_bin/2 — within
    rotary-embedding noise.
    """
    return dataclasses.replace(out, timestamps=out.timestamps.clamp(min=0.0, max=window_end))


class V2TrainWrapper(L.LightningModule):
    """POYO training wrapper using the tokenizer bypass path.

    Feeds either POYO's own tokenization ("fake") or an ntt tokenizer's output
    ("ntt") into ``POYOWithEmbeddingBypass`` through the adapter.
    """

    # Map tokenizer_mode → bound-method name. `getattr(self, name)` resolves
    # against the subclass MRO at call time, so FullWindowUnitV2TrainWrapper's
    # `_ntt_frozen_tokenize` override is picked up automatically.
    _TOKENIZE_METHODS: dict[str, str] = {
        "fake": "_fake_tokenize",
        "ntt": "_ntt_tokenize",
        "ntt_frozen": "_ntt_frozen_tokenize",
    }

    def __init__(
        self,
        cfg: DictConfig,
        model: nn.Module,
        modality_spec: ModalitySpec,
        tokenizer_mode: Literal["fake", "ntt", "ntt_frozen"] = "fake",
        unit_embedding_mode: Literal["none", "add", "concat"] = "add",
        sentinels: bool = False,
        ntt_tokenizer: nn.Module | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        # Declare the concrete POYO type: nn.Module.__getattr__ would otherwise
        # widen every `self.model.<attr>` to `Tensor | Module`.
        self.model = cast(POYO, model)
        self.modality_spec = modality_spec
        self.tokenizer_mode = tokenizer_mode
        self.unit_embedding_mode: Literal["none", "add", "concat"] = unit_embedding_mode
        self.use_sentinels = sentinels
        if tokenizer_mode in ("ntt", "ntt_frozen"):
            assert ntt_tokenizer is not None, f"ntt_tokenizer must be provided when tokenizer_mode='{tokenizer_mode}'"
        self.ntt_tokenizer = ntt_tokenizer

        # Latched once the run-info probe has fired on a real batch (training or
        # validation). Probing is deferred from on_fit_start to the first batch
        # so the sampler's generator state isn't perturbed and DataLoader
        # prefetch isn't wasted — see on_train_batch_start.
        self._run_info_logged = False

        # combines_units=True tokenizers (population / abstract-latent): the
        # adapter emits per-slot proxy unit_ids (slot k → first M input
        # unit_ids), so each (session, latent_slot) gets its own learnable
        # entry in unit_emb — mirroring the per-(session,unit) embeddings used
        # by spike_times and per-unit tokenizers. unit_identification FT
        # therefore works for these tokenizers too.

        target_dim = self.model.dim

        # Detect embedding-mode and build code_emb tables if needed.
        self.code_emb: nn.ModuleList | None = None
        is_embedding_mode = (
            tokenizer_mode == "ntt_frozen"
            and ntt_tokenizer is not None
            and getattr(ntt_tokenizer, "discrete_input_mode", "z_q") == "embedding"
        )
        if is_embedding_mode:
            # is_embedding_mode implies a FrozenNTTTokenizer (see its definition above).
            frozen_tok = cast("FrozenNTTTokenizer", ntt_tokenizer)
            assert frozen_tok.is_discrete, "internal: embedding mode requires a discrete-bottleneck tokenizer"
            code_embs = [nn.Embedding(frozen_tok.vocab_size, target_dim) for _ in range(frozen_tok.num_codebooks)]
            self.code_emb = nn.ModuleList(code_embs)
            for emb in code_embs:
                nn.init.normal_(emb.weight, std=0.02)
            n_params = sum(p.numel() for p in self.code_emb.parameters())
            logger.info(
                "Built code_emb: Q=%d tables x K=%d x D=%d (%d params)",
                frozen_tok.num_codebooks,
                frozen_tok.vocab_size,
                target_dim,
                n_params,
            )
            # input_proj is intentionally bypassed in embedding mode — code_emb
            # already produces target_dim-sized vectors.
            self.input_proj = None
        else:
            d_tok = self._infer_tokenizer_output_dim(tokenizer_mode, ntt_tokenizer, target_dim)
            # Guard against the -1 sentinel from FrozenNTTTokenizer.output_dim:
            # if we got here we're in z_q (continuous) mode and need a positive
            # D to size input_proj. Without this check the next branch would
            # call nn.Linear(-1, target_dim) and raise an opaque RuntimeError.
            if d_tok <= 0:
                tok_name = type(ntt_tokenizer).__name__ if ntt_tokenizer is not None else "<None>"
                raise ValueError(
                    f"Tokenizer output_dim probe failed for {tok_name} "
                    f"(returned {d_tok}). In z_q (continuous) mode the wrapper "
                    f"needs a positive D to build input_proj. Fix by either: "
                    f"(a) setting spec.z_dim or spec.embed_dim on the tokenizer, "
                    f"or (b) ensuring make_dummy_input(spec) produces an input "
                    f"that encode_with_timestamps() accepts. Embedding mode "
                    f"(discrete_input_mode='embedding') bypasses this entirely."
                )
            # Only build a projection when we actually need to change the dim.
            # For the spike_times path (D_tok == target_dim, mode in {none, add})
            # the projection is disabled to preserve the exact unit_emb + token_type_emb
            # formula from the POYO paper.
            if unit_embedding_mode == "concat":
                self.input_proj = _build_input_proj(d_tok + target_dim, target_dim)
            elif d_tok != target_dim:
                self.input_proj = _build_input_proj(d_tok, target_dim)
            else:
                self.input_proj = None

        assert isinstance(cfg, DictConfig), (
            f"cfg must be an OmegaConf DictConfig (from Hydra), got {type(cfg).__name__}"
        )
        self.save_hyperparameters(OmegaConf.to_container(cfg))

    @staticmethod
    def _infer_tokenizer_output_dim(
        tokenizer_mode: str,
        ntt_tokenizer: nn.Module | None,
        target_dim: int,
    ) -> int:
        """Return D_tok (tokenizer feature dim before projection)."""
        if tokenizer_mode == "fake":
            return target_dim  # _fake_tokenize uses token_type_emb (dim = model.dim)
        if tokenizer_mode == "ntt":
            # "ntt" tokenizers expose `token_dim`; nn.Module's __getattr__ can't see it.
            return cast(Any, ntt_tokenizer).token_dim
        if tokenizer_mode == "ntt_frozen":
            return cast("FrozenNTTTokenizer", ntt_tokenizer).output_dim
        raise ValueError(f"Unknown tokenizer_mode='{tokenizer_mode}'")

    def _tokenize(self, batch: dict) -> TokenizerOutput:
        """Dispatch to the tokenize method selected by ``self.tokenizer_mode``."""
        method_name = self._TOKENIZE_METHODS.get(self.tokenizer_mode)
        if method_name is None:
            raise ValueError(
                f"Unknown tokenizer_mode='{self.tokenizer_mode}'. Supported: {sorted(self._TOKENIZE_METHODS)}."
            )
        return getattr(self, method_name)(batch)

    def on_train_batch_start(self, batch, batch_idx):
        """Probe token stats on the first training batch we see.

        The probe runs here rather than in on_fit_start because calling
        ``next(iter(train_loader))`` there would (a) advance the sampler
        generator's RNG state and (b) waste ``2*num_workers`` of prefetch via
        persistent_workers. On the actual first batch no extra iterators are
        created and no loads are wasted.
        """
        if not self._run_info_logged:
            self._log_run_info(batch)

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        """Fallback for eval_only runs that never enter training.

        ``trainer.validate()`` / ``trainer.test()`` never fire
        ``on_train_batch_start``; without this hook, eval_only runs would
        leave ``run_info.json`` unwritten.
        """
        del batch_idx, dataloader_idx
        if not self._run_info_logged:
            self._log_run_info(batch)

    def _log_run_info(self, batch: dict) -> None:
        """Compute and persist run-level token stats. Idempotent.

        Latches ``_run_info_logged`` first so any failure in the probe
        doesn't loop on every subsequent batch — we'd rather lose the
        metric than tax the hot path.
        """
        self._run_info_logged = True

        # --- Trainable param count ---
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        with torch.no_grad():
            tok_out = self._tokenize(batch)

        # token_dim is the *effective* per-token feature width fed into POYO.
        # For float tokens it's the trailing tensor axis (D); for int64
        # codebook indices it's `model.dim` because the wrapper's `code_emb`
        # lookup expands [B, N, Q] → [B, N, model.dim] before POYO sees it.
        # Compute it before `token_dims_per_sec` so the saved
        # `tokens_per_second * token_dim == token_dims_per_sec` invariant
        # holds in both branches (the discrete path must not overwrite
        # `token_dim` after the multiplication).
        if torch.is_floating_point(tok_out.tokens):
            token_dim = int(tok_out.tokens.shape[-1])
        else:
            token_dim = int(self.model.dim)

        combines_units = tok_out.combines_units

        # --- Tokens per second ---
        # Single mask-driven path for both fixed-rate and dynamic
        # tokenizers. `tok_out.mask` is already flattened to [B, N_total]
        # in both per-unit (T'×U) and population (T') outputs, so its
        # row-sum is always the per-sample valid-token count regardless
        # of tokenizer family. Scaling per-unit outputs by an extra
        # `1/token_bin_s` would double-count by a factor of T' (the mask sum
        # already has T' baked in). See `metric_schema_version` below.
        valid_counts = tok_out.mask.float().sum(dim=1)  # [B]
        if tok_out.token_bin_size_s is not None:
            # Fixed-rate tokenizer: window length is exactly the model's
            # `sequence_length` (the sampler windows the data to this span).
            duration = float(self.model.sequence_length)
            tokens_per_second = (valid_counts / duration).mean().item()
        else:
            # Dynamic tokenizer (spike_times, fake): infer per-sample span
            # from valid timestamps. Min/max-mask via finite sentinels so
            # an empty sample ends up with a zero-width range and gets
            # clamped, instead of poisoning the batch mean with NaN.
            ts = tok_out.timestamps
            mask = tok_out.mask
            ts_lo = ts.masked_fill(~mask, float("inf"))
            ts_hi = ts.masked_fill(~mask, float("-inf"))
            t_min = ts_lo.min(dim=1).values
            t_max = ts_hi.max(dim=1).values
            durations = (t_max - t_min).clamp(min=0.1)  # ≥0.1 s floor
            tokens_per_second = (valid_counts / durations).mean().item()

        # --- Session ID ---
        session_id = batch.get("session_id", ["unknown"])[0]
        if isinstance(session_id, (list, tuple)):
            session_id = session_id[0]

        token_dims_per_sec = tokens_per_second * token_dim

        # metric_schema_version 2: `tokens_per_second` / `token_dims_per_sec`
        # are mask-driven (single path for all tokenizer families).
        run_info = {
            "metric_schema_version": 2,
            "trainable_params": trainable_params,
            "token_dim": token_dim,
            "tokens_per_second": tokens_per_second,
            "token_dims_per_sec": token_dims_per_sec,
            "combines_units": combines_units,
            "session": str(session_id),
            "tokenizer_mode": self.tokenizer_mode,
            "unit_embedding_mode": self.unit_embedding_mode,
            "sentinels": self.use_sentinels,
        }

        if not torch.is_floating_point(tok_out.tokens):
            run_info["discrete_input_mode"] = "embedding"
            run_info["num_codebooks"] = int(tok_out.tokens.shape[-1])
            run_info["vocab_size"] = int(tok_out.vocab_sizes[0]) if tok_out.vocab_sizes else None
        elif self.tokenizer_mode == "ntt_frozen" and self.ntt_tokenizer is not None:
            run_info["discrete_input_mode"] = getattr(self.ntt_tokenizer, "discrete_input_mode", "z_q")

        logger.info(f"Run info: {run_info}")

        # Rank-zero only for log + JSON write: WandbLogger's log_hyperparams
        # already self-gates, but writing run_info.json from every rank races
        # on the same path under shared filesystems. Keep the *computation*
        # above on every rank so each rank's logger.info trace shows the
        # token stats, but only rank zero persists artifacts.
        if self.trainer.is_global_zero:
            # Log to Lightning logger (wandb if available)
            if self.logger is not None:
                self.logger.log_hyperparams(run_info)

            # Save to JSON file
            log_dir = self.trainer.log_dir or self.trainer.default_root_dir
            if log_dir is not None:
                log_path = Path(log_dir)
                log_path.mkdir(parents=True, exist_ok=True)
                info_path = log_path / "run_info.json"
                with open(info_path, "w") as f:
                    json.dump(run_info, f, indent=2)
                logger.info(f"Saved run_info.json to {info_path}")

    def _fake_tokenize(self, batch: dict) -> TokenizerOutput:
        """Build a TokenizerOutput from POYO's own tokenize() output.

        Content = token_type_emb(token_type), so that adapter mode=add gives:
        embeddings = token_type_emb(token_type) + unit_emb(unit_index)
        which exactly equals POYO's `inputs` computation.

        Sentinels are already included by POYO's tokenize() as tokens with
        token_type=1 (START) and token_type=2 (END).
        """
        mi = batch["model_inputs"]
        input_token_type = mi["input_token_type"]  # [B, N]
        input_unit_index = mi["input_unit_index"]  # [B, N]
        input_timestamps = mi["input_timestamps"]  # [B, N]
        input_mask = mi["input_mask"]  # [B, N]

        # Content = token_type_emb lookup.
        tokens = self.model.token_type_emb(input_token_type)  # [B, N, D]

        return TokenizerOutput(
            tokens=tokens,
            timestamps=input_timestamps,
            unit_ids=input_unit_index,
            mask=input_mask,
            combines_units=False,
            token_bin_size_s=None,
        )

    def _ntt_tokenize(self, batch: dict) -> TokenizerOutput:
        """NTT spike-times mode: extract raw spikes from POYO batch → ntt tokenizer.

        POYO's tokenize() runs during data loading, producing tokens with
        token_type: 0=DEFAULT (spikes), 1=START sentinel, 2=END sentinel.
        We extract the spike tokens (type==0), build a SpikeTimesInput,
        and run the frozen ntt tokenizer on GPU.
        """
        mi = batch["model_inputs"]
        input_token_type = mi["input_token_type"]  # [B, N]
        input_unit_index = mi["input_unit_index"]  # [B, N]
        input_timestamps = mi["input_timestamps"]  # [B, N]
        input_mask = mi["input_mask"]  # [B, N]

        # DEFAULT token type = 0 (spikes). Sentinels are START=1, END=2.
        is_spike = (input_token_type == 0) & input_mask  # [B, N]

        B = input_token_type.shape[0]

        # Extract per-sample spike data and pad to common length.
        spike_ts_list: list[torch.Tensor] = []
        spike_uid_list: list[torch.Tensor] = []
        for b in range(B):
            mask_b = is_spike[b]  # [N]
            spike_ts_list.append(input_timestamps[b][mask_b])
            spike_uid_list.append(input_unit_index[b][mask_b])

        n_max = max(ts.shape[0] for ts in spike_ts_list)
        n_max = max(n_max, 1)  # at least 1 to avoid empty tensors

        device = input_timestamps.device
        spike_timestamps = torch.zeros(B, n_max, device=device, dtype=torch.float32)
        spike_unit_index = torch.zeros(B, n_max, device=device, dtype=torch.long)
        spike_pad_mask = torch.zeros(B, n_max, device=device, dtype=torch.long)

        for b in range(B):
            n_b = spike_ts_list[b].shape[0]
            if n_b > 0:
                spike_timestamps[b, :n_b] = spike_ts_list[b]
                spike_unit_index[b, :n_b] = spike_uid_list[b]
                spike_pad_mask[b, :n_b] = 1

        # Build SpikeTimesInput with dummy metadata fields. NTTSpikeTimesTokenizer
        # ignores unit_ids today (so this field is currently inert), but the
        # SpikeTimesInput contract is per-sample identity — see
        # ntt.core.model.tokenizer.validation.make_dummy_input which uses
        # `arange(n_neurons)` per sample. A batch-global unique-uid set
        # broadcast to every sample would inflate narrow samples' unit count
        # and replace per-sample identity with a global pool, so the set is
        # built per sample for any spike-times tokenizer that consumes it.
        unique_per_sample = [torch.unique(spike_unit_index[b][spike_pad_mask[b].bool()]) for b in range(B)]
        n_max_units = max(max(u.shape[0] for u in unique_per_sample), 1)
        unit_ids_padded = torch.zeros(B, n_max_units, device=device, dtype=torch.long)
        unit_ids_mask = torch.zeros(B, n_max_units, device=device, dtype=torch.long)
        for b, u in enumerate(unique_per_sample):
            n_u = u.shape[0]
            if n_u > 0:
                unit_ids_padded[b, :n_u] = u
                unit_ids_mask[b, :n_u] = 1

        spike_input = SpikeTimesInput(
            spike_unit_index=spike_unit_index,
            spike_timestamps=spike_timestamps,
            spike_pad_mask=spike_pad_mask,
            unit_ids=unit_ids_padded,
            unit_ids_pad_mask=unit_ids_mask,
            duration_sec=torch.ones(B, device=device),  # dummy
            session_id=("dummy",) * B,
            subject_id=("dummy",) * B,
            species=("dummy",) * B,
        )

        assert self.ntt_tokenizer is not None
        return self.ntt_tokenizer(spike_input)

    def _ntt_frozen_tokenize(self, batch: dict) -> TokenizerOutput:
        """NTT frozen-checkpoint mode: extract spikes, bin them, run frozen tokenizer.

        Extracts spike tokens from POYO batch (filtering sentinels), bins them
        at the frozen tokenizer's input_bin_size_ms, builds a BinnedSpikesInput,
        and calls the frozen ntt tokenizer.
        """
        from ntt_downstream.decoding.backends.poyo.frozen_ntt_tokenizer import FrozenNTTTokenizer  # noqa: F811

        assert isinstance(self.ntt_tokenizer, FrozenNTTTokenizer), (
            f"Expected FrozenNTTTokenizer, got {type(self.ntt_tokenizer).__name__}"
        )

        # Fixed-window specs (variable_t=False) must see a deterministic
        # [0, sequence_length] raster so the chunking inside the frozen
        # tokenizer's forward aligns to its native window. Variable-T specs
        # keep the spike-derived raster.
        spec = self.ntt_tokenizer.tokenizer.spec
        fixed_window = requires_fixed_input_window(spec)

        binned_input = self._bin_spikes(batch, fixed_window=fixed_window)

        out = self.ntt_tokenizer(binned_input)
        return _clamp_output_timestamps(out, float(self.model.sequence_length))

    def _bin_spikes(self, batch: dict, *, fixed_window: bool) -> BinnedSpikesInput:
        """Bin POYO spike tokens into a ``BinnedSpikesInput``.

        Extracts spike tokens (filtering sentinels and pad), bins them at the
        frozen tokenizer's ``input_bin_size_ms``, and assembles the per-sample
        spike matrix, pad mask, unit ids, and window bounds.

        When ``fixed_window`` (the spec cannot handle variable time lengths),
        the raster is pinned to ``[0, sequence_length]``: ``t_min=0`` and
        ``n_bins = round(sequence_length / bin_s)`` for every sample, so chunk
        boundaries align to the tokenizer's native window. ``has_any`` stays
        spike-derived so empty samples still get an all-zero pad mask. When not
        ``fixed_window``, the raster spans each sample's own ``[t_min, t_max]``
        (variable path, unchanged).
        """
        mi = batch["model_inputs"]
        input_token_type = mi["input_token_type"]  # [B, N]
        input_unit_index = mi["input_unit_index"]  # [B, N]
        input_timestamps = mi["input_timestamps"]  # [B, N]
        input_mask = mi["input_mask"]  # [B, N]

        # DEFAULT token type = 0 (spikes). Sentinels are START=1, END=2.
        is_spike = (input_token_type == 0) & input_mask  # [B, N]

        B = input_token_type.shape[0]
        device = input_timestamps.device

        bin_size_ms = cast("FrozenNTTTokenizer", self.ntt_tokenizer).input_bin_size_ms
        bin_size_s = bin_size_ms / 1000.0
        sampling_rate = 1000.0 / bin_size_ms

        # Vectorized binning over the whole batch (empty samples, all-sentinel
        # samples, single spikes, scatter collisions and boundary timestamps
        # are covered by tests). Two host syncs only (max_T, max_U).

        # has_any is spike-derived in both rasters (empty samples still get an
        # all-zero pad mask via the has_any factor on pad_and_gap_mask below),
        # so compute it once here rather than per arm.
        has_any = is_spike.any(dim=1)  # [B]

        if fixed_window:
            # Deterministic [0, sequence_length] raster so chunk boundaries
            # align to the tokenizer's native window. t_min is pinned to 0
            # (window start); n_bins is fixed for every sample.
            seq_len = float(self.model.sequence_length)
            n_bins_fixed = round(seq_len / bin_size_s)
            t_min = torch.zeros(B, dtype=input_timestamps.dtype, device=device)
            n_bins_per_b = torch.full((B,), n_bins_fixed, dtype=torch.long, device=device)
            max_T = n_bins_fixed
        else:
            # Per-sample (t_min, t_max) via masked min/max.
            INF = torch.finfo(input_timestamps.dtype).max
            ts_min_src = torch.where(is_spike, input_timestamps, input_timestamps.new_full((), INF))
            ts_max_src = torch.where(is_spike, input_timestamps, input_timestamps.new_full((), -INF))
            zero_t = torch.zeros((), dtype=input_timestamps.dtype, device=device)
            t_min = torch.where(has_any, ts_min_src.min(dim=1).values, zero_t)
            t_max = torch.where(has_any, ts_max_src.max(dim=1).values, zero_t)

            # n_bins in float64 so the bin count is exact for large windows.
            window_dur64 = (t_max.to(torch.float64) - t_min.to(torch.float64)) + bin_size_s
            n_bins_per_b = (window_dur64 / bin_size_s).long().clamp_min(1)
            n_bins_per_b = torch.where(has_any, n_bins_per_b, torch.ones_like(n_bins_per_b))
            max_T = int(n_bins_per_b.max().item())

        # bin_idx in float32 (scalar t_min cast preserves tensor dtype). Per-sample
        # clamp matches the loop's `clamp(0, n_bins - 1)`.
        bin_idx = ((input_timestamps - t_min[:, None]) / bin_size_s).long()
        bin_idx = bin_idx.clamp(min=0).clamp(max=(n_bins_per_b - 1)[:, None])

        # Per-sample unique-unit remap via stable sort + first-occurrence run mask.
        # Output unit_ids dtype is int64.
        uid64 = input_unit_index.to(torch.long)
        BIG = torch.iinfo(uid64.dtype).max
        uid_key = torch.where(is_spike, uid64, uid64.new_full((), BIG))
        sorted_uid, perm = uid_key.sort(dim=1, stable=True)
        valid_sorted = sorted_uid != BIG
        first_occ = torch.ones_like(sorted_uid, dtype=torch.bool)
        first_occ[:, 1:] = sorted_uid[:, 1:] != sorted_uid[:, :-1]
        contributes = first_occ & valid_sorted  # [B, N]
        rank_sorted = (contributes.cumsum(dim=1) - 1).clamp_min(0)
        remap = torch.zeros_like(uid64)
        remap.scatter_(1, perm, rank_sorted)

        n_units_per_b = contributes.sum(dim=1).clamp_min(1)
        n_units_per_b = torch.where(has_any, n_units_per_b, torch.ones_like(n_units_per_b))
        max_U = int(n_units_per_b.max().item())

        # Single batched scatter_add into a flat [B*max_T*max_U + 1] buffer; the
        # extra slot absorbs invalid (~is_spike) entries and is dropped after.
        b_offset = torch.arange(B, device=device).unsqueeze(1) * (max_T * max_U)
        flat_idx = b_offset + bin_idx * max_U + remap
        flat_idx = torch.where(is_spike, flat_idx, torch.full_like(flat_idx, B * max_T * max_U))
        contrib = is_spike.to(torch.int32)
        flat = torch.zeros(B * max_T * max_U + 1, device=device, dtype=torch.int32)
        flat.scatter_add_(0, flat_idx.reshape(-1), contrib.reshape(-1))
        spike_matrix = flat[:-1].view(B, max_T, max_U)

        # pad_and_gap_mask: True where (t < n_bins_per_b) & (u < n_units_per_b)
        # & (sample is non-empty). The has_any factor matches the loop's
        # all-zero pad mask for empty samples.
        t_valid = torch.arange(max_T, device=device)[None, :] < n_bins_per_b[:, None]
        u_valid = torch.arange(max_U, device=device)[None, :] < n_units_per_b[:, None]
        pad_and_gap_mask = (t_valid[:, :, None] & u_valid[:, None, :] & has_any[:, None, None]).to(torch.int32)

        # unit_ids: write sorted_uid at first-occurrence positions to column
        # rank_sorted; invalid writes go to a junk slot dropped after.
        unit_ids_flat = torch.zeros(B * max_U + 1, device=device, dtype=torch.long)
        flat_col = torch.where(
            contributes,
            torch.arange(B, device=device).unsqueeze(1) * max_U + rank_sorted,
            torch.full_like(rank_sorted, B * max_U),
        )
        src = torch.where(contributes, sorted_uid, torch.zeros_like(sorted_uid))
        unit_ids_flat.scatter_(0, flat_col.reshape(-1), src.reshape(-1))
        unit_ids = unit_ids_flat[:-1].view(B, max_U)
        unit_ids_pad_mask = u_valid.long()

        # Anchor token timestamps to the spike positions, not to 0.
        # Tokenizers compute `token_timestamps = abs_start + (i+0.5)*token_bin`
        # (see BaseTokenizer.encode_with_timestamps in ntt). The bin matrix is
        # already shifted left by t_min via `bin_idx = (ts - t_min) / bin`, so
        # passing abs_start=t_min preserves the spike→token positional
        # correspondence. Passing abs_start=0 would collapse sparse-start
        # windows (e.g. spikes only in [0.4, 0.6]s of a 1.0s window) onto
        # rotary positions [0, 0.2], breaking cross-window position
        # consistency and placing START sentinels (pinned at window_start=0
        # by the adapter) coincident with the first content token. The shift stays within [0, sequence_length] for any
        # POYO-normalised window, so the adapter's bounds check still passes.
        # Both rasters anchor at t_min: the fixed-window arm pins t_min to
        # all-zeros, so t_min.to(float64) already equals the [0, n_bins*bin_s]
        # window the fixed branch needs — no separate arm required.
        abs_start_tensor = t_min.to(torch.float64)
        abs_end_tensor = abs_start_tensor + max_T * bin_size_s

        binned_input = BinnedSpikesInput(
            spike_matrix=spike_matrix,
            pad_and_gap_mask=pad_and_gap_mask,
            sampling_rate=sampling_rate,
            unit_ids=unit_ids,
            unit_ids_pad_mask=unit_ids_pad_mask,
            duration_sec=torch.full((B,), max_T * bin_size_s, device=device),
            session_id=_session_id_field(batch, B),
            subject_id=("dummy",) * B,
            species=("dummy",) * B,
            absolute_start=abs_start_tensor,
            absolute_end=abs_end_tensor,
        )

        return binned_input

    def _build_input_embeddings(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run tokenizer + adapter → (embeddings, timestamps, mask)."""
        tok_out = self._tokenize(batch)

        pre = adapt(
            tok_out,
            unit_emb=self.model.unit_emb,
            token_type_emb=self.model.token_type_emb,
            mode=self.unit_embedding_mode,
            sentinels=self.use_sentinels,
            target_dim=self.model.dim,
            input_proj=self.input_proj,
            code_emb=self.code_emb,
            # Explicit bounds so the adapter's sentinel-placement check has
            # the model's window contract instead of the adapter's default.
            # POYO's tokenize() normalises spike timestamps to
            # [0, sequence_length], so window_start=0.0 stays correct here.
            window_start=0.0,
            window_end=float(self.model.sequence_length),
        )
        return pre.embeddings, pre.timestamps, pre.mask

    def training_step(self, batch, batch_idx):
        input_embeddings, input_timestamps, input_mask = self._build_input_embeddings(batch)

        mi = batch["model_inputs"]
        output_values = self.model(
            input_embeddings=input_embeddings,
            input_timestamps=input_timestamps,
            input_mask=input_mask,
            latent_index=mi["latent_index"],
            latent_timestamps=mi["latent_timestamps"],
            output_session_index=mi["output_session_index"],
            output_timestamps=mi["output_timestamps"],
            output_mask=mi["output_mask"],
        )

        mask = mi["output_mask"]
        output_values = output_values[mask]
        target_values = batch["target_values"][mask]
        target_weights = batch["target_weights"][mask]

        loss = self.modality_spec.loss_fn(output_values, target_values, target_weights)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        del dataloader_idx  # all tier loaders flow through one stitch evaluator
        input_embeddings, input_timestamps, input_mask = self._build_input_embeddings(batch)

        mi = batch["model_inputs"]
        output_values = self.model(
            input_embeddings=input_embeddings,
            input_timestamps=input_timestamps,
            input_mask=input_mask,
            latent_index=mi["latent_index"],
            latent_timestamps=mi["latent_timestamps"],
            output_session_index=mi["output_session_index"],
            output_timestamps=mi["output_timestamps"],
            output_mask=mi["output_mask"],
        )

        return DataForDecodingStitchEvaluator(
            timestamps=mi["output_timestamps"],
            preds=output_values,
            targets=batch["target_values"],
            eval_masks=batch["eval_mask"],
            session_ids=batch["session_id"],
            absolute_starts=batch["absolute_start"],
        )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx)

    def on_before_optimizer_step(self, optimizer) -> None:
        """Log gradient norms after backward, before the step.

        Fires after DDP grad sync, so per-rank values agree for synced
        params; we only log on rank zero to avoid duplicate scalar series
        in wandb. Three metrics:
          - `train/grad_norm_total`: global L2 over all params with grads
          - `train/grad_norm_<group>`: per-group L2 (`unit_session_emb` vs
            `model`) — useful since the SparseLamb groups have very
            different update mechanics.

        Gated to `cfg.log_every_n_steps` so the Python loop over named
        parameters + sparse-grad coalesce doesn't run every step.
        """
        del optimizer  # we walk model parameters directly

        every_n = int(self.cfg.log_every_n_steps)
        if every_n > 1 and (self.global_step % every_n) != 0:
            return

        emb_sq = torch.zeros((), device=self.device)
        rest_sq = torch.zeros((), device=self.device)
        # Walk the wrapper's full param tree so wrapper-level submodules
        # (input_proj, unfrozen ntt_tokenizer) are counted under the model
        # bucket alongside the POYO core. Names get a `model.` prefix for
        # POYO params, so the unit/session-emb filter has to match that.
        for n, p in self.named_parameters():
            g = p.grad
            if g is None:
                continue
            # Sparse grads (unit/session emb) need special norm handling.
            sq = g.coalesce()._values().pow(2).sum() if g.is_sparse else g.pow(2).sum()
            if "model.unit_emb" in n or "model.session_emb" in n:
                emb_sq = emb_sq + sq
            else:
                rest_sq = rest_sq + sq

        total_norm = torch.sqrt(emb_sq + rest_sq)
        self.log("train/grad_norm_total", total_norm, on_step=True, on_epoch=False, prog_bar=False, rank_zero_only=True)
        self.log(
            "train/grad_norm_unit_session_emb", torch.sqrt(emb_sq), on_step=True, on_epoch=False, rank_zero_only=True
        )
        self.log("train/grad_norm_model", torch.sqrt(rest_sq), on_step=True, on_epoch=False, rank_zero_only=True)

    def configure_optimizers(self):
        """Same optimizer setup as POYO (SparseLamb + OneCycleLR)."""
        # Scale by effective (DDP-aggregate) batch size: SparseLamb sees only
        # per-rank batch_size, but DDP averages grads across world_size ranks,
        # so the per-step update behaves like a `batch_size * world_size`
        # batch. Without this, NGPUS=2 BATCH_SIZE=256 peaks at 5e-4 instead
        # of the 1e-3 a single-GPU bs=512 run hits.
        world_size = int(self.cfg.gpus) * int(self.cfg.num_nodes)
        effective_batch_size = self.cfg.batch_size * world_size
        max_lr = self.cfg.optim.base_lr * effective_batch_size

        special_emb_params = list(self.model.unit_emb.parameters()) + list(self.model.session_emb.parameters())

        # Optional third group: trainable params on the wrapped NTT tokenizer
        # (e.g. when `tokenizer.freeze=False` for end-to-end fine-tuning).
        # Without this, freeze=False is a no-op — params would receive grads
        # but the optimizer would never step them. Scaled by
        # `optim.tokenizer_lr_factor` (default 1.0) so the tokenizer can use
        # a smaller peak LR than the freshly-initialised POYO encoder.
        tokenizer_trainable: list[nn.Parameter] = []
        if self.ntt_tokenizer is not None:
            tokenizer_trainable = [p for p in self.ntt_tokenizer.parameters() if p.requires_grad]
        # Identify these by id so we can exclude them from `remaining_params`.
        # The wrapper exposes ntt_tokenizer params via `self.named_parameters()`,
        # so without this exclusion the same params would land in two groups
        # and `Optimizer.add_param_group` would raise "some parameters appear
        # in more than one parameter group".
        tokenizer_trainable_ids: set[int] = {id(p) for p in tokenizer_trainable}

        # Walk the wrapper, not just `self.model`, so wrapper-level submodules
        # (`input_proj`, a frozen `ntt_tokenizer`) actually receive updates.
        # POYO params show up with a `model.` prefix here; the emb filter has
        # to match that.
        remaining_params = [
            p
            for n, p in self.named_parameters()
            if "model.unit_emb" not in n and "model.session_emb" not in n and id(p) not in tokenizer_trainable_ids
        ]

        param_groups: list[dict] = [
            {"params": special_emb_params, "sparse": True},
            {"params": remaining_params},
        ]
        max_lrs: list[float] = [max_lr, max_lr]
        if tokenizer_trainable:
            tokenizer_lr_factor = float(self.cfg.optim.get("tokenizer_lr_factor", 1.0))
            tokenizer_max_lr = max_lr * tokenizer_lr_factor
            param_groups.append({"params": tokenizer_trainable, "lr": tokenizer_max_lr})
            max_lrs.append(tokenizer_max_lr)
            logger.info(
                f"Tokenizer fine-tune enabled: {sum(p.numel() for p in tokenizer_trainable):,} "
                f"params, peak LR = {tokenizer_max_lr:.2e} "
                f"(factor={tokenizer_lr_factor})"
            )

        optimizer = SparseLamb(
            param_groups,
            lr=max_lr,
            weight_decay=self.cfg.optim.weight_decay,
        )

        # Catch the class of bug where a wrapper-level submodule silently
        # never trains because configure_optimizers forgot to include it.
        trainable = {id(p) for p in self.parameters() if p.requires_grad}
        covered = {id(p) for g in optimizer.param_groups for p in g["params"]}
        missing = trainable - covered
        assert not missing, f"{len(missing)} trainable params are not in any optimizer group"

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            # estimated_stepping_batches is typed int | float (inf when max_steps is unset).
            total_steps=cast(int, self.trainer.estimated_stepping_batches),
            pct_start=self.cfg.optim.warmup_pct,
            anneal_strategy="cos",
            div_factor=self.cfg.optim.div_factor,
            final_div_factor=self.cfg.optim.final_div_factor,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
