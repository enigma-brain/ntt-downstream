"""Lightning module for image encoding models."""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from collections.abc import Callable
from typing import Any, cast

import lightning as L
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ntt_downstream.encoding.models.enc.hiera.evals import (
    OnlinePearsonAccumulator,
    TemporalCorrAccumulator,
    batch_temporal_correlation,
    rebin_time_axis,
)
from ntt_downstream.encoding.models.enc.hiera.model_factory import create_model
from ntt_downstream.encoding.models.enc.hiera.noise_corrected_r2 import noise_corrected_r2
from ntt_downstream.encoding.tokenizer_wrappers.encoding_wrapper import TokenizerWrapper
from ntt_downstream.encoding.training.prediction import predict_and_align

logger = logging.getLogger(__name__)

# We gather per-session metric dicts ourselves (see `_gather_session_tensor_dict`)
# and then log on rank-zero only. Lightning still emits a "recommended to use
# sync_dist=True" warning for every per-session self.log call — silence it.
warnings.filterwarnings(
    "ignore",
    message=r".*It is recommended to use `self\.log\(.*sync_dist=True\).*",
)


def _ddp_is_active() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _gather_session_tensor_dict(session_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """All-gather per-session tensor dicts across DDP ranks and merge.

    Sessions are partitioned 1-to-1 across ranks by
    ``DistributedSessionBatchSampler``, so the per-rank dicts have disjoint
    keys. The merged dict contains every session's tensor on every rank,
    making downstream logging identical across ranks.
    """
    if not _ddp_is_active():
        return session_dict
    world_size = dist.get_world_size()
    # Serialize through CPU so all_gather_object is safe across devices.
    cpu_dict = {k: v.detach().cpu() for k, v in session_dict.items()}
    gathered: list = [None] * world_size
    dist.all_gather_object(gathered, cpu_dict)
    merged: dict[str, torch.Tensor] = {}
    for rank_dict in gathered:
        if not rank_dict:
            continue
        for k, v in rank_dict.items():
            merged[k] = v
    return merged


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------


class EncodingLitModule(L.LightningModule):
    """Lightning module for image -> neural response encoding.

    Supports both base (ImageEncodingHiera) and MAR (ImageEncodingHieraMAR) models.
    Model type is determined by ``cfg.model.model_type``.

    Validation metrics:

    - ``val/corr/``: Per-image temporal Pearson correlation (T > 1 only).
    - ``val/rate_corr/``: Cross-image Pearson correlation on time-summed rates.
    - ``val/rate_corr_trial_avg/``: Same, on repeat-averaged responses.
    - ``val/corr_trial_avg/``: Temporal corr on repeat-averaged responses.
    - ``val_nc_r2/``: Noise-corrected R² (Pospisil & Bair 2021).

    Early stopping / checkpointing monitors ``val/rate_corr/overall``.
    """

    def __init__(self, cfg, session_info: dict[str, int] | None = None, subject_info: dict[str, str] | None = None):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self._session_info = session_info
        self._subject_info = subject_info
        self.model: Any | None = None
        self.tokenizer_wrapper: TokenizerWrapper | None = None

        # Validation accumulators (reset each epoch)
        self._temporal_corr = TemporalCorrAccumulator()
        self._rate_corr = OnlinePearsonAccumulator()

        # Set by train.py after DataModule setup
        self._val_avg_dataloader_fn: Callable[[], DataLoader] | None = None
        self._val_dataloader_fn: Callable[[], DataLoader] | None = None
        self._train_dataloader_fn: Callable[[], DataLoader] | None = None

        # Eval-time neuron trace plotting: a fixed set of random neurons is
        # selected once per target session and reused across val epochs so the
        # logged traces are comparable over training.
        self._plot_target_session: str | None = None
        self._plot_random_neurons: dict[str, list[int]] = {}
        self._plot_val_counter: int = 0

        # Per-session codebook decode tables for the Q=1 CE marginal eval path
        # (see ``ce_marginal.compute_ce_q1_marginal_rate``). Populated lazily on
        # first val pass per session, lives on CPU. Same-session entries are
        # reused across val cadences and across train/val cycles.
        self._ce_codebook_rates_cache: dict[str, list[torch.Tensor]] = {}

        # LDNS-specific: cached background latent trajectory per session, used
        # to splice the encoder's short prediction into a real-distribution
        # full-length latent before decoding (see _predict_and_align). The LDNS
        # AE's InstanceNorm1d collapses on short inputs; splicing into a real
        # bg trajectory keeps the decoder in its trained operating regime.
        # Populated lazily on first val pass per session.
        self._ldns_bg_bank: dict[str, torch.Tensor] = {}

    # ``self.model`` / ``self.tokenizer_wrapper`` start as None and are filled in
    # by ``configure_model``. Every use through these accessors runs after that
    # (and, for the tokenizer, only when tokenizer.enabled), so narrow once here
    # instead of at each call site. Sites that must tolerate None keep using
    # ``self.model`` / ``self.tokenizer_wrapper`` with an explicit check.
    @property
    def _net(self) -> Any:
        assert self.model is not None, "configure_model() has not run"
        return self.model

    @property
    def _tok(self) -> TokenizerWrapper:
        assert self.tokenizer_wrapper is not None, "tokenizer is not enabled"
        return self.tokenizer_wrapper

    def set_val_average_dataloader(self, dataloader_fn):
        """Register a callable that returns the repeat-averaged val dataloader."""
        self._val_avg_dataloader_fn = dataloader_fn

    def set_val_dataloader(self, dataloader_fn):
        """Register a callable that returns the single-trial val dataloader."""
        self._val_dataloader_fn = dataloader_fn

    def set_train_dataloader(self, dataloader_fn):
        """Register a callable that returns the train dataloader (for train-set eval)."""
        self._train_dataloader_fn = dataloader_fn

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------

    def configure_model(self):
        """Lazy model creation -- called once before training starts."""
        if self.model is not None:
            return

        cfg = self.cfg
        loss_mode = str(cfg.model.get("loss_mode", "poisson"))
        needs_tokenizer = loss_mode not in ("poisson", "mse_rate")

        if needs_tokenizer:
            self.tokenizer_wrapper = self._build_tokenizer()

            # LDNS-only knob: refuses to silently no-op for any other tokenizer.
            # See _predict_and_align for the matching decode-side splice.
            if cfg.model.get("compress_n_target_slice") is not None:
                tk_kind = (self._tok.tokenizer_kind or "").lower()
                if tk_kind != "ldns":
                    raise ValueError(
                        f"model.compress_n_target_slice is LDNS-specific (matches the "
                        f"InstanceNorm1d length-sensitivity in the LDNS AE) but tokenizer "
                        f"kind is {tk_kind!r}. Either drop the knob or pick an LDNS checkpoint."
                    )

            # Calibrate latent scale if needed (requires a dataloader).
            # Use num_workers=0 to avoid CUDA-in-forked-subprocess errors:
            # configure_model runs after Lightning initializes CUDA, so forked
            # DataLoader workers would crash on .to(device).
            if getattr(self.tokenizer_wrapper, "_needs_calibration", False):
                if self._train_dataloader_fn is not None:
                    from torch.utils.data import DataLoader

                    from ntt_downstream.encoding.dataloading.cowley_v4 import SessionBatchSampler, numpy_collate_fn

                    src_dl = self._train_dataloader_fn()
                    batch_size = cfg.dataloader.get("batch_size", 64)
                    cal_dl = DataLoader(
                        src_dl.dataset,
                        batch_sampler=SessionBatchSampler(src_dl.dataset, batch_size=batch_size, shuffle=False),
                        collate_fn=numpy_collate_fn,
                        num_workers=0,
                    )
                    self._tok.tokenizer.to("cuda")
                    if getattr(self.tokenizer_wrapper, "_per_channel_calibration", False):
                        self._tok.calibrate_latent_scale_per_channel(cal_dl)
                    else:
                        self._tok.calibrate_latent_scale(cal_dl)
                    self._tok.tokenizer.to("cpu")

        # Derive n_time_bins from data config so model output matches response shape
        response_window = float(cfg.data.get("response_window", 0.200))
        bin_size = float(cfg.data.get("bin_size", 0.001))
        n_time_bins = int(round(response_window / bin_size))
        from omegaconf import OmegaConf

        OmegaConf.update(cfg, "model.n_time_bins", n_time_bins, force_add=True)

        model_kwargs = {}
        if self.tokenizer_wrapper is not None:
            model_kwargs["tokenizer_wrapper"] = self.tokenizer_wrapper

            # Auto-derive token_prediction_dim from tokenizer spec when not set.
            # compress_n (LDNS, WaNeural, etc.): emits (B, T, U=1, D=latent_n);
            # the model predicts D=latent_n, set by `compress_n_latent_n`.
            # Wavelet has compress_n=True too but our wavelet arm emits
            # (B, S=U_total, T_lat=K, D=z_channels) — token_prediction_dim should
            # match z_channels, NOT latent_n (= U_total). Treat wavelet as "compress_t-like"
            # for the auto-derive path: the wrapper's `latent_dim` falls through to
            # `spec.z_channels` for wavelet via the existing fallback chain.
            is_cosmos = bool(self.tokenizer_wrapper.is_cosmos)
            if (not self.tokenizer_wrapper.is_compress_n) or is_cosmos:
                if is_cosmos:
                    # wavelet's `latent_dim` property would treat it as compress_n
                    # (=> latent_n) — bypass and read z_channels directly.
                    latent_dim = int(cast(Any, self._tok.tokenizer).spec.z_channels)
                else:
                    latent_dim = self.tokenizer_wrapper.latent_dim
                if cfg.model.get("token_prediction_dim", None) is None:
                    # Default rule depends on the loss mode:
                    #   - cross_entropy: token_prediction_dim is the WIDTH of the
                    #     classifier input feeding ce_head=Linear(D, V*Q). The CE
                    #     target is token INDICES, so there is NO latent-shape
                    #     constraint tying D to the tokenizer z_dim — it is a free
                    #     hyperparameter. Tying it to a tiny z_dim (=1 for transformer-vq)
                    #     bottlenecks the head and crushes the encoder's image
                    #     signal. Decouple: never auto-derive below 32.
                    #   - flow_matching / diffusion / mse / energy: the head
                    #     REGRESSES the continuous latent, so D MUST equal z_dim.
                    #   - bce (BSQ): predicts z_dim bits directly, so D == z_dim.
                    if loss_mode == "cross_entropy":
                        token_prediction_dim = max(int(latent_dim), 32)
                        OmegaConf.update(cfg, "model.token_prediction_dim", token_prediction_dim, force_add=True)
                        logger.info(
                            f"Auto-derived token_prediction_dim={token_prediction_dim} for cross_entropy "
                            f"(max(z_dim={latent_dim}, 32) — decoupled from z_dim: CE predicts token "
                            f"indices, so the ce_head=Linear(D, V) input width is a free hyperparameter "
                            f"and must NOT bottleneck the classifier below 32)."
                        )
                    else:
                        # Latent-regression / per-bit heads: width MUST match z_dim.
                        OmegaConf.update(cfg, "model.token_prediction_dim", latent_dim, force_add=True)
                        logger.info(
                            f"Auto-derived token_prediction_dim={latent_dim} from tokenizer spec (z_dim) "
                            f"for loss_mode={loss_mode!r} (latent-regression/per-bit head — width tied to z_dim)."
                        )

                # Loud guard: a resolved CE classifier-input width below 16 (whether
                # pinned by the user or arrived at via some path) crushes the
                # encoder's image features through a near-scalar bottleneck before
                # the softmax. Warn prominently with the concrete head shape.
                if loss_mode == "cross_entropy":
                    resolved_token_dim = cfg.model.get("token_prediction_dim", None)
                    if resolved_token_dim is not None and int(resolved_token_dim) < 16:
                        logger.warning(
                            "="
                            * 78
                            + f"\n[cross_entropy] token_prediction_dim={int(resolved_token_dim)} is DANGEROUSLY "
                            "SMALL. The classifier is ce_head=Linear(token_prediction_dim, V): the "
                            "encoder's rich image features are squeezed through a "
                            f"{int(resolved_token_dim)}-dim bottleneck before the softmax, crushing the "
                            "image signal. For CE the head width is a FREE hyperparameter (unlike "
                            "flow/mse, where it must equal z_dim) — recommend >= 64 (e.g. set "
                            "++model.token_prediction_dim=128 via job.extra).\n" + "=" * 78
                        )

                # Auto-derive diffusion channel dims from tokenizer when not set
                loss_mode = str(cfg.model.get("loss_mode", "poisson"))
                if loss_mode == "diffusion":
                    diffusion_cfg = cfg.model.get("diffusion")
                    if diffusion_cfg is None:
                        OmegaConf.update(
                            cfg,
                            "model.diffusion",
                            {
                                "target_channels": latent_dim,
                                "z_channels": latent_dim,
                                "depth": 3,
                                "width": 128,
                                "num_sampling_steps": "100",
                            },
                            force_add=True,
                        )
                        logger.info(f"Auto-created diffusion config with target/z_channels={latent_dim}")
                    else:
                        for key_name in ("target_channels", "z_channels"):
                            if diffusion_cfg.get(key_name, None) is None:
                                OmegaConf.update(cfg, f"model.diffusion.{key_name}", latent_dim, force_add=True)
                                logger.info(f"Auto-derived diffusion.{key_name}={latent_dim} from tokenizer")

                # Auto-derive ce_vocab_size from tokenizer for cross-entropy mode
                loss_mode_for_ce = str(cfg.model.get("loss_mode", "poisson"))
                if loss_mode_for_ce == "cross_entropy" and cfg.model.get("ce_vocab_size", None) is None:
                    vocab_size = cast(Any, self._tok.tokenizer).get_vocab_size()
                    OmegaConf.update(cfg, "model.ce_vocab_size", vocab_size, force_add=True)
                    logger.info(f"Auto-derived ce_vocab_size={vocab_size} from tokenizer.get_vocab_size()")

                # Auto-derive num_codebooks for residual / multi-codebook VQ.
                # The wrapper probes the tokenizer at construction time and
                # reports the trailing Q dim of the discrete-indices output;
                # for transformer-vq / single-codebook bottlenecks this is 1.
                if loss_mode_for_ce == "cross_entropy" and cfg.model.get("num_codebooks", None) is None:
                    num_codebooks = int(self.tokenizer_wrapper.num_codebooks_effective)
                    OmegaConf.update(cfg, "model.num_codebooks", num_codebooks, force_add=True)
                    logger.info(
                        f"Auto-derived num_codebooks={num_codebooks} from tokenizer probe "
                        f"(indices_source={self.tokenizer_wrapper.indices_source!r})"
                    )

                # Validate temporal compatibility
                t_comp_factor = self.tokenizer_wrapper.temporal_compression_factor
                if t_comp_factor is not None:
                    if n_time_bins % t_comp_factor != 0:
                        logger.warning(
                            f"n_time_bins={n_time_bins} is not divisible by tokenizer "
                            f"temporal_compression={t_comp_factor}. The tokenizer may "
                            f"produce unexpected latent shapes."
                        )
                    t_comp = n_time_bins // t_comp_factor
                    logger.info(
                        f"Tokenizer temporal compression: {n_time_bins} bins -> "
                        f"{t_comp} latent time step(s) (factor={t_comp_factor}x)"
                    )
                    model_type = str(cfg.model.get("model_type", "image_encoding"))
                    if model_type == "image_encoding_mar":
                        # MAR handles t_comp>1 natively via n_temporal_tokens.
                        if cfg.model.get("n_temporal_tokens", None) is None:
                            OmegaConf.update(cfg, "model.n_temporal_tokens", t_comp, force_add=True)
                            logger.info(
                                f"Auto-derived n_temporal_tokens={t_comp} for MAR with "
                                f"compress_t tokenizer (n_time_bins={n_time_bins}, "
                                f"temporal_compression={t_comp_factor})."
                            )
                    else:
                        model_t_latent = int(self.cfg.model.get("token_t_latent", 1))
                        if t_comp != model_t_latent:
                            raise ValueError(
                                f"Tokenizer produces t_comp={t_comp} latent time steps but "
                                f"model.token_t_latent={model_t_latent}. These must match. "
                                f"Either set model.token_t_latent={t_comp} or adjust "
                                f"data.response_window / bin_size so t_comp matches."
                            )

            # For compress_n tokenizers: override readout neuron count to num_chunks
            # and pre-populate chunking plans so validation works before first training step
            if self.tokenizer_wrapper.is_compress_n and not self.tokenizer_wrapper.is_cosmos:
                assert self._session_info is not None, "compress_n requires session_info"
                n_neurons_override = {}
                for sid, n in self._session_info.items():
                    self.tokenizer_wrapper.get_or_create_plan(sid, n)
                    n_neurons_override[sid] = self.tokenizer_wrapper.get_num_chunks(sid, n)
                model_kwargs["n_neurons_override"] = n_neurons_override
            elif self.tokenizer_wrapper.is_cosmos:
                # Wavelet: U_total = ceil(N / n_chunk) * U_per_chunk. We probe
                # U_per_chunk from the zero-pad bank lazily, but at this point
                # in configure_model the tokenizer might be on CPU and the bank
                # cache is empty. Eagerly populate it on CPU.
                bank_dev = next(self.tokenizer_wrapper.tokenizer.parameters()).device
                bank_dtype = next(self.tokenizer_wrapper.tokenizer.parameters()).dtype
                bank = self.tokenizer_wrapper._ensure_cosmos_zero_pad_bank(bank_dev, bank_dtype)
                u_per_chunk = int(bank.shape[2])
                n_chunk = self.tokenizer_wrapper.cosmos_n_chunk
                assert self._session_info is not None, "wavelet-* requires session_info"
                n_neurons_override = {}
                for sid, n in self._session_info.items():
                    n_chunks = (int(n) + n_chunk - 1) // n_chunk
                    n_neurons_override[sid] = n_chunks * u_per_chunk
                    # Warm the chunk-plan cache so encode/decode don't allocate
                    # on the validation-sanity path.
                    self.tokenizer_wrapper._cosmos_chunk_cache[sid] = (
                        int(n),
                        self.tokenizer_wrapper._cosmos_chunk_slices(int(n)),
                    )
                model_kwargs["n_neurons_override"] = n_neurons_override
                logger.info(
                    f"[wavelet] U_per_chunk={u_per_chunk}, n_chunk={n_chunk}; "
                    f"n_neurons_override built for {len(n_neurons_override)} sessions"
                )

                # For MAR: auto-derive n_temporal_tokens from n_time_bins (compress_n preserves time)
                model_type = str(cfg.model.get("model_type", "image_encoding"))
                if model_type == "image_encoding_mar" and cfg.model.get("n_temporal_tokens", None) is None:
                    OmegaConf.update(cfg, "model.n_temporal_tokens", n_time_bins, force_add=True)
                    logger.info(f"Auto-derived n_temporal_tokens={n_time_bins} for MAR compress_n")

        assert self._session_info is not None, "session_info is required to build the model"
        self.model = create_model(
            cfg,
            session_info=self._session_info,
            model_type=cfg.model.get("model_type", "image_encoding"),
            device="cpu",
            subject_info=self._subject_info,
            **model_kwargs,
        )

        # Compute per-neuron response mean/std for z-scored MSE loss
        if getattr(self.model, "response_zscore", False) and self._train_dataloader_fn is not None:
            self._calibrate_response_stats()

        # Class-imbalance reweighting for the discrete CE/BCE loss paths.
        # The tokenizer is FROZEN here, so the target token distribution is
        # fixed and can be estimated once over a setup probe (mirrors
        # calibrate_latent_scale). Both branches default OFF and no-op unless
        # the corresponding knob is enabled.
        if self.tokenizer_wrapper is not None and self._train_dataloader_fn is not None:
            if loss_mode == "cross_entropy" and str(cfg.model.get("ce_class_weight", "none")) == "inv_freq":
                self._calibrate_ce_class_weights()
            if loss_mode == "bce" and str(cfg.model.get("bce_pos_weight", "none")) == "auto":
                self._calibrate_bce_pos_weight()

        # Restore fine-tuned decoder weights from checkpoint (if resuming)
        pending_sd = getattr(self, "_pending_decoder_state_dict", None)
        if pending_sd is not None and self.tokenizer_wrapper is not None:
            tok_params = dict(self.tokenizer_wrapper.tokenizer.named_parameters())
            for name, data in pending_sd.items():
                if name in tok_params:
                    tok_params[name].data.copy_(data)
            logger.info(f"Restored {len(pending_sd)} fine-tuned decoder parameters from checkpoint")
            del self._pending_decoder_state_dict

    def _calibrate_response_stats(self):
        """Compute per-session per-neuron response mean/std from the training set.

        These are used for z-scored MSE loss so that each neuron contributes
        equally-sized gradients regardless of its firing rate. Matches Cowley
        et al.'s z-scoring in script_train_ensemble_model.py.
        """
        import logging

        logger = logging.getLogger(__name__)
        logger.info("Computing per-neuron response stats for z-scored MSE loss...")

        from collections import defaultdict

        # Accumulate per-session sufficient statistics: sum, sum_sq, count per neuron
        stats: dict[str, dict] = defaultdict(lambda: {"sum": None, "sum_sq": None, "count": 0})

        # Use num_workers=0 to avoid CUDA-in-forked-subprocess errors.
        # configure_model runs after Lightning may have initialized CUDA,
        # so forked DataLoader workers would fail.
        from torch.utils.data import DataLoader

        from ntt_downstream.encoding.dataloading.cowley_v4 import SessionBatchSampler, numpy_collate_fn

        assert self._train_dataloader_fn is not None, "train dataloader is required"
        src_dl = self._train_dataloader_fn()
        batch_size = self.cfg.dataloader.get("batch_size", 64)
        cal_dl = DataLoader(
            src_dl.dataset,
            batch_sampler=SessionBatchSampler(src_dl.dataset, batch_size=batch_size, shuffle=False),
            collate_fn=numpy_collate_fn,
            num_workers=0,
        )
        for key, batch_dict in cal_dl:
            resp = batch_dict["response"]  # (B, N, T)
            # Sum over time to get scalar rate per neuron per image, matching mse_rate loss
            rate = resp.sum(dim=2).float()  # (B, N)
            s = stats[key]
            if s["sum"] is None:
                s["sum"] = rate.sum(dim=0)
                s["sum_sq"] = (rate**2).sum(dim=0)
            else:
                s["sum"] += rate.sum(dim=0)
                s["sum_sq"] += (rate**2).sum(dim=0)
            s["count"] += rate.shape[0]

        session_stats = {}
        for sid, s in stats.items():
            n = s["count"]
            mean = s["sum"] / n  # (N,)
            var = s["sum_sq"] / n - mean**2
            std = var.clamp(min=1e-8).sqrt()
            session_stats[sid] = (mean, std)
            logger.info(
                f"  {sid}: {mean.shape[0]} neurons, "
                f"mean rate range [{mean.min():.2f}, {mean.max():.2f}], "
                f"std range [{std.min():.3f}, {std.max():.3f}]"
            )

        self._net.set_response_stats(session_stats)
        logger.info(f"Response stats calibrated for {len(session_stats)} sessions.")

    def _imbalance_probe_dataloader(self, n_probe: int):
        """Build a num_workers=0 train dataloader for the imbalance setup probe.

        Mirrors the calibrate_latent_scale / _calibrate_response_stats setup:
        forked CUDA workers crash after Lightning initializes CUDA, so this
        dataloader is single-process. ``n_probe`` is the caller-supplied batch
        cap (the iteration itself stops early).
        """
        from torch.utils.data import DataLoader

        from ntt_downstream.encoding.dataloading.cowley_v4 import SessionBatchSampler, numpy_collate_fn

        assert self._train_dataloader_fn is not None, "train dataloader is required"
        src_dl = self._train_dataloader_fn()
        batch_size = self.cfg.dataloader.get("batch_size", 64)
        return DataLoader(
            src_dl.dataset,
            batch_sampler=SessionBatchSampler(src_dl.dataset, batch_size=batch_size, shuffle=False),
            collate_fn=numpy_collate_fn,
            num_workers=0,
        )

    @torch.no_grad()
    def _calibrate_ce_class_weights(self):
        """Estimate inverse-frequency CE class weights from a setup probe.

        The frozen tokenizer fixes the target token distribution, so we count
        token occurrences over ``ce_class_weight_n_probe`` (default 100) train
        batches — encoding responses to indices on the SAME raw_discrete path
        the CE target uses — then form tempered inverse-frequency weights:

            freq   = (counts + 1) / (counts + 1).sum()    # Laplace-smoothed
            weight = (1 / freq) ** beta                     # beta tempers it
            weight = weight / weight.mean()                 # mean-1 normalize

        ``beta`` is ``model.ce_class_weight_beta`` (default 0.5). The result is
        handed to the model via ``set_ce_class_weight`` and applied in the CE
        loss; with the knob disabled this method is never called.

        Q>1 (residual / multi-codebook VQ) LIMITATION: per-codebook class
        weighting is not implemented. We log a warning and leave the weight at
        None (plain unweighted CE), which is the safe no-op — never raising.
        Q=1 (transformer-vq, single-codebook FSQ) is fully supported.
        """
        num_codebooks = int(getattr(self.model, "num_codebooks", 1))
        if num_codebooks > 1:
            logger.warning(
                "ce_class_weight=inv_freq is not implemented for multi-codebook "
                f"tokenizers (num_codebooks={num_codebooks}); falling back to "
                "unweighted CE. Class weighting only supports Q=1 (e.g. transformer-vq/FSQ)."
            )
            return

        vocab_size = int(self._net.ce_vocab_size)
        n_probe = int(self.cfg.model.get("ce_class_weight_n_probe", 100))
        beta = float(self.cfg.model.get("ce_class_weight_beta", 0.5))

        cal_dl = self._imbalance_probe_dataloader(n_probe)
        self._tok.tokenizer.to("cuda")
        device = next(self._tok.tokenizer.parameters()).device

        counts = torch.zeros(vocab_size, dtype=torch.long, device=device)
        n_batches = 0
        for i, (key, batch_dict) in enumerate(cal_dl):
            if i >= n_probe:
                break
            # Dataset returns (B, N, T); encode expects (B, T, N), same as the
            # CE target path in base_model._training_loss.
            responses = batch_dict["response"].to(device, torch.float32).transpose(1, 2)
            with torch.autocast(device_type="cuda", enabled=False):
                target_indices = self._tok.encode(responses, key, raw_discrete=True)["target"]
            counts += torch.bincount(target_indices.reshape(-1).long(), minlength=vocab_size)
            n_batches += 1
        self._tok.tokenizer.to("cpu")

        if n_batches == 0 or counts.sum().item() == 0:
            logger.warning("_calibrate_ce_class_weights: no tokens observed; leaving CE unweighted.")
            return

        smoothed = counts.float() + 1.0
        freq = smoothed / smoothed.sum()
        weight = (1.0 / freq).pow(beta)
        weight = weight / weight.mean()
        # Model is on CPU here; Lightning moves it (and its buffers) to GPU
        # afterward. Register on CPU so the buffer rides along like the others.
        self._net.set_ce_class_weight(weight.cpu())
        logger.info(
            f"ce_class_weight=inv_freq calibrated over {n_batches} batches (beta={beta}): "
            f"V={vocab_size}, weight range [{weight.min().item():.3f}, {weight.max().item():.3f}], "
            f"nonzero classes={int((counts > 0).sum().item())}/{vocab_size}"
        )

    @torch.no_grad()
    def _calibrate_bce_pos_weight(self):
        """Estimate the per-bit BCE positive weight (BSQ) from a setup probe.

        BSQ targets are ``(target > 0)`` bits. We accumulate per-bit positive
        counts over the probe (same encode path as the bce loss target, which
        uses ``skip_target_scaling=True``), then form per-bit
        ``pos_weight = neg / pos`` clamped to ``bce_pos_weight_max`` (default
        100) to avoid blowups on near-constant bits. Handed to the model via
        ``set_bce_pos_weight``; disabled by default.
        """
        n_probe = int(self.cfg.model.get("ce_class_weight_n_probe", 100))
        clamp_max = float(self.cfg.model.get("bce_pos_weight_max", 100.0))

        cal_dl = self._imbalance_probe_dataloader(n_probe)
        self._tok.tokenizer.to("cuda")
        device = next(self._tok.tokenizer.parameters()).device

        pos_sum = None  # (n_bits,)
        total = 0
        n_batches = 0
        for i, (key, batch_dict) in enumerate(cal_dl):
            if i >= n_probe:
                break
            responses = batch_dict["response"].to(device, torch.float32).transpose(1, 2)
            with torch.autocast(device_type="cuda", enabled=False):
                # Mirror the bce loss target: continuous encode with target
                # scaling skipped, then threshold at 0 → per-bit indicator.
                target = self._tok.encode(responses, key, skip_target_scaling=True)["target"]
            bits = (target > 0).float()  # (B, S, T, n_bits)
            n_bits = bits.shape[-1]
            flat = bits.reshape(-1, n_bits)
            batch_pos = flat.sum(dim=0)
            if pos_sum is None:
                pos_sum = batch_pos
            else:
                pos_sum += batch_pos
            total += flat.shape[0]
            n_batches += 1
        self._tok.tokenizer.to("cpu")

        if n_batches == 0 or total == 0 or pos_sum is None:
            logger.warning("_calibrate_bce_pos_weight: no bits observed; leaving BCE unweighted.")
            return

        # neg/pos per bit; +1 in the denominator guards all-negative bits before
        # the clamp also bounds the result.
        neg = float(total) - pos_sum
        pos_weight = (neg / (pos_sum + 1.0)).clamp(max=clamp_max)
        # See _calibrate_ce_class_weights: register on CPU so the buffer moves
        # with the model when Lightning relocates it to GPU.
        self._net.set_bce_pos_weight(pos_weight.cpu())
        logger.info(
            f"bce_pos_weight=auto calibrated over {n_batches} batches: "
            f"n_bits={pos_weight.numel()}, pos_weight range "
            f"[{pos_weight.min().item():.3f}, {pos_weight.max().item():.3f}] (clamp_max={clamp_max})"
        )

    def _build_tokenizer(self):
        from ntt_downstream.encoding.tokenizer_wrappers.encoding_wrapper import build_tokenizer_wrapper

        return build_tokenizer_wrapper(self.cfg, device="cpu")

    def _ensure_tokenizer_device(self):
        """Move tokenizer to model device if needed (it's not an nn.Module on the LitModule)."""
        if self.tokenizer_wrapper is not None:
            self.tokenizer_wrapper.tokenizer.to(self.device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _get_warmup_config(self):
        """Return (warmup_loss_mode, warmup_steps) or (None, 0) if not configured."""
        warmup_mode = self.cfg.trainer.get("warmup_loss_mode", None)
        warmup_steps = int(self.cfg.trainer.get("warmup_steps", 0))
        if warmup_mode is None or warmup_steps <= 0:
            return None, 0
        return str(warmup_mode), warmup_steps

    def on_train_epoch_start(self):
        """Advance the batch sampler's epoch so shuffling varies each epoch.

        LightningDataModule has no `on_train_epoch_start` hook in Lightning 2.x,
        so this must live on the LightningModule instead — otherwise the
        sampler's RNG is re-seeded with `epoch=0` every epoch and batch order
        becomes identical across epochs.
        """
        dm = getattr(self.trainer, "datamodule", None)
        sampler = getattr(dm, "train_sampler", None) if dm is not None else None
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.current_epoch)

    def training_step(self, batch, batch_idx):
        key, batch_dict = batch
        images = batch_dict["image"]
        responses = batch_dict["response"]

        if batch_idx == 0:
            self._ensure_tokenizer_device()

        # Staged warmup: use a simpler loss mode for the first N steps to
        # pre-train the readout before switching to the target loss mode.
        warmup_mode, warmup_steps = self._get_warmup_config()
        in_warmup = warmup_mode is not None and self.global_step < warmup_steps

        if in_warmup:
            original_mode = self._net.loss_mode
            self._net.loss_mode = warmup_mode

        # responses from dataset: (B, N, T) -> model expects (B, T, N) for tokenizer
        responses_btn = responses.transpose(1, 2)

        out = self._net(images, key, responses=responses_btn)
        loss = out["loss"]

        if in_warmup:
            self._net.loss_mode = original_mode
            # Log transition once
            if self.global_step == warmup_steps - 1:
                logger.info(f"Warmup complete at step {warmup_steps}: switching from {warmup_mode} to {original_mode}")

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        if in_warmup:
            self.log("train/warmup", 1.0, on_step=True, on_epoch=False)
        # Keys handled explicitly below (with custom log args) are excluded from the generic loop.
        _explicit_log_keys = {"energy_diversity_ratio"}
        for k, v in out.items():
            if k != "loss" and k not in _explicit_log_keys and isinstance(v, torch.Tensor) and v.ndim == 0:
                self.log(f"train/{k}", v, on_step=True, on_epoch=False)

        # Promote energy diversity ratio to prog_bar — collapse is fast to spot.
        if "energy_diversity_ratio" in out:
            self.log(
                "train/energy_diversity_ratio",
                out["energy_diversity_ratio"],
                prog_bar=True,
                on_step=True,
                on_epoch=False,
            )
        return loss

    # ------------------------------------------------------------------
    # Eval helpers
    # ------------------------------------------------------------------

    def _get_eval_skip_bins(self, sampling_rate_hz: float | None = None) -> int:
        """Compute the number of initial time bins to skip during eval.

        Configured via ``trainer.eval_skip_seconds`` (default 0.050 = 50ms).
        """
        skip_sec = float(self.cfg.trainer.get("eval_skip_seconds", 0.050))
        if skip_sec <= 0:
            return 0
        if sampling_rate_hz is None:
            bin_size = float(self.cfg.data.get("bin_size", 0.001))
            sampling_rate_hz = 1.0 / bin_size
        return int(round(skip_sec * sampling_rate_hz))

    def _get_eval_rebin_hz(self) -> float | None:
        """Return the unified eval rebin rate, or None to skip rebinning."""
        val = self.cfg.trainer.get("eval_rebin_hz", None)
        return float(val) if val is not None else None

    def _get_native_hz(self) -> float:
        """Return the native sampling rate derived from data.bin_size."""
        return 1.0 / float(self.cfg.data.get("bin_size", 0.001))

    def _rebin_and_trim(self, pred, target, pred_hz, target_hz):
        """Rebin pred & target to eval_rebin_hz and apply skip window.

        Returns:
            (pred, target) as (B, T', N) tensors (time dimension preserved).
        """
        rebin_hz = self._get_eval_rebin_hz()
        if rebin_hz is not None:
            pred = rebin_time_axis(pred, from_hz=pred_hz, to_hz=rebin_hz)
            target = rebin_time_axis(target, from_hz=target_hz, to_hz=rebin_hz)
        skip = self._get_eval_skip_bins(rebin_hz)
        min_t = min(pred.shape[1], target.shape[1])
        return pred[:, skip:min_t, :], target[:, skip:min_t, :]

    def _ldns_splice_active(self) -> bool:
        """Return True iff the LDNS encode-long / predict-short / splice-decode
        path is configured. Triggers only when the tokenizer is LDNS *and*
        ``model.compress_n_target_slice`` is set."""
        if self.tokenizer_wrapper is None:
            return False
        if (self.tokenizer_wrapper.tokenizer_kind or "").lower() != "ldns":
            return False
        return self.cfg.model.get("compress_n_target_slice") is not None

    def _cosmos_active(self) -> bool:
        """Return True iff wavelet-style pad-to-1s encode + spliced decode is in use.

        The wrapper's `is_cosmos` arm handles encode/decode internally
        (zero-pad input, K-slice latents, expand at decode with cached
        zero-pad bank). The encoding module's only responsibility is to
        slice predictions and targets to the decoder-safe horizon
        (``cosmos_t_safe``) before metric computation, since the small
        decoder lookahead pollutes the last few output bins of the
        response window.
        """
        if self.tokenizer_wrapper is None:
            return False
        return bool(self.tokenizer_wrapper.is_cosmos) and self.tokenizer_wrapper.cosmos_t_safe is not None

    def _get_ldns_bg(self, key: str, responses_btn: torch.Tensor) -> torch.Tensor:
        """Return a cached background latent trajectory for ``key``.

        Lazily encoded from the first sample of the first val batch we see
        for that session. Lives in scaled-target space (same space as the
        model's flow-matching output), so we can splice in the model's
        prediction directly before unscaling and decoding.

        Returns shape ``(1, S, T_full, D)``.
        """
        cached = self._ldns_bg_bank.get(key)
        if cached is not None:
            return cached.to(responses_btn.device)
        # Background source. Default: the first real val sample — keeps the decoder
        # at its trained sequence length, but is image-invariant, so it leaks the
        # session-average temporal profile into every decode and its InstanceNorm1d
        # stats are set by a fixed reference rather than the current image. With
        # ``model.ldns_bg_zero=true`` use a content-free ZEROS window instead
        # (wavelet zero-pad analogue) to test/remove that leakage.
        bg_zero = bool(self.cfg.model.get("ldns_bg_zero", False))
        bg_input = responses_btn[:1].float()
        if bg_zero:
            bg_input = torch.zeros_like(bg_input)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=False):
                enc = self.tokenizer_wrapper.encode(bg_input, key)  # type: ignore[union-attr]
        bg = enc["target"].detach()
        self._ldns_bg_bank[key] = bg.cpu()
        logger.info(f"[ldns_bg_bank] populated for session={key!r}, shape={tuple(bg.shape)}, zero={bg_zero}")
        return bg

    def _predict_and_align(self, images, responses, key):
        """Run model prediction and return aligned (pred, target) as (B, T', N).

        Handles all model types (poisson, tokenizer, MAR) and applies skip + rebin.
        Time dimension is preserved for proper per-image temporal correlation.

        Honors ``trainer.val_micro_batch`` (default 0 = off): if set, chunks the
        input batch into mini-batches of that size and concatenates the
        predictions. Independent of which eval path runs underneath — bounds
        memory for huge-vocab CE (where the ``(B*S*T_tok, V)`` softmax tensor is
        the dominant consumer) and any other path that scales with B.
        """
        val_micro = int(self.cfg.trainer.get("val_micro_batch", 0))
        if val_micro > 0 and images.shape[0] > val_micro:
            preds: list[torch.Tensor] = []
            targets: list[torch.Tensor] = []
            for b_start in range(0, images.shape[0], val_micro):
                b_end = min(b_start + val_micro, images.shape[0])
                p, t = self._predict_and_align(images[b_start:b_end], responses[b_start:b_end], key)
                preds.append(p)
                targets.append(t)
            return torch.cat(preds, dim=0), torch.cat(targets, dim=0)

        loss_mode = str(self.cfg.model.get("loss_mode", "poisson"))
        model_type = str(self.cfg.model.get("model_type", "image_encoding"))

        # The full loss-mode dispatch + LDNS/Wavelet slice + rebin lives in the
        # shared `prediction.predict_and_align`, which final_eval also calls —
        # the single source of truth keeps the checkpoint-selection metric and
        # the reported metric on the same time support. The periodic path uses
        # n_samples=1 (no MC averaging); final_eval passes its configured count.
        #
        # Sampling knobs are resolved HERE (caller-owned), not inside the helper:
        # the periodic path reads them from cfg.trainer.tokenizer_eval / mar_eval,
        # whereas final_eval pins its own run() args. These sources legitimately
        # differ (e.g. flow_solver defaults to "euler" in tokenizer_eval but
        # "heun" in final_eval), so the helper must never consult cfg itself.
        tok_eval = self.cfg.trainer.get("tokenizer_eval", {})
        mar_eval = self.cfg.trainer.get("mar_eval", {})
        return predict_and_align(
            self,
            images,
            responses,
            key,
            loss_mode=loss_mode,
            model_type=model_type,
            n_samples=1,
            flow_solver=str(tok_eval.get("flow_solver", "euler")),
            n_flow_steps=int(tok_eval.get("n_flow_steps", 5)),
            sample_temperature=float(tok_eval.get("sample_temperature", 1.0)),
            cfg_scale=float(tok_eval.get("cfg_scale", 1.0)),
            flow_shift=float(tok_eval.get("flow_shift", 1.0)),
            mar_n_mar_steps=int(mar_eval.get("n_mar_steps", 8)),
            mar_n_flow_steps=int(mar_eval.get("n_flow_steps", 5)),
            mar_temperature=float(mar_eval.get("temperature", 1.0)),
            mar_flow_solver=str(mar_eval.get("flow_solver", "euler")),
            mar_flow_shift=float(mar_eval.get("flow_shift", 1.0)),
            mar_cfg_scale=float(mar_eval.get("cfg_scale", 1.0)),
            ce_cache=self._ce_codebook_rates_cache,
        )

    def _guarded_predict_and_align(self, images, responses, key, *, context: str):
        """``_predict_and_align`` wrapped in a narrow CUDA-OOM guard.

        Returns the aligned ``(pred, target)`` pair, or ``None`` if the
        prediction ran out of GPU memory — in which case the caller skips this
        batch's metric update. Eval must never abort a training run: freeing the
        cache and dropping one batch's contribution is strictly better than
        crashing. Heavy passes (validation_step, latent-quality, loader-corr,
        noise-corrected-R²) all route through here so the guard lives in one
        place.

        The guard is deliberately NARROW. Only out-of-memory is swallowed:
        - ``torch.cuda.OutOfMemoryError`` (the explicit class), and
        - a generic ``RuntimeError`` whose message contains "out of memory"
          (older / nested allocator failures still surface this way).
        Any other ``RuntimeError`` — notably "CUDA error: device-side assert
        triggered" from an out-of-range codebook/embedding index — is a genuine
        bug and MUST propagate loudly; silently skipping it would hide model or
        data corruption behind a quietly missing batch.

        ``context`` names the calling pass and only appears in the warning log.

        DDP note: returning ``None`` makes a caller skip a per-batch metric
        update inside its loop. This is gather-safe — every collective in the
        eval epoch (``_gather_session_tensor_dict`` / ``all_reduce`` / barrier)
        fires unconditionally once per rank AFTER the loop, regardless of how
        many batches each rank processed or what its accumulators contain. See
        the longer note in ``on_validation_epoch_end``.
        """
        try:
            return self._predict_and_align(images, responses, key)
        except torch.cuda.OutOfMemoryError as e:
            logger.warning(f"{context}: CUDA OOM in _predict_and_align (session={key}); skipping batch metrics. {e!r}")
            torch.cuda.empty_cache()
            return None
        except RuntimeError as e:
            # Narrow to OOM-only: a device-side assert / illegal index also
            # arrives as a RuntimeError but is NOT recoverable and must not be
            # swallowed, so we re-raise everything that isn't out-of-memory.
            if "out of memory" in str(e).lower():
                logger.warning(
                    f"{context}: CUDA out-of-memory in _predict_and_align "
                    f"(session={key}); skipping batch metrics. {e!r}"
                )
                torch.cuda.empty_cache()
                return None
            raise

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        key, batch_dict = batch
        images = batch_dict["image"]
        responses = batch_dict["response"]
        n_units = batch_dict["n_units"]

        # Eval must never break training: an OOM in any prediction path
        # (sample_tokens, marginal einsum, codebook-table build for huge V)
        # skips this batch's metric update. See _guarded_predict_and_align.
        result = self._guarded_predict_and_align(images, responses, key, context="validation_step")
        if result is None:
            return
        pred, target = result

        T = pred.shape[1]

        # Rate correlation: always computed (sum over time → cross-image Pearson)
        pred_rate = pred.float().sum(dim=1)  # (B, N)
        target_rate = target.float().sum(dim=1)  # (B, N)
        self._rate_corr.update(key, pred_rate, target_rate, n_units)

        # Temporal correlation: only meaningful when T > 1
        if T > 1:
            corr = batch_temporal_correlation(pred.float(), target.float())  # (B, N)
            self._temporal_corr.update(key, corr, n_units)

    def _log_corr_results(self, corrs: dict[str, torch.Tensor], prefix: str, prog_bar: bool = False) -> float | None:
        """Log per-session and overall correlation, return overall value.

        The incoming ``corrs`` dict has already been all-gathered across ranks
        (see ``_gather_session_tensor_dict``), so values are identical on every
        rank. We log rank-zero-only to avoid Lightning's sync_dist allreduce
        per key — those collectives risk a DDP validation deadlock and are pure
        overhead here.
        """
        all_corrs = []
        subject_corrs: dict[str, list[torch.Tensor]] = {}

        log_kwargs: dict[str, Any] = {"rank_zero_only": True}

        for key, corr in corrs.items():
            self.log(f"{prefix}/{key}", corr.nanmean().item(), prog_bar=False, **log_kwargs)
            all_corrs.append(corr)
            if self._subject_info and key in self._subject_info:
                subj = self._subject_info[key]
                subject_corrs.setdefault(subj, []).append(corr)

        if not all_corrs:
            return None

        overall = torch.hstack(all_corrs).nanmean().item()
        self.log(f"{prefix}/overall", overall, prog_bar=prog_bar, **log_kwargs)
        for subj, corr_list in subject_corrs.items():
            self.log(
                f"{prefix}_subject/{subj}",
                torch.hstack(corr_list).nanmean().item(),
                prog_bar=False,
                **log_kwargs,
            )
        return overall

    def on_validation_epoch_end(self):
        # --- Rate correlation (always available) ---
        # DDP: sessions are partitioned across ranks, so gather per-session
        # dicts before reducing to overall/subject means.
        #
        # Batch-skipping (e.g. validation_step dropping an OOM batch) is
        # gather-safe: `_gather_session_tensor_dict` uses `all_gather_object`,
        # which is collective-symmetric per rank REGARDLESS of dict contents —
        # objects may differ across ranks and a rank may even contribute an
        # empty dict. The gathers and the closing `dist.barrier()` all fire
        # unconditionally once per rank here, independent of how many batches
        # each rank processed. Skipping a batch changes accumulator contents,
        # never the number of collective calls, so it cannot desync NCCL. No
        # zero-count placeholder is needed.
        rate_corrs = _gather_session_tensor_dict(self._rate_corr.compute())
        self._log_corr_results(rate_corrs, "val/rate_corr", prog_bar=True)

        # --- Temporal correlation (T > 1 only) ---
        temporal_corrs = _gather_session_tensor_dict(self._temporal_corr.compute())
        if temporal_corrs:
            self._log_corr_results(temporal_corrs, "val/corr", prog_bar=True)

        # --- Neuron trace plots (uses rate_corrs for diverse-by-perf selection). ---
        # Plotting must never crash training. Late-bound imports inside the
        # plotting path (e.g. ntt_downstream.encoding.models.enc.viz) can disappear if
        # the working tree is checked out to a branch where they don't exist
        # while training is running — editable install means the import is a
        # live filesystem read. Catch broadly, log, continue.
        try:
            self._maybe_plot_neuron_traces(rate_corrs)
        except Exception as e:
            logger.warning(f"plot: neuron-trace plotting failed; continuing training: {e!r}")

        self._rate_corr.reset()
        self._temporal_corr.reset()

        # Skip extra eval passes during sanity check — the above is enough
        # for a step-0 baseline. The full passes are slow.
        if self.trainer is not None and self.trainer.sanity_checking:
            return

        # Scalar logs below are already aggregated across ranks inside the
        # compute helpers. Log rank-zero-only so Lightning's sync_dist machinery
        # doesn't schedule per-key allreduces (which would deadlock if any
        # upstream branch exited asymmetrically).
        rz: dict[str, Any] = {"rank_zero_only": True}

        # --- Val repeat-averaged metrics ---
        if self._val_avg_dataloader_fn is not None:
            val_avg_max = self.cfg.trainer.get("eval_max_batches_val_avg", None)
            if val_avg_max is None or int(val_avg_max) > 0:
                avg = self._compute_loader_correlation(
                    self._val_avg_dataloader_fn,
                    max_batches=None if val_avg_max is None else int(val_avg_max),
                )
                if avg is not None:
                    self.log("val/rate_corr_trial_avg/overall", cast(float, avg["rate_corr"]), prog_bar=True, **rz)
                    if avg["temporal_corr"] is not None:
                        self.log("val/corr_trial_avg/overall", avg["temporal_corr"], prog_bar=False, **rz)

        # --- Noise-corrected R² (Pospisil & Bair 2021) ---
        if self._val_dataloader_fn is not None:
            nc_r2_max = int(self.cfg.trainer.get("eval_max_batches_nc_r2", 200))
            if nc_r2_max > 0:
                r2 = self._compute_noise_corrected_r2(max_batches=nc_r2_max)
                if r2 is not None:
                    self.log("val_nc_r2/overall", r2["mean"], prog_bar=True, **rz)
                    self.log("val_nc_r2/median", r2["median"], prog_bar=False, **rz)
                    for sess, val in r2["per_session"].items():
                        self.log(f"val_nc_r2/{sess}", val, prog_bar=False, **rz)

        # --- Latent quality for generative modes (diffusion / flow_matching) ---
        latent_q = self._compute_latent_quality(
            max_batches=int(self.cfg.trainer.get("eval_max_batches_latent_quality", 20))
        )
        if latent_q is not None:
            self.log("val/latent_cosine_sim", latent_q["cosine_sim"], prog_bar=False, **rz)
            self.log("val/latent_r2", latent_q["r2"], prog_bar=False, **rz)

        # --- Train correlation (limited batches) ---
        if self._train_dataloader_fn is not None:
            train = self._compute_loader_correlation(
                self._train_dataloader_fn,
                max_batches=int(self.cfg.trainer.get("eval_max_batches_train", 50)),
            )
            if train is not None:
                self.log("train/rate_corr/overall", cast(float, train["rate_corr"]), prog_bar=False, **rz)
                if train["temporal_corr"] is not None:
                    self.log("train/corr/overall", train["temporal_corr"], prog_bar=False, **rz)

        # Hard resync before returning so any rank that raced through the eval
        # passes faster waits for the rest. Cheap insurance against subtle
        # drift between ranks.
        if _ddp_is_active():
            dist.barrier()

    # ------------------------------------------------------------------
    # Eval-time neuron trace plotting
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _maybe_plot_neuron_traces(self, rate_corrs: dict[str, torch.Tensor]) -> None:
        """Log two figures per validation to MLflow: random-fixed + diverse-by-perf.

        Runs once per validation epoch when enabled by ``trainer.plot_enabled``
        (default True) and the cadence gate ``trainer.plot_every_n_val``.
        Skipped during sanity-check.
        """
        if self.trainer is not None and self.trainer.sanity_checking:
            return
        if not bool(self.cfg.trainer.get("plot_enabled", True)):
            return
        # MLflow log_figure is non-collective; only rank 0 uploads to avoid
        # duplicate artifacts in DDP runs.
        if _ddp_is_active() and not self.trainer.is_global_zero:
            return

        self._plot_val_counter += 1
        every_n = int(self.cfg.trainer.get("plot_every_n_val", 1))
        if every_n <= 0 or (self._plot_val_counter % every_n) != 0:
            return
        if self._val_dataloader_fn is None or self._session_info is None:
            return

        # Resolve target session once.
        if self._plot_target_session is None:
            configured = self.cfg.trainer.get("plot_session_id", None)
            if configured is not None and str(configured) in self._session_info:
                self._plot_target_session = str(configured)
            else:
                self._plot_target_session = sorted(self._session_info.keys())[0]

        target_session = self._plot_target_session

        try:
            loader = self._val_dataloader_fn()
        except Exception as e:
            logger.warning(f"plot: could not create val dataloader: {e}")
            return

        # Grab one single-trial batch matching the target session. Cap the scan
        # in case the target session is deep in the sampler ordering.
        model_dtype = next(self._net.parameters()).dtype
        max_scan = int(self.cfg.trainer.get("plot_max_scan_batches", 50))
        found = None
        for i, (key, batch_dict) in enumerate(loader):
            if key == target_session:
                found = (key, batch_dict)
                break
            if i + 1 >= max_scan:
                break
        if found is None:
            logger.warning(f"plot: no val batch found for session {target_session} within {max_scan} batches")
            return
        key, batch_dict = found

        images = batch_dict["image"].to(self.device, model_dtype)
        responses = batch_dict["response"].to(self.device, model_dtype)

        self._net.eval()
        result = self._guarded_predict_and_align(images, responses, key, context="plot_neuron_traces")
        if result is None:
            return
        pred, target = result  # (B, T, N)

        # Take first trial.
        pred0 = pred[0]  # (T, N)
        target0 = target[0]  # (T, N)
        n_total = pred0.shape[1]

        pred_psth = None
        target_psth = None
        image_ids = batch_dict.get("image_id", None)
        if image_ids is not None and image_ids.numel() == pred.shape[0]:
            plot_image_id = int(image_ids[0].item())
            pred_psth, target_psth = self._lookup_repeat_averaged_trace(
                session_id=target_session,
                image_id=plot_image_id,
                model_dtype=model_dtype,
                max_scan=max_scan,
            )

        from ntt_downstream.encoding.models.enc.viz import (
            pick_diverse_neurons,
            pick_random_neurons,
        )

        if target_session not in self._plot_random_neurons:
            seed = int(self.cfg.trainer.get("plot_seed", 0))
            self._plot_random_neurons[target_session] = pick_random_neurons(n_total=n_total, k=10, seed=seed)
        rand_idx = self._plot_random_neurons[target_session]

        session_rate_corrs = rate_corrs.get(target_session, None)
        if session_rate_corrs is not None:
            diverse_idx = pick_diverse_neurons(session_rate_corrs, k=10)
        else:
            diverse_idx = rand_idx  # fallback

        # Time axis in seconds — use eval rebin rate if set, else native.
        rebin_hz = self._get_eval_rebin_hz()
        time_hz = rebin_hz if rebin_hz is not None else self._get_native_hz()

        self._log_trace_figure(
            pred0,
            target0,
            rand_idx,
            target_session,
            kind="random",
            pearson_r=session_rate_corrs,
            time_axis_hz=time_hz,
            pred_psth=pred_psth,
            target_psth=target_psth,
        )
        self._log_trace_figure(
            pred0,
            target0,
            diverse_idx,
            target_session,
            kind="diverse",
            pearson_r=session_rate_corrs,
            time_axis_hz=time_hz,
            pred_psth=pred_psth,
            target_psth=target_psth,
        )

    def _log_trace_figure(
        self,
        pred0: torch.Tensor,
        target0: torch.Tensor,
        neuron_idx: list[int],
        session_id: str,
        *,
        kind: str,
        pearson_r: torch.Tensor | None,
        time_axis_hz: float | None,
        pred_psth: torch.Tensor | None = None,
        target_psth: torch.Tensor | None = None,
    ) -> None:
        """Build one figure, log it to MLflow, and close it."""
        import matplotlib.pyplot as plt

        from ntt_downstream.encoding.models.enc.viz import plot_neuron_traces

        fig = plot_neuron_traces(
            pred0,
            target0,
            neuron_idx=neuron_idx,
            session_id=session_id,
            global_step=self.global_step,
            kind=kind,
            pearson_r=pearson_r,
            time_axis_hz=time_axis_hz,
            pred_psth=pred_psth,
            target_psth=target_psth,
        )
        try:
            # Lightning's ``Logger`` base has no ``experiment``/``run_id``; they are
            # probed with hasattr below, so treat it as an MLflow-style logger.
            mlflow_logger = cast(Any, self.logger)
            if mlflow_logger is not None and hasattr(mlflow_logger, "experiment") and hasattr(mlflow_logger, "run_id"):
                artifact_file = f"neuron_traces/{kind}/step_{self.global_step:07d}.png"
                mlflow_logger.experiment.log_figure(mlflow_logger.run_id, fig, artifact_file)
        except Exception as e:
            logger.warning(f"plot: failed to log figure to MLflow: {e}")
        finally:
            plt.close(fig)

    def _lookup_repeat_averaged_trace(
        self,
        *,
        session_id: str,
        image_id: int,
        model_dtype: torch.dtype,
        max_scan: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return repeat-averaged prediction/response for one ``(session, image_id)``.

        Uses the repeat-averaged validation dataloader, which has already
        averaged all presentations of each image in the dataset. This avoids
        constructing a PSTH from whatever repeats happen to share a raw
        single-trial batch.
        """
        if self._val_avg_dataloader_fn is None:
            return None, None

        try:
            loader = self._val_avg_dataloader_fn()
        except Exception as e:
            logger.warning(f"plot: could not create repeat-averaged val dataloader: {e}")
            return None, None

        for i, (key, batch_dict) in enumerate(loader):
            if key == session_id:
                image_ids = batch_dict.get("image_id", None)
                if image_ids is not None:
                    matches = image_ids == image_id
                    if bool(matches.any().item()):
                        images = batch_dict["image"][matches].to(self.device, model_dtype)
                        responses = batch_dict["response"][matches].to(self.device, model_dtype)
                        result = self._guarded_predict_and_align(
                            images, responses, key, context="repeat_averaged_trace"
                        )
                        if result is None:
                            return None, None
                        pred, target = result
                        return pred[0], target[0]
            if i + 1 >= max_scan:
                break

        logger.debug(
            "plot: no repeat-averaged trace found for image_id=%s in session %s within %s batches",
            image_id,
            session_id,
            max_scan,
        )
        return None, None

    # ------------------------------------------------------------------
    # Latent quality eval (for generative modes: diffusion / flow_matching / energy)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_latent_quality(self, max_batches: int = 20) -> dict[str, float] | None:
        """Compare sampled latents to encoded ground truth.

        Only for base model (image_encoding) with diffusion/flow_matching, where
        training diagnostics are meaningless (conditioning ≠ prediction). For MAR,
        latent quality is captured by training diagnostics (MSE modes) or
        val/rate_corr (generative modes). Uses cross-batch variance for R².
        """
        if self.tokenizer_wrapper is None or self._val_dataloader_fn is None:
            return None

        model_type = str(self.cfg.model.get("model_type", "image_encoding"))
        if model_type != "image_encoding":
            return None

        loss_mode = str(self.cfg.model.get("loss_mode", "poisson"))
        if loss_mode not in ("diffusion", "flow_matching", "energy"):
            return None

        try:
            loader = self._val_dataloader_fn()
        except Exception:
            return None

        # Read ``self.model`` directly (not ``self._net``): tests bind this method
        # onto lightweight stand-ins that only carry a ``model`` attribute.
        model = self.model
        assert model is not None, "configure_model() has not run"
        model.eval()
        self._ensure_tokenizer_device()
        model_dtype = next(model.parameters()).dtype

        all_cos = []
        all_sampled = []
        all_target = []

        for i, (key, batch_dict) in enumerate(loader):
            if i >= max_batches:
                break
            # The flow/diffusion sample_tokens + tokenizer encode below is the
            # most memory-hungry eval path (MC sampling on the primary generative
            # recipes). Wrap it in the same NARROW OOM guard as
            # `_guarded_predict_and_align`: on out-of-memory we free the cache and
            # skip the batch, but any other RuntimeError (e.g. a device-side
            # assert from an out-of-range codebook index) must propagate. Skipping
            # is safe here — this method has no collectives, the accumulators
            # tolerate fewer batches, and it is called BEFORE any DDP barrier, so
            # an unguarded asymmetric OOM would abort one rank / hang NCCL.
            try:
                images = batch_dict["image"].to(self.device, model_dtype)
                responses = batch_dict["response"].to(self.device, model_dtype)

                # Sample latents from generative head
                model_out = model(images, key, return_token_prediction=True)
                eval_cfg = self.cfg.trainer.get("tokenizer_eval", {})
                sampled = model.sample_tokens(
                    model_out["token_prediction"],
                    n_flow_steps=int(eval_cfg.get("n_flow_steps", 5)),
                    temperature=float(eval_cfg.get("sample_temperature", 1.0)),
                    cfg_scale=float(eval_cfg.get("cfg_scale", 1.0)),
                    flow_solver=str(eval_cfg.get("flow_solver", "euler")),
                    flow_shift=float(eval_cfg.get("flow_shift", 1.0)),
                )

                # Encode ground truth to latent space
                responses_btn = responses.transpose(1, 2)
                with torch.autocast(device_type="cuda", enabled=False):
                    target = self.tokenizer_wrapper.encode(responses_btn.float(), key)["target"]

                # LDNS encode-long path: target is the full 250-bin LDNS latent but
                # `sampled` only covers the post-stim slice. Match shapes before
                # flattening so the cosine-sim / MSE comparison is well-defined.
                if self._ldns_splice_active():
                    slice_t = int(self.cfg.model.compress_n_target_slice)
                    target = target[:, :, :slice_t, :]

                s_f = sampled.float().reshape(-1, sampled.shape[-1])
                t_f = target.float().reshape(-1, target.shape[-1])

                cos = torch.nn.functional.cosine_similarity(s_f, t_f, dim=-1)
            except torch.cuda.OutOfMemoryError as e:
                logger.warning(f"latent_quality: CUDA OOM in sample/encode (session={key}); skipping batch. {e!r}")
                torch.cuda.empty_cache()
                continue
            except RuntimeError as e:
                # Narrow to OOM-only; re-raise device-side asserts and friends.
                if "out of memory" in str(e).lower():
                    logger.warning(
                        f"latent_quality: CUDA out-of-memory in sample/encode (session={key}); skipping batch. {e!r}"
                    )
                    torch.cuda.empty_cache()
                    continue
                raise

            all_cos.append(cos)
            all_sampled.append(s_f)
            all_target.append(t_f)

        if not all_cos:
            return None

        cos_mean = torch.cat(all_cos).mean().item()
        # R² with cross-batch variance (more stable than per-batch)
        all_s = torch.cat(all_sampled)
        all_t = torch.cat(all_target)
        ss_res = ((all_s - all_t) ** 2).mean(dim=0)
        ss_tot = all_t.var(dim=0).clamp(min=1e-8)
        r2 = (1 - ss_res / ss_tot).mean().item()

        return {"cosine_sim": cos_mean, "r2": r2}

    # ------------------------------------------------------------------
    # Full-pass correlation eval
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_loader_correlation(
        self,
        dataloader_fn,
        max_batches: int | None = None,
    ) -> dict[str, float | None] | None:
        """Run correlation eval over an arbitrary dataloader.

        Returns:
            ``{"rate_corr": float, "temporal_corr": float | None}`` or None on failure.

        DDP invariant: every collective below (the two gathers + the temporal
        flag all-reduce) must execute on every rank. Do NOT add ``try/except``
        that swallows an error on one rank — that produces a silent NCCL hang.
        """
        loader = dataloader_fn()

        rate_acc = OnlinePearsonAccumulator()
        temporal_acc = TemporalCorrAccumulator()
        has_temporal = False

        self._net.eval()
        model_dtype = next(self._net.parameters()).dtype
        n_batches = 0
        for key, batch_dict in loader:
            images = batch_dict["image"].to(self.device, model_dtype)
            responses = batch_dict["response"].to(self.device, model_dtype)
            n_units = batch_dict["n_units"]

            # OOM-guard the prediction. On OOM we `continue` (skip this batch's
            # metric update) rather than `return`, so the post-loop collectives
            # below still fire on every rank — see the DDP invariant note.
            result = self._guarded_predict_and_align(images, responses, key, context="loader_correlation")
            n_batches += 1
            if result is not None:
                pred, target = result
                pred_rate = pred.float().sum(dim=1)
                target_rate = target.float().sum(dim=1)
                rate_acc.update(key, pred_rate, target_rate, n_units)

                if pred.shape[1] > 1:
                    has_temporal = True
                    corr = batch_temporal_correlation(pred.float(), target.float())
                    temporal_acc.update(key, corr, n_units)

            if max_batches is not None and n_batches >= max_batches:
                break

        # --- Collectives: every rank must reach these unconditionally. ---
        rate_corrs = _gather_session_tensor_dict(rate_acc.compute())

        if _ddp_is_active():
            flag = torch.tensor(1 if has_temporal else 0, device=self.device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            has_temporal = bool(flag.item())

        # Always call the temporal gather when any rank reported temporal data,
        # so the collective count stays balanced across ranks. Doing it inside
        # `if has_temporal` is safe because every rank sees the same flag here.
        temporal_corrs: dict[str, torch.Tensor] = {}
        if has_temporal:
            temporal_corrs = _gather_session_tensor_dict(temporal_acc.compute())
        # --- End collectives. ---

        if not rate_corrs:
            return None

        rate_overall = torch.hstack(list(rate_corrs.values())).nanmean().item()
        temporal_overall = None
        if temporal_corrs:
            temporal_overall = torch.hstack(list(temporal_corrs.values())).nanmean().item()

        return {"rate_corr": rate_overall, "temporal_corr": temporal_overall}

    # ------------------------------------------------------------------
    # Noise-corrected R²
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_noise_corrected_r2(self, max_batches: int | None = None) -> dict | None:
        """Compute noise-corrected R² (Pospisil & Bair 2021) on val data.

        Uses ``_predict_and_align`` for all model types, groups predictions by
        (session, image_id), and returns per-session + overall statistics.

        Returns:
            ``{"mean": float, "median": float, "per_session": {sess: float}}``
            or None on failure.

        DDP invariant: the ``_gather_session_tensor_dict`` call near the end
        must fire on every rank. Do NOT wrap loader creation in try/except
        that returns None — asymmetric exits deadlock NCCL.
        """
        assert self._val_dataloader_fn is not None, "val dataloader is required"
        loader = self._val_dataloader_fn()

        self._net.eval()
        model_dtype = next(self._net.parameters()).dtype
        native_hz = self._get_native_hz()

        # End-of-window truncation (None = use full trace after skip)
        eval_end_sec = self.cfg.trainer.get("eval_end_seconds", None)

        session_data: dict[str, dict[int, dict[str, list]]] = defaultdict(
            lambda: defaultdict(lambda: {"pred": [], "resp": []})
        )
        session_n_units: dict[str, int] = {}

        n_batches = 0
        for key, batch_dict in loader:
            images = batch_dict["image"].to(self.device, model_dtype)
            responses = batch_dict["response"].to(self.device, model_dtype)
            image_ids = batch_dict["image_id"]
            n_units = batch_dict["n_units"]
            session_n_units[key] = n_units

            # Unified prediction path for all model types, OOM-guarded. On OOM we
            # `continue` (skip this batch) rather than `return`, so the gather
            # below still fires on every rank — see the DDP invariant note.
            result = self._guarded_predict_and_align(images, responses, key, context="noise_corrected_r2")
            if result is None:
                n_batches += 1
                if max_batches is not None and n_batches >= max_batches:
                    break
                continue
            pred_btn, resp_btn = result
            # (B, T', N) — skip + rebin already applied

            # Apply eval_end_seconds truncation if configured
            if eval_end_sec is not None:
                skip_sec = float(self.cfg.trainer.get("eval_skip_seconds", 0.0))
                rebin_hz = self._get_eval_rebin_hz() or native_hz
                keep_bins = int(round((float(eval_end_sec) - skip_sec) * rebin_hz))
                pred_btn = pred_btn[:, :keep_bins, :]
                resp_btn = resp_btn[:, :keep_bins, :]

            # Sum across time → scalar per neuron per image
            pred_scalar = pred_btn.sum(dim=1).float()  # (B, N)
            resp_scalar = resp_btn.sum(dim=1).float()  # (B, N)

            for i in range(images.shape[0]):
                iid = int(image_ids[i].item())
                session_data[key][iid]["pred"].append(pred_scalar[i].clone())  # (N,)
                session_data[key][iid]["resp"].append(resp_scalar[i].clone())  # (N,)

            n_batches += 1
            if max_batches is not None and n_batches >= max_batches:
                break

        # DDP: every rank computed R² for its own disjoint sessions. The
        # gather below must run on every rank — so don't short-circuit here.
        per_session_r2_tensor: dict[str, torch.Tensor] = {}
        for sid, image_dict in session_data.items():
            n_units = session_n_units[sid]
            image_ids_sorted = sorted(image_dict.keys())
            n_images = len(image_ids_sorted)
            max_repeats = max(len(image_dict[iid]["resp"]) for iid in image_ids_sorted)

            # Need at least 2 images and 2 repeats for meaningful R²
            if n_images < 2 or max_repeats < 2:
                continue

            resp_all = torch.full((n_units, n_images, max_repeats), float("nan"), device=self.device)
            pred_mean = torch.zeros(n_units, n_images, device=self.device)

            for j, iid in enumerate(image_ids_sorted):
                resps = torch.stack(image_dict[iid]["resp"])  # (n_reps, N)
                preds = torch.stack(image_dict[iid]["pred"])  # (n_reps, N)
                n_reps = resps.shape[0]
                resp_all[:, j, :n_reps] = resps.T
                pred_mean[:, j] = preds.mean(dim=0)

            r2 = noise_corrected_r2(resp_all, pred_mean)  # (N_units,)
            per_session_r2_tensor[sid] = r2

        # Gather full per-session tensors across ranks, then reduce to scalars.
        per_session_r2_tensor = _gather_session_tensor_dict(per_session_r2_tensor)
        if not per_session_r2_tensor:
            return None

        per_session: dict[str, float] = {sid: r2.nanmean().item() for sid, r2 in per_session_r2_tensor.items()}
        all_r2s = list(per_session_r2_tensor.values())
        all_r2_cat = torch.cat(all_r2s)
        return {
            "mean": all_r2_cat.nanmean().item(),
            "median": all_r2_cat.nanmedian().item(),
            "per_session": per_session,
        }

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        cfg = self.cfg
        base_lr = cfg.trainer.get("lr", 1e-3)
        backbone_lr_mult = cfg.trainer.get("backbone_lr_multiplier", 0.1)
        backbone_lr = cfg.trainer.get("backbone_lr", base_lr * backbone_lr_mult)

        backbone_params = list(self._net.backbone.parameters())
        backbone_param_ids = {id(p) for p in backbone_params}

        energy_params: list[torch.nn.Parameter] = []
        if getattr(self.model, "energy_loss", None) is not None:
            energy_params = list(self._net.energy_loss.parameters())
        energy_param_ids = {id(p) for p in energy_params}

        other_params = [
            p for p in self._net.parameters() if id(p) not in backbone_param_ids and id(p) not in energy_param_ids
        ]

        param_groups = [
            {"params": other_params, "lr": base_lr},
            {"params": backbone_params, "lr": backbone_lr},
        ]

        if energy_params:
            energy_mult = float(cfg.trainer.get("energy_lr_multiplier", 0.25))
            energy_lr = float(cfg.trainer.get("energy_lr", base_lr * energy_mult))
            param_groups.append({"params": energy_params, "lr": energy_lr})
            logger.info(
                f"Energy head: {sum(p.numel() for p in energy_params):,} params "
                f"at lr={energy_lr} (base_lr={base_lr}, mult={energy_mult})"
            )

        # Decoder fine-tuning: add tokenizer decoder params as a third group
        tokenizer_mode = str(cfg.trainer.get("tokenizer_mode", "frozen"))
        if tokenizer_mode == "finetune_decoder" and self.tokenizer_wrapper is not None:
            decoder_params = self.tokenizer_wrapper.get_decoder_params()
            if decoder_params:
                decoder_lr = cfg.trainer.get("decoder_lr", None)
                if decoder_lr is None:
                    decoder_lr = float(base_lr) * 0.1
                else:
                    decoder_lr = float(decoder_lr)
                param_groups.append({"params": decoder_params, "lr": decoder_lr})
                logger.info(
                    f"Decoder fine-tuning: {sum(p.numel() for p in decoder_params):,} params at lr={decoder_lr}"
                )

        optimizer = AdamW(
            param_groups,
            betas=tuple(cfg.trainer.get("betas", [0.9, 0.999])),
            weight_decay=cfg.trainer.get("weight_decay", 0.01),
            eps=cfg.trainer.get("eps", 1e-8),
        )

        scheduler_cfg = cfg.trainer.get("scheduler", {})
        T_max = int(scheduler_cfg.get("T_max", 100000))
        eta_min = float(scheduler_cfg.get("eta_min", 1e-4))
        warmup_steps = int(scheduler_cfg.get("warmup_steps", 1000))

        # When `trainer.max_steps: -1` is used (n_epochs drives the run), the
        # default `scheduler.T_max: ${trainer.max_steps}` interpolation resolves
        # to -1, which makes CosineAnnealingLR collapse to eta_min after the
        # first post-warmup step. Fall back to Lightning's estimate.
        if T_max <= 0:
            estimated = self.trainer.estimated_stepping_batches
            if estimated is None or estimated == float("inf") or int(estimated) <= 0:
                raise ValueError(
                    f"scheduler.T_max={T_max} and trainer.estimated_stepping_batches="
                    f"{estimated}. Set scheduler.T_max explicitly or bound the run with "
                    f"trainer.max_steps / trainer.n_epochs."
                )
            T_max = int(estimated)
            logger.info(f"scheduler.T_max <= 0; falling back to estimated_stepping_batches={T_max}")

        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

        if warmup_steps > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
            )
        else:
            scheduler = cosine

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ------------------------------------------------------------------
    # Checkpoint hooks for decoder fine-tuning
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint):
        """Save fine-tuned decoder parameters alongside the main checkpoint."""
        if self.tokenizer_wrapper is None:
            return
        if str(self.cfg.trainer.get("tokenizer_mode", "frozen")) != "finetune_decoder":
            return
        decoder_sd = {}
        for name, param in self.tokenizer_wrapper.tokenizer.named_parameters():
            if param.requires_grad:
                decoder_sd[name] = param.data.cpu()
        if decoder_sd:
            checkpoint["decoder_state_dict"] = decoder_sd

    def on_load_checkpoint(self, checkpoint):
        """Stash decoder state dict for restoration after configure_model()."""
        decoder_sd = checkpoint.pop("decoder_state_dict", None)
        if decoder_sd is not None:
            self._pending_decoder_state_dict = decoder_sd
