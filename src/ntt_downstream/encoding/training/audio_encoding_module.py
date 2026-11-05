"""Lightning module for auditory encoding models."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any, cast

import lightning as L
import torch
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ntt_downstream.encoding.models.enc.audio.base_model import AudioEncodingModel
from ntt_downstream.encoding.models.enc.audio.model_factory import create_audio_model
from ntt_downstream.encoding.models.enc.hiera.evals import (
    OnlinePearsonAccumulator,
    TemporalCorrAccumulator,
    batch_temporal_correlation,
)
from ntt_downstream.encoding.tokenizer_wrappers.encoding_wrapper import TokenizerWrapper

logger = logging.getLogger(__name__)


class AudioEncodingLitModule(L.LightningModule):
    """Lightning module for spectrogram -> neural response encoding.

    Validation metrics:
    - val/corr/: Per-stimulus temporal Pearson correlation.
    - val/rate_corr/: Cross-stimulus Pearson on time-summed rates.
    - val/rate_corr_trial_avg/overall: Same, on repeat-averaged responses.
    - val/corr_trial_avg/overall: Temporal corr on repeat-averaged responses.

    Early stopping / checkpointing monitors val/rate_corr/overall.
    """

    def __init__(self, cfg, session_info: dict[str, int] | None = None, subject_info: dict[str, str] | None = None):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self._session_info = session_info
        self._subject_info = subject_info
        self.model: AudioEncodingModel | None = None
        self.tokenizer_wrapper: TokenizerWrapper | None = None

        self._temporal_corr = TemporalCorrAccumulator()
        self._rate_corr = OnlinePearsonAccumulator()

        self._val_avg_dataloader_fn: Callable[[], DataLoader] | None = None
        self._val_dataloader_fn: Callable[[], DataLoader] | None = None
        self._train_dataloader_fn: Callable[[], DataLoader] | None = None

        # Eval-time neuron trace plotting.
        self._plot_target_session: str | None = None
        self._plot_random_neurons: dict[str, list[int]] = {}
        self._plot_val_counter: int = 0

    # ``self.model`` / ``self.tokenizer_wrapper`` start as None and are filled in
    # by ``configure_model``. Every use below runs after that (or is guarded by
    # an explicit ``is not None`` check), so these narrow once instead of at
    # each call site.
    @property
    def _net(self) -> AudioEncodingModel:
        assert self.model is not None, "configure_model() has not run"
        return self.model

    @property
    def _tok(self) -> TokenizerWrapper:
        assert self.tokenizer_wrapper is not None, "tokenizer is not enabled"
        return self.tokenizer_wrapper

    def set_val_average_dataloader(self, dataloader_fn):
        self._val_avg_dataloader_fn = dataloader_fn

    def set_val_dataloader(self, dataloader_fn):
        self._val_dataloader_fn = dataloader_fn

    def set_train_dataloader(self, dataloader_fn):
        self._train_dataloader_fn = dataloader_fn

    def configure_model(self):
        if self.model is not None:
            return

        cfg = self.cfg
        needs_tokenizer = cfg.get("tokenizer", {}).get("enabled", False)
        token_prediction_dim = None
        n_neurons_override = None

        if needs_tokenizer:
            self.tokenizer_wrapper = self._build_tokenizer()

            # Calibrate latent scale if needed and dataloader is available
            if getattr(self.tokenizer_wrapper, "_needs_calibration", False):
                if self._train_dataloader_fn is not None:
                    self._calibrate_latent_scale()
                else:
                    logger.warning(
                        "Tokenizer needs latent scale calibration but train dataloader "
                        "is not yet available. Calibration will be skipped — set "
                        "tokenizer.latent_scale explicitly or ensure set_train_dataloader() "
                        "is called before configure_model()."
                    )

            # Auto-derive token_prediction_dim from tokenizer
            token_prediction_dim = self._tok.latent_dim

            # Auto-derive ce_vocab_size from discrete tokenizer for CE mode.
            if (
                str(cfg.model.get("loss_mode", "poisson")) == "cross_entropy"
                and cfg.model.get("ce_vocab_size", None) is None
            ):
                vocab_size = cast(Any, self._tok.tokenizer).get_vocab_size()
                OmegaConf.update(cfg, "model.ce_vocab_size", int(vocab_size), force_add=True)
                logger.info(f"Auto-derived ce_vocab_size={vocab_size} from tokenizer.get_vocab_size()")

            # compress_n: create session plans and override neuron counts with chunk counts
            if self._tok.is_compress_n:
                assert self._session_info is not None, "compress_n requires session_info"
                n_neurons_override = {}
                for sid, n in self._session_info.items():
                    self._tok.get_or_create_plan(sid, n)
                    n_neurons_override[sid] = self._tok.get_num_chunks(sid, n)

            # Validate window alignment
            bin_size = float(cfg.data.get("bin_size", 0.01))
            window_size = int(cfg.model.get("window_size", 10))
            model_window_s = window_size * bin_size
            tok_window_s = self._tok.tokenizer_window_length_seconds
            if tok_window_s is not None and abs(model_window_s - tok_window_s) > 1e-6:
                raise ValueError(
                    f"Model window_size={window_size} × bin_size={bin_size} = {model_window_s}s "
                    f"does not match tokenizer_window_length_seconds={tok_window_s}s. "
                    f"Set window_size={int(round(tok_window_s / bin_size))} to align."
                )

        # Derive tokens_per_window for multi-token readout
        n_tokens_per_neuron = None
        if needs_tokenizer:
            n_tokens_per_neuron = self._compute_tokens_per_window()

        assert self._session_info is not None, "session_info is required to build the model"
        self.model = create_audio_model(
            cfg,
            session_info=self._session_info,
            subject_info=self._subject_info,
            token_prediction_dim=token_prediction_dim,
            n_neurons_override=n_neurons_override,
            n_tokens_per_neuron=n_tokens_per_neuron,
        )

        if self.tokenizer_wrapper is not None:
            self.model.tokenizer_wrapper = self.tokenizer_wrapper

    def _compute_tokens_per_window(self) -> int:
        """Compute T_tok — tokens per window — from tokenizer spec.

        For tokenizers with ``token_bin_size_ms`` (e.g. Conv VAE), T_tok =
        tokenizer_window_length_seconds / (token_bin_size_ms / 1000). For others
        (e.g. cs-*), T_tok = 1 (one latent per window).
        """
        spec = getattr(self._tok.tokenizer, "spec", None)
        tbms = getattr(spec, "token_bin_size_ms", None) if spec else None
        lw = self._tok.tokenizer_window_length_seconds
        if tbms is not None and lw is not None:
            token_bin_s = float(tbms) / 1000.0
            t_tok = int(round(lw / token_bin_s))
            logger.info(f"Derived n_tokens_per_neuron={t_tok} from token_bin_size_ms={tbms}, window={lw}s")
            return max(t_tok, 1)
        return 1

    def _build_tokenizer(self):
        """Build a TokenizerWrapper from config."""
        from ntt_downstream.encoding.tokenizer_wrappers.encoding_wrapper import build_tokenizer_wrapper

        wrapper = build_tokenizer_wrapper(self.cfg, device="cpu")
        assert wrapper is not None, "tokenizer.enabled is set but no tokenizer wrapper was built"
        tok_cfg = self.cfg.get("tokenizer", {})
        lw = tok_cfg.get("tokenizer_window_length_seconds", None)
        if lw is not None:
            object.__setattr__(wrapper, "tokenizer_window_length_seconds", float(lw))
            wrapper.__post_init__()
        return wrapper

    def _calibrate_latent_scale(self, max_batches: int = 50):
        """Calibrate tokenizer latent scale from training data.

        Supports both global (``auto``) and per-channel (``auto_per_channel``)
        calibration modes.  Runs the tokenizer on CUDA when available.
        """
        from torch.utils.data import DataLoader

        from ntt_downstream.encoding.dataloading.samplers import SessionBatchSampler
        from ntt_downstream.encoding.dataloading.wingert_auditory import WingertEvalCollate

        per_channel = getattr(self.tokenizer_wrapper, "_per_channel_calibration", False)
        mode_str = "per-channel" if per_channel else "global"
        logger.info(f"Calibrating tokenizer latent scale ({mode_str})...")

        assert self._train_dataloader_fn is not None, "train dataloader is required for calibration"
        src_dl = self._train_dataloader_fn()
        batch_size = self.cfg.dataloader.get("batch_size", 32)
        cal_dl = DataLoader(
            src_dl.dataset,
            batch_sampler=SessionBatchSampler(src_dl.dataset, batch_size=batch_size, shuffle=False),
            collate_fn=WingertEvalCollate(),
            num_workers=0,
        )

        # Run tokenizer on CUDA during calibration for speed
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tok.tokenizer.to(device)

        # Temporarily disable any scaling so we measure raw tokens
        old_scale = self._tok.latent_scale
        old_means = self._tok.latent_channel_means
        old_stds = self._tok.latent_channel_stds
        object.__setattr__(self.tokenizer_wrapper, "latent_scale", 1.0)
        object.__setattr__(self.tokenizer_wrapper, "latent_channel_means", None)
        object.__setattr__(self.tokenizer_wrapper, "latent_channel_stds", None)

        all_tokens: list[torch.Tensor] | None = [] if per_channel else None
        total_sum = 0.0
        total_sum_sq = 0.0
        total_count = 0
        n_batches = 0

        with torch.no_grad():
            for i, batch in enumerate(cal_dl):
                if i >= max_batches:
                    break
                responses = batch["response"].to(device, torch.float32)
                key = batch["session_id"]
                tok_out = self._tok.encode_windowed(responses, key)
                flat = tok_out["target"].float()

                if per_channel:
                    assert all_tokens is not None
                    # Flatten to (-1, D) — trials may have different lengths
                    all_tokens.append(flat.reshape(-1, flat.shape[-1]).cpu())
                else:
                    total_sum += flat.sum().item()
                    total_sum_sq += flat.pow(2).sum().item()
                    total_count += flat.numel()
                n_batches += 1

        # Move tokenizer back to CPU to free GPU memory before training allocates
        self._tok.tokenizer.to("cpu")

        if per_channel:
            if not all_tokens:
                logger.warning("Latent scale calibration: no data, keeping defaults")
                object.__setattr__(self.tokenizer_wrapper, "latent_scale", old_scale)
                object.__setattr__(self.tokenizer_wrapper, "latent_channel_means", old_means)
                object.__setattr__(self.tokenizer_wrapper, "latent_channel_stds", old_stds)
                return
            cat = torch.cat(all_tokens, dim=0)  # (total_tokens, D)
            means = cat.mean(dim=0)  # (D,)
            stds = cat.std(dim=0).clamp(min=1e-6)  # (D,)
            # Reshape for broadcasting: (1, 1, 1, D) to match (B, W*T_tok, S, D)
            shape = [1, 1, 1, -1]
            object.__setattr__(self.tokenizer_wrapper, "latent_channel_means", means.reshape(*shape))
            object.__setattr__(self.tokenizer_wrapper, "latent_channel_stds", stds.reshape(*shape))
            logger.info(
                f"Latent scale calibrated (per-channel, {n_batches} batches): "
                f"means={means.tolist()}, stds={stds.tolist()}"
            )
        else:
            if total_count == 0:
                logger.warning("Latent scale calibration: no data, keeping scale=1.0")
                return
            global_mean = total_sum / total_count
            global_var = total_sum_sq / total_count - global_mean**2
            global_std = max(global_var, 0.0) ** 0.5
            if global_std < 1e-6:
                logger.warning(f"Latent scale calibration: std={global_std:.6f} near zero, keeping scale=1.0")
                return
            object.__setattr__(self.tokenizer_wrapper, "latent_scale", global_std)
            logger.info(f"Latent scale calibrated: {global_std:.4f} (from {n_batches} batches)")

    def on_train_epoch_start(self):
        """Advance the batch sampler's epoch so shuffling varies each epoch.

        LightningDataModule has no `on_train_epoch_start` hook in Lightning 2.x,
        so this must live on the LightningModule — otherwise the sampler's RNG
        is re-seeded with `epoch=0` every epoch and batch order is identical
        across epochs.
        """
        dm = getattr(self.trainer, "datamodule", None)
        sampler = getattr(dm, "train_sampler", None) if dm is not None else None
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.current_epoch)

    def training_step(self, batch, batch_idx):
        spec = batch["spectrogram"]
        resp = batch["response"]
        key = batch["session_id"]
        lengths = batch.get("lengths", None)
        tokenizer_targets = None
        if "tokenizer_target" in batch:
            tokenizer_targets = {
                "target": batch["tokenizer_target"],
                "tokens_per_window": int(batch["tokenizer_tokens_per_window"]),
                "n_windows": int(batch["tokenizer_n_windows"]),
            }

        # Move tokenizer to model device on first batch
        if batch_idx == 0 and self.tokenizer_wrapper is not None and tokenizer_targets is None:
            self.tokenizer_wrapper.tokenizer.to(self.device)

        # Warmup: temporarily override loss mode for initial training steps
        warmup_mode = self.cfg.trainer.get("warmup_loss_mode", None)
        warmup_steps = int(self.cfg.trainer.get("warmup_steps", 0))
        in_warmup = warmup_mode is not None and warmup_steps > 0 and self.global_step < warmup_steps

        if in_warmup:
            original_mode = self._net.loss_mode
            self._net.loss_mode = str(warmup_mode)

        out = self._net(spec, key, responses=resp, lengths=lengths, tokenizer_targets=tokenizer_targets)
        loss = out["loss"]

        if in_warmup:
            self._net.loss_mode = original_mode

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        for key_name in (
            "aux_loss",
            "energy_diversity_abs",
            "energy_diversity_ratio",
            "latent_cosine_sim",
            "latent_r2",
        ):
            if key_name in out:
                self.log(f"train/{key_name}", out[key_name], prog_bar=False, on_step=True, on_epoch=False)
        return loss

    def _decode_latent_to_rates(self, z_pred: torch.Tensor, session_key: str) -> torch.Tensor:
        """Decode tokenizer-mode latent predictions to firing rates.

        Routes through ``model.sample_tokens`` first so flow_matching / diffusion
        / energy / cross_entropy modes produce concrete tokens before decoding.
        For mse_latent / smooth_l1_latent this is a no-op; z is returned unchanged.

        Args:
            z_pred: (B, W, N, T_tok, D) or (B, W, N, D) per-window latent predictions.
            session_key: Session identifier.

        Returns:
            (B, T, N) decoded firing rates.
        """
        self._tok.tokenizer.to(z_pred.device)
        with torch.autocast(device_type="cuda", enabled=False):
            sampled = self._net.sample_tokens(z_pred.float())

            # CE path: sampled is (B, W, N, T_tok) or (B, W, N) integer indices.
            # Unsqueeze a trailing channel dim so the wrapper's 4D decode API accepts them.
            if self._net.loss_mode == "cross_entropy":
                if sampled.ndim == 4:  # (B, W, N, T_tok)
                    B, W, N, T_tok = sampled.shape
                    indices = sampled.permute(0, 2, 1, 3).reshape(B, N, W * T_tok).unsqueeze(-1)
                else:  # (B, W, N)
                    indices = sampled.transpose(1, 2).unsqueeze(-1)
                return self._tok.decode_differentiable(indices, session_key=session_key)

            # BCE / continuous: reshape (B, W, N, T_tok, D) → (B, W*T_tok, N, D)
            if sampled.ndim == 5:
                B, W, N, T_tok, D = sampled.shape
                sampled = sampled.permute(0, 1, 3, 2, 4).reshape(B, W * T_tok, N, D)

            # BCE: already in raw tokenizer latent space.
            # Continuous: predictions are in *scaled* target space; unscale to raw latent.
            if self._net.loss_mode == "bce":
                raw = sampled
            else:
                raw = self._tok._target_to_tokens(sampled.float())

            W_total = raw.shape[1]
            decoded_windows = []
            for w in range(W_total):
                z_w = raw[:, w, :, :].unsqueeze(2)  # (B, N, 1, D)
                rates_w = self._tok.decode_differentiable(z_w, session_key=session_key)
                decoded_windows.append(rates_w)
            return torch.cat(decoded_windows, dim=1)

    @torch.no_grad()
    def _predict_val_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one val batch through the model and return (pred, target) as (B, T, N)."""
        spec = batch["spectrogram"]
        resp = batch["response"]
        key = batch["session_id"]

        pred = self._net(spec, key)
        if not self._net._is_rate_mode and self.tokenizer_wrapper is not None:
            pred = self._decode_latent_to_rates(pred, key)

        T_pred = pred.shape[1]
        target = resp[:, :T_pred, :]
        return pred, target

    def validation_step(self, batch, batch_idx):
        key = batch["session_id"]
        lengths = batch.get("lengths", None)

        pred, target = self._predict_val_batch(batch)
        T_pred = pred.shape[1]

        if lengths is not None:
            effective_lengths = lengths.clamp(max=T_pred)
            t_idx = torch.arange(T_pred, device=pred.device).unsqueeze(0)
            mask = t_idx < effective_lengths.unsqueeze(1)
            pred_sum = (pred * mask.unsqueeze(2)).sum(dim=1)
            target_sum = (target * mask.unsqueeze(2)).sum(dim=1)
        else:
            pred_sum = pred.sum(dim=1)
            target_sum = target.sum(dim=1)

        n_units = pred.shape[2]
        self._rate_corr.update(key, pred_sum, target_sum, n_units)

        corr = batch_temporal_correlation(pred, target)
        self._temporal_corr.update(key, corr, n_units)

    def on_validation_epoch_end(self):
        try:
            rate_corr = self._rate_corr.compute()
            self._log_corr_results(rate_corr, "val/rate_corr", prog_bar=True)

            temp_corr = self._temporal_corr.compute()
            if temp_corr:
                self._log_corr_results(temp_corr, "val/corr")

            # Plot traces before resetting accumulators — diverse-by-perf needs rate_corr.
            self._maybe_plot_neuron_traces(rate_corr)

            self._rate_corr.reset()
            self._temporal_corr.reset()

            # Skip the extra repeat-avg pass during sanity check — above is enough.
            if self.trainer is not None and self.trainer.sanity_checking:
                return

            # Repeat-averaged validation: primary eval signal for Wingert.
            if self._val_avg_dataloader_fn is not None:
                avg = self._compute_loader_correlation(self._val_avg_dataloader_fn)
                if avg is not None:
                    self.log("val/rate_corr_trial_avg/overall", cast(float, avg["rate_corr"]), prog_bar=True)
                    if avg["temporal_corr"] is not None:
                        self.log("val/corr_trial_avg/overall", avg["temporal_corr"], prog_bar=False)
        finally:
            # Cached-target training should not keep the tokenizer resident on GPU
            # between validation passes.
            if self.tokenizer_wrapper is not None and not self._net._is_rate_mode:
                self.tokenizer_wrapper.tokenizer.to("cpu")

    # ------------------------------------------------------------------
    # Eval-time neuron trace plotting
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _maybe_plot_neuron_traces(self, rate_corrs: dict[str, torch.Tensor]) -> None:
        """Log two figures per validation to MLflow: random-fixed + diverse-by-perf."""
        if self.trainer is not None and self.trainer.sanity_checking:
            return
        if not bool(self.cfg.trainer.get("plot_enabled", True)):
            return
        # Rank-zero only — MLflow log_figure is non-collective.
        if self.trainer is not None and not self.trainer.is_global_zero:
            return

        self._plot_val_counter += 1
        every_n = int(self.cfg.trainer.get("plot_every_n_val", 1))
        if every_n <= 0 or (self._plot_val_counter % every_n) != 0:
            return
        if self._val_dataloader_fn is None or self._session_info is None:
            return

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

        # Grab one batch matching the target session. Cap the scan in case
        # the target session is deep in the sampler ordering.
        max_scan = int(self.cfg.trainer.get("plot_max_scan_batches", 50))
        found = None
        for i, b in enumerate(loader):
            if b["session_id"] == target_session:
                found = b
                break
            if i + 1 >= max_scan:
                break
        if found is None:
            logger.warning(f"plot: no val batch found for session {target_session} within {max_scan} batches")
            return

        model_dtype = next(self._net.parameters()).dtype
        batch = {
            "spectrogram": found["spectrogram"].to(self.device, model_dtype),
            "response": found["response"].to(self.device, model_dtype),
            "session_id": found["session_id"],
            "lengths": found.get("lengths", None),
        }

        self._net.eval()  # defensive — dropout would otherwise skew the plotted traces
        pred, target = self._predict_val_batch(batch)  # (B, T, N)

        pred0 = pred[0]  # (T, N)
        target0 = target[0]
        n_total = pred0.shape[1]

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
            diverse_idx = rand_idx

        # Time axis from data.bin_size (audio doesn't use a rebin step).
        bin_size = float(self.cfg.data.get("bin_size", 0.01))
        time_hz = 1.0 / bin_size if bin_size > 0 else None

        self._log_trace_figure(
            pred0,
            target0,
            rand_idx,
            target_session,
            kind="random",
            pearson_r=session_rate_corrs,
            time_axis_hz=time_hz,
        )
        self._log_trace_figure(
            pred0,
            target0,
            diverse_idx,
            target_session,
            kind="diverse",
            pearson_r=session_rate_corrs,
            time_axis_hz=time_hz,
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

    @torch.no_grad()
    def _compute_loader_correlation(
        self,
        dataloader_fn,
        max_batches: int | None = None,
    ) -> dict[str, float | None] | None:
        """Run correlation eval over an arbitrary dataloader.

        Mirrors the per-batch logic in ``validation_step`` but uses its own
        accumulators so it can be called outside Lightning's val loop (e.g.
        for the repeat-averaged dataloader).
        """
        try:
            loader = dataloader_fn()
        except Exception as e:
            logger.warning(f"Could not create dataloader for eval: {e}")
            return None

        rate_acc = OnlinePearsonAccumulator()
        temporal_acc = TemporalCorrAccumulator()
        has_temporal = False

        self._net.eval()
        model_dtype = next(self._net.parameters()).dtype
        n_batches = 0
        for batch in loader:
            spec = batch["spectrogram"].to(self.device, model_dtype)
            resp = batch["response"].to(self.device, model_dtype)
            key = batch["session_id"]
            lengths = batch.get("lengths", None)
            if lengths is not None:
                lengths = lengths.to(self.device)

            pred = self._net(spec, key)
            if not self._net._is_rate_mode and self.tokenizer_wrapper is not None:
                pred = self._decode_latent_to_rates(pred, key)

            T_pred = pred.shape[1]
            target = resp[:, :T_pred, :]

            if lengths is not None:
                effective_lengths = lengths.clamp(max=T_pred)
                t_idx = torch.arange(T_pred, device=pred.device).unsqueeze(0)
                mask = t_idx < effective_lengths.unsqueeze(1)
                pred_sum = (pred * mask.unsqueeze(2)).sum(dim=1)
                target_sum = (target * mask.unsqueeze(2)).sum(dim=1)
            else:
                pred_sum = pred.sum(dim=1)
                target_sum = target.sum(dim=1)

            n_units = pred.shape[2]
            rate_acc.update(key, pred_sum.float(), target_sum.float(), n_units)

            if T_pred > 1:
                has_temporal = True
                corr = batch_temporal_correlation(pred.float(), target.float())
                temporal_acc.update(key, corr, n_units)

            n_batches += 1
            if max_batches is not None and n_batches >= max_batches:
                break

        rate_corrs = rate_acc.compute()
        if not rate_corrs:
            return None

        rate_overall = torch.hstack(list(rate_corrs.values())).nanmean().item()
        temporal_overall = None
        if has_temporal:
            temporal_corrs = temporal_acc.compute()
            if temporal_corrs:
                temporal_overall = torch.hstack(list(temporal_corrs.values())).nanmean().item()

        return {"rate_corr": rate_overall, "temporal_corr": temporal_overall}

    def _log_corr_results(self, corrs: dict[str, torch.Tensor], prefix: str, prog_bar: bool = False) -> float | None:
        """Log per-session, per-subject, and overall correlation."""
        all_corrs = []
        subject_corrs: dict[str, list[torch.Tensor]] = {}

        for key, corr in corrs.items():
            self.log(f"{prefix}/{key}", corr.nanmean().item(), prog_bar=False)
            all_corrs.append(corr)
            if self._subject_info and key in self._subject_info:
                subj = self._subject_info[key]
                subject_corrs.setdefault(subj, []).append(corr)

        if not all_corrs:
            return None

        overall = torch.hstack(all_corrs).nanmean().item()
        self.log(f"{prefix}/overall", overall, prog_bar=prog_bar)
        for subj, corr_list in subject_corrs.items():
            self.log(f"{prefix}_subject/{subj}", torch.hstack(corr_list).nanmean().item(), prog_bar=False)
        return overall

    def configure_optimizers(self):
        lr = float(self.cfg.trainer.lr)
        wd = float(self.cfg.trainer.get("weight_decay", 0.01))
        betas = tuple(self.cfg.trainer.get("betas", [0.9, 0.999]))
        eps = float(self.cfg.trainer.get("eps", 1e-8))

        backbone_mult = float(self.cfg.trainer.get("backbone_lr_multiplier", 1.0))
        backbone_params = []
        readout_params = []
        for name, param in self._net.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("conv_blocks") or name.startswith("temporal_attn"):
                backbone_params.append(param)
            else:
                readout_params.append(param)

        param_groups = [
            {"params": readout_params, "lr": lr},
            {"params": backbone_params, "lr": lr * backbone_mult},
        ]

        optimizer = AdamW(param_groups, weight_decay=wd, betas=betas, eps=eps)

        max_steps = int(self.cfg.trainer.get("max_steps", 100000))
        warmup_steps = int(self.cfg.trainer.scheduler.get("warmup_steps", 1000))
        eta_min = float(self.cfg.trainer.scheduler.get("eta_min", 1e-5))

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
            return eta_min / lr + (1 - eta_min / lr) * 0.5 * (1 + math.cos(progress * math.pi))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
