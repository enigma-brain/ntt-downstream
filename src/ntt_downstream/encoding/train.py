"""Single entry point for training encoding models.

Dispatches to image (Cowley V4) or audio (Wingert auditory) encoding based on
``model.model_type`` in the Hydra config.

Usage:
    python -m ntt_downstream.encoding.train                           # default config (Cowley V4)
    python -m ntt_downstream.encoding.train model.model_type=image_encoding_mar  # MAR
    python -m ntt_downstream.encoding.train --config-name wingert_base  # audio encoding
"""

import os
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, cast

# DDP + Hydra workaround: Lightning's subprocess launcher re-execs this script
# with the same argv per rank. Without this shim, Hydra resolves `${now:...}`
# independently in each child, producing N different output dirs → empty dirs
# for ranks > 0 and racing ModelCheckpoint/TensorBoard writers. Rank 0 exports
# `NTT_HYDRA_RUN_DIR` (its resolved output dir) in env before `trainer.fit`;
# children inherit the env var and we inject `hydra.run.dir=<parent dir>` into
# argv here so Hydra routes them into the same dir. Must run before hydra.main.
_child_run_dir = os.environ.get("NTT_HYDRA_RUN_DIR")
if _child_run_dir and int(os.environ.get("LOCAL_RANK", "0")) > 0:
    if not any(a.startswith("hydra.run.dir=") for a in sys.argv[1:]):
        sys.argv.append(f"hydra.run.dir={_child_run_dir}")

import hydra  # noqa: E402
import lightning as L  # noqa: E402
import rootutils  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint  # noqa: E402
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from ntt_downstream.common.results.callback import MetricSnapshotCallback  # noqa: E402
from ntt_downstream.common.results.writer import check_config_resolves  # noqa: E402

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def _resolve_devices(cfg: DictConfig) -> int:
    """Return the integer device count for this run (defaults to 1)."""
    devices = cfg.trainer.get("devices", 1)
    if isinstance(devices, (list, tuple)):
        return len(devices)
    try:
        return int(devices)
    except (TypeError, ValueError):
        return 1


def _log_config_to_mlflow(logger: MLFlowLogger, cfg: DictConfig):
    """Log the full Hydra config as flattened MLflow params."""
    from omegaconf import OmegaConf

    flat = {}
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    assert isinstance(cfg_dict, dict)
    for key, value in cfg_dict.items():
        _flatten_dict(flat, str(key), value)

    # MLflow limits: 500 params, 250-char values
    for k, v in flat.items():
        v_str = str(v)
        if len(v_str) > 250:
            v_str = v_str[:247] + "..."
        flat[k] = v_str

    # log_hyperparams handles batching internally
    logger.log_hyperparams(flat)


def _flatten_dict(out: dict, prefix: str, value):
    """Recursively flatten a nested dict into dot-separated keys."""
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten_dict(out, f"{prefix}.{k}", v)
    elif isinstance(value, (list, tuple)):
        out[prefix] = str(value)
    else:
        out[prefix] = value


def _build_callbacks(cfg: DictConfig, checkpoint_dir: Path, *, has_logger: bool):
    """Build the trainer callbacks; returns ``(callbacks, snapshot_cb)``.

    ``LearningRateMonitor`` is only registered when the trainer has a logger:
    Lightning raises at train start if it is present without one, which is the
    ``trainer.logging=false`` case. With a logger, the set and order are the
    same as they have always been.
    """
    callbacks: list = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            monitor="val/rate_corr/overall",
            mode="max",
            save_top_k=3,
            filename="{epoch}-{step}-{val/rate_corr/overall:.4f}",
            save_last=True,
        ),
    ]
    if has_logger:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    snapshot_cb = MetricSnapshotCallback(monitor="val/rate_corr/overall", mode="max")
    callbacks.append(snapshot_cb)

    # Defensive step-based checkpoint: only fires if the user has set
    # save_every_n_train_steps. This is independent of validation, so a
    # pre-validation crash still leaves a recoverable last.ckpt on disk.
    save_every_n_steps = cfg.trainer.get("save_every_n_train_steps", None)
    if save_every_n_steps is not None and int(save_every_n_steps) > 0:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                every_n_train_steps=int(save_every_n_steps),
                save_top_k=1,
                save_last=True,
                filename="step-{step}",
            )
        )
    return callbacks, snapshot_cb


def _build_loggers(cfg: DictConfig, run_name: str, output_dir: str):
    """Build the experiment loggers; returns ``(mlflow_logger, loggers)``.

    ``trainer.logging=false`` returns ``(None, [])``: no MLflow, no
    TensorBoard, nothing that needs a tracking server. With logging on,
    MLflow is the primary logger and a missing tracking URI is a hard error
    here, before training starts.
    """
    if not cfg.trainer.get("logging", True):
        return None, []

    tags = dict(cfg.get("tags", {}))
    description = cfg.get("description", "")
    if description:
        tags["mlflow.note.content"] = description

    # Strip so a blank value counts as unset, matching decoding and the sweeps.
    tracking_uri = (cfg.trainer.get("mlflow_tracking_uri") or os.environ.get("MLFLOW_TRACKING_URI") or "").strip()
    if not tracking_uri:
        raise RuntimeError(
            "MLflow tracking URI is unset. Set MLFLOW_TRACKING_URI in your "
            "environment (e.g. via a .env file) or override "
            "trainer.mlflow_tracking_uri on the CLI. To run without MLflow, "
            "pass trainer.logging=false."
        )
    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.trainer.get("experiment_name", "ntt-downstream-encoding"),
        tracking_uri=tracking_uri,
        run_name=run_name,
        tags=tags,
        log_model=False,
    )
    _log_config_to_mlflow(mlflow_logger, cfg)
    mlflow_logger.experiment.log_param(mlflow_logger.run_id, "run_name", run_name)
    loggers: list = [mlflow_logger]

    # TensorBoard: local events for offline analysis, alongside MLflow.
    if cfg.trainer.get("tensorboard_logging", True):
        loggers.append(
            TensorBoardLogger(
                save_dir=output_dir,
                name="tb",
                version=run_name,
                default_hp_metric=False,
            )
        )
    return mlflow_logger, loggers


def _run_trainer(cfg: DictConfig, model, dm):
    """Shared trainer setup: callbacks, logger, and fit.

    Returns:
        ``(trainer, mlflow_logger, output_dir, checkpoint_dir, run_name, snapshot_cb)``
        so the caller can run final eval and then write the results record.
    """
    # Run name: model tag + short random suffix
    model_tag = cfg.get("tags", {}).get("model", "run")
    run_name = f"{model_tag}_{uuid.uuid4().hex[:8]}"

    # Checkpoint directory: Hydra output dir (respects experiment layout), overrideable
    output_dir = HydraConfig.get().runtime.output_dir
    checkpoint_base = cfg.trainer.get("local_checkpoint_dir", None) or output_dir
    checkpoint_dir = Path(checkpoint_base) / "checkpoints" / run_name

    # Loggers first: whether there is one decides the callback set.
    mlflow_logger, loggers = _build_loggers(cfg, run_name, output_dir)
    callbacks, snapshot_cb = _build_callbacks(cfg, checkpoint_dir, has_logger=bool(loggers))

    # Log model parameter count (after configure_model materialises weights)
    model.configure_model()
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if mlflow_logger is not None:
        mlflow_logger.experiment.log_param(mlflow_logger.run_id, "model_params", n_params)
        mlflow_logger.experiment.log_param(mlflow_logger.run_id, "model_trainable_params", n_trainable)

    # Pre-fit tokenizer-ceiling diagnostic (rank-0 only; skip on resume / no-tokenizer).
    _maybe_run_startup_roundtrip(cfg, model, dm, mlflow_logger)

    # DDP: when devices>1, disable Lightning's auto sampler-wrapping (we
    # provide DistributedSessionBatchSampler via the DataModule) and pick a
    # strategy. Default to "ddp_find_unused_parameters_true" so the tokenizer
    # path (which may skip some model params in some batches) stays safe.
    n_devices = _resolve_devices(cfg)

    max_epochs = cfg.trainer.get("max_epochs", None)
    if max_epochs is None:
        max_epochs = cfg.trainer.get("n_epochs", None)

    trainer_kwargs: dict = {
        "max_steps": int(cfg.trainer.get("max_steps", 100000)),
        "max_epochs": max_epochs,
        "precision": cfg.trainer.get("precision", "bf16-mixed"),
        "gradient_clip_val": cfg.trainer.get("gradient_clipping", 1.0),
        "accumulate_grad_batches": cfg.trainer.get("gradient_accumulation_steps", 1),
        "val_check_interval": cfg.trainer.get("logging_frequency_validation", 3000),
        "callbacks": callbacks,
        "logger": loggers if loggers else False,
        "enable_progress_bar": True,
        "log_every_n_steps": cfg.trainer.get("log_every_n_steps", cfg.trainer.get("logging_frequency_loss", 50)),
        "default_root_dir": output_dir,
        "accelerator": cfg.trainer.get("accelerator", "auto"),
        "devices": cfg.trainer.get("devices", 1),
    }

    if cfg.trainer.get("check_val_every_n_epoch") is not None:
        trainer_kwargs["check_val_every_n_epoch"] = cfg.trainer.get("check_val_every_n_epoch")
    if cfg.trainer.get("num_sanity_val_steps") is not None:
        trainer_kwargs["num_sanity_val_steps"] = cfg.trainer.get("num_sanity_val_steps")

    if n_devices > 1:
        trainer_kwargs["strategy"] = cfg.trainer.get("strategy", "ddp_find_unused_parameters_true")
        trainer_kwargs["use_distributed_sampler"] = False
    elif cfg.trainer.get("strategy") is not None:
        trainer_kwargs["strategy"] = cfg.trainer.get("strategy")

    # Optional overrides useful for smoke-testing / debugging
    if cfg.trainer.get("limit_train_batches") is not None:
        trainer_kwargs["limit_train_batches"] = cfg.trainer.get("limit_train_batches")
    if cfg.trainer.get("limit_val_batches") is not None:
        trainer_kwargs["limit_val_batches"] = cfg.trainer.get("limit_val_batches")

    trainer = L.Trainer(**trainer_kwargs)

    resume_ckpt = cfg.trainer.get("resume_from_ckpt", None)
    if resume_ckpt is not None:
        resume_ckpt = str(resume_ckpt)
        if not Path(resume_ckpt).is_file():
            raise FileNotFoundError(f"trainer.resume_from_ckpt is set but checkpoint not found: {resume_ckpt}")
        print(f"[train] resuming from checkpoint: {resume_ckpt}")

    # Resume loads with weights_only=False: our checkpoints pickle the
    # OmegaConf hparams, which a weights_only load rejects. Only this
    # package's own checkpoints are resumed here.
    trainer.fit(model, dm, ckpt_path=resume_ckpt, weights_only=False)

    # The results record is written by the caller after final eval, so it is
    # written once with every block (see _persist_encoding_results).
    return trainer, mlflow_logger, Path(output_dir), checkpoint_dir, run_name, snapshot_cb


def _build_single_rank_val_loader(dm, dataset):
    """Build a non-DDP val-style dataloader. Used to evaluate every session on
    rank 0 after fit completes, even when training ran with DDP (where the
    standard val_dataloader shards sessions across ranks)."""
    from torch.utils.data import DataLoader

    from ntt_downstream.encoding.dataloading.cowley_v4 import SessionBatchSampler, numpy_collate_fn

    batch_size = dm.cfg.dataloader.get("batch_size", 64)
    sampler = SessionBatchSampler(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0, collate_fn=numpy_collate_fn)


def _test_loader_fns(dm):
    """Lazy single-rank ``(test, test_average)`` loader factories for final eval.

    ``trainer.fit`` only runs ``dm.setup("fit")``, so the test datasets are
    built here on first use, once.
    """

    def _dataset(name):
        if getattr(dm, name, None) is None:
            dm.setup("test")
        return getattr(dm, name)

    return (
        lambda: _build_single_rank_val_loader(dm, _dataset("test_dataset")),
        lambda: _build_single_rank_val_loader(dm, _dataset("test_avg_dataset")),
    )


def _maybe_run_startup_roundtrip(cfg, model, dm, mlflow_logger):
    """Pre-fit tokenizer-ceiling diagnostic.

    Encodes → decodes a small slice of the val set with the calibrated tokenizer
    wrapper, logs aggregate metrics + two figures to MLflow. Rank 0 only;
    silently skipped when no tokenizer is in use, when resuming, or when
    logging is disabled. Failures are logged and swallowed.
    """
    sr_cfg = cfg.trainer.get("startup_roundtrip", {})
    if not bool(sr_cfg.get("enabled", True)):
        print("[startup_roundtrip] skipped: trainer.startup_roundtrip.enabled=false")
        return
    if int(os.environ.get("LOCAL_RANK", "0")) > 0:
        return
    if mlflow_logger is None:
        print("[startup_roundtrip] skipped: logging disabled")
        return
    if cfg.trainer.get("resume_from_ckpt", None) is not None:
        print("[startup_roundtrip] skipped: resuming from checkpoint")
        return
    wrapper = getattr(model, "tokenizer_wrapper", None)
    if wrapper is None:
        print("[startup_roundtrip] skipped: no tokenizer (poisson / mse_rate)")
        return

    import torch

    from ntt_downstream.encoding.training.tokenizer_startup_roundtrip import run_startup_roundtrip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bin_size = float(cfg.data.get("bin_size", 0.001))
    native_hz = 1.0 / bin_size
    eval_rebin_hz = cfg.trainer.get("eval_rebin_hz", None)
    eval_rebin_hz = float(eval_rebin_hz) if eval_rebin_hz is not None else None
    max_batches = int(sr_cfg.get("max_batches", 20))
    # When strict, a broken-tokenizer signal (non-finite / degenerate recon)
    # raises RuntimeError out of run_startup_roundtrip and we let it surface —
    # that is a real modeling failure, not an infra hiccup.
    strict = bool(sr_cfg.get("strict", False))

    try:
        wrapper.tokenizer.to(device)
        loader = dm.val_dataloader()
        metrics, fig_traces, fig_scatter = run_startup_roundtrip(
            wrapper,
            loader,
            native_hz=native_hz,
            eval_rebin_hz=eval_rebin_hz,
            device=device,
            max_batches=max_batches,
            strict=strict,
        )
        print("[startup_roundtrip] " + ", ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in metrics.items()))
        for k, v in metrics.items():
            mlflow_logger.experiment.log_metric(mlflow_logger.run_id, k, float(v), step=0)
        mlflow_logger.experiment.log_figure(mlflow_logger.run_id, fig_traces, "tokenizer_ceiling/roundtrip_traces.png")
        mlflow_logger.experiment.log_figure(
            mlflow_logger.run_id, fig_scatter, "tokenizer_ceiling/roundtrip_scatter.png"
        )
        import matplotlib.pyplot as plt

        plt.close(fig_traces)
        plt.close(fig_scatter)
    except RuntimeError:
        # A RuntimeError here is the broken-tokenizer signal raised under
        # strict=True. This is NOT an infra failure to swallow — re-raise so the
        # run fails loudly. (Infra problems below raise other exception types.)
        if strict:
            print("[startup_roundtrip] FAILED HARD: tokenizer roundtrip produced a broken recon under strict mode.")
            raise
        # Non-strict: a RuntimeError can still come from genuine infra issues
        # (e.g. an upstream tokenizer call). Log prominently and continue.
        import traceback

        print("[startup_roundtrip] WARNING: roundtrip raised a RuntimeError; skipping diagnostic.")
        traceback.print_exc()
    except Exception as e:
        # Infra failure (no loader, matplotlib backend, device, etc.): the
        # diagnostic is best-effort and must never abort an otherwise-fine run.
        import traceback

        print(f"[startup_roundtrip] WARNING: roundtrip skipped due to infra failure: {e!r}")
        traceback.print_exc()
    finally:
        # Restore tokenizer to CPU (matches post-calibration state in encoding_module).
        try:
            wrapper.tokenizer.to("cpu")
        except Exception:
            pass


def _select_final_eval_checkpoint(trainer, checkpoint_dir):
    """Pick the checkpoint final_eval should evaluate.

    Returns ``(path, kind)`` where ``kind`` is "best" or "last", or ``None`` if
    no checkpoint exists. The record's val-selected checkpoint is the
    val-BEST snapshot (``ModelCheckpoint(monitor="val/rate_corr/overall")
    .best_model_path``), so final_eval must score that same checkpoint — not
    ``last.ckpt`` — or it would evaluate a different model than the one reported.
    We fall back to ``last.ckpt`` ONLY when ``best_model_path`` is empty/missing.
    """
    from lightning.pytorch.callbacks import ModelCheckpoint

    best_path = ""
    for cb in trainer.callbacks:
        if isinstance(cb, ModelCheckpoint) and cb.monitor == "val/rate_corr/overall":
            best_path = cb.best_model_path or ""
            break

    if best_path and Path(best_path).is_file():
        return Path(best_path), "best"

    last_ckpt = checkpoint_dir / "last.ckpt"
    if last_ckpt.is_file():
        return last_ckpt, "last"
    return None


def _load_final_eval_state_dict(model, ckpt_path, root_device):
    """Load ``ckpt_path`` into ``model``, validating the state_dict strictly.

    Loads with ``strict=False`` only so we can inspect the returned
    ``IncompatibleKeys`` and raise a precise error; ANY missing or unexpected key
    RAISES. There is no key allowlist: the ``tokenizer_wrapper`` is a
    ``@dataclass`` held as a plain attribute of the model (an ``nn.Module``), so
    PyTorch never registers it in ``_modules`` and its inner tokenizer params
    never appear in the LitModule state_dict — in either direction. Hence in
    normal operation there are zero tokenizer keys to tolerate, and a substring
    allowlist would only risk silently swallowing a genuinely-required weight
    (e.g. ``model.tokenizer_adapter.weight``) and reintroducing an unfaithful evaluation.

    WHY raise instead of warn-and-continue: falling back to in-memory weights on a
    load error would let a structurally-wrong checkpoint produce a
    plausible-looking but unfaithful headline number.
    """
    import torch

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(ckpt["state_dict"], strict=False)

    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "final_eval: checkpoint state_dict mismatch (cannot evaluate wrong "
            f"weights). missing={list(incompatible.missing_keys)}, "
            f"unexpected={list(incompatible.unexpected_keys)}. "
            f"Checkpoint: {ckpt_path}"
        )
    model.to(root_device)


def _maybe_run_final_eval(cfg, model, dm, trainer, mlflow_logger, output_dir, checkpoint_dir):
    """End-of-training: load the REPORTED (val-best) checkpoint and run final_eval.

    Rank 0 only, image-only path. final_eval must score the checkpoint selected
    on val, so we select it via the ``ModelCheckpoint`` callback and fall back to
    ``last.ckpt`` only when no best exists. Any state_dict mismatch RAISES rather
    than silently scoring wrong weights; a genuinely-absent checkpoint is logged
    and skipped (not an error).

    Returns ``(table, status)``: the final-eval table (val, with the test table
    nested under ``"test"``) or ``None``, and ``{"status": "ok" | "failed" |
    "disabled", "error"?: str, "ckpt_kind"?: "best" | "last", "test"?: "disabled"
    | "failed", "test_error"?: str}``. The status goes into the record's metadata
    so a missing ``test`` block is explained, never silently replaced by val.
    A state_dict mismatch raises (callers persist the record first; see
    ``_finalize_run``).
    """
    fe_cfg = cfg.trainer.get("final_eval", {})
    if not bool(fe_cfg.get("enabled", True)):
        return None, {"status": "disabled"}
    if not trainer.is_global_zero:
        return None, {"status": "disabled", "error": "non-zero rank"}

    selected = _select_final_eval_checkpoint(trainer, checkpoint_dir)
    if selected is None:
        print(f"[final_eval] no best/last checkpoint found under {checkpoint_dir}, skipping")
        return None, {"status": "failed", "error": f"no checkpoint under {checkpoint_dir}"}
    ckpt_path, ckpt_kind = selected

    from ntt_downstream.encoding.training import final_eval as fe

    # Load the selected checkpoint, validating its state_dict. A mismatch raises
    # (we will not score the wrong weights); FileNotFound was already ruled out
    # by _select_final_eval_checkpoint.
    print(f"[final_eval] loading {ckpt_kind} checkpoint: {ckpt_path}")
    _load_final_eval_state_dict(model, ckpt_path, trainer.strategy.root_device)

    # Override DDP-sharded val loaders with single-rank versions so rank 0 sees
    # every session. Non-DDP runs aren't affected (sampler choice is identical).
    model.set_val_dataloader(lambda: _build_single_rank_val_loader(dm, dm.val_dataset))
    model.set_val_average_dataloader(lambda: _build_single_rank_val_loader(dm, dm.val_avg_dataset))
    # Score the same val-selected weights on the test split too (selection
    # never looks at test). Built lazily: the test datasets only exist after
    # dm.setup("test"), which trainer.fit() never runs.
    test_loader_fns = _test_loader_fns(dm) if bool(fe_cfg.get("test", True)) else None

    n_samples = int(fe_cfg.get("n_samples", 32))
    flow_solver = str(fe_cfg.get("flow_solver", "heun"))
    n_flow_steps_raw = fe_cfg.get("n_flow_steps", None)
    n_flow_steps = int(n_flow_steps_raw) if n_flow_steps_raw is not None else None
    # None -> run() falls back to cfg.trainer.tokenizer_eval.flow_shift, so a
    # configured shift is honored at final eval without a final_eval-specific key.
    flow_shift_raw = fe_cfg.get("flow_shift", None)
    flow_shift = float(flow_shift_raw) if flow_shift_raw is not None else None

    try:
        table = fe.run(
            model,
            n_samples=n_samples,
            flow_solver=flow_solver,
            n_flow_steps=n_flow_steps,
            flow_shift=flow_shift,
            output_dir=output_dir,
            mlflow_logger=mlflow_logger,
            test_loader_fns=test_loader_fns,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[final_eval] eval failed: {e!r}")
        traceback.print_exc()
        return None, {"status": "failed", "error": repr(e)}
    # ckpt_kind: "best" is the val-selected checkpoint; "last" is the fallback
    # when no best exists, which a consumer may want to treat differently.
    status: dict = {"status": "ok", "ckpt_kind": ckpt_kind}
    if test_loader_fns is None:
        status["test"] = "disabled"
    elif "test_error" in table:
        status["test"] = "failed"
        status["test_error"] = table["test_error"]
    return table, status


def _tokenizer_checkpoint(cfg) -> str | None:
    """The frozen tokenizer checkpoint an encoding run evaluates (None for tokenizer-free runs)."""
    tok = cfg.get("tokenizer") or {}
    if not tok.get("enabled", False):
        return None
    ckpt = tok.get("checkpoint_path")
    return str(ckpt) if ckpt else None


def _write_encoding_results(
    trainer, cfg, model, output_dir, run_name, mlflow_logger, snapshot_cb, *, final_table, final_eval
):
    """Build and write the run's ``downstream_encoding_results.json`` (rank 0).

    ``val_best`` / ``val_last`` come from the periodic-val metric snapshots;
    ``val_final`` / ``test`` from the final-eval table (absent when final eval
    did not produce one, with the reason in ``metadata.final_eval``). Written
    once, after final eval, so the record is never half-updated.
    """
    from lightning.pytorch.callbacks import ModelCheckpoint
    from omegaconf import OmegaConf

    from ntt_downstream.common.results.provenance import git_sha, hostname, now_iso, ntt_provenance
    from ntt_downstream.common.results.schema import ResultsDocument, RunMetadata
    from ntt_downstream.common.results.writer import write_results
    from ntt_downstream.encoding.training.final_eval import table_to_metric_block
    from ntt_downstream.encoding.training.results_parser import parse_encoding_block

    session_info = dict(getattr(model, "_session_info", {}) or {})

    last_snap = snapshot_cb.last_val or {}
    best_snap = snapshot_cb.best_val or last_snap
    metrics = {
        "val_best": parse_encoding_block(best_snap, session_info=session_info),
        "val_last": parse_encoding_block(last_snap, session_info=session_info),
    }
    if final_table is not None:
        metrics["val_final"] = table_to_metric_block(final_table)
        if "test" in final_table:
            metrics["test"] = table_to_metric_block(final_table["test"])

    checkpoint: dict = {}
    for cb in trainer.callbacks:
        if isinstance(cb, ModelCheckpoint) and cb.monitor == "val/rate_corr/overall":
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

    sessions = sorted(session_info.keys())
    metadata = RunMetadata(
        git_sha=git_sha(),
        timestamp=now_iso(),
        hostname=hostname(),
        run_name=run_name,
        mlflow_run_id=getattr(mlflow_logger, "run_id", None),
        config=cast("dict[str, Any]", OmegaConf.to_container(cfg, resolve=True)),
        checkpoint=checkpoint,
        sessions=sessions,
        n_sessions=len(sessions),
        tokenizer_checkpoint=_tokenizer_checkpoint(cfg),
        **ntt_provenance(),
        extra={"final_eval": final_eval},
    )
    doc = ResultsDocument(task="encoding", kind="run", metadata=metadata, metrics=metrics)
    return write_results(doc, output_dir, loggers=trainer.loggers)


def _persist_encoding_results(trainer, *args, **kwargs) -> None:
    """Rank-0 wrapper around ``_write_encoding_results`` that fails the run if no record lands.

    The record is the run's deliverable: downstream aggregation reads only
    ``downstream_encoding_results.json``. A run that trains, exits 0 and
    leaves no record looks successful and is silently dropped from every
    table, which is worse than a crash. So any error building the record, and
    a local write that did not happen (``write_results`` is best-effort and
    returns ``None``), raises. Uploading the record to a logger stays
    best-effort: the local file is the source of truth.
    """
    if not trainer.is_global_zero:
        return
    try:
        path = _write_encoding_results(trainer, *args, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"failed to write the results record: {exc!r}") from exc
    if path is None:
        raise RuntimeError("failed to write the results record (see the warning above for the cause)")


def _finalize_run(
    cfg, model, dm, trainer, mlflow_logger, output_dir, checkpoint_dir, run_name, snapshot_cb, *, run_final_eval: bool
) -> None:
    """Post-fit: run final eval (image path) and write the run's results record.

    The record is written whatever final eval does. If final eval raises (a
    state_dict mismatch, an unloadable checkpoint), the record is persisted
    first with ``final_eval.status="failed"`` so the diagnostic blocks
    survive, then the exception propagates so the run still fails loudly.
    A failure to write the record also fails the run; when both happen, the
    final-eval error is the one raised (it is the root cause) and the record
    error is printed.
    """
    final_table = None
    final_status: dict = {"status": "disabled", "error": "this path has no final eval"}
    if run_final_eval:
        try:
            final_table, final_status = _maybe_run_final_eval(
                cfg, model, dm, trainer, mlflow_logger, output_dir, checkpoint_dir
            )
        except Exception as exc:
            try:
                _persist_encoding_results(
                    trainer,
                    cfg,
                    model,
                    output_dir,
                    run_name,
                    mlflow_logger,
                    snapshot_cb,
                    final_table=None,
                    final_eval={"status": "failed", "error": repr(exc)},
                )
            except Exception:  # noqa: BLE001
                print("[train] ERROR: also failed to write the results record:")
                traceback.print_exc()
            raise
    _persist_encoding_results(
        trainer,
        cfg,
        model,
        output_dir,
        run_name,
        mlflow_logger,
        snapshot_cb,
        final_table=final_table,
        final_eval=final_status,
    )


def _train_image(cfg: DictConfig):
    """Train an image encoding model (Cowley V4)."""
    from ntt_downstream.encoding.training.data_module import CowleyV4DataModule
    from ntt_downstream.encoding.training.encoding_module import EncodingLitModule

    dm = CowleyV4DataModule(cfg)
    dm.setup("fit")

    session_info = dm.get_session_info("train")
    subject_info = dm.get_subject_info("train")

    model = EncodingLitModule(cfg, session_info=session_info, subject_info=subject_info)

    # Register extra eval dataloaders
    model.set_val_dataloader(dm.val_dataloader)
    model.set_val_average_dataloader(dm.val_average_dataloader)
    model.set_train_dataloader(dm.train_dataloader)

    trainer, mlflow_logger, output_dir, checkpoint_dir, run_name, snapshot_cb = _run_trainer(cfg, model, dm)
    _finalize_run(
        cfg,
        model,
        dm,
        trainer,
        mlflow_logger,
        output_dir,
        checkpoint_dir,
        run_name,
        snapshot_cb,
        run_final_eval=True,
    )


def _train_audio(cfg: DictConfig):
    """Train an audio encoding model (Wingert auditory cortex)."""
    from ntt_downstream.encoding.training.audio_encoding_module import AudioEncodingLitModule
    from ntt_downstream.encoding.training.wingert_data_module import WingertAuditoryDataModule

    dm = WingertAuditoryDataModule(cfg)
    dm.setup("fit")

    session_info = dm.get_session_info("train")
    subject_info = dm.get_subject_info("train")

    model = AudioEncodingLitModule(cfg, session_info=session_info, subject_info=subject_info)

    # Register extra eval dataloaders
    model.set_val_dataloader(dm.val_dataloader)
    model.set_val_average_dataloader(dm.val_average_dataloader)
    model.set_train_dataloader(dm.train_dataloader)

    trainer, mlflow_logger, output_dir, checkpoint_dir, run_name, snapshot_cb = _run_trainer(cfg, model, dm)
    # Audio path has no final_eval yet, so its record carries no test block.
    _finalize_run(
        cfg,
        model,
        dm,
        trainer,
        mlflow_logger,
        output_dir,
        checkpoint_dir,
        run_name,
        snapshot_cb,
        run_final_eval=False,
    )


@hydra.main(config_path=f"{root}/configs/encoding", config_name="cowley_v4_base", version_base=None)
def main(cfg: DictConfig):
    # Export rank 0's resolved Hydra output dir so DDP children (spawned via
    # Lightning's subprocess launcher after trainer.fit) route their Hydra
    # output into the same location — see the argv shim at module top.
    if int(os.environ.get("LOCAL_RANK", "0")) == 0 and "NTT_HYDRA_RUN_DIR" not in os.environ:
        os.environ["NTT_HYDRA_RUN_DIR"] = HydraConfig.get().runtime.output_dir

    # The results record (and the MLflow params, when logging) store the fully
    # resolved config. Resolve it here so a broken interpolation stops the run
    # here rather than after training, when the record is written.
    check_config_resolves(cfg, task="encoding")

    # Global determinism: seed before any model/data construction.
    L.seed_everything(cfg.seed, workers=True)

    model_type = str(cfg.model.get("model_type", "image_encoding"))

    if model_type == "audio_encoding":
        _train_audio(cfg)
    else:
        _train_image(cfg)


if __name__ == "__main__":
    main()
