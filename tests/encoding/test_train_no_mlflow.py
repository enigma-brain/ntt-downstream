"""Encoding must run end to end without an MLflow server.

Users of the published tokenizers have no tracking server. With
``trainer.logging=false`` an encoding run has to train, evaluate and write its
``downstream_encoding_results.json`` without touching MLflow; with logging on
and no URI it has to fail early with the clear "tracking URI is unset" error;
with a URI, loggers and callbacks stay as they were.

Three things must hold for this, each covered below:
- ``trainer.mlflow_tracking_uri: ${oc.env:MLFLOW_TRACKING_URI}`` needs a default,
  otherwise any read or full resolve of the config raises.
- ``LearningRateMonitor`` is only registered when there is a logger; Lightning
  rejects it at train start when the trainer has none.
- The record writer must not swallow its own errors, otherwise the run exits 0
  with no record.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightning as L
import pytest
import torch
from hydra import compose, initialize_config_dir
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import OmegaConf

from ntt_downstream.common.results.callback import MetricSnapshotCallback
from ntt_downstream.common.results.writer import check_config_resolves
from ntt_downstream.encoding import train as enc_train

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "encoding"
_RECIPES = sorted(p.stem for p in _CONFIG_DIR.glob("*.yaml"))


@pytest.fixture
def no_mlflow_env(monkeypatch):
    """An environment with no MLflow configuration at all (an outside user)."""
    for var in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"):
        monkeypatch.delenv(var, raising=False)


def _compose(recipe: str = "cowley_v4_base", overrides: list[str] | None = None):
    with initialize_config_dir(str(_CONFIG_DIR), version_base=None):
        return compose(recipe, overrides=overrides or [])


# ---------------------------------------------------------------------------
# Config: the tracking-URI default
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("recipe", _RECIPES)
def test_every_recipe_resolves_without_mlflow_env(no_mlflow_env, recipe):
    """The record stores the fully resolved config, so every recipe must resolve."""
    cfg = _compose(recipe)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)
    assert resolved["trainer"]["mlflow_tracking_uri"] is None


def test_tracking_uri_reads_env_when_set(no_mlflow_env, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking.example:5000")
    assert _compose().trainer.mlflow_tracking_uri == "http://tracking.example:5000"


def test_tracking_uri_cli_override_wins(no_mlflow_env, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking.example:5000")
    cfg = _compose(overrides=["trainer.mlflow_tracking_uri=file:///elsewhere"])
    assert cfg.trainer.mlflow_tracking_uri == "file:///elsewhere"


# ---------------------------------------------------------------------------
# Loggers
# ---------------------------------------------------------------------------


def test_logging_off_builds_no_loggers(no_mlflow_env, tmp_path):
    cfg = _compose(overrides=["trainer.logging=false"])
    mlflow_logger, loggers = enc_train._build_loggers(cfg, "run", str(tmp_path))
    assert mlflow_logger is None and loggers == []


def test_logging_on_without_uri_fails_with_clear_message(no_mlflow_env, tmp_path):
    cfg = _compose()
    with pytest.raises(RuntimeError, match="MLflow tracking URI is unset"):
        enc_train._build_loggers(cfg, "run", str(tmp_path))


def test_logging_on_with_blank_uri_fails_with_clear_message(no_mlflow_env, monkeypatch, tmp_path):
    """A whitespace-only URI counts as unset, as it does for decoding and the sweeps."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "  ")
    cfg = _compose()
    with pytest.raises(RuntimeError, match="MLflow tracking URI is unset"):
        enc_train._build_loggers(cfg, "run", str(tmp_path))


def test_logging_on_with_uri_builds_mlflow_and_tensorboard(no_mlflow_env, tmp_path):
    from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    cfg = _compose(overrides=[f"trainer.mlflow_tracking_uri={uri}"])
    mlflow_logger, loggers = enc_train._build_loggers(cfg, "run", str(tmp_path))
    assert isinstance(mlflow_logger, MLFlowLogger)
    assert [type(lg) for lg in loggers] == [MLFlowLogger, TensorBoardLogger]
    assert loggers[0] is mlflow_logger


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def test_callbacks_with_logger_keep_lr_monitor_in_place(no_mlflow_env, tmp_path):
    cfg = _compose()
    callbacks, snapshot_cb = enc_train._build_callbacks(cfg, tmp_path, has_logger=True)
    # Same set and order as with MLflow configured.
    assert [type(cb) for cb in callbacks] == [
        ModelCheckpoint,
        LearningRateMonitor,
        MetricSnapshotCallback,
        ModelCheckpoint,
    ]
    assert callbacks[2] is snapshot_cb


def test_callbacks_without_logger_drop_lr_monitor(no_mlflow_env, tmp_path):
    cfg = _compose(overrides=["trainer.logging=false"])
    callbacks, snapshot_cb = enc_train._build_callbacks(cfg, tmp_path, has_logger=False)
    assert not any(isinstance(cb, LearningRateMonitor) for cb in callbacks)
    assert snapshot_cb in callbacks
    assert any(isinstance(cb, ModelCheckpoint) and cb.monitor == "val/rate_corr/overall" for cb in callbacks)


class _TinyLit(L.LightningModule):
    """Smallest module that trains, validates and logs the monitored metric."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 1)

    def training_step(self, batch, _idx):
        return self.lin(batch[0]).pow(2).mean()

    def validation_step(self, batch, _idx):
        self.log("val/rate_corr/overall", self.lin(batch[0]).mean())

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def _loader(self):
        ds = torch.utils.data.TensorDataset(torch.randn(4, 2))
        return torch.utils.data.DataLoader(ds, batch_size=2)

    def train_dataloader(self):
        return self._loader()

    def val_dataloader(self):
        return self._loader()


def test_logger_free_trainer_fits_with_built_callbacks(no_mlflow_env, tmp_path):
    """Real Lightning fit with logger=False: the callbacks must not include LearningRateMonitor."""
    cfg = _compose(overrides=["trainer.logging=false"])
    callbacks, snapshot_cb = enc_train._build_callbacks(cfg, tmp_path / "ckpt", has_logger=False)
    trainer = L.Trainer(
        max_epochs=1,
        logger=False,
        callbacks=callbacks,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
    )
    trainer.fit(_TinyLit())
    assert snapshot_cb.last_val is not None
    assert (tmp_path / "ckpt" / "last.ckpt").is_file()


# ---------------------------------------------------------------------------
# Results record: written with no MLflow, and never lost silently
# ---------------------------------------------------------------------------


class _Trainer:
    """Trainer stand-in for the record writer: no callbacks, no loggers, rank 0."""

    is_global_zero = True
    callbacks: list = []
    loggers: list = []


def _snap_cb():
    cb = MetricSnapshotCallback(monitor="val/rate_corr/overall", mode="max")
    cb.best_val = {"val/rate_corr/overall": 0.3}
    cb.last_val = {"val/rate_corr/overall": 0.2}
    return cb


def _persist(cfg, output_dir):
    enc_train._persist_encoding_results(
        _Trainer(),
        cfg,
        torch.nn.Module(),
        str(output_dir),
        "run",
        None,
        _snap_cb(),
        final_table=None,
        final_eval={"status": "disabled"},
    )


def test_record_written_without_mlflow(no_mlflow_env, tmp_path):
    cfg = _compose(overrides=["trainer.logging=false"])
    _persist(cfg, tmp_path)
    doc = json.loads((tmp_path / "downstream_encoding_results.json").read_text())
    assert doc["schema_version"] == 2
    assert doc["metadata"]["mlflow_run_id"] is None
    assert doc["metadata"]["config"]["trainer"]["mlflow_tracking_uri"] is None


def test_record_failure_fails_the_run(tmp_path):
    """An unresolvable config fails the run (non-zero exit), not just a warning."""
    cfg = OmegaConf.create({"trainer": {"x": "${oc.env:NTT_TEST_SURELY_UNSET_VAR}"}})
    with pytest.raises(RuntimeError, match="results record"):
        _persist(cfg, tmp_path)


def test_record_write_error_fails_the_run(no_mlflow_env, tmp_path):
    """``write_results`` is best-effort (returns None); encoding treats None as a failure."""
    cfg = _compose(overrides=["trainer.logging=false"])
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("")
    with pytest.raises(RuntimeError, match="results record"):
        _persist(cfg, blocker)


def test_finalize_reraises_final_eval_error_even_if_record_fails(tmp_path, monkeypatch):
    """When final eval raises, its error stays the one that propagates."""

    def _boom(*_a, **_kw):
        raise ValueError("final eval exploded")

    def _record_boom(*_a, **_kw):
        raise OSError("disk full")

    monkeypatch.setattr(enc_train, "_maybe_run_final_eval", _boom)
    monkeypatch.setattr(enc_train, "_write_encoding_results", _record_boom)
    with pytest.raises(ValueError, match="final eval exploded"):
        enc_train._finalize_run(
            OmegaConf.create({}),
            torch.nn.Module(),
            None,
            _Trainer(),
            None,
            str(tmp_path),
            tmp_path,
            "run",
            _snap_cb(),
            run_final_eval=True,
        )


# ---------------------------------------------------------------------------
# Fail fast: a config that can't resolve stops the run before training
# ---------------------------------------------------------------------------


def test_check_config_resolves_passes_without_mlflow(no_mlflow_env):
    check_config_resolves(_compose(overrides=["trainer.logging=false"]), task="encoding")


def test_main_checks_config_before_training(monkeypatch):
    """main() resolves the config before seeding, so nothing trains on a broken one."""
    calls = []

    def _boom(cfg, *, task):
        calls.append(("check", task))
        raise RuntimeError("config check ran")

    # Skip main()'s HydraConfig lookup, which needs a live Hydra run.
    monkeypatch.setenv("NTT_HYDRA_RUN_DIR", "unused")
    monkeypatch.setattr(enc_train, "check_config_resolves", _boom)
    monkeypatch.setattr(enc_train.L, "seed_everything", lambda *a, **k: calls.append("seed"))
    with pytest.raises(RuntimeError, match="config check ran"):
        enc_train.main.__wrapped__(OmegaConf.create({"seed": 0}))
    assert calls == [("check", "encoding")]
