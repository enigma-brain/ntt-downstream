"""Decoding ``train.py`` with ``mlflow.enabled=true`` and no tracking URI.

Without a URI, MLflow falls back to a local ``./mlruns`` file store, which
current MLflow refuses ("filesystem tracking backend ... is in maintenance
mode"), so ``MLFlowLogger`` would fail with a message that never mentions the
missing URI. ``_build_loggers`` fails first with a message naming the missing URI.
"""

from __future__ import annotations

import pytest
from lightning.pytorch.loggers import CSVLogger, MLFlowLogger
from omegaconf import OmegaConf

from ntt_downstream.decoding import train as dec_train


def _cfg(tmp_path, *, mlflow_enabled: bool):
    return OmegaConf.create(
        {
            "log_dir": str(tmp_path),
            "session": "S",
            "wandb": {"enabled": False},
            "mlflow": {
                "enabled": mlflow_enabled,
                "experiment_name": "e",
                "run_name": None,
                "log_model": False,
                "parent_run_id": None,
                "tags": {},
            },
        }
    )


def test_mlflow_disabled_needs_no_uri(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    loggers = dec_train._build_loggers(_cfg(tmp_path, mlflow_enabled=False), "fake")
    assert [type(lg) for lg in loggers] == [CSVLogger]


def test_mlflow_enabled_without_uri_fails_with_clear_message(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(RuntimeError, match=r"MLFLOW_TRACKING_URI.*mlflow\.enabled=false"):
        dec_train._build_loggers(_cfg(tmp_path, mlflow_enabled=True), "fake")


def test_mlflow_enabled_with_uri_builds_logger(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    loggers = dec_train._build_loggers(_cfg(tmp_path, mlflow_enabled=True), "fake")
    assert [type(lg) for lg in loggers] == [CSVLogger, MLFlowLogger]
