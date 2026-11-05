"""The decoding run's v2 record: blocks, location, and provenance."""

from __future__ import annotations

import json
from types import SimpleNamespace

from omegaconf import OmegaConf

from ntt_downstream.common.results.callback import MetricSnapshotCallback
from ntt_downstream.decoding import train as dec_train


def _snap() -> MetricSnapshotCallback:
    snap = MetricSnapshotCallback(monitor="average_val_metric", mode="max")
    snap.best_val = {"average_val_metric": 0.7, "s1": 0.7}
    snap.last_val = {"average_val_metric": 0.6, "s1": 0.6}
    snap.test = {"average_test_metric": 0.65, "s1": 0.65}
    return snap


def _trainer(tmp_path, snap):
    # The record goes to the per-run Lightning version dir, next to run_result.json,
    # because that is the tree the benchmark sweep walks and uploads.
    return SimpleNamespace(
        callbacks=[snap],
        loggers=[],
        log_dir=str(tmp_path / "csv_logs" / "version_0"),
        default_root_dir=str(tmp_path),
    )


def test_decoding_record_v2(tmp_path):
    cfg = OmegaConf.create({"tokenizer": {"ntt_checkpoint": "/ckpt/a.ckpt"}, "log_dir": str(tmp_path)})

    path = dec_train._write_decoding_results(_trainer(tmp_path, _snap()), cfg, session_ids=["s1"])

    assert path == tmp_path / "csv_logs" / "version_0" / "downstream_decoding_results.json"
    doc = json.loads(path.read_text())
    assert (doc["schema_version"], doc["kind"], doc["task"]) == (2, "run", "decoding")
    assert set(doc["metrics"]) == {"val_best", "val_last", "test"}
    assert doc["metrics"]["test"]["per_session"]["s1"]["r2"] == 0.65
    assert doc["metrics"]["val_best"]["aggregated"]["r2"]["value"] == 0.7
    assert doc["metadata"]["tokenizer_checkpoint"] == "/ckpt/a.ckpt"
    assert doc["metadata"]["ntt_path"]


def test_decoding_record_without_test_phase(tmp_path):
    snap = _snap()
    snap.test = None
    cfg = OmegaConf.create({"tokenizer": {}, "log_dir": str(tmp_path)})

    path = dec_train._write_decoding_results(_trainer(tmp_path, snap), cfg, session_ids=["s1"])

    doc = json.loads(path.read_text())
    assert set(doc["metrics"]) == {"val_best", "val_last"}
    assert doc["metadata"]["tokenizer_checkpoint"] is None


def test_v2_record_written_even_if_sweep_bookkeeping_raises(monkeypatch, tmp_path):
    """The contract record goes first; a run_result.json (sweep bookkeeping) failure still
    surfaces (the sweep then sees an incomplete cell) but can't block it."""
    import pytest

    calls = []
    monkeypatch.setattr(
        dec_train, "_write_decoding_results", lambda *a, **k: calls.append("v2") or tmp_path / "rec.json"
    )

    def _bookkeeping_boom(*a, **k):
        calls.append("bookkeeping")
        raise OSError("disk full")

    monkeypatch.setattr(dec_train, "_save_results", _bookkeeping_boom)
    trainer = SimpleNamespace(is_global_zero=True)

    with pytest.raises(OSError):
        dec_train._persist_results(trainer, OmegaConf.create({}), train_ds=None, session_ids=["s1"])
    assert calls == ["v2", "bookkeeping"]


def test_persist_results_noop_on_nonzero_rank(monkeypatch):
    calls = []
    monkeypatch.setattr(dec_train, "_write_decoding_results", lambda *a, **k: calls.append("v2"))
    monkeypatch.setattr(dec_train, "_save_results", lambda *a, **k: calls.append("bookkeeping"))
    dec_train._persist_results(
        SimpleNamespace(is_global_zero=False), OmegaConf.create({}), train_ds=None, session_ids=None
    )
    assert calls == []


# ---------------------------------------------------------------------------
# A run that can't write its record fails. The record is what the cohort and
# downstream readers consume; a cell that exits 0 without it looks like a
# success but drops out of the cohort silently.
# ---------------------------------------------------------------------------


def _boom_record(*a, **k):
    raise OSError("record dir not writable")


def test_persist_results_fails_run_when_record_write_raises(monkeypatch):
    import pytest

    calls = []
    monkeypatch.setattr(dec_train, "_write_decoding_results", _boom_record)
    monkeypatch.setattr(dec_train, "_save_results", lambda *a, **k: calls.append("bookkeeping"))

    with pytest.raises(RuntimeError, match="results record") as excinfo:
        dec_train._persist_results(
            SimpleNamespace(is_global_zero=True), OmegaConf.create({}), train_ds=None, session_ids=["s1"]
        )
    # No run_result.json either: calibrate's --missing-only resume treats one as
    # "cell done", so writing it here would make the failed cell never rerun.
    assert calls == []
    assert isinstance(excinfo.value.__cause__, OSError)


def test_persist_results_fails_run_when_record_not_written(monkeypatch):
    """write_results returns None (and only logs) when the file can't be written."""
    import pytest

    calls = []
    monkeypatch.setattr(dec_train, "_write_decoding_results", lambda *a, **k: None)
    monkeypatch.setattr(dec_train, "_save_results", lambda *a, **k: calls.append("bookkeeping"))

    with pytest.raises(RuntimeError, match="results record"):
        dec_train._persist_results(
            SimpleNamespace(is_global_zero=True), OmegaConf.create({}), train_ds=None, session_ids=["s1"]
        )
    assert calls == []


def test_persist_results_writes_bookkeeping_after_record(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        dec_train, "_write_decoding_results", lambda *a, **k: calls.append("v2") or tmp_path / "rec.json"
    )
    monkeypatch.setattr(dec_train, "_save_results", lambda *a, **k: calls.append("bookkeeping"))
    dec_train._persist_results(
        SimpleNamespace(is_global_zero=True), OmegaConf.create({}), train_ds=None, session_ids=["s1"]
    )
    assert calls == ["v2", "bookkeeping"]


def test_write_record_into_a_file_path_fails_the_run(tmp_path, monkeypatch):
    """End to end through the real writer: a log dir that is a file can't hold the record."""
    import pytest

    blocker = tmp_path / "csv_logs"
    blocker.write_text("not a directory")
    monkeypatch.setattr(dec_train, "_save_results", lambda *a, **k: None)
    trainer = _trainer(tmp_path, _snap())
    trainer.is_global_zero = True
    cfg = OmegaConf.create({"tokenizer": {}, "log_dir": str(tmp_path)})

    with pytest.raises(RuntimeError, match="results record"):
        dec_train._persist_results(trainer, cfg, train_ds=None, session_ids=["s1"])


def test_main_checks_config_before_training(monkeypatch):
    """main() resolves the config before seeding, so a broken interpolation
    stops the run before training rather than when the record is written."""
    import pytest

    calls = []

    def _boom(cfg, *, task):
        calls.append(("check", task))
        raise RuntimeError("config check ran")

    monkeypatch.setattr(dec_train, "check_config_resolves", _boom)
    monkeypatch.setattr(dec_train, "seed_everything", lambda *a, **k: calls.append("seed"))
    with pytest.raises(RuntimeError, match="config check ran"):
        dec_train.main.__wrapped__(OmegaConf.create({"seed": 0}))
    assert calls == [("check", "decoding")]
