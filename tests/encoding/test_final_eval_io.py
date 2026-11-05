"""IO + checkpoint-selection guards for end-of-training final eval.

Two concerns:

1. **Checkpoint selection** — ``train._maybe_run_final_eval`` must
   evaluate the *reported* (val-best) checkpoint, not ``last.ckpt``. The
   headline ``results.json`` reports the ``ModelCheckpoint(monitor=
   "val/rate_corr/overall").best_model_path`` snapshot; loading anything else,
   or silently falling back to the in-memory weights on a load error, would
   make final_eval score a *different* model than the one reported. These
   tests pin: best is loaded when present,
   ``last.ckpt`` is the fallback only when best is empty, and ANY state_dict
   mismatch RAISES rather than silently evaluating wrong weights (there is no
   key allowlist — the tokenizer wrapper never appears in the state_dict).

2. **Result storage** — ``final_eval.write_outputs`` must persist the
   canonical machine-readable record through the shared
   ``common.results.serialization.write_json`` (RFC-8259 conformant, shared
   schema), not a bespoke ``json.dump``. The fixed-width table print and the
   human-readable CSV may remain as supplementary report artifacts.

No GPU. Tiny CPU fakes throughout; we mock ``final_eval.run`` to a no-op so
the checkpoint-selection tests focus purely on checkpoint selection and state_dict validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from ntt_downstream.encoding import train as enc_train
from ntt_downstream.encoding.training import final_eval

# ---------------------------------------------------------------------------
# Fakes — checkpoint selection + state_dict validation
# ---------------------------------------------------------------------------


class _FakeStrategy:
    """Carries just the ``root_device`` that ``_maybe_run_final_eval`` reads."""

    root_device = torch.device("cpu")


from lightning.pytorch.callbacks import ModelCheckpoint as _RealModelCheckpoint


class _FakeModelCheckpoint(_RealModelCheckpoint):
    """ModelCheckpoint stand-in matched by ``monitor`` + ``best_model_path``.

    ``_maybe_run_final_eval`` looks up the checkpoint callback exactly as
    ``_write_encoding_results`` does (``isinstance(cb, ModelCheckpoint) and
    cb.monitor == "val/rate_corr/overall"``). We subclass the real
    ModelCheckpoint (without running its heavyweight ``__init__``) so the
    production ``isinstance`` check matches while we set only the few
    attributes the lookup reads.
    """

    monitor = "val/rate_corr/overall"

    def __init__(self, best_model_path: str = "", last_model_path: str = ""):
        # Deliberately skip super().__init__(): we only need the attributes the
        # callback lookup reads, not a configured checkpoint writer.
        self.best_model_path = best_model_path
        self.last_model_path = last_model_path
        self.best_model_score = None
        self.mode = "max"


class _FakeTrainer:
    """Trainer stand-in exposing ``callbacks``, ``strategy``, ``is_global_zero``."""

    is_global_zero = True

    def __init__(self, callbacks):
        self.callbacks = callbacks
        self.strategy = _FakeStrategy()
        self.loggers = []


class _FakeModel(torch.nn.Module):
    """LitModule stand-in with one parameter so a sentinel weight is observable.

    ``set_val_dataloader`` / ``set_val_average_dataloader`` are no-op setters
    that ``_maybe_run_final_eval`` calls before invoking ``fe.run``.
    """

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(3))

    def set_val_dataloader(self, fn):
        self._vfn = fn

    def set_val_average_dataloader(self, fn):
        self._vafn = fn


def _make_cfg() -> OmegaConf:
    return OmegaConf.create({"trainer": {"final_eval": {"enabled": True, "n_samples": 4, "flow_solver": "heun"}}})


def _write_ckpt(path: Path, state_dict: dict) -> None:
    """Write a minimal Lightning-style checkpoint carrying ``state_dict``."""
    torch.save({"state_dict": state_dict}, str(path))


@pytest.fixture
def _noop_run(monkeypatch):
    """Replace ``final_eval.run`` with a recorder no-op.

    The checkpoint-selection tests are about *which* checkpoint reaches the model — not the eval itself —
    so we stub the heavy loader walk and just record that run was reached.
    """
    calls = {"n": 0}

    def _fake_run(lit, **kw):
        calls["n"] += 1
        return {}

    monkeypatch.setattr(final_eval, "run", _fake_run)
    return calls


def test_loads_best_checkpoint(tmp_path, _noop_run):
    """final_eval loads the BEST checkpoint, not last.ckpt.

    The best ckpt carries a sentinel weight; after ``_maybe_run_final_eval``
    the model parameter must equal the sentinel (proving best, not last/
    in-memory, reached the model)."""
    sentinel = torch.tensor([1.0, 2.0, 3.0])
    best = tmp_path / "best.ckpt"
    last = tmp_path / "last.ckpt"
    _write_ckpt(best, {"w": sentinel})
    _write_ckpt(last, {"w": torch.tensor([9.0, 9.0, 9.0])})

    model = _FakeModel()
    cb = _FakeModelCheckpoint(best_model_path=str(best), last_model_path=str(last))
    trainer = _FakeTrainer([cb])

    enc_train._maybe_run_final_eval(
        _make_cfg(),
        model,
        dm=None,
        trainer=trainer,
        mlflow_logger=None,
        output_dir=str(tmp_path),
        checkpoint_dir=tmp_path,
    )

    assert torch.equal(model.w.detach(), sentinel)
    assert _noop_run["n"] == 1


def test_raises_on_state_dict_mismatch(tmp_path, _noop_run):
    """An unexpected key must RAISE, not silently continue.

    Loading with ``strict=False`` and swallowing exceptions would ignore a
    structurally-wrong checkpoint and evaluate the in-memory weights. The loader
    inspects the IncompatibleKeys and raises on ANY missing/unexpected key."""
    best = tmp_path / "best.ckpt"
    _write_ckpt(best, {"w": torch.zeros(3), "bogus.layer.weight": torch.zeros(2)})

    model = _FakeModel()
    cb = _FakeModelCheckpoint(best_model_path=str(best))
    trainer = _FakeTrainer([cb])

    with pytest.raises(RuntimeError, match="state_dict"):
        enc_train._maybe_run_final_eval(
            _make_cfg(),
            model,
            dm=None,
            trainer=trainer,
            mlflow_logger=None,
            output_dir=str(tmp_path),
            checkpoint_dir=tmp_path,
        )
    # Eval must not have run on wrong weights.
    assert _noop_run["n"] == 0


def test_raises_on_tokenizer_named_required_weight(tmp_path, _noop_run):
    """A REQUIRED weight whose name merely contains 'tokenizer' must RAISE.

    There is no key allowlist: the ``tokenizer_wrapper`` is a ``@dataclass`` held
    as a plain attribute of the model, so PyTorch never registers it in
    ``_modules`` and its params never appear in the LitModule state_dict in
    either direction. A substring allowlist (matching "tokenizer") would have
    silently swallowed a genuinely-required weight like
    ``model.tokenizer_adapter.weight`` — reintroducing an unfaithful evaluation. So a missing
    such key must surface as a loud failure, not be tolerated."""
    best = tmp_path / "best.ckpt"
    # Checkpoint is MISSING the live model's "w" and carries a tokenizer-named
    # unexpected key instead: a structural mismatch that must raise.
    _write_ckpt(best, {"tokenizer_adapter.weight": torch.zeros(2)})

    model = _FakeModel()
    cb = _FakeModelCheckpoint(best_model_path=str(best))
    trainer = _FakeTrainer([cb])

    with pytest.raises(RuntimeError, match="state_dict"):
        enc_train._maybe_run_final_eval(
            _make_cfg(),
            model,
            dm=None,
            trainer=trainer,
            mlflow_logger=None,
            output_dir=str(tmp_path),
            checkpoint_dir=tmp_path,
        )
    assert _noop_run["n"] == 0


def test_falls_back_to_last_when_no_best(tmp_path, _noop_run):
    """Empty best_model_path + present last.ckpt -> load last.ckpt."""
    last = tmp_path / "last.ckpt"
    sentinel = torch.tensor([7.0, 7.0, 7.0])
    _write_ckpt(last, {"w": sentinel})

    model = _FakeModel()
    cb = _FakeModelCheckpoint(best_model_path="", last_model_path=str(last))
    trainer = _FakeTrainer([cb])

    enc_train._maybe_run_final_eval(
        _make_cfg(),
        model,
        dm=None,
        trainer=trainer,
        mlflow_logger=None,
        output_dir=str(tmp_path),
        checkpoint_dir=tmp_path,
    )
    assert torch.equal(model.w.detach(), sentinel)
    assert _noop_run["n"] == 1


def test_skips_when_no_checkpoint(tmp_path, _noop_run):
    """Neither best nor last present -> log + return (NOT an error)."""
    model = _FakeModel()
    cb = _FakeModelCheckpoint(best_model_path="", last_model_path="")
    trainer = _FakeTrainer([cb])

    # Must not raise; final_eval simply doesn't run.
    enc_train._maybe_run_final_eval(
        _make_cfg(),
        model,
        dm=None,
        trainer=trainer,
        mlflow_logger=None,
        output_dir=str(tmp_path),
        checkpoint_dir=tmp_path,
    )
    assert _noop_run["n"] == 0


# ---------------------------------------------------------------------------
# Shared result storage
# ---------------------------------------------------------------------------


class _RunLit:
    """Minimal ``lit`` stand-in for driving ``final_eval.run`` with stubbed walks.

    ``run`` reads ``cfg``, calls ``model.eval()`` / ``_ensure_tokenizer_device``,
    runs the tokenizer warm-up (short-circuited via ``tokenizer_wrapper=None``),
    then walks the val loaders (stubbed out by the test). ``tokenizer_eval`` /
    ``seed`` live under ``cfg`` so the up-front MC seeding and n_flow_steps
    resolution succeed."""

    def __init__(self):
        self.cfg = OmegaConf.create(
            {
                "seed": 0,
                "model": {"loss_mode": "poisson", "model_type": "image_encoding"},
                "trainer": {"tokenizer_eval": {"n_flow_steps": 5}, "mar_eval": {}},
            }
        )
        self.model = _FakeModel()
        self.tokenizer_wrapper = None
        self._val_dataloader_fn = lambda: iter(())
        self._val_avg_dataloader_fn = lambda: iter(())

    def _ensure_tokenizer_device(self):
        pass


def _tiny_table() -> dict:
    """A minimal aggregate table shaped like ``final_eval.aggregate`` output."""
    return {
        "sessions": ["sessA"],
        "n_units_by_session": {"sessA": 5},
        "per_session": {
            "sessA": {
                "rate_pearson": {"mean": 0.5, "std": 0.1, "n_valid": 5, "n_total": 5},
                "temp_pearson": {"mean": 0.3, "std": 0.2, "n_valid": 5, "n_total": 5},
            }
        },
        "overall": {
            "rate_pearson": {"mean": 0.5, "se": 0.05, "n_sessions": 1},
            "temp_pearson": {"mean": 0.3, "se": 0.04, "n_sessions": 1},
            "nc_r2": {"mean": float("nan"), "se": float("nan"), "n_sessions": 0},
            "rate_pearson_psth": {"mean": 0.6, "se": 0.06, "n_sessions": 1},
            "temp_pearson_psth": {"mean": 0.4, "se": 0.03, "n_sessions": 1},
        },
        "config": {"loss_mode": "poisson", "model_type": "image_encoding"},
    }


def _stub_walks(monkeypatch, per_fn: dict) -> list:
    """Stub ``_walk_loader`` to return ``per_fn[loader_fn]`` and record the order."""
    calls = []

    def _fake_walk(lit, loader_fn, **kw):
        calls.append(loader_fn)
        return per_fn[loader_fn]

    monkeypatch.setattr(final_eval, "_walk_loader", _fake_walk)
    return calls


def _one_session_metrics(value: float) -> dict:
    return {"sessA": {"rate_pearson": torch.full((4,), value)}}


def test_run_without_test_loaders_is_val_only(tmp_path, monkeypatch):
    """No test loaders -> the val table, marked as such, no ``test`` key, no test CSV."""
    lit = _RunLit()
    _stub_walks(
        monkeypatch,
        {lit._val_dataloader_fn: _one_session_metrics(0.5), lit._val_avg_dataloader_fn: {}},
    )

    table = final_eval.run(lit, n_samples=1, output_dir=str(tmp_path))

    assert table["split"] == "valid"
    assert "test" not in table
    assert (tmp_path / "final_eval.csv").is_file()
    assert not (tmp_path / "final_eval_test.csv").exists()


def test_run_with_test_loaders_nests_test_table(tmp_path, monkeypatch):
    """Test loaders -> val table with the test table nested under ``test``.

    encoding/train.py maps these to the record's ``val_final`` / ``test``
    blocks. Val is walked first, then test."""
    lit = _RunLit()
    test_fn, test_avg_fn = (lambda: iter(())), (lambda: iter(()))
    calls = _stub_walks(
        monkeypatch,
        {
            lit._val_dataloader_fn: _one_session_metrics(0.5),
            lit._val_avg_dataloader_fn: {},
            test_fn: _one_session_metrics(0.25),
            test_avg_fn: {},
        },
    )

    table = final_eval.run(lit, n_samples=1, output_dir=str(tmp_path), test_loader_fns=(test_fn, test_avg_fn))

    assert calls == [lit._val_dataloader_fn, lit._val_avg_dataloader_fn, test_fn, test_avg_fn]
    assert table["overall"]["rate_pearson"]["mean"] == pytest.approx(0.5)
    assert table["test"]["split"] == "test"
    assert table["test"]["overall"]["rate_pearson"]["mean"] == pytest.approx(0.25)

    assert table["split"] == "valid"
    assert (tmp_path / "final_eval_test.csv").is_file()


def test_run_reseeds_before_test_walk(tmp_path, monkeypatch):
    """The test walk starts from the same RNG state as the val walk.

    MC-sampled loss modes draw from the global RNG; reseeding before each split
    makes the test numbers independent of how much sampling the val walk did."""
    lit = _RunLit()
    test_fn, test_avg_fn = (lambda: iter(())), (lambda: iter(()))
    first_draw = {}

    def _fake_walk(lit_, loader_fn, **kw):
        if loader_fn in (lit._val_dataloader_fn, test_fn):
            first_draw[loader_fn] = torch.rand(1).item()
        torch.rand(100)  # simulate MC sampling consuming RNG state
        return {}

    monkeypatch.setattr(final_eval, "_walk_loader", _fake_walk)
    final_eval.run(lit, n_samples=1, output_dir=None, test_loader_fns=(test_fn, test_avg_fn))

    assert first_draw[test_fn] == first_draw[lit._val_dataloader_fn]


def test_log_to_mlflow_prefixes_test_metrics():
    """Test-split scalars go under ``final_test/`` so they never overwrite val's ``final/``."""

    class _Rec:
        def __init__(self):
            self.metrics = {}

        def log_metrics(self, m):
            self.metrics.update(m)

    table = _tiny_table()
    table["test"] = _tiny_table()
    table["test"]["overall"]["rate_pearson"]["mean"] = 0.1
    rec = _Rec()

    final_eval.log_to_mlflow(table, rec)

    assert rec.metrics["final/rate_pearson/overall_mean"] == 0.5
    assert rec.metrics["final_test/rate_pearson/overall_mean"] == 0.1


class _FakeDataModule:
    """Data module stand-in: records ``setup`` stages and builds named datasets."""

    def __init__(self):
        self.stages = []
        self.test_dataset = None
        self.test_avg_dataset = None

    def setup(self, stage=None):
        self.stages.append(stage)
        self.test_dataset = "test_ds"
        self.test_avg_dataset = "test_avg_ds"


def test_final_eval_passes_test_loaders_by_default(tmp_path, monkeypatch):
    """``_maybe_run_final_eval`` scores the test split by default, lazily.

    The test datasets are only built (``dm.setup("test")``) when a test loader
    is first walked, and each loader is built over the matching dataset."""
    captured = {}

    def _fake_run(lit, **kw):
        captured.update(kw)
        return {}

    built = []
    monkeypatch.setattr(final_eval, "run", _fake_run)
    monkeypatch.setattr(enc_train, "_build_single_rank_val_loader", lambda dm, ds: built.append(ds) or ds)

    best = tmp_path / "best.ckpt"
    _write_ckpt(best, {"w": torch.zeros(3)})
    dm = _FakeDataModule()
    enc_train._maybe_run_final_eval(
        _make_cfg(),
        _FakeModel(),
        dm=dm,
        trainer=_FakeTrainer([_FakeModelCheckpoint(str(best))]),
        mlflow_logger=None,
        output_dir=str(tmp_path),
        checkpoint_dir=tmp_path,
    )

    test_fn, test_avg_fn = captured["test_loader_fns"]
    assert dm.stages == []  # nothing built until the eval walks a test loader
    assert test_fn() == "test_ds" and test_avg_fn() == "test_avg_ds"
    assert dm.stages == ["test"]
    assert built == ["test_ds", "test_avg_ds"]


def test_final_eval_test_split_can_be_disabled(tmp_path, monkeypatch):
    """``trainer.final_eval.test=false`` -> val-only final eval."""
    captured = {}
    monkeypatch.setattr(final_eval, "run", lambda lit, **kw: captured.update(kw) or {})

    best = tmp_path / "best.ckpt"
    _write_ckpt(best, {"w": torch.zeros(3)})
    cfg = _make_cfg()
    cfg.trainer.final_eval.test = False
    enc_train._maybe_run_final_eval(
        cfg,
        _FakeModel(),
        dm=_FakeDataModule(),
        trainer=_FakeTrainer([_FakeModelCheckpoint(str(best))]),
        mlflow_logger=None,
        output_dir=str(tmp_path),
        checkpoint_dir=tmp_path,
    )

    assert captured["test_loader_fns"] is None


def test_table_to_metric_block_tags_final_aggregate_as_mean_of_means():
    """The final block's aggregate must declare it summarises mean-of-session-means
    (with std carrying SE), so a consumer can't confuse it with decoding's
    across-session sample std."""
    from ntt_downstream.encoding.training.final_eval import table_to_metric_block

    table = {
        "sessions": ["s1"],
        "n_units_by_session": {"s1": 10},
        "per_session": {"s1": {"rate_pearson": {"mean": 0.6, "std": 0.1}}},
        "overall": {"rate_pearson": {"mean": 0.7, "se": 0.02, "n_sessions": 3}},
    }
    block = table_to_metric_block(table)
    entry = block.aggregated["rate_pearson"]
    assert entry.value == 0.7
    assert entry.std == 0.02  # SE carried in std
    assert entry.n == 3
    assert entry.aggregated_over == "sessions:mean_of_means"


# ---------------------------------------------------------------------------
# v2 record: one downstream_encoding_results.json, written after final eval
# ---------------------------------------------------------------------------


def _snap_cb(best=None, last=None):
    from ntt_downstream.common.results.callback import MetricSnapshotCallback

    cb = MetricSnapshotCallback(monitor="val/rate_corr/overall", mode="max")
    cb.best_val = best or {"val/rate_corr/overall": 0.3}
    cb.last_val = last or {"val/rate_corr/overall": 0.2}
    return cb


def _enc_cfg():
    return OmegaConf.create({"tokenizer": {"enabled": True, "checkpoint_path": "/ckpt/tok.ckpt"}, "trainer": {}})


def _write_record(tmp_path, *, final_table, final_eval):
    return enc_train._write_encoding_results(
        _FakeTrainer([]),
        _enc_cfg(),
        _FakeModel(),
        str(tmp_path),
        "run",
        None,
        _snap_cb(),
        final_table=final_table,
        final_eval=final_eval,
    )


def test_encoding_record_has_v2_blocks(tmp_path):
    table = _tiny_table()
    table["test"] = _tiny_table()
    table["test"]["overall"]["rate_pearson"]["mean"] = 0.1

    path = _write_record(tmp_path, final_table=table, final_eval={"status": "ok"})

    doc = json.loads(path.read_text())
    assert path.name == "downstream_encoding_results.json"
    assert (doc["schema_version"], doc["kind"], doc["task"]) == (2, "run", "encoding")
    assert set(doc["metrics"]) == {"val_best", "val_last", "val_final", "test"}
    test_rp = doc["metrics"]["test"]["aggregated"]["rate_pearson"]
    assert test_rp["value"] == 0.1 and test_rp["aggregated_over"] == "sessions:mean_of_means"
    assert doc["metrics"]["val_final"]["aggregated"]["rate_pearson"]["value"] == 0.5
    assert doc["metadata"]["tokenizer_checkpoint"] == "/ckpt/tok.ckpt"
    assert doc["metadata"]["final_eval"] == {"status": "ok"}
    assert doc["metadata"]["ntt_path"]
    assert not (tmp_path / "final_eval_results.json").exists()
    assert not (tmp_path / "results.json").exists()


def test_final_eval_failure_writes_record_without_test(tmp_path, monkeypatch):
    def _boom(lit, **kw):
        raise RuntimeError("cuda oom")

    monkeypatch.setattr(final_eval, "run", _boom)
    best = tmp_path / "best.ckpt"
    _write_ckpt(best, {"w": torch.zeros(3)})

    table, status = enc_train._maybe_run_final_eval(
        _make_cfg(),
        _FakeModel(),
        dm=_FakeDataModule(),
        trainer=_FakeTrainer([_FakeModelCheckpoint(str(best))]),
        mlflow_logger=None,
        output_dir=str(tmp_path),
        checkpoint_dir=tmp_path,
    )
    assert table is None and status["status"] == "failed" and "cuda oom" in status["error"]

    doc = json.loads(_write_record(tmp_path, final_table=table, final_eval=status).read_text())
    assert set(doc["metrics"]) == {"val_best", "val_last"}
    assert doc["metadata"]["final_eval"]["status"] == "failed"


def test_final_eval_disabled_status(tmp_path):
    cfg = _make_cfg()
    cfg.trainer.final_eval.enabled = False
    table, status = enc_train._maybe_run_final_eval(
        cfg,
        _FakeModel(),
        dm=None,
        trainer=_FakeTrainer([]),
        mlflow_logger=None,
        output_dir=str(tmp_path),
        checkpoint_dir=tmp_path,
    )
    assert table is None and status == {"status": "disabled"}


def test_final_eval_ok_returns_table(tmp_path, _noop_run):
    best = tmp_path / "best.ckpt"
    _write_ckpt(best, {"w": torch.zeros(3)})
    table, status = enc_train._maybe_run_final_eval(
        _make_cfg(),
        _FakeModel(),
        dm=_FakeDataModule(),
        trainer=_FakeTrainer([_FakeModelCheckpoint(str(best))]),
        mlflow_logger=None,
        output_dir=str(tmp_path),
        checkpoint_dir=tmp_path,
    )
    assert table == {} and status == {"status": "ok", "ckpt_kind": "best"}


def test_write_outputs_writes_csvs_only(tmp_path):
    table = _tiny_table()
    table["test"] = _tiny_table()
    paths = final_eval.write_outputs(table, tmp_path)
    assert set(paths) == {"csv", "test_csv"}
    assert not list(tmp_path.glob("*.json"))


# ---------------------------------------------------------------------------
# Post-fit glue (_finalize_run): the record survives every final-eval outcome
# ---------------------------------------------------------------------------


def _finalize(tmp_path, *, trainer, run_final_eval=True, dm=None, cfg=None):
    enc_train._finalize_run(
        cfg or _make_cfg_with_tokenizer(),
        _FakeModel(),
        dm or _FakeDataModule(),
        trainer,
        None,
        str(tmp_path),
        tmp_path,
        "run",
        _snap_cb(),
        run_final_eval=run_final_eval,
    )
    return json.loads((tmp_path / "downstream_encoding_results.json").read_text())


def _make_cfg_with_tokenizer():
    cfg = _make_cfg()
    cfg.tokenizer = {"enabled": True, "checkpoint_path": "/ckpt/tok.ckpt"}
    return cfg


def test_finalize_persists_record_then_reraises_on_state_dict_mismatch(tmp_path, _noop_run):
    best = tmp_path / "best.ckpt"
    _write_ckpt(best, {"w": torch.zeros(3), "bogus.layer.weight": torch.zeros(2)})

    with pytest.raises(RuntimeError, match="state_dict"):
        _finalize(tmp_path, trainer=_FakeTrainer([_FakeModelCheckpoint(str(best))]))

    doc = json.loads((tmp_path / "downstream_encoding_results.json").read_text())
    assert set(doc["metrics"]) == {"val_best", "val_last"}
    assert doc["metadata"]["final_eval"]["status"] == "failed"
    assert "state_dict" in doc["metadata"]["final_eval"]["error"]


def test_finalize_ok_records_ckpt_kind(tmp_path, monkeypatch):
    table = _tiny_table()
    table["test"] = _tiny_table()
    monkeypatch.setattr(final_eval, "run", lambda lit, **kw: table)
    best = tmp_path / "best.ckpt"
    _write_ckpt(best, {"w": torch.zeros(3)})

    doc = _finalize(tmp_path, trainer=_FakeTrainer([_FakeModelCheckpoint(str(best))]))

    assert set(doc["metrics"]) == {"val_best", "val_last", "val_final", "test"}
    assert doc["metadata"]["final_eval"] == {"status": "ok", "ckpt_kind": "best"}


def test_finalize_without_final_eval_still_writes_record(tmp_path):
    """The audio path: no final eval, but the record (diagnostic blocks) is written."""
    doc = _finalize(tmp_path, trainer=_FakeTrainer([]), run_final_eval=False)
    assert set(doc["metrics"]) == {"val_best", "val_last"}
    assert doc["metadata"]["final_eval"]["status"] == "disabled"


def test_final_eval_test_disabled_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(final_eval, "run", lambda lit, **kw: _tiny_table())
    best = tmp_path / "best.ckpt"
    _write_ckpt(best, {"w": torch.zeros(3)})
    cfg = _make_cfg()
    cfg.trainer.final_eval.test = False

    _table, status = enc_train._maybe_run_final_eval(
        cfg,
        _FakeModel(),
        dm=_FakeDataModule(),
        trainer=_FakeTrainer([_FakeModelCheckpoint(str(best))]),
        mlflow_logger=None,
        output_dir=str(tmp_path),
        checkpoint_dir=tmp_path,
    )
    assert status == {"status": "ok", "ckpt_kind": "best", "test": "disabled"}


def test_no_checkpoint_is_failed_not_disabled(tmp_path, _noop_run):
    _table, status = enc_train._maybe_run_final_eval(
        _make_cfg(),
        _FakeModel(),
        dm=None,
        trainer=_FakeTrainer([_FakeModelCheckpoint("", "")]),
        mlflow_logger=None,
        output_dir=str(tmp_path),
        checkpoint_dir=tmp_path,
    )
    assert status["status"] == "failed" and "no checkpoint" in status["error"]


def test_test_split_failure_keeps_val_table(tmp_path, monkeypatch):
    """A crash in the test split (dm.setup('test'), test walk) must not discard the
    already-computed val table: run() records the error and returns val only."""
    lit = _RunLit()

    def _boom():
        raise RuntimeError("test split missing")

    _stub_walks(monkeypatch, {lit._val_dataloader_fn: _one_session_metrics(0.5), lit._val_avg_dataloader_fn: {}})
    real_walk = final_eval._walk_loader

    def _walk(lit_, fn, **kw):
        if fn is _boom:
            fn()
        return real_walk(lit_, fn, **kw)

    monkeypatch.setattr(final_eval, "_walk_loader", _walk)
    table = final_eval.run(lit, n_samples=1, output_dir=None, test_loader_fns=(_boom, _boom))

    assert "test" not in table and "test split missing" in table["test_error"]
    assert table["overall"]["rate_pearson"]["mean"] == pytest.approx(0.5)


def test_status_reports_test_failure(tmp_path, monkeypatch):
    table = _tiny_table()
    table["test_error"] = "RuntimeError('test split missing')"
    monkeypatch.setattr(final_eval, "run", lambda lit, **kw: table)
    best = tmp_path / "best.ckpt"
    _write_ckpt(best, {"w": torch.zeros(3)})

    _t, status = enc_train._maybe_run_final_eval(
        _make_cfg(),
        _FakeModel(),
        dm=_FakeDataModule(),
        trainer=_FakeTrainer([_FakeModelCheckpoint(str(best))]),
        mlflow_logger=None,
        output_dir=str(tmp_path),
        checkpoint_dir=tmp_path,
    )
    assert status["test"] == "failed" and "test split missing" in status["test_error"]


def test_tokenizer_checkpoint_respects_enabled():
    def cfg(enabled):
        return OmegaConf.create({"tokenizer": {"enabled": enabled, "checkpoint_path": "/x"}})

    assert enc_train._tokenizer_checkpoint(cfg(False)) is None
    assert enc_train._tokenizer_checkpoint(cfg(True)) == "/x"


def test_final_eval_blocks_carry_the_same_value_ranges_as_val_blocks():
    block = final_eval.table_to_metric_block(_tiny_table())
    assert block.aggregated["rate_pearson"].value_range == (-1.0, 1.0)
    assert block.aggregated["rate_pearson_psth"].value_range == (-1.0, 1.0)
    table = _tiny_table()
    table["overall"]["nc_r2"] = {"mean": 0.2, "se": 0.01, "n_sessions": 1}
    assert final_eval.table_to_metric_block(table).aggregated["nc_r2"].value_range == (None, 1.0)
