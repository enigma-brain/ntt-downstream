"""Generate the v2 results-record fixtures under ``tests/fixtures/results_v2/``.

The fixtures are produced by the **real** writers (encoding/decoding train
writers and the cohort builder) on small fake inputs, with provenance pinned to
fixed values so the output is deterministic. They exist for consumers: ntt's
aggregation reader tests against copies of these files, so it is exercised on
exactly what this repo writes. ``tests/common/test_v2_fixtures_current.py``
fails when a writer change makes them stale.

Usage:
    uv run python scripts/fixtures/make_results_v2_fixtures.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from omegaconf import OmegaConf

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "results_v2"

_PINNED = {
    "git_sha": "fixture-git-sha",
    "now_iso": "2026-01-01T00:00:00Z",
    "hostname": "fixture-host",
    "ntt": {
        "ntt_version": "0.0.0-fixture",
        "ntt_path": "/fixture/ntt/src/ntt/__init__.py",
        "ntt_commit": "0000000000000000000000000000000000000000",
    },
}


def _pinned_provenance():
    """Pin every provenance source the writers read, wherever it was imported."""
    import ntt_downstream.common.results.provenance as prov
    import ntt_downstream.decoding.cohort_results as cohort

    patches = []
    for module in (prov, cohort):
        patches += [
            mock.patch.object(module, "git_sha", lambda *a, **k: _PINNED["git_sha"]),
            mock.patch.object(module, "now_iso", lambda: _PINNED["now_iso"]),
            mock.patch.object(module, "hostname", lambda: _PINNED["hostname"]),
            mock.patch.object(module, "ntt_provenance", lambda: dict(_PINNED["ntt"])),
        ]
    return patches


def _final_eval_table(session: str, rate: float, nc_r2: float, split: str) -> dict[str, Any]:
    stat = lambda m: {"mean": m, "std": 0.1, "n_valid": 96, "n_total": 96}  # noqa: E731
    overall = lambda m: {"mean": m, "se": 0.01, "n_sessions": 1}  # noqa: E731
    return {
        "split": split,
        "sessions": [session],
        "n_units_by_session": {session: 96},
        "per_session": {session: {"rate_pearson": stat(rate), "nc_r2": stat(nc_r2)}},
        "overall": {"rate_pearson": overall(rate), "nc_r2": overall(nc_r2)},
    }


def _encoding_records(tmp: Path) -> dict[str, dict]:
    from ntt_downstream.common.results.callback import MetricSnapshotCallback
    from ntt_downstream.encoding import train as enc_train

    snap = MetricSnapshotCallback(monitor="val/rate_corr/overall", mode="max")
    snap.best_val = {"val/rate_corr/overall": 0.31, "val_nc_r2/overall": 0.22, "val/rate_corr/PE_201021": 0.31}
    snap.last_val = {"val/rate_corr/overall": 0.29, "val_nc_r2/overall": 0.20, "val/rate_corr/PE_201021": 0.29}
    trainer = SimpleNamespace(callbacks=[], loggers=[], is_global_zero=True)
    model = SimpleNamespace(_session_info={"PE_201021": 96})
    cfg = OmegaConf.create(
        {"tokenizer": {"enabled": True, "checkpoint_path": "mlflow-artifacts:/1/abc/artifacts/model.ckpt"}}
    )

    table = _final_eval_table("PE_201021", 0.30, 0.21, "valid")
    table["test"] = _final_eval_table("PE_201021", 0.28, 0.19, "test")
    out = {}
    for name, final_table, status in (
        ("encoding_run.json", table, {"status": "ok"}),
        ("encoding_run_final_eval_failed.json", None, {"status": "failed", "error": "RuntimeError('cuda oom')"}),
    ):
        d = tmp / name
        d.mkdir()
        path = enc_train._write_encoding_results(
            trainer, cfg, model, str(d), "fixture_run", None, snap, final_table=final_table, final_eval=status
        )
        out[name] = json.loads(path.read_text())
    return out


def _decoding_run(run_dir: Path, session: str, test_r2: float | None) -> Path:
    from ntt_downstream.common.results.callback import MetricSnapshotCallback
    from ntt_downstream.decoding import train as dec_train

    snap = MetricSnapshotCallback(monitor="average_val_metric", mode="max")
    snap.best_val = {"average_val_metric": 0.86, session: 0.86}
    snap.last_val = {"average_val_metric": 0.84, session: 0.84}
    snap.test = {"average_test_metric": test_r2, session: test_r2} if test_r2 is not None else None
    trainer = SimpleNamespace(
        callbacks=[snap], loggers=[], log_dir=str(run_dir / "csv_logs" / "version_0"), default_root_dir=str(run_dir)
    )
    cfg = OmegaConf.create({"tokenizer": {"ntt_checkpoint": "mlflow-artifacts:/1/abc/artifacts/model.ckpt"}})
    return dec_train._write_decoding_results(trainer, cfg, session_ids=[session])


def _decoding_multisession_with_nan(tmp: Path) -> dict:
    """A multi-session run with one diverged session and a real ModelCheckpoint.

    Exercises the shapes a reader must handle: a populated ``checkpoint`` dict,
    several sessions, and a null per-session R² excluded from the aggregate.
    """
    from lightning.pytorch.callbacks import ModelCheckpoint

    from ntt_downstream.common.results.callback import MetricSnapshotCallback
    from ntt_downstream.decoding import train as dec_train

    sessions = ["c_20131003_center_out_reaching", "t_20130819_center_out_reaching", "j_20160405_center_out_reaching"]
    test = dict(zip(sessions, (0.82, float("nan"), 0.74), strict=True))
    snap = MetricSnapshotCallback(monitor="average_val_metric", mode="max")
    snap.best_val = {"average_val_metric": 0.8, **dict.fromkeys(sessions, 0.8)}
    snap.last_val = {"average_val_metric": 0.79, **dict.fromkeys(sessions, 0.79)}
    snap.test = {"average_test_metric": 0.78, **test}
    snap.best_epoch, snap.best_step = 3, 1200
    ckpt = ModelCheckpoint(dirpath=str(tmp / "ckpts"), monitor="average_val_metric", mode="max")
    ckpt.best_model_path = "checkpoints/epoch=3-step=1200.ckpt"
    ckpt.last_model_path = "checkpoints/last.ckpt"
    run_dir = tmp / "multi"
    trainer = SimpleNamespace(
        callbacks=[snap, ckpt],
        loggers=[],
        log_dir=str(run_dir / "csv_logs" / "version_0"),
        default_root_dir=str(run_dir),
    )
    cfg = OmegaConf.create({"tokenizer": {"ntt_checkpoint": "mlflow-artifacts:/1/abc/artifacts/model.ckpt"}})
    return json.loads(dec_train._write_decoding_results(trainer, cfg, session_ids=sessions).read_text())


def _decoding_records(tmp: Path) -> dict[str, dict]:
    from ntt_downstream.decoding import cohort_results

    sessions = {"c_20131003_center_out_reaching": 0.85, "t_20130819_center_out_reaching": 0.81}
    dirs = {s: tmp / f"T__tokA__{s}" for s in sessions}
    for s, r2 in sessions.items():
        path = _decoding_run(dirs[s], s, r2)
    out = {
        "decoding_run.json": json.loads(path.read_text()),
        "decoding_run_multisession_nan.json": _decoding_multisession_with_nan(tmp),
    }

    for name, child_dirs in (
        ("decoding_cohort.json", dirs),
        ("decoding_cohort_incomplete.json", {**dirs, "j_20160405_center_out_reaching": tmp / "never_ran"}),
    ):
        doc = cohort_results.build_cohort_document(
            tokenizer_name="tokA",
            tokenizer_checkpoint="mlflow-artifacts:/1/abc/artifacts/model.ckpt",
            parent_run_id="fixture-parent-run",
            child_dirs=child_dirs,
        )
        # Absolute temp paths would make the fixture non-deterministic.
        for child in doc.metadata.extra["cohort"]["children"].values():
            child["record_path"] = str(Path(child["record_path"]).relative_to(tmp))
        d = tmp / name.removesuffix(".json")
        out[name] = json.loads(cohort_results.write_cohort_record(doc, d).read_text())
    return out


def build_all() -> dict[str, dict]:
    """Every fixture name -> its JSON content, produced by the real writers."""
    patches = _pinned_provenance()
    for p in patches:
        p.start()
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            (tmp / "enc").mkdir()
            (tmp / "dec").mkdir()
            return {**_encoding_records(tmp / "enc"), **_decoding_records(tmp / "dec")}
    finally:
        for p in patches:
            p.stop()


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    for name, doc in build_all().items():
        (FIXTURES_DIR / name).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        print(f"wrote {FIXTURES_DIR / name}")


if __name__ == "__main__":
    main()
