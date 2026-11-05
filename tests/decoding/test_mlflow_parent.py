"""Unit tests for the ntt_downstream.decoding.utils.mlflow_parent helper.

The module lazily imports ``mlflow`` (only inside the subcommands that talk to a
server), so its pure helpers — URI synthesis, tag parsing, metric filtering,
the mean/median aggregation, and the ``--disabled`` / empty-run-id no-op
contracts the orchestrator depends on — are all testable offline with no
MLflow installed or running.
"""

from __future__ import annotations

import argparse
import types

import pytest

from ntt_downstream.decoding.utils import mlflow_parent as mp


def _fake_run(run_id, status, tags, metrics):
    return types.SimpleNamespace(
        info=types.SimpleNamespace(run_id=run_id, status=status),
        data=types.SimpleNamespace(tags=tags, metrics=metrics),
    )


def test_metrics_table_is_tidy_per_child_with_facet_columns():
    children = [
        _fake_run("r1", "FINISHED", {"mlflow.runName": "n1", "seed": "42", "ft_mode": "full"},
                  {"test_r2": 0.8, "val_best_r2": 0.9}),
        _fake_run("r2", "FINISHED", {"mlflow.runName": "n2", "seed": "43", "calib_session": "s_b"},
                  {"test_r2": 0.7}),
    ]  # fmt: skip
    fields, rows = mp._metrics_table(children)
    assert fields[:3] == ["run_id", "run_name", "status"]
    # facet tags become columns; internal mlflow.* tags are dropped (name surfaced separately)
    assert {"tag.seed", "tag.ft_mode", "tag.calib_session"} <= set(fields)
    assert {"metric.test_r2", "metric.val_best_r2"} <= set(fields)
    assert not any(f.startswith("tag.mlflow") for f in fields)
    assert rows[0]["run_id"] == "r1" and rows[0]["run_name"] == "n1" and rows[0]["tag.seed"] == "42"
    assert rows[0]["metric.test_r2"] == "0.8"
    # missing facets / metrics are empty strings (union of keys across children), not errors
    assert rows[0]["tag.calib_session"] == "" and rows[1]["metric.val_best_r2"] == ""


def test_cmd_export_empty_run_id_is_noop():
    assert mp.cmd_export(argparse.Namespace(run_id="", out=None)) == 0


# --- _build_run_uri ---


def test_build_run_uri_http_scheme():
    uri = mp._build_run_uri("https://mlflow.example.com/", "12", "abc123")
    assert uri == "https://mlflow.example.com/#/experiments/12/runs/abc123"


def test_build_run_uri_file_scheme_is_not_browsable():
    assert mp._build_run_uri("file:///tmp/mlruns", "0", "abc123") == ("local-mlflow-run:abc123")
    # Empty scheme (bare local path) is treated the same way.
    assert mp._build_run_uri("/tmp/mlruns", "0", "xyz") == "local-mlflow-run:xyz"


# --- _parse_tag_args ---


def test_parse_tag_args_basic_and_empty_value():
    out = mp._parse_tag_args(["sweep=cmp-foo", "mlflow.note.content="])
    assert out == {"sweep": "cmp-foo", "mlflow.note.content": ""}


def test_parse_tag_args_value_may_contain_equals():
    assert mp._parse_tag_args(["k=a=b"]) == {"k": "a=b"}


def test_parse_tag_args_rejects_missing_equals():
    with pytest.raises(SystemExit):
        mp._parse_tag_args(["bad_tag"])


def test_parse_tag_args_rejects_empty_key():
    with pytest.raises(SystemExit):
        mp._parse_tag_args(["=value"])


# --- _is_aggregable (exact-key skip list, NOT suffix matching) ---


@pytest.mark.parametrize("key", ["epoch", "step"])
def test_is_aggregable_skips_bare_counters(key):
    assert mp._is_aggregable(key) is False


def test_is_aggregable_skips_system_prefix():
    assert mp._is_aggregable("system/cpu_utilization") is False


@pytest.mark.parametrize("key", ["val/standard_r2", "test_r2", "val_loss_step"])
def test_is_aggregable_keeps_real_metrics(key):
    # `val_loss_step` is kept on purpose: the skip list is exact-key, so a
    # `_step`-suffixed metric is NOT dropped (it isn't emitted today anyway).
    assert mp._is_aggregable(key) is True


# --- _aggregate_values (mean / median / count) ---


def test_aggregate_values_odd_count():
    assert mp._aggregate_values([3.0, 1.0, 2.0]) == (2.0, 2.0, 3)


def test_aggregate_values_even_count():
    assert mp._aggregate_values([1.0, 2.0, 3.0, 4.0]) == (2.5, 2.5, 4)


def test_aggregate_values_single():
    assert mp._aggregate_values([7.5]) == (7.5, 7.5, 1)


# --- soft-dependency no-op contracts callers rely on ---


def test_cmd_create_disabled_prints_empty_contract(capsys):
    rc = mp.cmd_create(argparse.Namespace(disabled=True))
    assert rc == 0
    out = capsys.readouterr().out
    assert out == "RUN_ID=\nRUN_URI=\n"


def test_cmd_create_no_tracking_uri_is_noop(capsys, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    rc = mp.cmd_create(argparse.Namespace(disabled=False, experiment_name="e", run_name="r", tag=[]))
    assert rc == 0
    assert capsys.readouterr().out == "RUN_ID=\nRUN_URI=\n"


def test_cmd_aggregate_empty_run_id_is_noop():
    # Returns 0 without importing mlflow or touching the network.
    assert mp.cmd_aggregate(argparse.Namespace(run_id="")) == 0


def _fake_mlflow(logged: list):
    class _Client:
        def log_artifact(self, run_id, path, artifact_path=None):
            logged.append((run_id, path, artifact_path))

    return types.SimpleNamespace(
        set_tracking_uri=lambda uri: None,
        tracking=types.SimpleNamespace(MlflowClient=_Client),
    )


def test_log_artifacts_uploads_v2_record_never_run_result(tmp_path, monkeypatch):
    """Per-child trees upload the v2 record (+ metrics.csv), never run_result.json;
    ``--file`` paths (the cohort record) land at the parent's artifact root."""
    from pathlib import Path

    v = tmp_path / "cell" / "csv_logs" / "version_0"
    v.mkdir(parents=True)
    for name in ("run_result.json", "downstream_decoding_results.json", "metrics.csv", "model.ckpt"):
        (v / name).write_text("{}")
    cohort = tmp_path / "cohort" / "downstream_decoding_results.json"
    cohort.parent.mkdir()
    cohort.write_text("{}")
    logged: list = []
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///tmp/unused")
    monkeypatch.setattr(mp, "_safe_import_mlflow", lambda: _fake_mlflow(logged))

    rc = mp.cmd_log_artifacts(argparse.Namespace(run_id="R", src=[str(tmp_path / "cell")], file=[str(cohort)]))

    assert rc == 0
    uploaded = sorted((Path(p).name, ap or "") for _, p, ap in logged)
    assert uploaded == [
        ("downstream_decoding_results.json", ""),
        ("downstream_decoding_results.json", "cell/csv_logs/version_0"),
        ("metrics.csv", "cell/csv_logs/version_0"),
    ]


def test_log_artifacts_cli_accepts_file_flag(monkeypatch):
    seen = {}
    monkeypatch.setattr(mp, "cmd_log_artifacts", lambda args: seen.setdefault("file", args.file) and 0)
    mp.main(["log-artifacts", "--run-id", "R", "--file", "/x.json", "--file", "/y.json"])
    assert seen["file"] == ["/x.json", "/y.json"]


def test_log_artifacts_skips_older_attempts(tmp_path, monkeypatch):
    """Only the newest csv_logs/version_N of a cell is uploaded: an earlier
    attempt's record next to the current one would mislead drill-down."""
    for n in (0, 2, 10):
        v = tmp_path / "cell" / "csv_logs" / f"version_{n}"
        v.mkdir(parents=True)
        (v / "downstream_decoding_results.json").write_text("{}")
    calib = tmp_path / "cell" / "calibration_full.csv"  # outside csv_logs: always uploaded
    calib.write_text("x")
    logged: list = []
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///tmp/unused")
    monkeypatch.setattr(mp, "_safe_import_mlflow", lambda: _fake_mlflow(logged))

    mp.cmd_log_artifacts(argparse.Namespace(run_id="R", src=[str(tmp_path / "cell")], file=[]))

    assert sorted(ap for _, _, ap in logged) == ["cell", "cell/csv_logs/version_10"]
