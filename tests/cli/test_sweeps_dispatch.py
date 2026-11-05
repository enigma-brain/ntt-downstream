"""Dispatch tests for the sweep launcher (calibrate/benchmark).

These verify the launcher loads configs/cli/<verb>.yaml, overlays the promoted
CLI flags, and drives the EXISTING orchestration engine verbatim through the
config-object seam — without re-implementing _resolve_calibrate/run_benchmark.
The engine's own --dry-run path is exercised, so no GPU/data is needed.
"""

import pytest

from ntt_downstream.cli import app as cli
from ntt_downstream.decoding import orchestration as orch


def test_calibrate_dry_run_drives_engine(tmp_path, monkeypatch):
    captured = []
    # _run_train receives the per-cell Hydra OVERRIDES (no _TRAIN_PREFIX).
    monkeypatch.setattr(orch, "_run_train", lambda argv, **k: captured.append(argv) or 0)
    monkeypatch.setattr(orch, "_mlflow_create_parent", lambda *a, **k: ("", ""))
    monkeypatch.setattr(orch, "_resolve_gpus", lambda *a, **k: ["0"])
    rc = cli.main(
        [
            "calibrate",
            "--config-name",
            "calibrate",
            "--tokenizer-ckpt",
            "/m/tokA/checkpoints/model.ckpt",
            "--tokenizer-kind",
            "frozen",
            "--tokenizer-name",
            "tokA",
            "--data-root",
            str(tmp_path),
            "--output-root",
            str(tmp_path),
            "--dry-run",
        ]
    )
    assert rc == 0
    assert captured, "engine should have resolved at least one cell"
    # The promoted tokenizer ckpt + kind must reach the per-cell overrides verbatim:
    assert any("tokenizer=frozen" in tok for cell in captured for tok in cell)
    assert any("tokenizer.ntt_checkpoint=/m/tokA/checkpoints/model.ckpt" in tok for cell in captured for tok in cell)


def test_unknown_sweep_config_key_rejected(tmp_path, capsys, monkeypatch):
    """A typo'd / stale sweep YAML key fails loudly (exit 2), never silently
    ignored — the engine seam validates keys against its real argparse dests."""
    bad = tmp_path / "calibrate.yaml"
    bad.write_text("jobs_per_gpus: 2\n")  # typo: extra 's'
    # The guard fires before any engine work, but stub the engine so a failure of
    # the guard is loud here rather than touching GPUs.
    monkeypatch.setattr(orch, "_run_train", lambda *a, **k: 0)
    monkeypatch.setattr(orch, "_resolve_gpus", lambda *a, **k: ["0"])
    rc = cli.main(
        [
            "calibrate",
            "--config-dir",
            str(tmp_path),
            "--config-name",
            "calibrate",
            "--dry-run",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown calibrate sweep config key 'jobs_per_gpus'" in err


def test_calibrate_print_resolved_config_never_touches_engine(capsys, monkeypatch):
    """--print-resolved-config prints the overlaid sweep mapping and exits 0
    WITHOUT driving the engine (it must never launch cells)."""

    def _boom(*a, **k):
        raise AssertionError("engine must not run under --print-resolved-config")

    monkeypatch.setattr(orch, "calibrate_config_from_mapping", _boom)
    monkeypatch.setattr(orch, "run_calibrate", _boom)
    rc = cli.main(["calibrate", "--tokenizer-name", "tokA", "--print-resolved-config"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "tokenizer_name: tokA" in out
    assert "jobs_per_gpu: 1" in out  # YAML engine knob survives the overlay


def test_benchmark_print_resolved_config_never_touches_engine(capsys, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("engine must not run under --print-resolved-config")

    monkeypatch.setattr(orch, "benchmark_from_mapping", _boom)
    rc = cli.main(["benchmark", "--tokenizer-name", "tokA", "--print-resolved-config"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "tokenizers:\n- tokA" in out


def test_benchmark_dry_run_drives_engine_and_ignores_single_tokenizer_ckpt(tmp_path, monkeypatch):
    """Benchmark's cohort overlay: --tokenizer-name -> a one-element `tokenizers`
    filter; --tokenizer-ckpt is intentionally IGNORED (a single ckpt can't
    parameterize a discovered cohort). Locks that documented behavior."""
    captured = []
    monkeypatch.setattr(orch, "_run_train", lambda argv, **k: captured.append(argv) or 0)
    monkeypatch.setattr(orch, "_mlflow_create_parent", lambda *a, **k: ("", ""))
    monkeypatch.setattr(orch, "_resolve_gpus", lambda *a, **k: ["0"])
    rc = cli.main(
        [
            "benchmark",
            "--tokenizer-name",
            "tokA",
            "--tokenizer-ckpt",
            "/ignored/x.ckpt",
            "--data-root",
            str(tmp_path),
            "--dry-run",
        ]
    )
    assert rc == 0
    assert captured, "engine should have resolved at least one benchmark cell"
    # --tokenizer-name -> the cohort filter: cells reference the tokA ckpt path.
    assert any("tokA" in tok for cell in captured for tok in cell)
    assert any("tokenizer=frozen" in tok for cell in captured for tok in cell)
    # --tokenizer-ckpt is ignored for the cohort model — it must NOT leak into any cell.
    assert not any("/ignored/x.ckpt" in tok for cell in captured for tok in cell)


def test_calibrate_mlflow_uri_ckpt_passes_through_verbatim(tmp_path, monkeypatch):
    """A mlflow-artifacts:/ ckpt URI reaches the per-cell train.py overrides
    UNCHANGED (per-cell FrozenNTTTokenizer resolves it); a local-file existence
    check must not apply to URIs."""
    captured = []
    monkeypatch.setattr(orch, "_run_train", lambda argv, **k: captured.append(argv) or 0)
    monkeypatch.setattr(orch, "_mlflow_create_parent", lambda *a, **k: ("", ""))
    monkeypatch.setattr(orch, "_resolve_gpus", lambda *a, **k: ["0"])
    uri = "mlflow-artifacts:/exp/run/artifacts/model.ckpt"
    rc = cli.main(
        [
            "calibrate",
            "--tokenizer-ckpt",
            uri,
            "--tokenizer-kind",
            "frozen",
            "--tokenizer-name",
            "tokA",
            "--data-root",
            str(tmp_path),
            "--output-root",
            str(tmp_path),
            "--dry-run",
        ]
    )
    assert rc == 0
    assert any(f"tokenizer.ntt_checkpoint={uri}" in tok for cell in captured for tok in cell)


def test_resolve_calibrate_uri_skips_is_file_check(monkeypatch):
    """Non-dry-run: a URI ckpt must NOT be is_file()-rejected; the name comes
    from --tokenizer-name (can't be derived from a URI path)."""
    monkeypatch.setattr(orch, "_resolve_gpus", lambda *a, **k: ["0"])
    cfg = orch.calibrate_config_from_mapping(
        {
            "tokenizer_kind": "frozen",
            "ckpt": "mlflow-artifacts:/e/r/model.ckpt",
            "tokenizer_name": "tokA",
            "no_crossattn": "true",
            "dry_run": False,
        }
    )
    assert cfg.tokenizer_name == "tokA"
    assert cfg.ckpt == "mlflow-artifacts:/e/r/model.ckpt"


def test_resolve_calibrate_uri_requires_tokenizer_name(monkeypatch):
    monkeypatch.setattr(orch, "_resolve_gpus", lambda *a, **k: ["0"])
    with pytest.raises(ValueError, match="tokenizer-name"):
        orch.calibrate_config_from_mapping(
            {
                "tokenizer_kind": "frozen",
                "ckpt": "mlflow-artifacts:/e/r/model.ckpt",
                "no_crossattn": "true",
                "dry_run": False,
            }
        )


def test_resolve_calibrate_local_missing_ckpt_still_raises(monkeypatch):
    """The local-path happy/failure path is unchanged: a missing local ckpt
    still fails loud on a non-dry-run resolve."""
    monkeypatch.setattr(orch, "_resolve_gpus", lambda *a, **k: ["0"])
    with pytest.raises(FileNotFoundError):
        orch.calibrate_config_from_mapping(
            {
                "tokenizer_kind": "frozen",
                "ckpt": "/definitely/not/here/model.ckpt",
                "tokenizer_name": "x",
                "no_crossattn": "true",
                "dry_run": False,
            }
        )


def test_overlay_common_defaults_repo_root_to_project_root():
    """The installed CLI is cwd-independent: _overlay_common anchors repo_root to
    the discovered project root when the sweep YAML leaves it null."""
    import argparse

    from ntt_downstream.cli import sweeps

    args = argparse.Namespace(data_root=None, hydra_overrides=None, dry_run=True)
    mapping: dict = {}
    sweeps._overlay_common(mapping, args)
    assert mapping["repo_root"] == str(sweeps._ROOT)


def test_overlay_common_preserves_explicit_repo_root():
    import argparse

    from ntt_downstream.cli import sweeps

    args = argparse.Namespace(data_root=None, hydra_overrides=None, dry_run=True)
    mapping = {"repo_root": "/custom/root"}
    sweeps._overlay_common(mapping, args)
    assert mapping["repo_root"] == "/custom/root"


def test_benchmark_no_tokenizers_exits_2_cleanly(tmp_path, capsys):
    """A benchmark config with an empty ckpt cohort + no spike-times baseline makes
    the engine raise a misconfig ValueError; the CLI surfaces it as `error: …`/exit
    2, not a raw traceback. (ckpt_root points at an empty tmp dir so the test does
    not depend on checkpoints present in the default ckpt_root.)"""
    (tmp_path / "empty_ckpts").mkdir()
    cfg = tmp_path / "benchmark.yaml"
    cfg.write_text(f"ckpt_root: {tmp_path / 'empty_ckpts'}\n")
    rc = cli.main(["benchmark", "--config-dir", str(tmp_path), "--config-name", "benchmark", "--dry-run"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "error:" in err and "tokenizer" in err.lower()
