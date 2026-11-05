import subprocess

from ntt_downstream.cli import app as cli


def test_decode_dry_run_builds_hydra_command(capsys, monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    rc = cli.main(["decode", "--tokenizer-kind", "fake", "--data-root", "/d", "--output-root", "/o", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "ntt_downstream.decoding.train" in out
    assert "--config-dir" in out and "--config-name" in out and "decode" in out
    assert "tokenizer=fake" in out and "data_root=/d" in out and "log_dir=/o" in out
    assert called["n"] == 0  # dry-run never subprocesses


def test_passthrough_after_promoted(capsys, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    cli.main(["decode", "--tokenizer-kind", "fake", "--data-root", "/d", "--dry-run", "batch_size=8", "seed=99"])
    out = capsys.readouterr().out
    # passthrough appears after the promoted-flag overrides
    assert out.index("batch_size=8") > out.index("tokenizer=fake")
    assert "seed=99" in out


def test_external_config_name_and_dir(capsys, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    cli.main(["encode", "--config-dir", "/ext/cfgs", "--config-name", "myexp", "--data-root", "/d", "--dry-run"])
    out = capsys.readouterr().out
    assert "--config-dir /ext/cfgs" in out.replace("  ", " ") or "/ext/cfgs" in out
    assert "myexp" in out


def test_missing_required_promoted_for_decode_ok_without(capsys, monkeypatch):
    # decode with no promoted flags still launches with the default entrypoint config
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    rc = cli.main(["decode", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0 and "--config-name" in out and "decode" in out
