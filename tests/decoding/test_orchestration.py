"""Unit tests for the decoding orchestration engine and its pure helpers.

Covers the deterministic, side-effect-free logic (no GPUs, no MLflow server, no
real training): bool/cksum/name helpers, the (size, crossattn) -> config map,
tokenizer/mlflow override construction, the per-cell train.py argv contract, the
idempotence helpers (completeness check + plain-eval mirror), tokenizer
discovery, and the config_loader / calibration_csv modules.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import types
from pathlib import Path

import pytest

from ntt_downstream.decoding import calibration_csv, config_loader
from ntt_downstream.decoding import orchestration as orch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CFG_DIR = _REPO_ROOT / "configs" / "decoding" / "single_session"

# ── bool / cksum / name helpers ─────────────────────────────────────────────


@pytest.mark.parametrize("v", ["true", "True", "TRUE", "1", True])
def test_parse_bool_true(v):
    assert orch._parse_bool(v, "x") is True


@pytest.mark.parametrize("v", ["false", "False", "FALSE", "0", False])
def test_parse_bool_false(v):
    assert orch._parse_bool(v, "x") is False


def test_parse_bool_rejects_garbage():
    with pytest.raises(ValueError):
        orch._parse_bool("maybe", "x")


def test_short_id_identity_under_limit():
    assert orch._short_id("short-name", 120) == "short-name"


def test_short_id_truncates_with_cksum_suffix_to_exact_limit():
    raw = "x" * 200
    out = orch._short_id(raw, 120)
    assert len(out) == 120
    head, _, suffix = out.rpartition("_")
    assert head == "x" * (120 - len(suffix) - 1)
    assert suffix.isdigit()


@pytest.mark.skipif(shutil.which("cksum") is None, reason="needs POSIX cksum")
@pytest.mark.parametrize("s", [b"", b"abc", b"123456789", b"c_20151103_center_out_reaching"])
def test_posix_cksum_matches_system_cksum(s):
    expected = int(subprocess.run(["cksum"], input=s, capture_output=True).stdout.split()[0])
    assert orch._posix_cksum(s) == expected


# ── (size, crossattn) -> (MODEL_CFG, VARIANT) ───────────────────────────────


def test_model_cfg_and_variant_mapping():
    assert orch._model_cfg_and_variant("1.3M", True) == ("poyo_1.3M_no_crossattn", "1.3M_nocrossattn")
    assert orch._model_cfg_and_variant("1.3M", False) == ("poyo_1.3M", "1.3M_crossattn")
    assert orch._model_cfg_and_variant("11.8M", False) == ("poyo_11.8M", "11.8M_crossattn")


def test_model_cfg_and_variant_rejects_unsupported():
    with pytest.raises(ValueError):
        orch._model_cfg_and_variant("11.8M", True)  # no config
    with pytest.raises(ValueError):
        orch._model_cfg_and_variant("2M", True)


# ── tokenizer / mlflow / worker-tag override helpers ────────────────────────


def test_tokenizer_args_frozen_carries_ckpt_and_sentinels():
    args = orch._tokenizer_args("frozen", "/abs/m.ckpt", "add", "false")
    assert args == [
        "tokenizer=frozen",
        "tokenizer.ntt_checkpoint=/abs/m.ckpt",
        "tokenizer.unit_embedding_mode=add",
        "tokenizer.sentinels=false",
    ]


def test_tokenizer_args_spike_times_omits_sentinels_by_default():
    # Benchmark policy: spike_times inherits the tokenizer config's sentinels
    # default (the benchmark deliberately omits the override).
    args = orch._tokenizer_args("spike_times", "", "concat", "true")
    assert args == ["tokenizer=spike_times", "tokenizer.unit_embedding_mode=concat"]


def test_tokenizer_args_spike_times_emits_sentinels_when_requested():
    # Calibrate policy: spike_times carries tokenizer.sentinels, so the
    # resolved sentinels setting is honored rather than silently dropped in favor of the tokenizer config default.
    args = orch._tokenizer_args("spike_times", "", "concat", "false", spike_times_sentinels=True)
    assert args == [
        "tokenizer=spike_times",
        "tokenizer.unit_embedding_mode=concat",
        "tokenizer.sentinels=false",
    ]


def test_mlflow_parent_override():
    assert orch._mlflow_parent_override("") == "mlflow.parent_run_id=null"
    assert orch._mlflow_parent_override("abc123") == "mlflow.parent_run_id=abc123"


def test_worker_tags():
    assert orch._worker_tags(["0", "1"], 2) == ["0:0", "0:1", "1:0", "1:1"]


def test_cell_env_pins_pythonhashseed_and_cublas(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    env = orch._cell_env("3", 43, ":4096:8")
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["PYTHONHASHSEED"] == "43"  # pinned to the cell seed (authoritative)
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"  # from the YAML default


def test_cell_env_explicit_cublas_env_wins_but_seed_is_authoritative(monkeypatch):
    # CUBLAS honors a run-time env override (an inherited env value beats the YAML default)...
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    env = orch._cell_env("0", 7, ":4096:8")
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    # ...but PYTHONHASHSEED is always re-pinned to the cell seed.
    assert env["PYTHONHASHSEED"] == "7"


def test_as_list():
    assert orch._as_list(None) == []
    assert orch._as_list(["a", 1]) == ["a", "1"]
    assert orch._as_list("a b c") == ["a", "b", "c"]


# ── per-cell train.py argv contract ─────────────────────────────────────────


def _fake_cfg(**over):
    base = {
        "tokenizer_name": "tok",
        "tokenizer_kind": "frozen",
        "ckpt": "/abs/m.ckpt",
        "data_root": "/data",
        "model_cfg": "poyo_1.3M_no_crossattn",
        "variant": "1.3M_nocrossattn",
        "no_crossattn": True,
        "num_latents_per_step": "16",
        "ue_mode": "add",
        "sentinels": "false",
        "temporal_split_enabled": "false",
        "base_max_steps": "3400",
        "ft_max_steps": "800",
        "eval_every_n_steps": "200",
        "ft_eval_every": "100",
        "ckpt_every_n_steps": "200",
        "ft_ckpt_every": "100",
        "batch_size": "128",
        "num_workers": "8",
        "precision": "bf16-mixed",
        "force_flash_attn": True,
        "mlflow_enabled": False,
        "mlflow_experiment_name": "exp",
        "mlflow_log_model": "false",
        "multi_session": False,
        "hydra_data_config": "calibration_multi_base",
        "train_sessions": ["s_a", "s_b"],
        "extra": [],
    }
    base.update(over)
    return types.SimpleNamespace(**base)


_SWEEP = {"timestamp": "TS", "group_name": "GN", "group": "G"}


def test_common_wandb_tags_contains_expected_facets():
    tags = orch._common_wandb_tags(_fake_cfg(), "s_train", 42).split(",")
    assert "tok-tok" in tags
    assert "kind-frozen" in tags
    assert "arch-poyo_1.3M_no_crossattn" in tags
    assert "size-1.3M" in tags
    assert "crossattn-false" in tags
    assert "sess-s_train" in tags
    assert "seed-42" in tags
    assert "ckpt-m" in tags  # basename of /abs/m.ckpt sans .ckpt


def test_base_argv_has_no_finetune_keys():
    argv = orch._base_argv(_fake_cfg(), "s_train", 42, "/out", "", _SWEEP)
    joined = " ".join(argv)
    assert "max_steps=3400" in argv
    assert "+mlflow.tags.stage=base" in argv
    assert "mlflow.parent_run_id=null" in argv
    assert "init_ckpt" not in joined and "freeze_mode" not in joined and "eval_only" not in joined


def test_ft_argv_n_trials_zero_is_eval_only():
    argv = orch._ft_argv(
        _fake_cfg(), "s_train", "s_eval", "unit_identification", 0, 42, "/d", "/base.ckpt", "PID", _SWEEP
    )
    assert "eval_only=true" in argv
    assert "max_steps=1" in argv
    assert "freeze_mode=unit_identification" in argv
    assert "train_decimation.trials=0" in argv
    assert "init_ckpt=/base.ckpt" in argv
    assert "mlflow.parent_run_id=PID" in argv


def test_ft_argv_full_mode_maps_to_freeze_none_and_runs():
    argv = orch._ft_argv(_fake_cfg(), "s_train", "s_eval", "full", 8, 42, "/d", "/base.ckpt", "", _SWEEP)
    assert "freeze_mode=none" in argv
    assert "eval_only=false" in argv
    assert "max_steps=800" in argv
    assert "train_decimation.trials=8" in argv


def test_crossattn_cfg_adds_num_latents_per_step():
    cfg = _fake_cfg(no_crossattn=False, model_cfg="poyo_1.3M", variant="1.3M_crossattn")
    assert "model.num_latents_per_step=16" in orch._base_argv(cfg, "s", 42, "/o", "", _SWEEP)
    assert "model.num_latents_per_step=16" not in orch._base_argv(_fake_cfg(), "s", 42, "/o", "", _SWEEP)


def test_calibrate_spike_times_argv_carries_sentinels():
    # End-to-end calibrate guard: a spike_times base/FT cell must pass
    # tokenizer.sentinels (anchored to the calibration config)
    # rather than dropping it and inheriting the config default.
    cfg = _fake_cfg(tokenizer_kind="spike_times", ckpt="", sentinels="true")
    assert "tokenizer.sentinels=true" in orch._base_argv(cfg, "s", 42, "/o", "", _SWEEP)
    ft = orch._ft_argv(cfg, "s", "o", "full", 8, 42, "/d", "/b.ckpt", "", _SWEEP)
    assert "tokenizer.sentinels=true" in ft


def test_multisession_base_argv_wires_pool_not_single_session():
    cfg = _fake_cfg(multi_session=True, tokenizer_kind="spike_times", ckpt="", train_sessions=["s_a", "s_b", "s_c"])
    base = orch._base_argv(cfg, None, 42, "/out", "PID", _SWEEP)
    # Pool wired via the Hydra preset; the single-session `session=` is gone.
    assert "+data=calibration_multi_base" in base
    assert "_train_sessions_list=[s_a,s_b,s_c]" in base
    assert not any(a.startswith("session=") for a in base)
    # Pool tags replace +mlflow.tags.train_session.
    assert "+mlflow.tags.pool_size=3" in base
    assert any(a.startswith("+mlflow.tags.pool_hash=") for a in base)
    assert not any(a.startswith("+mlflow.tags.train_session=") for a in base)
    # wandb common carries the pool facets, not a single sess-/trainsess- facet.
    tags = next(a for a in base if a.startswith("wandb.tags="))
    assert "pool-3sess" in tags and "trainsess-" not in tags


def test_multisession_ft_argv_drops_train_session_facets():
    cfg = _fake_cfg(multi_session=True, tokenizer_kind="spike_times", ckpt="", train_sessions=["s_a", "s_b"])
    ft = orch._ft_argv(cfg, None, "s_eval", "full", 8, 42, "/d", "/b.ckpt", "PID", _SWEEP)
    assert "session=s_eval" in ft  # FT still fine-tunes on the held-out session
    assert "+mlflow.tags.calib_session=s_eval" in ft
    assert not any(a.startswith("+mlflow.tags.train_session=") for a in ft)
    tags = next(a for a in ft if a.startswith("wandb.tags="))
    assert "trainsess-" not in tags and "calibsess-s_eval" in tags


def test_pool_hash_is_order_invariant_and_dedups():
    # The pool hash is order- and duplicate-insensitive.
    assert orch._pool_hash(["s_b", "s_a"]) == orch._pool_hash(["s_a", "s_b", "s_a"])


def test_multisession_requires_two_train_sessions(monkeypatch):
    monkeypatch.setattr(orch, "_preflight_trials", lambda c: {})
    monkeypatch.setattr(orch, "_mlflow_create_parent", lambda *a, **k: ("", ""))
    with pytest.raises(ValueError, match="2"):
        orch.main([
            "calibrate", "--multi-session", "--repo-root", str(_REPO_ROOT),
            "--tokenizer-kind", "spike_times", "--train-sessions", "only_one",
            "--dry-run", "--gpus", "0",
        ])  # fmt: skip


# ── idempotence helpers ─────────────────────────────────────────────────────


def _write_run_result(run_dir, version, payload):
    v = run_dir / "csv_logs" / f"version_{version}"
    v.mkdir(parents=True, exist_ok=True)
    (v / "run_result.json").write_text(json.dumps(payload, indent=2))
    return v


def test_latest_run_result_complete(tmp_path):
    rd = tmp_path / "cell"
    assert orch._latest_run_result_complete(rd) is False  # nothing yet
    _write_run_result(rd, 0, {"val_last_r2": 0.5, "test_r2": 0.4})
    assert orch._latest_run_result_complete(rd) is True


def test_latest_run_result_incomplete_is_false(tmp_path):
    rd = tmp_path / "cell"
    v = rd / "csv_logs" / "version_0"
    v.mkdir(parents=True)
    (v / "run_result.json").write_text('{"other": 1}')  # missing val_last_r2
    assert orch._latest_run_result_complete(rd) is False


def test_mirror_plain_eval_copies_result(tmp_path):
    src = tmp_path / "src"
    _write_run_result(src, 0, {"val_last_r2": 0.9, "test_r2": 0.8})
    dst = tmp_path / "dst"
    assert orch._mirror_plain_eval(src, dst) is True
    mirrored = json.loads((dst / "csv_logs" / "version_0" / "run_result.json").read_text())
    assert mirrored["val_last_r2"] == 0.9
    assert (dst / "run.log").is_file()


def test_mirror_plain_eval_missing_source_returns_false(tmp_path):
    assert orch._mirror_plain_eval(tmp_path / "nope", tmp_path / "dst") is False


def test_discover_tokenizers(tmp_path):
    root = tmp_path / "ckpts"
    for name in ("vae_a", "bsq_b"):
        ck = root / name / "checkpoints"
        ck.mkdir(parents=True)
        (ck / "model.ckpt").write_text("x")
    (root / "no_ckpt").mkdir()  # ignored: no model.ckpt
    found = orch._discover_tokenizers(root, [], 0)
    assert [n for n, _ in found] == ["bsq_b", "vae_a"]  # sorted
    assert orch._discover_tokenizers(root, [], 1) == found[:1]


def test_discover_tokenizers_missing_raises_but_dry_run_warns(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError):
        orch._discover_tokenizers(missing, ["x"], 0)
    # dry-run tolerates a missing ckpt (still lists it for the plan)
    assert orch._discover_tokenizers(missing, ["x"], 0, dry_run=True) == [
        ("x", str(missing / "x" / "checkpoints" / "model.ckpt"))
    ]


# ── config_loader / calibration_csv ──────────────────────────────────────────────────────


def test_config_loader_extends_and_deep_merge(tmp_path):
    (tmp_path / "base.yaml").write_text("a: 1\nnest: {x: 1, y: 2}\nlst: [1, 2]\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\na: 99\nnest: {y: 5}\nlst: [9]\n")
    cfg = config_loader.load_yaml(tmp_path / "child.yaml")
    assert cfg["a"] == 99  # override wins
    assert cfg["nest"] == {"x": 1, "y": 5}  # deep-merged
    assert cfg["lst"] == [9]  # lists replaced wholesale
    assert config_loader.walk(cfg, "nest.x") == 1


def test_calibration_csv_build(tmp_path):
    root = tmp_path / "base_run"
    _write_run_result(
        root / "unit_identification" / "s_eval" / "ft_trials_000",
        0,
        {"val_best_r2": 0.81, "test_r2": 0.79, "kept_trials": 0},
    )
    _write_run_result(
        root / "unit_identification" / "s_eval" / "ft_trials_008",
        0,
        {"val_best_r2": 0.91, "test_r2": 0.88, "kept_trials": 8},
    )
    calibration_csv.build_csv(
        experiments_root=root, ft_mode="unit_identification", ft_trials=[0, 8], other_sessions=["s_eval"]
    )
    rows = (root / "calibration_unit_identification.csv").read_text().splitlines()
    # Full ordered schema: three metric blocks in _METRIC_FIELDS order, each long
    # across the axis. kept_trials renders as an integer (not %.6f) and lands last.
    assert rows == [
        "metric,ft_trials,s_eval",
        "val_best_r2,0,0.810000",
        "val_best_r2,8,0.910000",
        "test_r2,0,0.790000",
        "test_r2,8,0.880000",
        "kept_trials,0,0",
        "kept_trials,8,8",
    ]


# ── strict _yaml_defaults ─────────────────────────────────────────


def test_yaml_defaults_strict_raises_on_missing_key(tmp_path):
    (tmp_path / "c.yaml").write_text("a:\n  b: 1\n")
    with pytest.raises(ValueError, match=r"a\.z"):
        orch._yaml_defaults(tmp_path / "c.yaml", {"a.b": "AB", "a.z": "AZ"})


def test_yaml_defaults_collects_all_missing(tmp_path):
    (tmp_path / "c.yaml").write_text("a: 1\n")
    with pytest.raises(ValueError) as ei:
        orch._yaml_defaults(tmp_path / "c.yaml", {"x": "X", "y": "Y"})
    msg = str(ei.value)
    assert "x" in msg and "y" in msg  # both reported in one error, not fail-on-first


def test_yaml_defaults_happy_path_on_real_configs():
    # The shipped calibrate configs must keep every whitelisted
    # key, else strict mode would (correctly) abort a default run.
    tcfg = orch._yaml_defaults(_CFG_DIR / "tokenizer_training.yaml", config_loader.TRAINING_KEYS)
    assert tcfg["MODEL_SIZE"] == "1.3M"
    assert tcfg["TOKENIZER_KIND"] == "frozen"
    dcfg = orch._yaml_defaults(_CFG_DIR / "data.yaml", config_loader.DATA_KEYS)
    assert dcfg["DATA_ROOT"]  # present and non-empty
    bcfg = orch._yaml_defaults(_CFG_DIR / "baseline_training.yaml", config_loader.TRAINING_KEYS)
    assert bcfg["TOKENIZER_KIND"] == "spike_times"


# ── trial-eligibility clamp ───────────────────────────────────────


@pytest.mark.parametrize(
    "n_valid,expected",
    [
        (123, [0, 4, 8, 16, 32, 64, 86]),  # int(0.7*123)=86 -> 86 kept (boundary)
        (120, [0, 4, 8, 16, 32, 64]),  # int(0.7*120)=84 -> 86 dropped
        (10, [0, 4]),  # int(0.7*10)=7
        (0, [0]),  # no data -> only the plain-eval cell survives
    ],
)
def test_eligible_trials_clamp(n_valid, expected):
    assert orch._eligible_trials(n_valid, [4, 8, 16, 32, 64, 86]) == expected


# ── exact per-cell train.py override contract ─────────────────────


def _arg_keys(argv):
    return [a.split("=", 1)[0] for a in argv]


_BASE_KEYS = [
    "model", "session", "data_root", "log_dir",
    "tokenizer", "tokenizer.ntt_checkpoint", "tokenizer.unit_embedding_mode", "tokenizer.sentinels",
    "temporal_split.enabled", "max_steps", "eval_every_n_steps", "checkpoint_every_n_steps",
    "batch_size", "seed", "num_workers", "precision", "force_flash_attn",
    "wandb.name", "wandb.group", "wandb.tags",
    "mlflow.enabled", "mlflow.experiment_name", "mlflow.run_name", "mlflow.log_model", "mlflow.parent_run_id",
    "+mlflow.tags.sweep_id", "+mlflow.tags.sweep_timestamp", "+mlflow.tags.tokenizer_name",
    "+mlflow.tags.tokenizer_kind", "+mlflow.tags.stage", "+mlflow.tags.train_session", "+mlflow.tags.seed",
]  # fmt: skip

_FT_KEYS = [
    "model", "session", "data_root", "log_dir",
    "tokenizer", "tokenizer.ntt_checkpoint", "tokenizer.unit_embedding_mode", "tokenizer.sentinels",
    "temporal_split.enabled", "max_steps", "eval_every_n_steps", "checkpoint_every_n_steps",
    "batch_size", "seed", "num_workers", "precision", "force_flash_attn",
    "wandb.name", "wandb.group", "wandb.tags",
    "mlflow.enabled", "mlflow.experiment_name", "mlflow.run_name", "mlflow.log_model", "mlflow.parent_run_id",
    "+mlflow.tags.sweep_id", "+mlflow.tags.sweep_timestamp", "+mlflow.tags.tokenizer_name",
    "+mlflow.tags.tokenizer_kind", "+mlflow.tags.stage", "+mlflow.tags.ft_mode", "+mlflow.tags.ft_trials",
    "+mlflow.tags.train_session", "+mlflow.tags.calib_session", "+mlflow.tags.seed",
    "train_decimation.trials", "train_decimation.seed", "init_ckpt", "freeze_mode", "eval_only",
]  # fmt: skip


def test_base_argv_exact_override_keys():
    # Exact key list catches a missing/extra/renamed base override.
    assert _arg_keys(orch._base_argv(_fake_cfg(), "s_train", 42, "/out", "", _SWEEP)) == _BASE_KEYS
    # The crossattn variant inserts model.num_latents_per_step right after the
    # four tokenizer args (indices 4-7), i.e. at index 8.
    cfg = _fake_cfg(no_crossattn=False, model_cfg="poyo_1.3M", variant="1.3M_crossattn")
    keys = _arg_keys(orch._base_argv(cfg, "s_train", 42, "/out", "", _SWEEP))
    assert keys[8] == "model.num_latents_per_step"
    assert keys.count("model.num_latents_per_step") == 1


def test_ft_argv_exact_override_keys_and_eval_only_clamp():
    assert _arg_keys(orch._ft_argv(_fake_cfg(), "s", "o", "full", 8, 42, "/d", "/b.ckpt", "PID", _SWEEP)) == _FT_KEYS
    # n_trials=0 -> eval-only: the schedule knobs are clamped to 1 regardless of cfg.
    ev = orch._ft_argv(_fake_cfg(), "s", "o", "unit_identification", 0, 42, "/d", "/b.ckpt", "", _SWEEP)
    assert _arg_keys(ev) == _FT_KEYS
    assert "max_steps=1" in ev and "eval_every_n_steps=1" in ev and "checkpoint_every_n_steps=1" in ev
    assert "eval_only=true" in ev


# ── worker pool ───────────────────────────────────────────────────


def test_run_pool_runs_all_jobs_on_workers():
    seen = []
    lock = threading.Lock()

    def run_job(job, gpu):
        with lock:
            seen.append((job, gpu))
        return 0

    rc = orch._run_pool(list(range(20)), orch._worker_tags(["0", "1"], 2), run_job)
    assert rc == 0
    assert sorted(j for j, _ in seen) == list(range(20))  # every job pulled exactly once
    assert {g for _, g in seen} <= {"0", "1"}


def test_run_pool_flags_nonzero_return():
    assert orch._run_pool([1, 2, 3], ["0:0"], lambda job, gpu: 1 if job == 2 else 0) == 1


def test_run_pool_flags_raised_exception():
    def run_job(job, gpu):
        if job == 2:
            raise RuntimeError("boom")
        return 0

    assert orch._run_pool([1, 2, 3], ["0:0", "0:1"], run_job) == 1


def test_run_pool_raises_when_jobs_but_no_workers():
    """Degenerate pool (empty worker_tags from jobs_per_gpu<1 or no gpus) must
    fail loud instead of silently draining nothing and reporting success."""
    with pytest.raises(ValueError, match="no worker tags"):
        orch._run_pool([1, 2], [], lambda job, gpu: 0)


def test_run_pool_empty_jobs_is_valid_noop():
    assert orch._run_pool([], [], lambda job, gpu: 0) == 0


@pytest.mark.parametrize("bad", [0, -1, "abc", 1.9, True])
def test_resolve_calibrate_rejects_bad_jobs_per_gpu(bad, monkeypatch):
    """Reject non-positive, non-numeric, non-integer-float, and bool values — the
    Hydra seam bypasses argparse's type=int so a YAML 1.9/true could slip in."""
    monkeypatch.setattr(orch, "_resolve_gpus", lambda *a, **k: ["0"])
    with pytest.raises(ValueError, match="jobs_per_gpu"):
        orch.calibrate_config_from_mapping({"tokenizer_kind": "spike_times", "jobs_per_gpu": bad, "dry_run": True})


@pytest.mark.parametrize("good,expected", [(1, 1), (3, 3), ("4", 4), (2.0, 2)])
def test_coerce_positive_int_accepts_valid(good, expected):
    assert orch._coerce_positive_int(good, "jobs_per_gpu") == expected


def test_resolve_gpus_whitespace_flag_falls_back(monkeypatch):
    """A whitespace-only --gpus must not yield an empty list (which would later
    index gpus[0] out of range); it falls through to auto-detect/default."""
    monkeypatch.setattr(orch.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError))
    assert orch._resolve_gpus("   ") == ["0"]


def test_read_metrics_non_dict_json_is_treated_as_corrupt(tmp_path):
    """Valid JSON that isn't an object (bare list/number) must not AttributeError
    out of _read_metrics — it degrades to defaults like a truncated file."""
    rd = tmp_path / "cell"
    v0 = rd / "csv_logs" / "version_0"
    v0.mkdir(parents=True)
    (v0 / "run_result.json").write_text("[1, 2, 3]")
    m = calibration_csv._read_metrics(rd)
    assert all(v is None for v in m.values())


def test_cell_env_empty_cublas_env_falls_back_to_default(monkeypatch):
    """An inherited but EMPTY CUBLAS_WORKSPACE_CONFIG must fall back to the cell
    default (the same as an unset variable), not be preserved as "" (which
    setdefault did, silently disabling determinism)."""
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "")
    env = orch._cell_env("0", 7, ":4096:8")
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_read_metrics_falls_back_to_older_version_on_corrupt_json(tmp_path):
    """A truncated newest run_result.json falls back to the next-older version
    rather than aborting the whole base run's CSV."""
    rd = tmp_path / "cell"
    _write_run_result(rd, 0, {"val_best_r2": 0.7, "test_r2": 0.6, "kept_trials": 4})
    v1 = rd / "csv_logs" / "version_1"
    v1.mkdir(parents=True)
    (v1 / "run_result.json").write_text('{"val_best_r2": 0.9')  # truncated
    m = calibration_csv._read_metrics(rd)
    assert m["val_best_r2"] == 0.7  # fell back to the valid v0


def test_read_metrics_all_corrupt_returns_defaults(tmp_path):
    rd = tmp_path / "cell"
    v0 = rd / "csv_logs" / "version_0"
    v0.mkdir(parents=True)
    (v0 / "run_result.json").write_text('{"val_best_r2"')  # truncated
    m = calibration_csv._read_metrics(rd)
    assert all(v is None for v in m.values())


# ── base-ckpt selection min-steps guard ───────────────────────────


def test_latest_base_ckpt_min_steps_guard(tmp_path):
    torch = pytest.importorskip("torch")
    base = tmp_path / "base"

    def write_ckpt(version, step):
        d = base / "csv_logs" / f"version_{version}" / "checkpoints"
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"global_step": step}, d / "last.ckpt")
        return d / "last.ckpt"

    v0 = write_ckpt(0, 5000)  # older, complete
    write_ckpt(2, 1000)  # newest, partial (< min)
    # Newest version is partial -> falls back to the older complete one.
    assert orch._latest_base_ckpt(base, 3400) == str(v0)
    # When the newest meets the bar, it wins.
    v3 = write_ckpt(3, 9000)
    assert orch._latest_base_ckpt(base, 3400) == str(v3)
    # None qualify.
    assert orch._latest_base_ckpt(base, 100000) is None


# ── MLflow parent create stdout parsing ───────────────────────────


def test_mlflow_create_parent_parses_run_id_and_uri(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return types.SimpleNamespace(stdout="RUN_ID=abc\nRUN_URI=http://x/#/experiments/1/runs/abc\n")

    monkeypatch.setattr(orch.subprocess, "run", fake_run)
    rid, uri = orch._mlflow_create_parent(
        tmp_path, experiment_name="e", run_name="r", tags={"k": "v"}, disabled=False, dry_run=False
    )
    assert (rid, uri) == ("abc", "http://x/#/experiments/1/runs/abc")
    assert "--tag" in captured["cmd"] and "k=v" in captured["cmd"]


def test_mlflow_create_parent_dry_run_skips_subprocess(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("subprocess.run must not run on dry-run")

    monkeypatch.setattr(orch.subprocess, "run", boom)
    assert orch._mlflow_create_parent(
        tmp_path, experiment_name="e", run_name="r", tags={}, disabled=False, dry_run=True
    ) == ("", "")


def test_mlflow_create_parent_disabled_appends_flag(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return types.SimpleNamespace(stdout="RUN_ID=\nRUN_URI=\n")

    monkeypatch.setattr(orch.subprocess, "run", fake_run)
    rid, uri = orch._mlflow_create_parent(
        tmp_path, experiment_name="e", run_name="r", tags={}, disabled=True, dry_run=False
    )
    assert "--disabled" in captured["cmd"]
    assert (rid, uri) == ("", "")


# ── run_benchmark defaults ────────────────────────────────────────


def test_run_benchmark_default_overrides(monkeypatch):
    captured = []

    def fake_run_train(argv, **_):
        captured.append(argv)
        return 0

    monkeypatch.setattr(orch, "_run_train", fake_run_train)
    monkeypatch.setattr(orch, "_mlflow_create_parent", lambda *a, **k: ("", ""))
    monkeypatch.setattr(orch, "_resolve_gpus", lambda flag: ["0"])
    rc = orch.main([
        "benchmark", "--repo-root", str(_REPO_ROOT), "--tokenizers", "x",
        "--dry-run", "--data-root", "/data", "--sessions", "S",
    ])  # fmt: skip
    assert rc == 0
    assert captured, "no benchmark cell was emitted"
    argv = captured[0]
    for expected in (
        "epochs=400", "batch_size=128", "seed=42",
        "num_workers=8", "precision=bf16-mixed", "force_flash_attn=true",
    ):  # fmt: skip
        assert expected in argv


def test_run_benchmark_spike_times_only_baseline(monkeypatch):
    # Baseline mode: spike_times only,
    # no frozen discovery (so no CKPT_ROOT needed), crossattn poyo_1.3M default.
    cells = []
    parents = []

    def fake_run_train(argv, **_):
        cells.append(argv)
        return 0

    def fake_create_parent(repo_root, *, experiment_name, run_name, tags, disabled, dry_run):
        parents.append(tags)
        return ("", "")

    monkeypatch.setattr(orch, "_run_train", fake_run_train)
    monkeypatch.setattr(orch, "_mlflow_create_parent", fake_create_parent)
    monkeypatch.setattr(orch, "_resolve_gpus", lambda flag: ["0"])
    rc = orch.main([
        "benchmark", "--repo-root", str(_REPO_ROOT), "--spike-times-only", "true",
        "--dry-run", "--data-root", "/data", "--sessions", "S1 S2",
    ])  # fmt: skip
    assert rc == 0
    assert len(cells) == 2  # spike_times only, one cell per session — no frozen tokenizers
    for argv in cells:
        assert "tokenizer=spike_times" in argv
        assert "model=poyo_1.3M" in argv  # crossattn baseline default, not poyo_1.3M_no_crossattn
        assert not any(a.startswith("tokenizer=frozen") for a in argv)
    # A single spike_times parent, tagged as the baseline orchestrator.
    assert len(parents) == 1
    assert parents[0]["orchestrator"] == "baseline_benchmark"
    assert parents[0]["tokenizer_kind"] == "spike_times"


# ── run_calibrate: missing-only skip + plain-eval mirror interaction ──


def _calib_cfg(tmp_path, **over):
    base = {
        "repo_root": tmp_path,
        "data_root": "/data",
        "tokenizer_kind": "frozen",
        "ckpt": "/abs/m.ckpt",
        "tokenizer_name": "tok",
        "model_cfg": "poyo_1.3M_no_crossattn",
        "variant": "1.3M_nocrossattn",
        "no_crossattn": True,
        "num_latents_per_step": "16",
        "ue_mode": "add",
        "sentinels": "false",
        "train_sessions": ["A"],
        "other_sessions": ["A", "B"],
        "ft_trial_counts": [8],
        "ft_modes": ["unit_identification", "full"],
        "seeds": [42],
        "base_max_steps": "3400",
        "ft_max_steps": "800",
        "eval_every_n_steps": "200",
        "ft_eval_every": "100",
        "ckpt_every_n_steps": "200",
        "ft_ckpt_every": "100",
        "batch_size": "128",
        "num_workers": "8",
        "precision": "bf16-mixed",
        "cublas_workspace_config": ":4096:8",
        "force_flash_attn": True,
        "temporal_split_enabled": "false",
        "brainset": "bs",
        "calib_root": str(tmp_path / "calib"),
        "mlflow_enabled": False,
        "mlflow_experiment_name": "exp",
        "mlflow_log_model": "false",
        "gpus": ["0"],
        "jobs_per_gpu": 1,
        "reuse_base_if_exists": False,
        "missing_only": True,
        "skip_base_train": False,
        "base_ckpt": "",
        "multi_session": False,
        "hydra_data_config": "calibration_multi_base",
        "extra": [],
        "dry_run": False,
    }
    base.update(over)
    return orch.CalibrateConfig(**base)


def test_run_calibrate_missing_only_skips_complete_and_mirrors_plain_eval(tmp_path, monkeypatch):
    cfg = _calib_cfg(tmp_path)
    runs = []  # FT cells actually dispatched to _run_train
    csv_calls = []

    def fake_run_train(argv, **_):
        d = dict(a.split("=", 1) for a in argv if "=" in a)
        if "train_decimation.trials" in d:  # an FT cell (base cell has no decimation)
            runs.append((d["+mlflow.tags.ft_mode"], d["+mlflow.tags.calib_session"], int(d["train_decimation.trials"])))
        return 0

    def fake_build_csv(*, experiments_root, ft_mode, other_sessions, ft_trials):
        csv_calls.append((ft_mode, tuple(ft_trials), tuple(other_sessions)))

    monkeypatch.setattr(orch, "_preflight_trials", lambda c: {"B": 10**6})  # no clamp
    monkeypatch.setattr(orch, "_mlflow_create_parent", lambda *a, **k: ("", ""))
    monkeypatch.setattr(orch, "_mlflow_aggregate", lambda *a, **k: None)
    monkeypatch.setattr(orch, "_latest_base_ckpt", lambda *a, **k: "/fake/base.ckpt")
    monkeypatch.setattr(orch, "_run_train", fake_run_train)
    monkeypatch.setattr(orch.calibration_csv, "build_csv", fake_build_csv)

    # others = other_sessions minus the train session -> ["B"].
    base_run = "single_tok_A_1.3M_nocrossattn_seed42"
    exp_root = tmp_path / "calib" / base_run
    # Pre-seed the unit_identification cells complete so missing-only skips them.
    _write_run_result(exp_root / "unit_identification" / "B" / "ft_trials_008", 0, {"val_last_r2": 0.5})
    _write_run_result(exp_root / "unit_identification" / "B" / "ft_trials_000", 0, {"val_last_r2": 0.5})

    assert orch.run_calibrate(cfg) == 0

    # Only the genuinely-missing cell ran; the two pre-seeded cells were skipped,
    # and full/B/ft_trials_000 was mirrored from the unit_identification plain eval.
    assert runs == [("full", "B", 8)]
    mirrored = exp_root / "full" / "B" / "ft_trials_000" / "csv_logs" / "version_0" / "run_result.json"
    assert mirrored.is_file()
    # build_csv is called once per ft_mode with the FULL [0, 8] axis (clamped/missing
    # cells become NaN rows rather than vanishing).
    assert csv_calls == [
        ("unit_identification", (0, 8), ("B",)),
        ("full", (0, 8), ("B",)),
    ]


# ── skip-base-train ckpt validation ──────────────────────────


def test_ckpt_global_step(tmp_path):
    torch = pytest.importorskip("torch")
    good = tmp_path / "x.ckpt"
    torch.save({"global_step": 1234}, good)
    assert orch._ckpt_global_step(good) == 1234
    assert orch._ckpt_global_step(tmp_path / "missing.ckpt") is None  # unreadable -> None
    (tmp_path / "garbage.ckpt").write_text("not a checkpoint")
    assert orch._ckpt_global_step(tmp_path / "garbage.ckpt") is None


def test_preflight_missing_hdf5_is_strict_on_real_runs(tmp_path):
    # A typo'd session must fail loudly on a real run (not silently disable the
    # 0.7 clamp); --dry-run stays permissive so a no-data plan still resolves.
    real = _calib_cfg(tmp_path, other_sessions=["ghost"], data_root=str(tmp_path), brainset="bs", dry_run=False)
    with pytest.raises(FileNotFoundError):
        orch._preflight_trials(real)
    dry = _calib_cfg(tmp_path, other_sessions=["ghost"], data_root=str(tmp_path), brainset="bs", dry_run=True)
    assert orch._preflight_trials(dry)["ghost"] >= 1 << 30


def test_skip_base_train_rejects_missing_ckpt(tmp_path, monkeypatch):
    # --skip-base-train must refuse a base ckpt that doesn't exist;
    # the engine must not silently fine-tune from a bogus path.
    cfg = _calib_cfg(tmp_path, skip_base_train=True, base_ckpt=str(tmp_path / "nope.ckpt"))
    trained = []
    monkeypatch.setattr(orch, "_preflight_trials", lambda c: {"B": 10**6})
    monkeypatch.setattr(orch, "_mlflow_create_parent", lambda *a, **k: ("", ""))
    monkeypatch.setattr(orch, "_mlflow_aggregate", lambda *a, **k: None)
    monkeypatch.setattr(orch, "_mlflow_log_artifacts", lambda *a, **k: None)
    monkeypatch.setattr(orch, "_run_train", lambda *a, **k: trained.append(a) or 0)
    monkeypatch.setattr(orch.calibration_csv, "build_csv", lambda **k: None)
    # The worker catches the FileNotFoundError -> rc=1; nothing is trained.
    assert orch.run_calibrate(cfg) == 1
    assert trained == []


# ── log-artifacts archived onto the MLflow parent, by default ────────


def test_calibrate_logs_artifacts_to_parent(tmp_path, monkeypatch):
    cfg = _calib_cfg(tmp_path, ft_modes=["full"], ft_trial_counts=[])
    logged = []
    monkeypatch.setattr(orch, "_preflight_trials", lambda c: {"B": 10**6})
    monkeypatch.setattr(orch, "_mlflow_create_parent", lambda *a, **k: ("PID", "uri"))
    monkeypatch.setattr(orch, "_mlflow_aggregate", lambda *a, **k: None)
    monkeypatch.setattr(orch, "_latest_base_ckpt", lambda *a, **k: "/fake/base.ckpt")
    monkeypatch.setattr(orch, "_run_train", lambda *a, **k: 0)
    monkeypatch.setattr(orch.calibration_csv, "build_csv", lambda **k: None)
    monkeypatch.setattr(
        orch,
        "_mlflow_log_artifacts",
        lambda repo, run_id, srcs, *, dry_run: logged.append((run_id, [str(s) for s in srcs])),
    )
    assert orch.run_calibrate(cfg) == 0
    # Each base run's exp_root is archived onto the (non-empty) parent.
    assert logged == [("PID", [str(tmp_path / "calib" / "single_tok_A_1.3M_nocrossattn_seed42")])]


def test_mlflow_log_artifacts_noops_without_parent_or_srcs(tmp_path, monkeypatch):
    # Soft no-op contract: empty run_id or no existing src dir -> never spawns.
    def boom(*a, **k):
        raise AssertionError("subprocess must not run for a no-op log-artifacts")

    monkeypatch.setattr(orch.subprocess, "run", boom)
    orch._mlflow_log_artifacts(tmp_path, "", [tmp_path], dry_run=False)  # empty run_id
    orch._mlflow_log_artifacts(tmp_path, "PID", [tmp_path / "nope"], dry_run=False)  # no real src


def test_mirror_plain_eval_copies_v2_record(tmp_path):
    src = tmp_path / "src"
    _write_run_result(src, 3, {"val_last_r2": 0.9})
    (src / "csv_logs" / "version_3" / "downstream_decoding_results.json").write_text('{"k": 1}')
    dst = tmp_path / "dst"

    assert orch._mirror_plain_eval(src, dst) is True

    copied = dst / "csv_logs" / "version_0" / "downstream_decoding_results.json"
    assert json.loads(copied.read_text()) == {"k": 1}


def test_mlflow_log_artifacts_passes_root_files(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(orch.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    src = tmp_path / "cell"
    src.mkdir()
    cohort = tmp_path / "cohort.json"
    cohort.write_text("{}")

    orch._mlflow_log_artifacts(tmp_path, "R", [src], dry_run=False, files=[cohort, tmp_path / "missing.json"])

    (cmd,) = calls
    assert cmd[cmd.index("--file") + 1] == str(cohort)
    assert str(tmp_path / "missing.json") not in cmd


def _fake_train_writing_records(records_for: set[str]):
    """A ``_run_train`` stand-in that writes a v2 child record into the cell's log_dir.

    Cells whose session is not in ``records_for`` "crash" (no record), so the
    cohort must list them as missing.
    """

    def fake_run_train(argv, **_):
        run_dir = Path(next(a.split("=", 1)[1] for a in argv if a.startswith("log_dir=")))
        session = next(a.split("=", 1)[1] for a in argv if a.startswith("session="))
        v = run_dir / "csv_logs" / "version_0"
        v.mkdir(parents=True, exist_ok=True)
        if session in records_for:
            block = {"aggregated": {"r2": {"value": 0.5}}, "per_session": {session: {"r2": 0.5}}}
            doc = {
                "schema_version": 2,
                "task": "decoding",
                "kind": "run",
                "metrics": {"test": block, "val_best": block, "val_last": block},
                "metadata": {"timestamp": orch.cohort_results.now_iso()},
            }
            (v / "downstream_decoding_results.json").write_text(json.dumps(doc))
        return 0

    return fake_run_train


def _run_offline_benchmark(tmp_path, monkeypatch, records_for):
    # repo_root = tmp_path so outputs land in the sandbox; the benchmark reads its
    # session/config YAMLs from <repo_root>/configs, so link the real ones in.
    (tmp_path / "configs").symlink_to(_REPO_ROOT / "configs")
    monkeypatch.setattr(orch, "_run_train", _fake_train_writing_records(records_for))
    monkeypatch.setattr(orch, "_mlflow_create_parent", lambda *a, **k: ("", ""))
    monkeypatch.setattr(orch, "_resolve_gpus", lambda flag: ["0"])
    rc = orch.main([
        "benchmark", "--repo-root", str(tmp_path), "--spike-times-only", "true", "--mlflow", "false",
        "--data-root", "/data", "--sessions", "S1 S2", "--comparison-tag", "T",
    ])  # fmt: skip
    cohort_dirs = list((tmp_path / "experiments_benchmark").glob("*/*__cohort"))
    return rc, cohort_dirs


def test_benchmark_builds_cohort_without_mlflow(tmp_path, monkeypatch):
    rc, cohort_dirs = _run_offline_benchmark(tmp_path, monkeypatch, records_for={"S1", "S2"})

    assert rc == 0
    (cohort_dir,) = cohort_dirs
    doc = json.loads((cohort_dir / "downstream_decoding_results.json").read_text())
    assert doc["kind"] == "cohort"
    assert doc["metadata"]["cohort"]["complete"] is True
    assert doc["metadata"]["cohort"]["found"] == ["S1", "S2"]


def test_benchmark_cohort_reports_crashed_cell(tmp_path, monkeypatch):
    _rc, (cohort_dir,) = _run_offline_benchmark(tmp_path, monkeypatch, records_for={"S1"})

    cohort = json.loads((cohort_dir / "downstream_decoding_results.json").read_text())["metadata"]["cohort"]
    assert cohort["complete"] is False and cohort["missing"] == ["S2"]


def test_benchmark_rerun_crashing_early_does_not_reuse_stale_records(tmp_path, monkeypatch):
    """Same-tag rerun where every cell dies before Lightning makes a new version
    dir: the previous attempt's records are still 'newest' but predate this
    sweep, so the cohort must be incomplete, not a stale 'complete'."""
    import time

    real_now_iso = orch.cohort_results.now_iso
    _rc, (cohort_dir,) = _run_offline_benchmark(tmp_path, monkeypatch, records_for={"S1", "S2"})
    first = json.loads((cohort_dir / "downstream_decoding_results.json").read_text())["metadata"]["cohort"]
    assert first["complete"] is True

    time.sleep(1.1)  # timestamps have 1 s resolution
    monkeypatch.setattr(orch, "_run_train", lambda argv, **_: 1)  # crash before any version dir
    orch.main([
        "benchmark", "--repo-root", str(tmp_path), "--spike-times-only", "true", "--mlflow", "false",
        "--data-root", "/data", "--sessions", "S1 S2", "--comparison-tag", "T",
    ])  # fmt: skip

    second = json.loads((cohort_dir / "downstream_decoding_results.json").read_text())["metadata"]["cohort"]
    assert second["complete"] is False and second["stale"] == ["S1", "S2"]
    assert real_now_iso is orch.cohort_results.now_iso


# ── MLflow without a tracking server ────────────────────────────────────────
# Benchmark and calibrate default to mlflow=auto: on when MLFLOW_TRACKING_URI is
# set, off otherwise (without a URI, MLflow falls back to ./mlruns, which current
# MLflow refuses). Asking for MLflow explicitly with no URI fails before any
# cell launches instead of crashing each cell.


@pytest.fixture
def no_mlflow_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)


@pytest.mark.parametrize("value", [None, "auto", "AUTO"])
def test_resolve_mlflow_auto_is_off_without_uri(no_mlflow_uri, value):
    assert orch._resolve_mlflow_enabled(value) is False


@pytest.mark.parametrize("value", [None, "auto"])
def test_resolve_mlflow_auto_is_on_with_uri(monkeypatch, value):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking.example:5000")
    assert orch._resolve_mlflow_enabled(value) is True


def test_resolve_mlflow_auto_ignores_blank_uri(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "   ")
    assert orch._resolve_mlflow_enabled("auto") is False


@pytest.mark.parametrize("value", ["false", False])
def test_resolve_mlflow_explicit_false(monkeypatch, value):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking.example:5000")
    assert orch._resolve_mlflow_enabled(value) is False


@pytest.mark.parametrize("value", ["true", True])
def test_resolve_mlflow_explicit_true_without_uri_fails_early(no_mlflow_uri, value):
    with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
        orch._resolve_mlflow_enabled(value)


def test_resolve_mlflow_explicit_true_with_uri(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking.example:5000")
    assert orch._resolve_mlflow_enabled("true") is True


def _benchmark_cells(monkeypatch, extra_args=()):
    cells, parents = [], []

    def fake_create_parent(repo_root, *, experiment_name, run_name, tags, disabled, dry_run):
        parents.append(disabled)
        return ("", "")

    monkeypatch.setattr(orch, "_run_train", lambda argv, **_: cells.append(argv) or 0)
    monkeypatch.setattr(orch, "_mlflow_create_parent", fake_create_parent)
    monkeypatch.setattr(orch, "_resolve_gpus", lambda flag: ["0"])
    rc = orch.main([
        "benchmark", "--repo-root", str(_REPO_ROOT), "--spike-times-only", "true",
        "--dry-run", "--data-root", "/data", "--sessions", "S", *extra_args,
    ])  # fmt: skip
    return rc, cells, parents


def test_benchmark_default_without_uri_disables_mlflow_per_cell(no_mlflow_uri, monkeypatch):
    rc, cells, parents = _benchmark_cells(monkeypatch)
    assert rc == 0 and cells
    assert all("mlflow.enabled=false" in argv for argv in cells)
    assert parents and all(parents)  # parent runs created with disabled=True


def test_benchmark_default_with_uri_keeps_mlflow(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking.example:5000")
    rc, cells, parents = _benchmark_cells(monkeypatch)
    assert rc == 0 and cells
    assert all("mlflow.enabled=true" in argv for argv in cells)
    assert parents and not any(parents)


def test_benchmark_explicit_mlflow_without_uri_fails_before_cells(no_mlflow_uri, monkeypatch):
    with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
        _benchmark_cells(monkeypatch, ["--mlflow", "true"])


def test_benchmark_cli_yaml_defaults_to_auto():
    """The Hydra sweep seam reads configs/cli/benchmark.yaml, not argparse defaults."""
    cli = config_loader.load_yaml(_REPO_ROOT / "configs" / "cli" / "benchmark.yaml")
    assert str(cli["mlflow"]).lower() == "auto"


def test_calibrate_default_without_uri_disables_mlflow(no_mlflow_uri, monkeypatch):
    monkeypatch.setattr(orch, "_resolve_gpus", lambda *a, **k: ["0"])
    cfg = orch.calibrate_config_from_mapping({"tokenizer_kind": "spike_times", "dry_run": True})
    assert cfg.mlflow_enabled is False


def test_calibrate_default_with_uri_enables_mlflow(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking.example:5000")
    monkeypatch.setattr(orch, "_resolve_gpus", lambda *a, **k: ["0"])
    cfg = orch.calibrate_config_from_mapping({"tokenizer_kind": "spike_times", "dry_run": True})
    assert cfg.mlflow_enabled is True


def test_calibrate_explicit_mlflow_without_uri_fails_early(no_mlflow_uri, monkeypatch):
    monkeypatch.setattr(orch, "_resolve_gpus", lambda *a, **k: ["0"])
    with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
        orch.calibrate_config_from_mapping({"tokenizer_kind": "spike_times", "mlflow": "true", "dry_run": True})
