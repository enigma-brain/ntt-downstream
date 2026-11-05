"""Stable orchestration engine + CLI for POYO decoding calibration / benchmark.

This module holds the orchestration logic for calibration sweeps and tokenizer
benchmarks. The ``*.sbatch`` wrappers, the ``ntt job=downstream`` launcher, and
local/CI runs are all thin clients of the CLI exposed here, so editing a
``#SBATCH`` knob can never change the orchestration logic. See
``docs/decoding_orchestration.md``.

Two subcommands: ``calibrate`` (single-session or multi-session fine-tune
calibration) and ``benchmark`` (one cohort of tokenizers, optionally plus the
spike-times baseline). Each cell is a ``train.py`` subprocess with a fixed
override list; the engine owns the job queue / GPU pool, idempotent resume,
the MLflow parent-run lifecycle and CSV aggregation.

Design boundary (see the doc):
  * pure logic is imported in-process (``config_loader``, ``calibration_csv``,
    ``mlflow_utils``);
  * ``ntt_downstream.decoding.train`` and the stateful
    ``ntt_downstream.decoding.utils.mlflow_parent create|aggregate`` are invoked
    as subprocesses, per cell, so Hydra's one-run-per-process rule and per-cell
    CUDA isolation hold and a flaky MLflow server can't kill the loop.

``--dry-run`` resolves everything and prints the job queue + each cell's exact
``train.py`` override list **without** executing; the orchestration tests
assert on that output.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import queue
import shlex
import subprocess
import sys
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ntt_downstream.decoding import calibration_csv, cohort_results
from ntt_downstream.decoding.cohort_results import version_dirs
from ntt_downstream.decoding.config_loader import DATA_KEYS, TRAINING_KEYS, load_yaml, walk

log = logging.getLogger("ntt_downstream.decoding.orchestration")

_TRUE = {"true", "True", "TRUE", "1"}
_FALSE = {"false", "False", "FALSE", "0"}

# cuBLAS deterministic-workspace fallback, used when the training YAML omits
# runtime.cublas_workspace_config and for the benchmark (which has no training
# YAML).
_DEFAULT_CUBLAS_WORKSPACE_CONFIG = ":4096:8"

# Checkpoint URI schemes that are NOT local files; the per-cell train.py resolves
# them via FrozenNTTTokenizer._resolve_ckpt_path (the canonical resolver — keep
# this tuple in sync with it). Kept inline so the import-light engine needn't
# import the heavy poyo backend just to detect a URI.
_CKPT_URI_SCHEMES = ("mlflow-artifacts:", "mlflow:")


def _is_ckpt_uri(ckpt: str) -> bool:
    """True if ``ckpt`` is a resolvable URI rather than a local filesystem path."""
    return str(ckpt).startswith(_CKPT_URI_SCHEMES)


# ─────────────────────────── small pure helpers ────────────────────────────


def _parse_bool(value: str | bool, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"{name} must be true|false, got: {value!r}")


def _resolve_mlflow_enabled(value: str | bool | None) -> bool:
    """Resolve the sweep-level ``mlflow`` knob (``auto`` | true | false) to a bool.

    ``auto`` (and ``None``, an unset knob) means "log to MLflow when a tracking
    server is configured": on iff ``MLFLOW_TRACKING_URI`` is set. This is the
    default so a benchmark or calibrate sweep runs for users with no MLflow
    server. Without it every cell got ``mlflow.enabled=true``, MLflow fell back
    to a local ``./mlruns`` file store, current MLflow refuses that, and every
    cell crashed. An explicit ``true`` with no URI raises here, before any cell
    launches, instead of crashing each cell.
    """
    has_uri = bool(os.environ.get("MLFLOW_TRACKING_URI", "").strip())
    if value is None or (isinstance(value, str) and value.strip().lower() == "auto"):
        if not has_uri:
            log.info("mlflow=auto and MLFLOW_TRACKING_URI is unset: MLflow logging is off for this sweep")
        return has_uri
    enabled = _parse_bool(value, "mlflow")
    if enabled and not has_uri:
        raise ValueError(
            "mlflow=true but MLFLOW_TRACKING_URI is unset. Set MLFLOW_TRACKING_URI, "
            "or use mlflow=auto (the default) / mlflow=false to run without MLflow."
        )
    return enabled


def _bool_str(b: bool) -> str:
    return "true" if b else "false"


def _eligible_trials(n_valid: int, trial_counts: Sequence[int]) -> list[int]:
    """``[0, *trial_counts]`` keeping only counts ``<= floor(0.7 * n_valid)``.

    ``n=0`` (plain eval) always survives. Clamps requested FT-trial counts to
    70% of a session's valid trials, so a session never fine-tunes on more than
    it can hold out."""
    threshold = int(0.7 * n_valid)
    return [n for n in [0, *trial_counts] if n <= threshold]


def _pool_hash(sessions: Sequence[str]) -> str:
    """Stable id of a multi-session base run: the POSIX ``cksum`` of the
    newline-joined, sorted, de-duplicated train sessions."""
    data = "".join(f"{s}\n" for s in sorted(set(sessions)))
    return str(_posix_cksum(data.encode()))


# POSIX `cksum` (CRC-32/CKSUM): CRC over the data, then over the byte-length,
# then complemented. Implemented here (rather than via zlib.crc32, which is a
# different CRC variant) so truncated wandb/MLflow run names stay stable and
# reproducible with the shell `cksum` tool.
def _posix_cksum(data: bytes) -> int:
    poly = 0x04C11DB7
    table = []
    for i in range(256):
        c = i << 24
        for _ in range(8):
            c = ((c << 1) ^ poly) & 0xFFFFFFFF if (c & 0x80000000) else (c << 1) & 0xFFFFFFFF
        table.append(c)
    crc = 0
    for b in data:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ table[((crc >> 24) ^ b) & 0xFF]
    n = len(data)
    while n:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ table[((crc >> 24) ^ (n & 0xFF)) & 0xFF]
        n >>= 8
    return (~crc) & 0xFFFFFFFF


def _short_id(raw: str, limit: int = 120) -> str:
    """Shorten a run id: identity unless ``len > limit``, in which case truncate
    and append ``_<posix-cksum>`` so distinct long ids stay distinct."""
    if len(raw) <= limit:
        return raw
    h = str(_posix_cksum(raw.encode()))
    return f"{raw[: limit - len(h) - 1]}_{h}"


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _model_cfg_and_variant(model_size: str, no_crossattn: bool) -> tuple[str, str]:
    """Map (model size, crossattn flag) to ``(model config name, variant tag)``."""
    arch = "nocrossattn" if no_crossattn else "crossattn"
    key = f"{model_size}_{arch}"
    mapping = {
        "1.3M_nocrossattn": ("poyo_1.3M_no_crossattn", "1.3M_nocrossattn"),
        "1.3M_crossattn": ("poyo_1.3M", "1.3M_crossattn"),
        "11.8M_crossattn": ("poyo_11.8M", "11.8M_crossattn"),
    }
    if key == "11.8M_nocrossattn":
        raise ValueError("MODEL_SIZE=11.8M with no_crossattn is unsupported (no config).")
    if key not in mapping:
        raise ValueError(f"MODEL_SIZE must be 1.3M|11.8M, got: {model_size!r}")
    return mapping[key]


# ──────────────────────────── MLflow subprocess ────────────────────────────


# Stateful MLflow ops live in a utils module and are invoked as a subprocess (so
# a flaky tracking server can't take down the orchestrator loop), run from
# repo_root so relative paths resolve.
_MLFLOW_PARENT_MODULE = "ntt_downstream.decoding.utils.mlflow_parent"


def _mlflow_create_parent(
    repo_root: Path,
    *,
    experiment_name: str,
    run_name: str,
    tags: dict[str, str],
    disabled: bool,
    dry_run: bool,
) -> tuple[str, str]:
    """Create a parent run via the script; return (run_id, run_uri). Soft-fails
    to ('', '') so the orchestrator degrades gracefully."""
    cmd = [
        sys.executable,
        "-m",
        _MLFLOW_PARENT_MODULE,
        "create",
        "--experiment-name",
        experiment_name,
        "--run-name",
        run_name,
    ]
    for k, v in tags.items():
        cmd += ["--tag", f"{k}={v}"]
    if disabled or dry_run:
        cmd.append("--disabled")
    if dry_run:
        log.info("[dry-run] mlflow create: %s", shlex.join(cmd))
        return "", ""
    try:
        out = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True, check=False).stdout
    except Exception as exc:  # noqa: BLE001
        log.warning("_mlflow_parent.create failed: %s", exc)
        return "", ""
    run_id = run_uri = ""
    for line in out.splitlines():
        if line.startswith("RUN_ID="):
            run_id = line[len("RUN_ID=") :]
        elif line.startswith("RUN_URI="):
            run_uri = line[len("RUN_URI=") :]
    return run_id, run_uri


def _mlflow_aggregate(repo_root: Path, run_id: str, *, dry_run: bool) -> None:
    if not run_id:
        return
    cmd = [sys.executable, "-m", _MLFLOW_PARENT_MODULE, "aggregate", "--run-id", run_id]
    if dry_run:
        log.info("[dry-run] mlflow aggregate: %s", shlex.join(cmd))
        return
    try:
        subprocess.run(cmd, cwd=str(repo_root), check=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("MLflow aggregation failed for parent=%s: %s", run_id, exc)


def _mlflow_log_artifacts(
    repo_root: Path, run_id: str, srcs: Sequence[Path], *, dry_run: bool, files: Sequence[Path] = ()
) -> None:
    """Upload each src tree's results record + ``metrics.csv`` + ``calibration_*.csv``
    onto the parent run, plus ``files`` at the parent's artifact root (the
    benchmark cohort record), so a sweep's deliverables are retrievable from one
    place. Best-effort: a no-op when MLflow is disabled (empty run_id) or there's
    nothing to upload; the helper never uploads checkpoints or run_result.json."""
    existing = [s for s in srcs if s.is_dir()]
    root_files = [f for f in files if Path(f).is_file()]
    if not run_id or not (existing or root_files):
        return
    cmd = [sys.executable, "-m", _MLFLOW_PARENT_MODULE, "log-artifacts", "--run-id", run_id]
    for s in existing:
        cmd += ["--src", str(s)]
    for f in root_files:
        cmd += ["--file", str(f)]
    if dry_run:
        log.info("[dry-run] mlflow log-artifacts: %s", shlex.join(cmd))
        return
    try:
        subprocess.run(cmd, cwd=str(repo_root), check=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("MLflow log-artifacts failed for parent=%s: %s", run_id, exc)


# ─────────────────────────── train.py invocation ───────────────────────────

# Launcher prefix for a single training cell. We're already inside the venv
# (the orchestrator is run via `uv run python -m …orchestration`), so spawn the
# trainer with the current interpreter — no nested `uv run`.
_TRAIN_PREFIX = [sys.executable, "-m", "ntt_downstream.decoding.train"]


def _tokenizer_args(
    kind: str,
    ckpt: str,
    ue_mode: str,
    sentinels: str | bool,
    *,
    spike_times_sentinels: bool = False,
) -> list[str]:
    """Hydra tokenizer overrides.

    ``frozen`` always carries an ntt_checkpoint and a ``tokenizer.sentinels``
    override — both subcommands agree there. For ``spike_times`` the two
    deliberately diverge: ``calibrate`` always passes ``tokenizer.sentinels``
    too (so the calibration config's value wins), whereas ``benchmark`` omits it
    so the run inherits the spike_times config default. ``spike_times_sentinels``
    selects which policy applies — calibrate passes ``True``, the benchmark
    leaves it ``False``.
    Without this, spike_times calibration would silently ignore the resolved
    ``TOKENIZER_SENTINELS`` and fall back to the tokenizer config default.
    """
    args = [f"tokenizer={kind}"]
    if ckpt:
        args.append(f"tokenizer.ntt_checkpoint={ckpt}")
    args.append(f"tokenizer.unit_embedding_mode={ue_mode}")
    if kind == "frozen" or spike_times_sentinels:
        args.append(f"tokenizer.sentinels={_bool_str(_parse_bool(sentinels, 'sentinels'))}")
    return args


def _mlflow_parent_override(parent_run_id: str) -> str:
    return f"mlflow.parent_run_id={parent_run_id}" if parent_run_id else "mlflow.parent_run_id=null"


def _cell_env(gpu: str, seed: int, cublas_workspace_config: str) -> dict[str, str]:
    """Per-cell subprocess environment: GPU pin + determinism knobs.

    Determinism is set by the engine rather than the SLURM wrapper:
      * ``PYTHONHASHSEED`` is pinned to the cell's seed (authoritative), so each
        seed is reproducible;
      * ``CUBLAS_WORKSPACE_CONFIG`` defaults to the training YAML's
        ``runtime.cublas_workspace_config`` but an explicit environment value
        still wins.
    """
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONHASHSEED=str(seed))
    # `or` (not setdefault) so an inherited EMPTY value falls back to the cell
    # default (setdefault would preserve a present-but-empty "", silently
    # disabling the determinism config).
    env["CUBLAS_WORKSPACE_CONFIG"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG") or cublas_workspace_config
    return env


def _run_train(
    argv_overrides: list[str],
    *,
    gpu: str,
    log_path: Path,
    dry_run: bool,
    seed: int,
    cublas_workspace_config: str,
    repo_root: Path,
) -> int:
    """Run one training cell as a subprocess with the per-cell determinism env.

    Runs from ``repo_root`` so relative
    Hydra/data/log paths resolve correctly even when the CLI is invoked from
    outside the repo."""
    cmd = [*_TRAIN_PREFIX, *argv_overrides]
    if dry_run:
        log.info("[dry-run] cell (GPU %s) -> %s\n    %s", gpu, log_path, shlex.join(cmd))
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as fh:
        return subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=_cell_env(gpu, seed, cublas_workspace_config),
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode


# ───────────────────────────── worker pool ─────────────────────────────────


def _worker_tags(gpus: Sequence[str], jobs_per_gpu: int) -> list[str]:
    return [f"{g}:{s}" for g in gpus for s in range(jobs_per_gpu)]


def _run_pool(jobs: list, worker_tags: list[str], run_job) -> int:
    """Dispatch ``jobs`` across worker threads (one per ``gpu:slot`` tag), each
    pulling atomically from a shared queue and calling ``run_job(job, gpu)``.
    Returns 1 if any job raised / returned non-zero, else 0. (An in-process
    Queue is used, so no file locking is needed on shared filesystems.)"""
    q: queue.Queue = queue.Queue()
    for j in jobs:
        q.put(j)
    # Defensive: jobs queued but no worker to drain them (empty gpus or
    # jobs_per_gpu<1) would otherwise hang-free no-op and falsely report success.
    if jobs and not worker_tags:
        raise ValueError(f"{len(jobs)} job(s) queued but no worker tags (empty gpus or jobs_per_gpu<1)")
    failed = threading.Event()

    def worker(tag: str) -> None:
        gpu = tag.split(":", 1)[0]
        while True:
            try:
                job = q.get_nowait()
            except queue.Empty:
                return
            try:
                if run_job(job, gpu) != 0:
                    failed.set()
            except Exception:  # noqa: BLE001
                log.exception("worker %s job failed: %r", tag, job)
                failed.set()
            finally:
                q.task_done()

    threads = [threading.Thread(target=worker, args=(t,), daemon=True) for t in worker_tags]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return 1 if failed.is_set() else 0


# ───────────────────────────── config resolution ───────────────────────────


def _yaml_defaults(path: Path, key_map: dict[str, str]) -> dict[str, object]:
    """Resolve a YAML through the dotted-path whitelist into ``{VAR: value}``.

    Every whitelisted key must be present — a missing key is a hard error, so a
    typo'd or dropped YAML key fails loudly here instead of silently reaching
    ``train.py`` as the string ``"None"`` (scalars) or an empty list. All missing
    keys are reported at once.
    """
    cfg = load_yaml(path)
    out: dict[str, object] = {}
    missing: list[str] = []
    for dotted, var in key_map.items():
        try:
            out[var] = walk(cfg, dotted)
        except KeyError:
            missing.append(dotted)
    if missing:
        raise ValueError(f"{path}: missing required config key(s): {', '.join(missing)}")
    return out


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return shlex.split(str(value))


@dataclasses.dataclass
class CalibrateConfig:
    repo_root: Path
    data_root: str
    tokenizer_kind: str
    ckpt: str
    tokenizer_name: str
    model_cfg: str
    variant: str
    no_crossattn: bool
    num_latents_per_step: str
    ue_mode: str
    sentinels: str
    train_sessions: list[str]
    other_sessions: list[str]
    ft_trial_counts: list[int]
    ft_modes: list[str]
    seeds: list[int]
    base_max_steps: str
    ft_max_steps: str
    eval_every_n_steps: str
    ft_eval_every: str
    ckpt_every_n_steps: str
    ft_ckpt_every: str
    batch_size: str
    num_workers: str
    precision: str
    cublas_workspace_config: str
    force_flash_attn: bool
    temporal_split_enabled: str
    brainset: str
    calib_root: str
    mlflow_enabled: bool
    mlflow_experiment_name: str
    mlflow_log_model: str
    gpus: list[str]
    jobs_per_gpu: int
    reuse_base_if_exists: bool
    missing_only: bool
    skip_base_train: bool
    base_ckpt: str
    multi_session: bool
    hydra_data_config: str
    extra: list[str]
    dry_run: bool


def _resolve_gpus(gpus_flag: str | None) -> list[str]:
    # A whitespace-only flag splits to [] — fall through to auto-detect rather
    # than returning an empty list that would later index gpus[0] out of range.
    if gpus_flag and gpus_flag.split():
        return gpus_flag.split()
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        gpus = out.split()
    except Exception:  # noqa: BLE001
        gpus = []
    return gpus or ["0"]


def _pick(flag: Any, default: Any) -> Any:
    """CLI flag wins when provided (not None); else the YAML default."""
    return default if flag is None else flag


def _coerce_positive_int(value, name: str) -> int:
    """Coerce ``value`` to an int >= 1, rejecting bools and non-integer floats.

    The Hydra CLI seam (``_namespace_from_mapping``) bypasses argparse's
    ``type=int``, so a YAML ``2.5``/``true`` could otherwise slip through
    ``int()`` (which truncates floats and maps bools to 0/1) and silently
    mis-size the worker pool.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got bool: {value!r}")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be a whole number, got: {value!r}")
    try:
        out = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive integer, got: {value!r}") from None
    if out < 1:
        raise ValueError(f"{name} must be >= 1, got: {out}")
    return out


def _resolve_calibrate(args: argparse.Namespace) -> CalibrateConfig:
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path.cwd()
    multi = bool(args.multi_session)
    # Config defaults follow the mode; an explicit --training/data-config still wins.
    subdir = "multi_session" if multi else "single_session"
    training_config = args.training_config or f"configs/decoding/{subdir}/tokenizer_training.yaml"
    data_config = args.data_config or f"configs/decoding/{subdir}/data.yaml"
    tcfg = _yaml_defaults(repo_root / training_config, TRAINING_KEYS)
    dcfg = _yaml_defaults(repo_root / data_config, DATA_KEYS)

    kind = _pick(args.tokenizer_kind, tcfg["TOKENIZER_KIND"])
    if kind not in ("frozen", "spike_times"):
        raise ValueError(f"tokenizer-kind must be frozen|spike_times, got: {kind!r}")
    ckpt = args.ckpt or ""
    if kind == "frozen":
        if not ckpt:
            raise ValueError("--ckpt is required for tokenizer-kind=frozen.")
        if _is_ckpt_uri(ckpt):
            # mlflow-artifacts:/ (or mlflow:) URIs are first-class: per-cell
            # train.py resolves+downloads them. Skip the local is_file() check,
            # but require an explicit name (it can't be derived from a URI path).
            if not args.tokenizer_name:
                raise ValueError(
                    "--tokenizer-name is required when --ckpt is an "
                    "mlflow-artifacts:/ or mlflow: URI (the run name / MLflow "
                    "tags cannot be derived from a URI path)."
                )
            tok_name = args.tokenizer_name
        else:
            if not args.dry_run and not Path(ckpt).is_file():
                raise FileNotFoundError(f"tokenizer checkpoint not found: {ckpt}")
            tok_name = args.tokenizer_name or Path(ckpt).resolve().parent.parent.name
    else:
        tok_name = args.tokenizer_name or kind

    no_crossattn = _parse_bool(_pick(args.no_crossattn, tcfg["NO_CROSSATTN"]), "no_crossattn")
    if kind == "frozen" and not no_crossattn:
        raise ValueError("tokenizer-kind=frozen requires no_crossattn=true.")
    model_cfg, variant = _model_cfg_and_variant(str(_pick(args.model_cfg, tcfg["MODEL_SIZE"])), no_crossattn)

    # Fail loud on a degenerate worker pool: jobs_per_gpu<1 or no GPUs would make
    # _worker_tags() return [] and _run_pool() silently drain nothing yet report
    # success. The Hydra CLI seam (_namespace_from_mapping) bypasses argparse
    # type=int, so coerce defensively here.
    gpus = _resolve_gpus(args.gpus)
    jobs_per_gpu = _coerce_positive_int(args.jobs_per_gpu, "jobs_per_gpu")
    if not gpus:
        raise ValueError("no GPUs resolved; pass --gpus or run where nvidia-smi reports a device")

    return CalibrateConfig(
        repo_root=repo_root,
        data_root=str(_pick(args.data_root, dcfg["DATA_ROOT"])),
        tokenizer_kind=kind,
        ckpt=ckpt,
        tokenizer_name=tok_name,
        model_cfg=model_cfg,
        variant=variant,
        no_crossattn=no_crossattn,
        num_latents_per_step=str(_pick(args.num_latents_per_step, tcfg["NUM_LATENTS_PER_STEP"])),
        ue_mode=str(_pick(args.unit_embedding_mode, tcfg["UE_MODE"])),
        sentinels=str(_pick(args.sentinels, tcfg["TOKENIZER_SENTINELS"])),
        train_sessions=_as_list(args.train_sessions) or _as_list(dcfg["TRAIN_SESSIONS"]),
        other_sessions=_as_list(args.other_sessions) or _as_list(dcfg["OTHER_SESSIONS"]),
        ft_trial_counts=[int(x) for x in (_as_list(args.ft_trial_counts) or _as_list(tcfg["FT_TRIAL_COUNTS"]))],
        ft_modes=_as_list(args.ft_modes) or _as_list(tcfg["FT_MODES"]),
        seeds=[int(x) for x in (_as_list(args.seeds) or _as_list(tcfg["SEEDS"]))],
        base_max_steps=str(_pick(args.base_max_steps, tcfg["BASE_MAX_STEPS"])),
        ft_max_steps=str(_pick(args.ft_max_steps, tcfg["FT_MAX_STEPS"])),
        eval_every_n_steps=str(_pick(args.eval_every_n_steps, tcfg["EVAL_EVERY_N_STEPS"])),
        ft_eval_every=str(_pick(args.ft_eval_every, tcfg["FT_EVAL_EVERY"])),
        ckpt_every_n_steps=str(_pick(args.ckpt_every_n_steps, tcfg["CKPT_EVERY_N_STEPS"])),
        ft_ckpt_every=str(_pick(args.ft_ckpt_every, tcfg["FT_CKPT_EVERY"])),
        batch_size=str(_pick(args.batch_size, tcfg["BATCH_SIZE"])),
        num_workers=str(_pick(args.num_workers, tcfg["NUM_WORKERS"])),
        precision=str(_pick(args.precision, tcfg["PRECISION"])),
        cublas_workspace_config=str(tcfg["CUBLAS_WORKSPACE_CONFIG"] or _DEFAULT_CUBLAS_WORKSPACE_CONFIG),
        force_flash_attn=_parse_bool(_pick(args.force_flash_attn, tcfg["FORCE_FLASH_ATTN"]), "force_flash_attn"),
        temporal_split_enabled=_bool_str(
            _parse_bool(_pick(args.temporal_split_enabled, dcfg["TEMPORAL_SPLIT_ENABLED"]), "temporal_split_enabled")
        ),
        brainset=str(_pick(args.brainset, dcfg["BRAINSET"])),
        calib_root=str(_pick(args.calib_root, tcfg["CALIB_ROOT"])),
        mlflow_enabled=_resolve_mlflow_enabled(_pick(args.mlflow, tcfg["MLFLOW_ENABLED"])),
        mlflow_experiment_name=str(_pick(args.mlflow_experiment_name, tcfg["MLFLOW_EXPERIMENT_NAME"])),
        mlflow_log_model=_bool_str(
            _parse_bool(_pick(args.mlflow_log_model, tcfg["MLFLOW_LOG_MODEL"]), "mlflow-log-model")
        ),
        gpus=gpus,
        jobs_per_gpu=jobs_per_gpu,
        reuse_base_if_exists=_parse_bool(args.reuse_base_if_exists, "reuse-base-if-exists"),
        missing_only=_parse_bool(args.missing_only, "missing-only"),
        skip_base_train=_parse_bool(args.skip_base_train, "skip-base-train"),
        base_ckpt=args.base_ckpt or "",
        multi_session=multi,
        hydra_data_config=args.hydra_data_config,
        extra=shlex.split(args.extra or ""),
        dry_run=args.dry_run,
    )


# ──────────────────────── config-object seam (Hydra CLI) ────────────────────
# The Hydra CLI (src/ntt_downstream/cli/) drives this engine from flat Hydra
# sweep configs rather than argparse. To reuse the EXACT field logic + argparse
# defaults below — without duplicating _resolve_calibrate's ~50 lines or
# hand-listing defaults — we synthesize a Namespace seeded with the subparser's
# real defaults (`parse_args([verb])`), overlay the Hydra-derived mapping, and
# delegate to the existing resolver. The engine bodies stay untouched.


def _namespace_from_mapping(verb: str, mapping: dict[str, object]) -> argparse.Namespace:
    """Seed an ``argparse.Namespace`` with ``verb``'s real argparse defaults, then
    overlay ``mapping``. Lets the Hydra-driven CLI reuse the exact
    ``_resolve_calibrate`` / ``run_benchmark`` field logic without duplicating it
    or hand-listing defaults. ``parse_args([verb])`` succeeds for both calibrate
    and benchmark (no required args), giving every dest its declared default.

    Note: because the mapping is ``setattr``'d directly, argparse ``type`` /
    ``choices`` coercion is bypassed — the downstream resolver re-validates the
    load-bearing fields (e.g. ``_resolve_calibrate`` rejects a bad
    ``tokenizer_kind``; numeric dests are coerced where the engine reads them)."""
    ns = _build_parser().parse_args([verb])
    for key, value in mapping.items():
        # Reject keys that aren't real argparse dests: a typo'd or stale sweep
        # YAML key would otherwise be setattr'd here and silently ignored by the
        # engine. This is the schema check for external sweep configs — argparse
        # stays the single source of truth for the key set. ValueError (not a
        # CLI-side error type) keeps the engine free of cli imports; app.main
        # already maps ValueError to `error: …` + exit 2.
        if not hasattr(ns, key):
            raise ValueError(f"unknown {verb} sweep config key {key!r}")
        setattr(ns, key, value)
    return ns


def calibrate_config_from_mapping(mapping: dict[str, object]) -> CalibrateConfig:
    """Build the existing ``CalibrateConfig`` from a (Hydra-derived) dest mapping."""
    return _resolve_calibrate(_namespace_from_mapping("calibrate", mapping))


def benchmark_from_mapping(mapping: dict[str, object]) -> int:
    """Run the existing benchmark engine from a (Hydra-derived) dest mapping."""
    return run_benchmark(_namespace_from_mapping("benchmark", mapping))


# ───────────────────────── trial-count preflight ───────────────────────────


def _preflight_trials(cfg: CalibrateConfig) -> dict[str, int]:
    """``{session: n_valid_trials}`` from the HDF5 files (in-process).

    On a real run a missing session HDF5 is a hard error — a typo'd other-sessions
    entry must fail loudly rather than silently disabling the 0.7 trial clamp. ``--dry-run`` stays
    permissive (warn + no-clamp sentinel) so a no-data plan still resolves."""
    counts: dict[str, int] = {}
    for sess in cfg.other_sessions:
        path = Path(cfg.data_root) / cfg.brainset / f"{sess}.h5"
        if not path.is_file():
            if not cfg.dry_run:
                raise FileNotFoundError(f"preflight: session HDF5 not found: {path}")
            log.warning("preflight: %s missing — assuming no trial clamp (dry-run)", path)
            counts[sess] = 1 << 30
            continue
        import h5py  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        import temporaldata  # noqa: PLC0415

        with h5py.File(path, "r") as h:
            data = temporaldata.Data.from_hdf5(h, lazy=True)
            # temporaldata attaches ``trials`` dynamically, so the stub doesn't know it.
            trials = data.trials  # pyright: ignore[reportAttributeAccessIssue]
            counts[sess] = int(np.asarray(trials.is_valid).astype(bool).sum())
    return counts


# ───────────────────── idempotence / reuse helpers ─────────────────────────


def _latest_run_result_complete(run_dir: Path) -> bool:
    """True iff the newest ``run_result.json`` is present, contains
    ``val_last_r2``, and ends with ``}`` (a complete, non-truncated write)."""
    for v in reversed(version_dirs(run_dir)):
        rr = v / "run_result.json"
        if rr.is_file():
            try:
                text = rr.read_text()
            except OSError:
                return False
            return '"val_last_r2"' in text and text.rstrip().endswith("}")
    return False


def _ckpt_global_step(ckpt: Path) -> int | None:
    """Read ``global_step`` from a Lightning checkpoint, or ``None`` if the file
    is unreadable / not a valid checkpoint. torch is imported lazily so the
    module stays light when no base ckpt is inspected."""
    try:
        import torch  # noqa: PLC0415

        return int(torch.load(ckpt, map_location="cpu", weights_only=False).get("global_step", -1))
    except Exception:  # noqa: BLE001
        return None


def _latest_base_ckpt(base_log_dir: Path, min_steps: int) -> str | None:
    """Newest ``last.ckpt`` whose ``global_step >= min_steps`` (guards against
    reusing a partial base), or None."""
    ckpt_dirs = sorted(
        base_log_dir.glob("csv_logs/version_*/checkpoints"),
        key=lambda p: int(p.parent.name.split("_", 1)[1]),
        reverse=True,
    )
    for d in ckpt_dirs:
        ckpt = d / "last.ckpt"
        if not ckpt.is_file():
            continue
        step = _ckpt_global_step(ckpt)
        if step is not None and step >= min_steps:
            return str(ckpt)
    return None


def _mirror_plain_eval(src_dir: Path, dst_dir: Path) -> bool:
    """Copy the canonical n_trials=0 artifacts into a parallel ft_mode subtree so
    the CSV layout stays symmetric. Returns False (with a warning) if the source
    has no result."""
    import shutil  # noqa: PLC0415

    versions = version_dirs(src_dir)
    src_v = versions[-1] if versions else None
    if src_v is None or not (src_v / "run_result.json").is_file():
        log.warning("no run_result.json under %s/csv_logs/ — cannot mirror", src_dir)
        return False
    dst_v = dst_dir / "csv_logs" / "version_0"
    dst_v.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_v / "run_result.json", dst_v / "run_result.json")
    for opt in ("run_info.json", "hparams.yaml", "metrics.csv", "downstream_decoding_results.json"):
        if (src_v / opt).is_file():
            shutil.copy(src_v / opt, dst_v / opt)
    (dst_dir / "run.log").write_text(
        f"# Plain-eval result mirrored from {src_v}/run_result.json\n"
        "# (n_trials=0 / eval_only=true is independent of ft_mode.)\n"
    )
    return True


# ─────────────────────────── calibrate driver ──────────────────────────────


def _common_wandb_tags(cfg: CalibrateConfig, train_session: str | None, seed: int) -> str:
    crossattn = "false" if cfg.no_crossattn else "true"
    tags = [
        _short_id(f"tok-{cfg.tokenizer_name}", 64),
        f"kind-{cfg.tokenizer_kind}",
        f"arch-{cfg.model_cfg}",
        f"size-{cfg.variant.split('_')[0]}",
        f"crossattn-{crossattn}",
    ]
    # Multi-session runs have no single train session: identify the pool by its
    # size + a stable hash instead of the single-session `sess-<id>` facet.
    if cfg.multi_session:
        tags += [f"pool-{len(cfg.train_sessions)}sess", f"poolhash-{_pool_hash(cfg.train_sessions)}"]
    else:
        tags.append(f"sess-{train_session}")
    tags.append(f"seed-{seed}")
    if cfg.ckpt:
        ckpt_name = Path(cfg.ckpt).name
        if ckpt_name.endswith(".ckpt"):
            ckpt_name = ckpt_name[: -len(".ckpt")]
        tags.append(_short_id(f"ckpt-{ckpt_name}", 64))
    return ",".join(tags)


def _base_run_name(cfg: CalibrateConfig, train_session: str | None, seed: int) -> str:
    """Per-(base, seed) run name: ``single_…`` keyed on the one train session, or
    ``multi_…`` keyed on the pool size + sorted-pool hash."""
    if cfg.multi_session:
        pool = cfg.train_sessions
        return f"multi_{cfg.tokenizer_name}_{len(pool)}sess_{_pool_hash(pool)}_{cfg.variant}_seed{seed}"
    return f"single_{cfg.tokenizer_name}_{train_session}_{cfg.variant}_seed{seed}"


def _base_argv(cfg, train_session, seed, run_dir, parent_run_id, sweep) -> list[str]:
    name = _short_id(_base_run_name(cfg, train_session, seed), 120)
    if cfg.multi_session:
        pool = cfg.train_sessions
        # The pool is wired into the Hydra preset's `_train_sessions_list: ???`
        # placeholder; this replaces the single-session `session=<one>`.
        session_args = [f"+data={cfg.hydra_data_config}", f"_train_sessions_list=[{','.join(pool)}]"]
        base_facet = ""  # no `trainsess-<id>` facet for a pool
        train_tags = [f"+mlflow.tags.pool_size={len(pool)}", f"+mlflow.tags.pool_hash={_pool_hash(pool)}"]
    else:
        session_args = [f"session={train_session}"]
        base_facet = f"trainsess-{train_session},"
        train_tags = [f"+mlflow.tags.train_session={train_session}"]
    argv = [
        f"model={cfg.model_cfg}",
        *session_args,
        f"data_root={cfg.data_root}",
        f"log_dir={run_dir}",
        *_tokenizer_args(cfg.tokenizer_kind, cfg.ckpt, cfg.ue_mode, cfg.sentinels, spike_times_sentinels=True),
    ]
    if not cfg.no_crossattn:
        argv.append(f"model.num_latents_per_step={cfg.num_latents_per_step}")
    common = _common_wandb_tags(cfg, train_session, seed)
    argv += [
        f"temporal_split.enabled={cfg.temporal_split_enabled}",
        f"max_steps={cfg.base_max_steps}",
        f"eval_every_n_steps={cfg.eval_every_n_steps}",
        f"checkpoint_every_n_steps={cfg.ckpt_every_n_steps}",
        f"batch_size={cfg.batch_size}",
        f"seed={seed}",
        f"num_workers={cfg.num_workers}",
        f"precision={cfg.precision}",
        f"force_flash_attn={_bool_str(cfg.force_flash_attn)}",
        f"wandb.name={name}",
        f"wandb.group={sweep['group']}",
        f"wandb.tags=[stage-base,{base_facet}{common}]",
        f"mlflow.enabled={_bool_str(cfg.mlflow_enabled)}",
        f"mlflow.experiment_name={cfg.mlflow_experiment_name}",
        f"mlflow.run_name={name}",
        f"mlflow.log_model={cfg.mlflow_log_model}",
        _mlflow_parent_override(parent_run_id),
        f"+mlflow.tags.sweep_id={sweep['group_name']}",
        f"+mlflow.tags.sweep_timestamp={sweep['timestamp']}",
        f"+mlflow.tags.tokenizer_name={cfg.tokenizer_name}",
        f"+mlflow.tags.tokenizer_kind={cfg.tokenizer_kind}",
        "+mlflow.tags.stage=base",
        *train_tags,
        f"+mlflow.tags.seed={seed}",
        *cfg.extra,
    ]
    return argv


def _ft_argv(
    cfg, train_session, other_session, ft_mode, n_trials, seed, run_dir, base_ckpt, parent_run_id, sweep
) -> list[str]:
    base_name = _base_run_name(cfg, train_session, seed)
    trials_tag = f"ft_trials_{n_trials:03d}"
    name = _short_id(f"{base_name}__{ft_mode}__{other_session}__{trials_tag}", 120)
    freeze_mode = "none" if ft_mode == "full" else ft_mode
    eval_only = n_trials == 0
    max_steps = "1" if eval_only else cfg.ft_max_steps
    eval_every = "1" if eval_only else cfg.ft_eval_every
    ckpt_every = "1" if eval_only else cfg.ft_ckpt_every
    common = _common_wandb_tags(cfg, train_session, seed)
    # A pooled base has no single train session — drop the trainsess wandb facet
    # and the +mlflow.tags.train_session tag (the multi base carries pool tags).
    train_facet = "" if cfg.multi_session else f"trainsess-{train_session},"
    train_tags = [] if cfg.multi_session else [f"+mlflow.tags.train_session={train_session}"]
    argv = [
        f"model={cfg.model_cfg}",
        f"session={other_session}",
        f"data_root={cfg.data_root}",
        f"log_dir={run_dir}",
        *_tokenizer_args(cfg.tokenizer_kind, cfg.ckpt, cfg.ue_mode, cfg.sentinels, spike_times_sentinels=True),
    ]
    if not cfg.no_crossattn:
        argv.append(f"model.num_latents_per_step={cfg.num_latents_per_step}")
    argv += [
        f"temporal_split.enabled={cfg.temporal_split_enabled}",
        f"max_steps={max_steps}",
        f"eval_every_n_steps={eval_every}",
        f"checkpoint_every_n_steps={ckpt_every}",
        f"batch_size={cfg.batch_size}",
        f"seed={seed}",
        f"num_workers={cfg.num_workers}",
        f"precision={cfg.precision}",
        f"force_flash_attn={_bool_str(cfg.force_flash_attn)}",
        f"wandb.name={name}",
        f"wandb.group={sweep['group']}",
        f"wandb.tags=[stage-ft,ftmode-{ft_mode},fttrials-{n_trials},calibsess-{other_session},{train_facet}{common}]",
        f"mlflow.enabled={_bool_str(cfg.mlflow_enabled)}",
        f"mlflow.experiment_name={cfg.mlflow_experiment_name}",
        f"mlflow.run_name={name}",
        f"mlflow.log_model={cfg.mlflow_log_model}",
        _mlflow_parent_override(parent_run_id),
        f"+mlflow.tags.sweep_id={sweep['group_name']}",
        f"+mlflow.tags.sweep_timestamp={sweep['timestamp']}",
        f"+mlflow.tags.tokenizer_name={cfg.tokenizer_name}",
        f"+mlflow.tags.tokenizer_kind={cfg.tokenizer_kind}",
        "+mlflow.tags.stage=ft",
        f"+mlflow.tags.ft_mode={ft_mode}",
        f"+mlflow.tags.ft_trials={n_trials}",
        *train_tags,
        f"+mlflow.tags.calib_session={other_session}",
        f"+mlflow.tags.seed={seed}",
        f"train_decimation.trials={n_trials}",
        f"train_decimation.seed={seed}",
        f"init_ckpt={base_ckpt}",
        f"freeze_mode={freeze_mode}",
        f"eval_only={_bool_str(eval_only)}",
        *cfg.extra,
    ]
    return argv


def run_calibrate(cfg: CalibrateConfig) -> int:
    for m in cfg.ft_modes:
        if m not in ("unit_identification", "full"):
            raise ValueError(f"unknown FT mode: {m!r}")
    # Cheap validations up front, before any tracking-server side effect, so a
    # misuse never leaks an orphan MLflow parent run.
    if cfg.multi_session:
        if len(cfg.train_sessions) < 2:
            raise ValueError("multi-session mode requires --train-sessions to have >=2 entries.")
    elif cfg.skip_base_train and len(cfg.train_sessions) > 1:
        raise ValueError(
            "skip_base_train is only valid with a single --train-sessions entry "
            "(one base ckpt can't serve multiple train sessions)."
        )

    session_trials = _preflight_trials(cfg)
    sweep = {
        "timestamp": _utc_stamp(),
        "tokenizer_name": cfg.tokenizer_name,
    }
    suffix = "_multi" if cfg.multi_session else ""
    sweep["group_name"] = f"{cfg.tokenizer_name}{suffix}__{sweep['timestamp']}"
    sweep["group"] = _short_id(sweep["group_name"], 120)
    parent_run_id, parent_uri = _mlflow_create_parent(
        cfg.repo_root,
        experiment_name=cfg.mlflow_experiment_name,
        run_name=sweep["group_name"],
        tags={
            "sweep_id": sweep["group_name"],
            "sweep_timestamp": sweep["timestamp"],
            "tokenizer_name": cfg.tokenizer_name,
            "tokenizer_kind": cfg.tokenizer_kind,
            "orchestrator": "multisession" if cfg.multi_session else "singlesession",
            "model_cfg": cfg.model_cfg,
        },
        disabled=not cfg.mlflow_enabled,
        dry_run=cfg.dry_run,
    )

    log.info(
        "calibrate kind=%s tokenizer=%s model=%s sweep=%s parent=%s",
        cfg.tokenizer_kind,
        cfg.tokenizer_name,
        cfg.model_cfg,
        sweep["group_name"],
        parent_uri or "(none)",
    )

    # Multi-session base-trains the whole pool once per seed (queue = seeds);
    # single-session is the (train_session, seed) product.
    jobs = (
        [(None, sd) for sd in cfg.seeds]
        if cfg.multi_session
        else [(ts, sd) for ts in cfg.train_sessions for sd in cfg.seeds]
    )

    def run_tuple(job, gpu: str) -> int:
        train_session, seed = job
        if cfg.multi_session:
            pool = set(cfg.train_sessions)
            others = [s for s in cfg.other_sessions if s not in pool]
        else:
            others = [s for s in cfg.other_sessions if s != train_session]
        base_run = _base_run_name(cfg, train_session, seed)
        base_log_dir = cfg.repo_root / "outputs" / base_run
        exp_root = Path(cfg.calib_root) / base_run
        if not cfg.dry_run:
            exp_root.mkdir(parents=True, exist_ok=True)

        # --- base ckpt: skip / reuse-if-exists / train ---
        base_steps = int(cfg.base_max_steps)
        placeholder = str(base_log_dir / "csv_logs" / "version_0" / "checkpoints" / "last.ckpt")
        reuse = _latest_base_ckpt(base_log_dir, base_steps) if (cfg.reuse_base_if_exists and not cfg.dry_run) else None
        if cfg.skip_base_train:
            if not cfg.base_ckpt:
                raise ValueError("skip_base_train requires --base-ckpt")
            base_ckpt = cfg.base_ckpt
            # The supplied base ckpt must exist and be
            # fully trained (global_step >= base_max_steps); refuse a missing or
            # partial ckpt rather than silently fine-tuning from it.
            if not cfg.dry_run:
                if not Path(base_ckpt).is_file():
                    raise FileNotFoundError(f"skip_base_train: base ckpt not found: {base_ckpt}")
                step = _ckpt_global_step(Path(base_ckpt))
                if step is None or step < base_steps:
                    raise ValueError(
                        f"skip_base_train: base ckpt {base_ckpt} is partial/unreadable "
                        f"(global_step={step} < base_max_steps={base_steps}); refusing."
                    )
            log.info("[base] skip; reusing %s", base_ckpt)
        elif reuse is not None:
            base_ckpt = reuse
            log.info("[base] reusing existing ckpt %s", base_ckpt)
        else:
            rc = _run_train(
                _base_argv(cfg, train_session, seed, str(base_log_dir), parent_run_id, sweep),
                gpu=gpu,
                log_path=base_log_dir / "base_train.log",
                dry_run=cfg.dry_run,
                seed=seed,
                cublas_workspace_config=cfg.cublas_workspace_config,
                repo_root=cfg.repo_root,
            )
            if cfg.dry_run:
                base_ckpt = placeholder
            elif rc != 0:
                log.error("base train failed: %s", base_run)
                return rc
            else:
                resolved = _latest_base_ckpt(base_log_dir, base_steps)
                if resolved is None:
                    log.error("base ckpt missing/partial under %s (interrupted?)", base_log_dir)
                    return 1
                base_ckpt = resolved

        # --- calibration sweep (missing-only skip + plain-eval mirror) ---
        plain_done: dict[str, Path] = {}
        failed = 0
        for ft_mode in cfg.ft_modes:
            for other in others:
                n_valid = session_trials.get(other, 0)
                threshold = int(0.7 * n_valid)
                eligible = set(_eligible_trials(n_valid, cfg.ft_trial_counts))
                for n_trials in [0, *cfg.ft_trial_counts]:
                    if n_trials not in eligible:
                        log.info(
                            "skip %s/%s/%s n=%d (>floor(0.7*%d)=%d)",
                            base_run,
                            ft_mode,
                            other,
                            n_trials,
                            n_valid,
                            threshold,
                        )
                        continue
                    run_dir = exp_root / ft_mode / other / f"ft_trials_{n_trials:03d}"
                    if cfg.missing_only and not cfg.dry_run and _latest_run_result_complete(run_dir):
                        log.info("[%s] %s n=%d already complete; skipping", ft_mode, other, n_trials)
                        if n_trials == 0:
                            plain_done.setdefault(other, run_dir)
                        continue
                    if (
                        n_trials == 0
                        and other in plain_done
                        and not cfg.dry_run
                        and _mirror_plain_eval(plain_done[other], run_dir)
                    ):
                        log.info("[%s] %s n=0 (mirrored)", ft_mode, other)
                        continue
                    rc = _run_train(
                        _ft_argv(
                            cfg,
                            train_session,
                            other,
                            ft_mode,
                            n_trials,
                            seed,
                            str(run_dir),
                            base_ckpt,
                            parent_run_id,
                            sweep,
                        ),
                        gpu=gpu,
                        log_path=run_dir / "run.log",
                        dry_run=cfg.dry_run,
                        seed=seed,
                        cublas_workspace_config=cfg.cublas_workspace_config,
                        repo_root=cfg.repo_root,
                    )
                    if rc != 0:
                        failed = 1
                    elif n_trials == 0:
                        plain_done.setdefault(other, run_dir)
        if not cfg.dry_run:
            for ft_mode in cfg.ft_modes:
                calibration_csv.build_csv(
                    experiments_root=exp_root,
                    ft_mode=ft_mode,
                    ft_trials=[0, *cfg.ft_trial_counts],
                    other_sessions=others,
                )
        return failed

    if cfg.dry_run:
        unit = "seed" if cfg.multi_session else "(train_session, seed)"
        log.info("queue: %d %s jobs on workers %s", len(jobs), unit, _worker_tags(cfg.gpus, cfg.jobs_per_gpu))
        for job in jobs:
            run_tuple(job, cfg.gpus[0])
        return 0

    rc = _run_pool(jobs, _worker_tags(cfg.gpus, cfg.jobs_per_gpu), run_tuple)
    _mlflow_aggregate(cfg.repo_root, parent_run_id, dry_run=cfg.dry_run)
    # Upload each base run's CSVs + per-cell results records onto the parent.
    exp_roots = [Path(cfg.calib_root) / _base_run_name(cfg, ts, sd) for ts, sd in jobs]
    _mlflow_log_artifacts(cfg.repo_root, parent_run_id, exp_roots, dry_run=cfg.dry_run)
    return rc


# ─────────────────────────── benchmark driver ──────────────────────────────


def _discover_tokenizers(
    ckpt_root: Path, explicit: list[str], limit: int, *, dry_run: bool = False
) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if explicit:
        for name in explicit:
            ckpt = ckpt_root / name / "checkpoints" / "model.ckpt"
            if not ckpt.is_file():
                if dry_run:
                    log.warning("[dry-run] tokenizer ckpt missing for %r: %s", name, ckpt)
                else:
                    raise FileNotFoundError(f"tokenizer ckpt missing for {name!r}: {ckpt}")
            found.append((name, str(ckpt)))
    elif not ckpt_root.is_dir():
        if dry_run:
            log.warning("[dry-run] CKPT_ROOT does not exist: %s", ckpt_root)
        else:
            raise FileNotFoundError(f"CKPT_ROOT does not exist: {ckpt_root}")
    else:
        for name in sorted(p.name for p in ckpt_root.iterdir() if p.is_dir()):
            ckpt = ckpt_root / name / "checkpoints" / "model.ckpt"
            if ckpt.is_file():
                found.append((name, str(ckpt)))
    if limit > 0:
        found = found[:limit]
    return found


def run_benchmark(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path.cwd()
    dcfg = load_yaml(repo_root / args.data_config)
    data_root = args.data_root or str(dcfg["data_root"])
    sessions = _as_list(args.sessions) or [str(s) for s in (dcfg.get("other_sessions") or [])]
    if not sessions:
        raise ValueError(f"no sessions resolved (data-config={args.data_config})")

    ckpt_root = Path(args.ckpt_root) if args.ckpt_root else repo_root / "results" / "tokenizer_checkpoints"
    comparison_tag = args.comparison_tag or f"tok_bench_{datetime.now(UTC).strftime('%Y%m%d')}"
    # Baseline mode: spike_times only,
    # no frozen tokenizers, and the crossattn poyo_1.3M default — vs the
    # no_crossattn default the frozen-tokenizer benchmark uses, since frozen
    # tokenizers require no_crossattn. Both still overridable via --model-cfg.
    spike_times_only = _parse_bool(args.spike_times_only, "spike-times-only")
    include_spike_times = spike_times_only or _parse_bool(args.include_spike_times, "include-spike-times")
    model_cfg = args.model_cfg or ("poyo_1.3M" if spike_times_only else "poyo_1.3M_no_crossattn")
    ue_mode = args.unit_embedding_mode or "add"
    sentinels = _bool_str(_parse_bool(args.sentinels if args.sentinels is not None else "false", "sentinels"))
    epochs = str(args.epochs if args.epochs is not None else 400)
    batch_size = str(args.batch_size if args.batch_size is not None else 128)
    seed = str(args.seed if args.seed is not None else 42)
    num_workers = str(args.num_workers if args.num_workers is not None else 8)
    mlflow_enabled = _resolve_mlflow_enabled(args.mlflow)
    mlflow_experiment = args.mlflow_experiment_name or "ntt-eval-poyo"
    mlflow_log_model = _bool_str(
        _parse_bool(
            args.mlflow_log_model if args.mlflow_log_model is not None else "false",
            "mlflow-log-model",
        )
    )
    extra = shlex.split(args.extra or "")
    gpus = _resolve_gpus(args.gpus)
    timestamp = _utc_stamp()
    exp_root = repo_root / "experiments_benchmark" / comparison_tag

    # Baseline mode skips frozen-tokenizer discovery entirely (no CKPT_ROOT needed).
    tokenizers = (
        []
        if spike_times_only
        else _discover_tokenizers(
            ckpt_root, _as_list(args.tokenizers), int(args.tokenizers_limit or 0), dry_run=args.dry_run
        )
    )
    if not tokenizers and not include_spike_times:
        raise ValueError("no tokenizers discovered and --no-include-spike-times")

    # Per-tokenizer MLflow parent (spike_times first).
    cohort: list[tuple[str, str, str]] = []  # (name, kind, ckpt)
    if include_spike_times:
        cohort.append(("spike_times", "spike_times", ""))
    cohort += [(name, "frozen", ckpt) for name, ckpt in tokenizers]

    parents: dict[str, str] = {}
    groups: dict[str, str] = {}
    for name, kind, _ in cohort:
        group_name = f"{name}__{timestamp}"
        groups[name] = _short_id(group_name, 120)
        pid, _uri = _mlflow_create_parent(
            repo_root,
            experiment_name=mlflow_experiment,
            run_name=group_name,
            tags={
                "sweep_id": comparison_tag,
                "sweep_timestamp": timestamp,
                "tokenizer_name": name,
                "tokenizer_kind": kind,
                "model_cfg": model_cfg,
                "seed": seed,
                "orchestrator": "baseline_benchmark" if spike_times_only else "tokenizer_benchmark",
            },
            disabled=not mlflow_enabled,
            dry_run=args.dry_run,
        )
        parents[name] = pid

    jobs = [(name, kind, ckpt, session) for (name, kind, ckpt) in cohort for session in sessions]
    log.info(
        "benchmark tag=%s tokenizers=%d sessions=%d jobs=%d gpus=%s",
        comparison_tag,
        len(tokenizers),
        len(sessions),
        len(jobs),
        gpus,
    )

    def run_cell(job, gpu: str) -> int:
        name, kind, ckpt, session = job
        run_name = f"{comparison_tag}__{name}__{session}"
        run_dir = exp_root / run_name
        wandb_name = _short_id(run_name, 120)
        wandb_tags = ",".join(
            _short_id(t, 64)
            for t in [
                "bench",
                f"cmp-{comparison_tag}",
                f"sweep-{timestamp}",
                f"tok-{name}",
                f"kind-{kind}",
                f"sess-{session}",
                f"arch-{model_cfg}",
                f"seed-{seed}",
            ]
        )
        argv = [
            f"model={model_cfg}",
            f"session={session}",
            f"data_root={data_root}",
            f"log_dir={run_dir}",
            *_tokenizer_args(kind, ckpt, ue_mode, sentinels),
            f"epochs={epochs}",
            f"batch_size={batch_size}",
            f"seed={seed}",
            f"num_workers={num_workers}",
            "precision=bf16-mixed",
            "force_flash_attn=true",
            f"wandb.name={wandb_name}",
            f"wandb.group={groups[name]}",
            f"wandb.tags=[{wandb_tags}]",
            f"mlflow.enabled={_bool_str(mlflow_enabled)}",
            f"mlflow.experiment_name={mlflow_experiment}",
            f"mlflow.run_name={wandb_name}",
            f"mlflow.log_model={mlflow_log_model}",
            _mlflow_parent_override(parents.get(name, "")),
            f"+mlflow.tags.sweep_id={comparison_tag}",
            f"+mlflow.tags.sweep_timestamp={timestamp}",
            f"+mlflow.tags.tokenizer_name={name}",
            f"+mlflow.tags.tokenizer_kind={kind}",
            f"+mlflow.tags.session={session}",
            *extra,
        ]
        return _run_train(
            argv,
            gpu=gpu,
            log_path=run_dir / "run.log",
            dry_run=args.dry_run,
            seed=int(seed),
            cublas_workspace_config=_DEFAULT_CUBLAS_WORKSPACE_CONFIG,
            repo_root=repo_root,
        )

    if args.dry_run:
        log.info("queue: %d (tokenizer, session) cells; output -> %s", len(jobs), exp_root)
        for job in jobs:
            run_cell(job, gpus[0])
        return 0

    # Records written before this point belong to an earlier attempt with the
    # same tag; the cohort must not count them (see cohort_results._is_stale).
    sweep_started = cohort_results.now_iso()
    rc = _run_pool(jobs, _worker_tags(gpus, 1), run_cell)
    # One cohort record per cohort member (not per MLflow parent), so offline /
    # MLflow-disabled sweeps get it too. It is what ntt's aggregation reads for
    # this tokenizer; the per-cell records are uploaded alongside for drill-down.
    for name, _kind, ckpt in cohort:
        child_dirs = {s: exp_root / f"{comparison_tag}__{name}__{s}" for s in sessions}
        cohort_dir = exp_root / f"{comparison_tag}__{name}{cohort_results.COHORT_DIRNAME_SUFFIX}"
        cohort_path = _write_benchmark_cohort(
            name, ckpt, parents.get(name), child_dirs, cohort_dir, not_before=sweep_started
        )
        if parents.get(name):
            _mlflow_aggregate(repo_root, parents[name], dry_run=args.dry_run)
            _mlflow_log_artifacts(
                repo_root,
                parents[name],
                list(child_dirs.values()),
                dry_run=args.dry_run,
                files=[cohort_path] if cohort_path else [],
            )
    return rc


def _write_benchmark_cohort(
    name: str,
    ckpt: str,
    parent_run_id: str | None,
    child_dirs: dict[str, Path],
    out_dir: Path,
    *,
    not_before: str | None = None,
) -> Path | None:
    """Build and write one tokenizer's cohort record; never fails the sweep."""
    try:
        doc = cohort_results.build_cohort_document(
            tokenizer_name=name,
            tokenizer_checkpoint=ckpt or None,  # spike-times baseline has no checkpoint
            parent_run_id=parent_run_id or None,
            child_dirs=child_dirs,
            not_before=not_before,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("cohort record for %s could not be built: %s", name, exc)
        return None
    path = cohort_results.write_cohort_record(doc, out_dir)
    cohort = doc.metadata.extra["cohort"]
    if cohort["missing"]:
        log.warning("cohort %s incomplete: missing sessions %s", name, ", ".join(cohort["missing"]))
    return path


# ─────────────────────────────────── CLI ───────────────────────────────────


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--repo-root", default=None, help="repo root (default: cwd)")
    p.add_argument("--data-root", default=None)
    p.add_argument("--gpus", default=None, help="space-separated GPU indices (default: nvidia-smi)")
    p.add_argument("--mlflow", default=None, help="auto|true|false (auto: on iff MLFLOW_TRACKING_URI is set)")
    p.add_argument("--mlflow-experiment-name", default=None)
    p.add_argument("--extra", default=None, help="raw Hydra overrides appended to every train.py call")
    p.add_argument("--dry-run", action="store_true", help="resolve + print the plan; do not execute")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ntt_downstream.decoding.orchestration", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="single-session POYO calibration sweep")
    _add_common(c)
    c.add_argument("--training-config", default=None, help="default: per-mode tokenizer_training.yaml")
    c.add_argument("--data-config", default=None, help="default: per-mode data.yaml")
    c.add_argument(
        "--multi-session",
        action="store_true",
        help="multi-session mode: joint base-train over the whole train pool (queue = seeds)",
    )
    c.add_argument(
        "--hydra-data-config",
        default="calibration_multi_base",
        help="Hydra data preset wired with the train pool for the multi-session base train",
    )
    c.add_argument("--tokenizer-kind", default=None, choices=[None, "frozen", "spike_times"])
    c.add_argument("--ckpt", default=None)
    c.add_argument("--tokenizer-name", default=None)
    c.add_argument("--train-sessions", default=None)
    c.add_argument("--other-sessions", default=None)
    c.add_argument("--ft-trial-counts", default=None)
    c.add_argument("--ft-modes", default=None)
    c.add_argument("--seeds", default=None)
    c.add_argument("--model-cfg", dest="model_cfg", default=None, help="MODEL_SIZE: 1.3M|11.8M")
    c.add_argument("--no-crossattn", default=None, help="true|false")
    c.add_argument("--unit-embedding-mode", default=None)
    c.add_argument("--sentinels", default=None)
    c.add_argument("--batch-size", default=None)
    c.add_argument("--num-workers", default=None)
    c.add_argument("--base-max-steps", default=None)
    c.add_argument("--ft-max-steps", default=None)
    # YAML-default-with-CLI-override knobs:
    # unset -> resolved from the training/data YAML; passed -> override at run time.
    c.add_argument("--num-latents-per-step", default=None)
    c.add_argument("--eval-every-n-steps", default=None)
    c.add_argument("--ckpt-every-n-steps", default=None)
    c.add_argument("--ft-eval-every", default=None)
    c.add_argument("--ft-ckpt-every", default=None)
    c.add_argument("--precision", default=None)
    c.add_argument("--force-flash-attn", default=None, help="true|false")
    c.add_argument("--temporal-split-enabled", default=None, help="true|false")
    c.add_argument("--brainset", default=None)
    c.add_argument("--mlflow-log-model", default=None, help="true|false")
    c.add_argument("--calib-root", default=None)
    c.add_argument("--jobs-per-gpu", type=int, default=1)
    c.add_argument("--reuse-base-if-exists", default="true")
    c.add_argument("--missing-only", default="true")
    c.add_argument("--skip-base-train", default="false")
    c.add_argument("--base-ckpt", default=None)
    c.set_defaults(func=lambda a: run_calibrate(_resolve_calibrate(a)))

    b = sub.add_parser("benchmark", help="multi-tokenizer POYO benchmark")
    _add_common(b)
    b.add_argument("--data-config", default="configs/decoding/single_session/data_benchmark.yaml")
    b.add_argument("--ckpt-root", default=None)
    b.add_argument("--tokenizers", default=None, help="space-separated names; else auto-discover")
    b.add_argument("--tokenizers-limit", type=int, default=0)
    b.add_argument("--include-spike-times", default="false")
    b.add_argument(
        "--spike-times-only",
        default="false",
        help="baseline mode: spike_times only, skip frozen-tokenizer discovery "
        "(implies --include-spike-times; defaults --model-cfg to poyo_1.3M)",
    )
    b.add_argument("--sessions", default=None)
    b.add_argument("--model-cfg", dest="model_cfg", default=None)
    b.add_argument("--unit-embedding-mode", default=None)
    b.add_argument("--sentinels", default=None)
    b.add_argument("--epochs", type=int, default=None)
    b.add_argument("--batch-size", type=int, default=None)
    b.add_argument("--seed", type=int, default=None)
    b.add_argument("--num-workers", type=int, default=None)
    b.add_argument("--comparison-tag", default=None)
    b.add_argument("--mlflow-log-model", default=None)
    b.set_defaults(func=run_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
