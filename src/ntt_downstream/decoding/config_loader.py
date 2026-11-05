"""Load + merge the decoding calibration/training YAML configs.

Pure, side-effect-free helpers used by the orchestration engine
(:mod:`ntt_downstream.decoding.orchestration`); importable in-process and
unit-testable (no subprocess).

``TRAINING_KEYS`` / ``DATA_KEYS`` are the contract between the YAML files and the
orchestrator: a whitelist of dotted YAML paths and the orchestrator variable
each maps to. Listing them explicitly (rather than walking the YAML blindly)
keeps the contract readable and makes a typo in the YAML loud at load time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

TRAINING_KEYS: dict[str, str] = {
    "model.size": "MODEL_SIZE",
    "model.no_crossattn": "NO_CROSSATTN",
    "model.num_latents_per_step": "NUM_LATENTS_PER_STEP",
    "training.ft_max_steps": "FT_MAX_STEPS",
    "training.base_max_steps": "BASE_MAX_STEPS",
    "training.ft_eval_every": "FT_EVAL_EVERY",
    "training.ft_ckpt_every": "FT_CKPT_EVERY",
    "training.eval_every_n_steps": "EVAL_EVERY_N_STEPS",
    "training.ckpt_every_n_steps": "CKPT_EVERY_N_STEPS",
    "training.ft_trial_counts": "FT_TRIAL_COUNTS",
    "training.ft_modes": "FT_MODES",
    "training.seeds": "SEEDS",
    "runtime.batch_size": "BATCH_SIZE",
    "runtime.num_workers": "NUM_WORKERS",
    "runtime.precision": "PRECISION",
    "runtime.force_flash_attn": "FORCE_FLASH_ATTN",
    "runtime.cublas_workspace_config": "CUBLAS_WORKSPACE_CONFIG",
    "runtime.calib_root": "CALIB_ROOT",
    "tokenizer.kind": "TOKENIZER_KIND",
    "tokenizer.unit_embedding_mode": "UE_MODE",
    "tokenizer.sentinels": "TOKENIZER_SENTINELS",
    "mlflow.enabled": "MLFLOW_ENABLED",
    "mlflow.experiment_name": "MLFLOW_EXPERIMENT_NAME",
    "mlflow.log_model": "MLFLOW_LOG_MODEL",
}

DATA_KEYS: dict[str, str] = {
    "data_root": "DATA_ROOT",
    "brainset": "BRAINSET",
    "temporal_split_enabled": "TEMPORAL_SPLIT_ENABLED",
    "train_sessions": "TRAIN_SESSIONS",
    "other_sessions": "OTHER_SESSIONS",
}


def walk(cfg: Any, dotted: str) -> Any:
    """Navigate ``cfg`` by a dotted key path; raise ``KeyError`` if absent."""
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"missing key '{dotted}' in YAML")
        cur = cur[part]
    return cur


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``; override wins on conflicts.

    Lists are replaced wholesale (no element-wise merge) — variant YAMLs are
    expected to override list-valued keys like ``training.seeds`` in full.
    """
    out: dict[str, Any] = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML and recursively resolve any ``extends:`` parent chain.

    The optional top-level ``extends:`` key holds a path (relative to the
    current YAML's directory) to a parent YAML whose values are inherited and
    then overridden by the current file. This lets variant configs (e.g.
    ``baseline_training.yaml``, ``tokenizer_training.yaml``) share a common base
    (``training.yaml``) without duplicating shared hparams.
    """
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"top-level YAML in {path} must be a mapping")
    parent_ref = cfg.pop("extends", None)
    if parent_ref is not None:
        parent_path = (path.parent / parent_ref).resolve()
        parent_cfg = load_yaml(parent_path)
        cfg = deep_merge(parent_cfg, cfg)
    return cfg
