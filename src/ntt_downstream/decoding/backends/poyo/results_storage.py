"""Structured results storage for POYO training runs.

Writes the flat per-run summary, ``run_result.json``: the local per-run
summary used by sweep resume (completion checks, plain-eval mirror) and the
calibration CSV. It is never uploaded; consumers read the v2 record
``downstream_decoding_results.json`` (``common/results``, docs/results.md).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_run_result(
    *,
    cfg: dict[str, Any],
    test_r2: float,
    val_best_r2: float | None,
    model_params: int,
    token_dims_per_sec: float | None,
    log_dir: str,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build a run result dict from config and metrics."""
    tokenizer_cfg = cfg.get("tokenizer", {})
    model_cfg = cfg.get("model", {})

    # Determine human-readable tokenizer type.
    mode = tokenizer_cfg.get("mode", "unknown")
    ntt_tok = tokenizer_cfg.get("ntt_tokenizer")
    if mode == "fake":
        tokenizer_type = "fake"
    elif ntt_tok == "frozen":
        tokenizer_type = "frozen"
    elif ntt_tok == "spike_times":
        tokenizer_type = "spike_times"
    elif ntt_tok == "poisson":
        tokenizer_type = "poisson"
    else:
        tokenizer_type = f"{mode}/{ntt_tok}"

    return {
        "timestamp": timestamp or datetime.now().isoformat(timespec="seconds"),
        "session": cfg.get("session", "unknown"),
        "tokenizer_type": tokenizer_type,
        "tokenizer_checkpoint": tokenizer_cfg.get("ntt_checkpoint"),
        "unit_embedding_mode": tokenizer_cfg.get("unit_embedding_mode", "unknown"),
        "sentinels": tokenizer_cfg.get("sentinels", False),
        "unit_dropout": cfg.get("unit_dropout", False),
        "model_depth": model_cfg.get("depth"),
        "model_dim": model_cfg.get("dim"),
        "model_params": model_params,
        "token_dims_per_sec": token_dims_per_sec,
        "epochs": cfg.get("epochs"),
        "max_steps": cfg.get("max_steps"),
        "test_r2": test_r2,
        "val_best_r2": val_best_r2,
        "log_dir": log_dir,
    }


def save_run_result(result: dict[str, Any], *, log_dir: str | Path) -> Path:
    """Write ``result`` (from ``build_run_result``) to ``<log_dir>/run_result.json``.

    Returns the path written.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_json_path = log_dir / "run_result.json"
    with open(run_json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Saved run_result.json to {run_json_path}")
    return run_json_path
