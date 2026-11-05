"""JSON serialization helpers for downstream results files.

Follows ntt's JSON conventions so that downstream results files are
RFC-8259 conformant (NaN/Inf -> null) with strict numpy/torch/OmegaConf handling
(unknown types raise rather than silently stringify).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf


def json_default(obj: Any) -> Any:
    """``default`` for ``json.dump``: numpy/torch/OmegaConf -> native types.

    Raises ``TypeError`` for unknown types instead of stringifying.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (DictConfig, ListConfig)):
        return OmegaConf.to_container(obj, resolve=True)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def replace_non_finite_floats(obj: Any) -> Any:
    """Recursively replace NaN/+Inf/-Inf python floats with ``None``.

    Apply immediately before ``json.dump`` so the written file is valid per
    RFC 8259 (CPython otherwise emits the bare tokens ``NaN``/``Infinity``).
    """
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: replace_non_finite_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [replace_non_finite_floats(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(replace_non_finite_floats(v) for v in obj)
    return obj


def sanitize_keys(obj: Any) -> Any:
    """Recursively convert non-JSON-scalar dict keys to strings."""
    if isinstance(obj, dict):
        return {
            (k if isinstance(k, (str, int, float, bool, type(None))) else str(k)): sanitize_keys(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [sanitize_keys(v) for v in obj]
    return obj


def write_json(path: str | Path, obj: Any) -> Path:
    """Serialize ``obj`` to ``path`` as RFC-8259-conformant JSON; return the path.

    Creates parent directories. Keys are sanitized and non-finite floats are
    nulled before dumping.

    Non-finite NATIVE python floats are nulled for RFC-8259 conformance; callers
    must pass values already coerced to native python types -- the results schema
    does this. numpy/torch scalars are converted by ``json_default`` at dump time
    and are not non-finite-checked.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = replace_non_finite_floats(sanitize_keys(obj))
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=json_default)
    return path
