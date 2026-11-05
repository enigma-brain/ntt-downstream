"""Parse decoding (POYO) ``callback_metrics`` snapshots into a shared ``MetricBlock``.

Decoding metrics: per-session R² (keyed by bare ``recording_id`` ``brainset/session``),
``average_{val,test}_metric`` aggregate, and tiered means logged by ``TierAggregator``
as ``<prefix>/<tier>_r2`` and ``<prefix>/<tier>_r2_<brainset_short>``. Train-loss and
grad-norm diagnostics are attached only to the ``val`` (last) block.
"""

from __future__ import annotations

import math
import statistics

from ntt_downstream.common.results.schema import MetricBlock, MetricEntry
from ntt_downstream.decoding.mlflow_utils import _aggregate_values

_R2_RANGE = (None, 1.0)
_RESERVED_PREFIXES = ("val/", "test/", "train/", "train_", "system/")


def _finite_or_none(value) -> float | None:
    """``value`` as a float if finite, else ``None`` (serialized as JSON null)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def parse_decoding_block(
    metrics: dict[str, float],
    *,
    session_ids: list[str] | None = None,
    prefix: str = "val",
) -> MetricBlock:
    """``prefix`` is ``"val"`` or ``"test"``; per-session keys are bare recording ids."""
    block = MetricBlock()

    # --- per session (built first so the aggregate can summarise it) ---
    sess_keys = session_ids if session_ids is not None else _infer_session_ids(metrics)
    per_session: dict[str, dict[str, float | None]] = {}
    for sess in sess_keys:
        if sess in metrics:
            per_session[sess] = {"r2": _finite_or_none(metrics[sess])}
    block.per_session = per_session

    # --- aggregate: value = authoritative logged mean (fallback = computed
    # mean over per-session R²s); median/std/n summarise across sessions.
    # Only finite R²s count: one diverged session must neither crash the
    # record (statistics.stdev raises on NaN) nor poison median/n. ---
    avg_key = f"average_{prefix}_metric"
    session_r2s = [row["r2"] for row in per_session.values() if row["r2"] is not None]
    if avg_key in metrics or session_r2s:
        entry = MetricEntry(direction="higher", value_range=_R2_RANGE, value=None)
        if session_r2s:
            mean, median, n = _aggregate_values(session_r2s)
            entry.median = median
            entry.n = n
            entry.std = statistics.stdev(session_r2s) if n > 1 else None
            entry.aggregated_over = "sessions:mean"
            entry.value = metrics[avg_key] if avg_key in metrics else mean
        else:
            entry.value = metrics[avg_key]
        block.aggregated["r2"] = entry

    # --- tiers ---
    tiers = _parse_tiers(metrics, prefix)
    if tiers:
        block.groups["tiers"] = tiers

    # --- train diagnostics (val/last block only) ---
    if prefix == "val":
        train: dict[str, float] = {}
        for out_name, key in (
            ("loss", "train_loss"),
            ("grad_norm_total", "train/grad_norm_total"),
        ):
            if key in metrics:
                train[out_name] = metrics[key]
        if train:
            block.groups["train"] = train

    return block


def _parse_tiers(metrics: dict[str, float], prefix: str) -> dict[str, dict]:
    """Parse ``<prefix>/<tier>_r2`` and ``..._r2_<sub>`` into nested tier dicts."""
    tier_prefix = f"{prefix}/"
    suffix = "_r2"
    tiers: dict[str, dict] = {}
    for key, val in metrics.items():
        if not key.startswith(tier_prefix):
            continue
        rest = key[len(tier_prefix) :]  # e.g. "tier1_val_r2" or "tier1_val_r2_odoherty"
        idx = rest.find(suffix)
        if idx == -1:
            continue
        tier_name = rest[:idx]  # "tier1_val"
        after = rest[idx + len(suffix) :]  # "" or "_odoherty"
        node = tiers.setdefault(tier_name, {})
        if after == "":
            node["r2"] = val
        elif after.startswith("_"):
            node.setdefault("by_brainset", {})[after[1:]] = val
    return tiers


def _infer_session_ids(metrics: dict[str, float]) -> list[str]:
    """Heuristic: bare ``brainset/session`` keys (contain '/', not a reserved prefix)."""
    out = [key for key in metrics if "/" in key and not key.startswith(_RESERVED_PREFIXES)]
    return sorted(out)
