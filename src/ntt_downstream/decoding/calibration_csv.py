"""Aggregate per-FT-run ``val_best_r2`` / ``test_r2`` / ``kept_trials`` into
calibration CSVs.

Pure, side-effect-free (filesystem-only) logic used by the orchestration
engine (:mod:`ntt_downstream.decoding.orchestration`); importable in-process and
unit-testable.

Walks an experiments tree of the form::

    <experiments_root>/<ft_mode>/<other_session>/ft_trials_<NNN>/csv_logs/version_*/run_result.json

and produces, per FT mode, one CSV ``<experiments_root>/calibration_<ft_mode>.csv``
(long across ``metric``, wide across sessions). Intentionally tolerant: missing
files / fields become NaN cells (with a warning) rather than aborting, so
partial sweeps still produce a usable CSV.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# `kept_trials` is not an R²-metric — it's the actual train trial count after the
# eligibility filter clamped `requested_trials`. Emitted as a third "metric"
# block so the wide-by-session schema is unchanged.
_METRIC_FIELDS = ("val_best_r2", "test_r2", "kept_trials")


def _format_trials(n: int) -> str:
    """Match the trial-count directory naming (``ft_trials_086`` etc.)."""
    return f"ft_trials_{int(n):03d}"


def _read_metrics(run_dir: Path) -> dict[str, float | None]:
    """Return ``{val_best_r2, test_r2, kept_trials}`` from the newest
    ``csv_logs/version_*/run_result.json`` under ``run_dir``.

    Lightning's CSVLogger increments ``version_<N>`` per fresh fit; we pick the
    highest version present so re-runs don't read stale results. Returns
    ``{field: None, ...}`` when no run_result.json is found, or for any field
    that's missing/null/NaN in the JSON.
    """
    out: dict[str, float | None] = dict.fromkeys(_METRIC_FIELDS)
    csv_logs = run_dir / "csv_logs"
    if not csv_logs.is_dir():
        return out
    versions = sorted(
        (p for p in csv_logs.iterdir() if p.is_dir() and p.name.startswith("version_")),
        key=lambda p: int(p.name.split("_", 1)[1]),
        reverse=True,
    )
    for v in versions:
        rr = v / "run_result.json"
        if rr.is_file():
            try:
                with open(rr) as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    # Valid JSON but not an object (e.g. a bare list/number) —
                    # `.get()` below would AttributeError; treat as corrupt.
                    raise ValueError(f"run_result.json is not a JSON object: {type(data).__name__}")
            except (json.JSONDecodeError, OSError, ValueError) as exc:
                # A run killed mid-write can leave a truncated/garbage run_result.json.
                # Fall back to the next-older version_* rather than aborting the
                # whole base run's CSV (mirrors the scratch_scaling summary).
                logger.warning("corrupt/unreadable %s: %s — trying older version", rr, exc)
                continue
            for field in _METRIC_FIELDS:
                raw = data.get(field)
                if raw is None:
                    continue
                # `test_r2` is written as float("nan") for runs that never
                # reached trainer.test(); coerce to None so the cell renders
                # as NaN rather than the string "nan".
                try:
                    fv = float(raw)
                except (TypeError, ValueError):
                    continue
                if fv != fv:  # NaN
                    continue
                out[field] = fv
            return out
    return out


def build_csv(
    *,
    experiments_root: Path,
    ft_mode: str,
    ft_trials: list[int],
    other_sessions: list[str],
) -> None:
    """Write a single ``calibration_<ft_mode>.csv`` for one base run × one FT mode.

    Output schema is long across ``metric`` and wide across sessions::

        metric,<axis_name>,<sessA>,<sessB>,...
        val_best_r2,0,0.81,nan,...
        test_r2,0,0.79,nan,...

    The sweep axis is the absolute train-trial count (``ft_trials``); it becomes
    the second column and drives the per-run directory lookup
    (``ft_trials_<NNN>``).
    """
    axis_name = "ft_trials"
    axis_values: list[tuple[str, str]] = [(str(int(n)), _format_trials(n)) for n in ft_trials]

    cells: dict[tuple[str, str, str], str] = {}
    # `kept_trials` is only meaningful for trial-budget sweeps and only on runs
    # whose train_decimation ran; don't warn on its NaN cells.
    expected_metric_fields = {"val_best_r2", "test_r2"}
    for axis_label, dir_tag in axis_values:
        for sess in other_sessions:
            run_dir = experiments_root / ft_mode / sess / dir_tag
            metrics = _read_metrics(run_dir)
            missing = [f for f, v in metrics.items() if v is None and f in expected_metric_fields]
            if missing:
                logger.warning(f"missing metrics {missing} for {ft_mode}/{sess}/{dir_tag} ({run_dir}) — recording NaN")
            for field in _METRIC_FIELDS:
                v = metrics[field]
                if v is None:
                    cells[(field, axis_label, sess)] = "nan"
                elif field == "kept_trials":
                    cells[(field, axis_label, sess)] = f"{int(v)}"
                else:
                    cells[(field, axis_label, sess)] = f"{v:.6f}"

    out_path = experiments_root / f"calibration_{ft_mode}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["metric", axis_name, *other_sessions]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for field in _METRIC_FIELDS:
            for axis_label, _ in axis_values:
                row = {"metric": field, axis_name: axis_label}
                for sess in other_sessions:
                    row[sess] = cells[(field, axis_label, sess)]
                writer.writerow(row)
    logger.info(f"Wrote {out_path}")
