"""Pure helpers for the MLflow parent-run lifecycle.

These are the side-effect-free pieces of the orchestrator's MLflow handling —
URI synthesis, ``--tag key=value`` parsing, the aggregable-metric filter, and
the mean/median arithmetic. They import no ``mlflow`` and touch no network, so
they're importable in-process by the orchestration engine and unit-testable
offline.

The *stateful* operations (open/aggregate parent runs against a tracking
server) live in ``ntt_downstream.decoding.utils.mlflow_parent`` and are invoked
as a subprocess, so a flaky server can't take down the orchestrator loop. That
module imports these helpers, so its unit tests keep exercising this
implementation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any
from urllib.parse import urlparse

# Metric keys we don't aggregate onto the parent: step-counters (meaningless to
# average across heterogeneous children) and system telemetry. Matched as exact
# keys (Lightning emits bare ``epoch`` / ``step``), not suffixes.
_SKIP_METRIC_KEYS = ("epoch", "step")
_SKIP_METRIC_PREFIXES = ("system/",)


def _build_run_uri(tracking_uri: str, experiment_id: str, run_id: str) -> str:
    """Synthesize the canonical MLflow UI URL for a run.

    MLflow has no first-class "UI URL for run" API; we glue it together from the
    tracking URI. File / local-mode URIs aren't browsable, so we surface the raw
    run id instead.
    """
    base = tracking_uri.rstrip("/")
    parsed = urlparse(base)
    if parsed.scheme in ("", "file"):
        return f"local-mlflow-run:{run_id}"
    return f"{base}/#/experiments/{experiment_id}/runs/{run_id}"


def _parse_tag_args(tag_args: Iterable[str]) -> dict[str, str]:
    """Parse ``--tag key=value`` repeats. Empty values pass through (useful for
    explicitly clearing a field such as ``mlflow.note.content=``)."""
    out: dict[str, str] = {}
    for raw in tag_args:
        if "=" not in raw:
            raise SystemExit(f"--tag expects key=value, got: {raw!r}")
        k, v = raw.split("=", 1)
        if not k:
            raise SystemExit(f"--tag has empty key: {raw!r}")
        out[k] = v
    return out


def _is_aggregable(metric_key: str) -> bool:
    """True unless ``metric_key`` is a step-counter or system-telemetry key."""
    if any(metric_key.startswith(p) for p in _SKIP_METRIC_PREFIXES):
        return False
    if metric_key in _SKIP_METRIC_KEYS:
        return False
    return True


def _aggregate_values(values: list[float]) -> tuple[float, float, int]:
    """Return ``(mean, median, count)`` over a non-empty list of floats.

    Median is the middle element for odd counts and the mean of the two middle
    elements for even counts.
    """
    values_sorted = sorted(values)
    n = len(values_sorted)
    mean = sum(values_sorted) / n
    mid = n // 2
    median = values_sorted[mid] if n % 2 == 1 else 0.5 * (values_sorted[mid - 1] + values_sorted[mid])
    return mean, median, n


def _metrics_table(children: Sequence[Any]) -> tuple[list[str], list[dict[str, str]]]:
    """Build a tidy per-subrun table ``(fieldnames, rows)`` from MLflow run objects.

    One row per child run: ``run_id``, ``run_name``, ``status``, every non-internal
    tag as ``tag.<key>`` (seed / ft_mode / calib_session / ft_trials / stage /
    tokenizer_name / …), and every last-logged metric as ``metric.<key>``. Pure
    (duck-types ``run.info.{run_id,status}`` + ``run.data.{tags,metrics}``), so it
    is unit-testable offline and a caller can dump it straight to CSV and pivot /
    group-by seed in pandas. Internal ``mlflow.*`` tags are dropped (the run name
    is surfaced separately)."""
    tag_keys = sorted({k for r in children for k in r.data.tags if not k.startswith("mlflow.")})
    metric_keys = sorted({k for r in children for k in r.data.metrics})
    fieldnames = [
        "run_id", "run_name", "status",
        *(f"tag.{k}" for k in tag_keys),
        *(f"metric.{k}" for k in metric_keys),
    ]  # fmt: skip
    rows: list[dict[str, str]] = []
    for r in children:
        row = {
            "run_id": r.info.run_id,
            "run_name": r.data.tags.get("mlflow.runName", ""),
            "status": r.info.status,
        }
        for k in tag_keys:
            row[f"tag.{k}"] = str(r.data.tags.get(k, ""))
        for k in metric_keys:
            v = r.data.metrics.get(k)
            row[f"metric.{k}"] = "" if v is None else repr(v) if isinstance(v, float) and v != v else str(v)
        rows.append(row)
    return fieldnames, rows
