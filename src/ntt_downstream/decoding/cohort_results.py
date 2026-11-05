"""Cohort record for a decoding benchmark: one v2 record per tokenizer.

A benchmark trains one POYO run per (tokenizer, session) cell; each cell writes
its own ``downstream_decoding_results.json``. ntt's aggregation wants one record
per tokenizer, so this module merges the cells into a ``kind="cohort"`` record:
per-session rows plus an across-session aggregate for each block.

It is built from the **local** child run dirs rather than from MLflow, for two
reasons: the orchestration knows which sessions were *expected*, so missing or
failed cells are detectable (and reported via ``complete: false``) rather than
silently averaged away; and it works for offline / MLflow-disabled sweeps.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from pathlib import Path
from typing import Any

from ntt_downstream.common.results.provenance import git_sha, hostname, now_iso, ntt_provenance
from ntt_downstream.common.results.schema import MetricBlock, MetricEntry, ResultsDocument, RunMetadata
from ntt_downstream.common.results.writer import results_filename, write_results
from ntt_downstream.decoding.mlflow_utils import _aggregate_values

logger = logging.getLogger(__name__)

# Sibling dir of the per-session cell dirs that holds the cohort record:
# experiments_benchmark/<tag>/<tag>__<tokenizer>__cohort/
COHORT_DIRNAME_SUFFIX = "__cohort"

_BLOCKS = ("test", "val_best", "val_last")
_RECORD_NAME = results_filename("decoding")
_R2_RANGE = (None, 1.0)


def version_dirs(run_dir: Path) -> list[Path]:
    """``csv_logs/version_<N>`` dirs under ``run_dir``, sorted ascending by N (version-aware)."""
    csv_logs = Path(run_dir) / "csv_logs"
    if not csv_logs.is_dir():
        return []
    # Ignore non-numeric names (e.g. a stray ``version_tmp``) rather than crash.
    return sorted(
        (p for p in csv_logs.iterdir() if p.is_dir() and p.name.startswith("version_") and p.name[8:].isdigit()),
        key=lambda p: int(p.name.split("_", 1)[1]),
    )


def newest_child_record(run_dir: Path) -> tuple[dict[str, Any], Path] | None:
    """The v2 record in the **newest** ``version_N`` of ``run_dir``, with its path.

    Only the newest version counts: a rerun that crashed after creating a new
    version dir must make the cell *missing*, never fall back to a stale record
    from an earlier attempt. ``None`` when that dir has no parseable record.
    """
    versions = version_dirs(run_dir)
    if not versions:
        return None
    path = versions[-1] / _RECORD_NAME
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or not isinstance(doc.get("metrics"), dict):
        return None
    return doc, path


def _is_stale(record: dict[str, Any], not_before: str | None) -> bool:
    """True when ``record`` predates the sweep (``not_before``, ISO-8601 UTC ``...Z``).

    "Newest version dir" alone can't tell a fresh result from the previous
    attempt's: a rerun that crashes before Lightning creates its version dir
    (data setup, checkpoint load, model build) leaves the old record newest.
    A record without a timestamp can't prove freshness and counts as stale.
    ISO-8601 UTC strings with a fixed format compare correctly as strings.
    """
    if not_before is None:
        return False
    ts = (record.get("metadata") or {}).get("timestamp")
    return not isinstance(ts, str) or ts < not_before


def _finite(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _child_r2(record: dict[str, Any], block: str, session: str) -> float | None:
    """A child's R² for ``block``: its own per-session row, else its block aggregate.

    Benchmark cells are single-session runs, so both normally agree; the
    fallback covers a child whose per-session key differs from the cohort's
    session name.
    """
    blk = (record.get("metrics") or {}).get(block) or {}
    row = (blk.get("per_session") or {}).get(session) or {}
    if "r2" in row:
        return _finite(row["r2"])
    return _finite(((blk.get("aggregated") or {}).get("r2") or {}).get("value"))


def _cohort_block(values: dict[str, float | None]) -> MetricBlock:
    """Per-session rows plus an across-session aggregate over the finite values."""
    block = MetricBlock(per_session={s: {"r2": v} for s, v in values.items()})
    finite = [v for v in values.values() if v is not None]
    if finite:
        mean, median, n = _aggregate_values(finite)
        block.aggregated["r2"] = MetricEntry(
            value=mean,
            direction="higher",
            value_range=_R2_RANGE,
            median=median,
            std=statistics.stdev(finite) if n > 1 else None,
            n=n,
            aggregated_over="sessions:mean",
        )
    return block


def build_cohort_document(
    *,
    tokenizer_name: str,
    tokenizer_checkpoint: str | None,
    parent_run_id: str | None,
    child_dirs: dict[str, Path],
    not_before: str | None = None,
) -> ResultsDocument:
    """Merge the benchmark cells of one tokenizer into a ``kind="cohort"`` record.

    ``child_dirs`` maps each *expected* session to its cell run dir. A session is
    *found* iff its newest record exists, parses, and has a ``test`` block; all
    others are listed in ``metadata.cohort.missing`` and ``complete`` is false.
    Every block (``test``, ``val_best``, ``val_last``) is always present, with an
    empty ``aggregated`` when no found child has a finite value.

    ``not_before`` (the sweep's start time, ISO-8601 UTC) additionally rejects
    records written before this sweep, listing them under ``stale`` too. A
    malformed child only makes that child missing, never the whole cohort.
    """
    expected = sorted(child_dirs)
    found: list[str] = []
    stale: list[str] = []
    children: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[str, Any]] = {}
    for session in expected:
        try:
            hit = newest_child_record(child_dirs[session])
        except Exception:  # noqa: BLE001 — one unreadable cell must not sink the cohort
            logger.warning("cohort: could not read cell %s; counting it missing", session, exc_info=True)
            hit = None
        if hit is None or "test" not in hit[0]["metrics"]:
            continue
        record, path = hit
        if _is_stale(record, not_before):
            stale.append(session)
            continue
        found.append(session)
        records[session] = record
        md = record.get("metadata") or {}
        children[session] = {
            "run_name": md.get("run_name"),
            "mlflow_run_id": md.get("mlflow_run_id"),
            "git_sha": md.get("git_sha"),
            "ntt_version": md.get("ntt_version"),
            "ntt_commit": md.get("ntt_commit"),
            "record_path": str(path),
        }
    missing = [s for s in expected if s not in records]
    provenance = {(c["git_sha"], c["ntt_version"], c["ntt_commit"]) for c in children.values()}

    metrics = {block: _cohort_block({s: _child_r2(records[s], block, s) for s in found}) for block in _BLOCKS}
    cohort = {
        "expected": expected,
        "found": found,
        "missing": missing,
        "stale": stale,
        "not_before": not_before,
        "complete": not missing,
        # A cohort stitched together across a code or ntt change (e.g. a
        # missing_only rerun on a newer checkout) is flagged, not hidden.
        "mixed_provenance": len(provenance) > 1,
        "children": children,
    }
    metadata = RunMetadata(
        git_sha=git_sha(),
        timestamp=now_iso(),
        hostname=hostname(),
        run_name=f"{tokenizer_name}{COHORT_DIRNAME_SUFFIX}",
        mlflow_run_id=parent_run_id,
        config={},
        checkpoint={},
        sessions=found,
        n_sessions=len(found),
        tokenizer_checkpoint=tokenizer_checkpoint,
        **ntt_provenance(),
        extra={"tokenizer_name": tokenizer_name, "cohort": cohort},
    )
    return ResultsDocument(task="decoding", kind="cohort", metadata=metadata, metrics=metrics)


def write_cohort_record(doc: ResultsDocument, out_dir: Path) -> Path | None:
    """Write ``doc`` to ``<out_dir>/downstream_decoding_results.json``; never raises.

    Building the cohort is a reporting step and must never fail a sweep.
    """
    try:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return write_results(doc, out_dir)
    except Exception:  # noqa: BLE001
        logger.warning("Failed to write the cohort record under %s; continuing.", out_dir, exc_info=True)
        return None
