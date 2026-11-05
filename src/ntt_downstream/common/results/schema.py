"""Dataclasses for the downstream results record (``downstream_{task}_results.json``).

Hybrid schema: a bespoke top-level layout fitted to downstream metrics, plus
``MetricResult``-like leaves (``MetricEntry``) where the extra context helps.

Schema v2 block semantics, identical for both tasks:

- ``test``: the authoritative score, i.e. the val-selected checkpoint evaluated
  on the held-out test split. Consumers (ntt's aggregation) read this block.
- ``val_best`` / ``val_last``: periodic-validation snapshots at the selected
  checkpoint / at the end of training. Diagnostics only.
- ``val_final`` (encoding only): the end-of-training final eval on val.

``kind`` is ``"run"`` for one training run and ``"cohort"`` for the record a
decoding benchmark builds over its per-session child runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = 2


@dataclass
class MetricEntry:
    """A single aggregate metric value with light context (MetricResult-like leaf)."""

    value: float | None
    direction: str = "higher"  # "higher" | "lower"
    value_range: tuple[float | None, float | None] = (None, None)
    median: float | None = None
    std: float | None = None
    n: int | None = None
    # What the median/std/n summarise across — e.g. "sessions" (decoding),
    # "sessions:mean_of_means" (encoding final), "neurons". Disambiguates the
    # dispersion axis, which differs by task. Omitted from the dict when None.
    aggregated_over: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "value": self.value,
            "direction": self.direction,
            "value_range": list(self.value_range),
        }
        if self.median is not None:
            d["median"] = self.median
        if self.std is not None:
            d["std"] = self.std
        if self.n is not None:
            d["n"] = self.n
        if self.aggregated_over is not None:
            d["aggregated_over"] = self.aggregated_over
        return d


@dataclass
class MetricBlock:
    """Metrics for one phase/epoch: aggregates + per-session + task-specific groups.

    ``groups`` holds task-specific sub-trees merged in at the block's top level
    (e.g. ``per_subject`` for encoding; ``tiers`` for decoding; ``train`` diagnostics).
    """

    aggregated: dict[str, MetricEntry] = field(default_factory=dict)
    # None = a non-finite value, serialized as JSON null (see docs/results.md).
    per_session: dict[str, dict[str, float | None]] = field(default_factory=dict)
    groups: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "aggregated": {k: v.to_dict() for k, v in self.aggregated.items()},
            "per_session": self.per_session,
        }
        d.update(self.groups)
        return d


@dataclass
class RunMetadata:
    """Run provenance + checkpoint info for the results document."""

    git_sha: str | None
    timestamp: str
    hostname: str
    run_name: str | None
    mlflow_run_id: str | None
    config: dict[str, Any]
    checkpoint: dict[str, Any]
    sessions: list[str]
    n_sessions: int
    # The frozen tokenizer checkpoint the run evaluated (path or mlflow URI),
    # so a consumer can check a tracker row points at the right model.
    tokenizer_checkpoint: str | None = None
    # Which ntt the run imported (see ``provenance.ntt_provenance``).
    ntt_version: str | None = None
    ntt_path: str | None = None
    ntt_commit: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "git_sha": self.git_sha,
            "timestamp": self.timestamp,
            "hostname": self.hostname,
            "run_name": self.run_name,
            "mlflow_run_id": self.mlflow_run_id,
            "config": self.config,
            "checkpoint": self.checkpoint,
            "sessions": self.sessions,
            "n_sessions": self.n_sessions,
            "tokenizer_checkpoint": self.tokenizer_checkpoint,
            "ntt_version": self.ntt_version,
            "ntt_path": self.ntt_path,
            "ntt_commit": self.ntt_commit,
        }
        d.update(self.extra)
        return d


@dataclass
class ResultsDocument:
    """Top-level results record; ``metrics`` keys per the module docstring."""

    task: str  # "encoding" | "decoding"
    metadata: RunMetadata
    metrics: dict[str, MetricBlock]
    kind: str = "run"  # "run" | "cohort"
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task": self.task,
            "kind": self.kind,
            "metadata": self.metadata.to_dict(),
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
        }
