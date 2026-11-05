"""Canonical results-record writing (``downstream_{task}_results.json``) for downstream jobs."""

from ntt_downstream.common.results.callback import MetricSnapshotCallback
from ntt_downstream.common.results.schema import (
    SCHEMA_VERSION,
    MetricBlock,
    MetricEntry,
    ResultsDocument,
    RunMetadata,
)
from ntt_downstream.common.results.writer import results_filename, write_results

__all__ = [
    "SCHEMA_VERSION",
    "MetricEntry",
    "MetricBlock",
    "RunMetadata",
    "ResultsDocument",
    "MetricSnapshotCallback",
    "results_filename",
    "write_results",
]
