"""Aggregated radar plots for tokenizers, baselined against a Poisson reference.

The flow is:

1. A results cache directory with the layout::

       tokenizer_results_artifacts/
         _reference/poisson_baseline.json     # ntt EvalArtifact w/ floor + ceiling
         <safe_tokenizer>/recon/eval_report.json
         manifest.csv

   ``manifest.csv`` has one row per tokenizer. ``RadarBundle.from_cache``
   reads the ``tokenizer`` column (display name) and ``eval_report_path``
   (report path relative to the cache directory). ``list_available_metrics``
   additionally scans any ``<source>_eval_report_path`` columns (for example
   ``latent_snr_eval_report_path``) for further report sources. The wide
   table passed to ``inject_sheet_metrics`` is a per-tokenizer DataFrame
   whose scalar columns use the prefixes ``recon_``, ``latent_input_corruption_``,
   ``latent_latent_corruption_``, ``latent_snr_``, ``encoding_`` and
   ``sheet_decoding_``.

2. ``RadarBundle.from_cache(...)`` loads that tree, deserializing each report
   through ``ntt.evaluation.core.types.EvaluationReport.from_dict``. A report
   carrying a category string that ``MetricCategory`` does not know is
   loaded with that category replaced by a valid one (the category is
   cosmetic here, see ``_to_known_category``).

3. Per-tokenizer normalized scores come from
   ``ntt.evaluation.aggregation.comparison.compute_normalized_comparisons``
   so direction-aware sign handling and the ``(model - floor) / (ceiling - floor)``
   formula stay consistent with ntt's own ``plot_normalized_comparison_radar``.

4. ``CategorySpec`` lets callers either lean on the ntt ``MetricCategory`` enum
   (``CategorySpec.from_metric_categories(report)``) or define arbitrary buckets
   that average a hand-picked metric list (``{"Single-unit fidelity":
   ["acg_correlation", "isi_kl_divergence"]}``). Metrics not present in a given
   tokenizer are skipped with a warning.

5. ``plot_radar`` overlays one polygon per tokenizer using a matplotlib polar
   axis. ``0`` is the Poisson floor and ``1`` is the perfect-reconstruction
   ceiling, matching ntt's plotting convention.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import RegularPolygon
from matplotlib.path import Path as MplPath
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from ntt.evaluation.aggregation.comparison import (
    MetricComparison,
    compute_normalized_comparisons,
)
from ntt.evaluation.core.types import (
    EvaluationReport,
    MetricCategory,
    MetricDirection,
    MetricLevel,
    MetricResult,
)

LOGGER = logging.getLogger("ntt_downstream.encoding.analysis.radar")


def _to_known_category(value: str) -> str:
    """Return ``value`` if ``MetricCategory`` accepts it, else a valid fallback.

    Never raises. ``compute_normalized_comparisons`` only reads ``direction``
    and ``raw_value``, and the radar's axes come from the user
    ``CategorySpec``, so a cosmetic category mislabel is harmless. The
    fallback is ``DOWNSTREAM_ENCODING``.
    """
    valid = {m.value for m in MetricCategory}
    if value in valid:
        return value
    return MetricCategory.DOWNSTREAM_ENCODING.value


def _sanitize_categories_inplace(report_dict: dict) -> None:
    """Rewrite category strings that ``MetricCategory`` does not know, in place.

    ``EvaluationReport.from_dict`` raises ``ValueError`` on an unknown
    category. The category is cosmetic for the radar (axes come from the user
    ``CategorySpec``), so degrading an unknown label beats aborting the whole
    load.
    """
    for entry in report_dict.get("results", {}).values():
        if not isinstance(entry, dict):
            continue
        cat = entry.get("category")
        if isinstance(cat, str):
            entry["category"] = _to_known_category(cat)
    cat_scores = report_dict.get("category_scores")
    if isinstance(cat_scores, dict):
        report_dict["category_scores"] = {_to_known_category(str(k)): v for k, v in cat_scores.items()}


def _report_from_dict_graceful(data: dict) -> EvaluationReport:
    """``EvaluationReport.from_dict`` with a graceful retry on unknown categories.

    Reports with known categories load on the first attempt. An unknown
    category makes ``from_dict`` raise ``ValueError``, which triggers a
    one-shot category sanitization and a retry, so the load degrades instead
    of aborting.
    """
    try:
        return EvaluationReport.from_dict(data)
    except ValueError:
        _sanitize_categories_inplace(data)
        return EvaluationReport.from_dict(data)


@dataclass
class CategorySpec:
    """Maps a display label to a list of metric names whose normalized scores get averaged."""

    categories: dict[str, list[str]]

    def labels(self) -> list[str]:
        return list(self.categories.keys())

    @classmethod
    def from_metric_categories(
        cls,
        report: EvaluationReport,
        only: list[MetricCategory] | None = None,
    ) -> CategorySpec:
        """Build a spec by grouping ``report``'s metric names by their declared category.

        ``only`` optionally restricts to a subset of MetricCategory values.
        """
        buckets: dict[str, list[str]] = {}
        for name, result in report.results.items():
            cat = result.category
            if only is not None and cat not in only:
                continue
            buckets.setdefault(cat.value, []).append(name)
        return cls(categories={k: sorted(v) for k, v in sorted(buckets.items())})


@dataclass
class RadarBundle:
    """All the reports needed to draw a multi-tokenizer radar."""

    baseline: EvaluationReport
    ceiling: EvaluationReport
    tokenizers: dict[str, EvaluationReport] = field(default_factory=dict)
    manifest: pd.DataFrame | None = None

    @classmethod
    def from_cache(
        cls,
        cache_dir: str | Path,
        tokenizers: list[str] | None = None,
        reference_path: str | Path | None = None,
    ) -> RadarBundle:
        cache = Path(cache_dir)
        manifest_path = cache / "manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"manifest.csv not found at {manifest_path}; build the results cache "
                "described in the module docstring first."
            )
        manifest = pd.read_csv(manifest_path)

        ref = Path(reference_path) if reference_path else cache / "_reference" / "poisson_baseline.json"
        if not ref.exists():
            raise FileNotFoundError(
                f"Poisson reference not found at {ref}; supply reference_path "
                "or place the reference at <cache>/_reference/poisson_baseline.json."
            )
        baseline, ceiling = _load_reference_split(ref)

        rows = manifest if tokenizers is None else manifest[manifest["tokenizer"].isin(tokenizers)]
        loaded: dict[str, EvaluationReport] = {}
        for _, row in rows.iterrows():
            tokenizer_name = str(row["tokenizer"])
            rel = row.get("eval_report_path")
            if not isinstance(rel, str) or not rel:
                continue
            path = cache / rel
            if not path.exists():
                LOGGER.warning("missing cached report for %s: %s", tokenizer_name, path)
                continue
            try:
                loaded[tokenizer_name] = load_report(path)
            except Exception as exc:
                LOGGER.warning("failed to load report for %s: %r", tokenizer_name, exc)
        return cls(baseline=baseline, ceiling=ceiling, tokenizers=loaded, manifest=manifest)


@dataclass
class SheetMetric:
    """Specification for a metric drawn from the wide per-tokenizer table / sheet snapshot.

    Use ``inject_sheet_metrics`` to materialize these into a ``RadarBundle`` so
    they ride the same normalization/category-grouping path as recon-report
    metrics.

    Attributes
    ----------
    name:
        Metric key to register on each report. Becomes the value you put in a
        ``CategorySpec`` entry.
    column:
        Column name in the wide DataFrame whose value becomes ``raw_value`` per
        tokenizer.
    direction:
        ``"higher"`` (default) or ``"lower"`` — drives the sign-orient step in
        ``compute_normalized_comparisons``.
    floor:
        Either a number (Poisson-floor value used for *every* tokenizer) or a
        tokenizer name to look up in the same column. ``None`` reuses
        ``raw_value`` of the *baseline* injection — usually wrong; prefer an
        explicit number.
    ceiling:
        Same shape as ``floor``, but for the perfect-reconstruction ceiling.
        ``1.0`` is the conventional choice for Pearson correlations / R².
    level:
        Maps to ``MetricLevel``; defaults to ``POPULATION`` (these are scalar
        per-session-aggregated metrics).
    category:
        Stored on the synthetic ``MetricResult``. Cosmetic — your
        ``CategorySpec`` is what assigns the radar axis. Defaults to
        ``downstream_encoding``.
    """

    name: str
    column: str
    direction: str = "higher"
    floor: float | str | None = None
    ceiling: float | str | None = None
    level: str = "population"
    category: str = "downstream_encoding"


def inject_sheet_metrics(
    bundle: RadarBundle,
    df: pd.DataFrame,
    metrics: list[SheetMetric],
    *,
    tokenizer_col: str = "tokenizer",
) -> list[str]:
    """Augment ``bundle`` with metrics drawn from the wide per-tokenizer table.

    For every ``SheetMetric`` we:

    1. Resolve numeric floor and ceiling values (constants or row lookups).
    2. Append a synthetic ``MetricResult`` to ``bundle.baseline`` and
       ``bundle.ceiling`` carrying those floor/ceiling values.
    3. For each tokenizer in ``bundle.tokenizers`` that has a finite value in
       ``df``, append a ``MetricResult`` with that ``raw_value``.

    Tokenizers missing from ``df`` (or with NaN) are silently skipped so the
    radar polygon for that tokenizer just doesn't extend along the new axis.
    Returns the list of metric names actually injected.
    """
    if tokenizer_col not in df.columns:
        raise KeyError(f"DataFrame is missing tokenizer key column {tokenizer_col!r}")

    by_tokenizer = {str(row[tokenizer_col]): row for _, row in df.iterrows() if isinstance(row[tokenizer_col], str)}

    injected: list[str] = []
    for spec in metrics:
        if spec.column not in df.columns:
            LOGGER.warning("sheet metric %r: column %r missing from DataFrame; skipping", spec.name, spec.column)
            continue

        floor_val = _resolve_reference_value(spec.floor, spec.column, by_tokenizer, role="floor")
        ceiling_val = _resolve_reference_value(spec.ceiling, spec.column, by_tokenizer, role="ceiling")
        if floor_val is None or ceiling_val is None:
            LOGGER.warning(
                "sheet metric %r: could not resolve floor=%r / ceiling=%r; skipping",
                spec.name,
                spec.floor,
                spec.ceiling,
            )
            continue

        try:
            direction = MetricDirection(spec.direction)
        except ValueError:
            LOGGER.warning("sheet metric %r: invalid direction %r; skipping", spec.name, spec.direction)
            continue
        try:
            level = MetricLevel(spec.level)
        except ValueError as exc:
            LOGGER.warning("sheet metric %r: %s; skipping", spec.name, exc)
            continue
        category = MetricCategory(_to_known_category(spec.category))

        bundle.baseline.results[spec.name] = _make_synthetic_result(
            spec.name, category, direction, level, raw_value=floor_val
        )
        bundle.ceiling.results[spec.name] = _make_synthetic_result(
            spec.name, category, direction, level, raw_value=ceiling_val
        )

        n_tokenizers_with_value = 0
        for tokenizer_name, report in bundle.tokenizers.items():
            row = by_tokenizer.get(tokenizer_name)
            if row is None:
                continue
            value = _coerce_scalar_float(row.get(spec.column))
            if value is None:
                continue
            report.results[spec.name] = _make_synthetic_result(spec.name, category, direction, level, raw_value=value)
            n_tokenizers_with_value += 1

        injected.append(spec.name)
        LOGGER.info(
            "injected sheet metric %r (column=%r) into %d/%d tokenizers (floor=%.4g ceiling=%.4g)",
            spec.name,
            spec.column,
            n_tokenizers_with_value,
            len(bundle.tokenizers),
            floor_val,
            ceiling_val,
        )

    return injected


def build_sheet_metrics(
    catalog: pd.DataFrame,
    selectors: list[str],
    *,
    default_range: tuple[float, float] = (0.0, 1.0),
    default_direction: str = "higher",
    overrides: Mapping[str, Mapping[str, object]] | None = None,
) -> list[SheetMetric]:
    """Build ``SheetMetric`` specs from catalog ``"source/metric"`` selectors.

    The selector itself is used as the metric ``name``, so the same string can
    be referenced inside a ``CategorySpec`` after injection — no naming drift
    between the catalog, the injection step, and the spec.

    Eval-report rows in the catalog (``where == "eval_report"``) are skipped:
    those metrics already live in the bundle and don't need synthetic
    injection. The skipped selectors are returned via the LOGGER so callers
    can spot them.

    Parameters
    ----------
    catalog:
        Output of ``list_available_metrics`` — must contain ``source``,
        ``metric``, ``column``, ``where`` columns.
    selectors:
        Identifiers of the form ``"<source>/<metric>"`` (e.g.,
        ``"encoding/nc_r2"``, ``"sheet_decoding/poyo"``). Use the slashes
        printed by ``print_metric_catalog``.
    default_range:
        ``(floor, ceiling)`` applied to every selector that doesn't have an
        override. ``(0.0, 1.0)`` is right for Pearson / R² / accuracy.
    default_direction:
        ``"higher"`` or ``"lower"``. Per-selector overrides take precedence.
    overrides:
        Per-selector dict with any of ``{"floor", "ceiling", "direction",
        "name", "category", "level"}``. Use this for metrics whose
        floor/ceiling differ from the default (e.g.,
        ``binned_rate_difference`` is lower-is-better with no natural unit
        bound — supply realistic numbers).
    """
    overrides_map = {} if overrides is None else dict(overrides)
    floor_default, ceiling_default = default_range
    out: list[SheetMetric] = []
    for selector in selectors:
        if "/" not in selector:
            LOGGER.warning("selector %r is missing '/'; expected '<source>/<metric>'", selector)
            continue
        source, metric = selector.split("/", 1)
        match = catalog[(catalog["source"] == source) & (catalog["metric"] == metric)]
        if match.empty:
            LOGGER.warning("selector %r not found in catalog; skipping", selector)
            continue
        row = match.iloc[0]
        if str(row.get("where", "")) == "eval_report":
            LOGGER.info(
                "selector %r is an eval_report metric — already in the bundle; "
                "use it directly in CategorySpec, no SheetMetric needed",
                selector,
            )
            continue
        column = str(row["column"])
        if not column:
            LOGGER.warning("selector %r has no CSV column; skipping", selector)
            continue
        ov = overrides_map.get(selector, {})
        spec = SheetMetric(
            name=str(ov.get("name", selector)),
            column=column,
            direction=str(ov.get("direction", default_direction)),
            floor=_coerce_override_number(ov.get("floor", floor_default)),
            ceiling=_coerce_override_number(ov.get("ceiling", ceiling_default)),
            level=str(ov.get("level", "population")),
            category=str(ov.get("category", "downstream_encoding")),
        )
        out.append(spec)
    return out


def _coerce_override_number(value: object) -> float | str | None:
    """Override values for floor/ceiling come from a Mapping[str, object]; coerce safely."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return value
    return None


def tokenizers_with_data(
    bundle: RadarBundle,
    metric_names: list[str],
    *,
    require: str = "any",
) -> list[str]:
    """Return tokenizers whose reports carry the given metrics.

    ``require="any"`` (default) keeps a tokenizer if at least one of the
    metrics is present; ``require="all"`` keeps it only when *every* metric is
    present. Useful for picking a sensible default subset before
    ``plot_radar`` — e.g., select tokenizers with encoding/decoding data when
    those axes are on the radar.
    """
    out: list[str] = []
    for tokenizer, report in bundle.tokenizers.items():
        present = {n for n in metric_names if n in report.results}
        if require == "all":
            if len(present) == len(metric_names):
                out.append(tokenizer)
        else:
            if present:
                out.append(tokenizer)
    return out


def _resolve_reference_value(
    spec: float | str | None,
    column: str,
    by_tokenizer: Mapping[str, pd.Series],
    *,
    role: str,
) -> float | None:
    if spec is None:
        return None
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return float(spec)
    if isinstance(spec, str):
        row = by_tokenizer.get(spec)
        if row is None:
            LOGGER.warning("%s spec references unknown tokenizer %r", role, spec)
            return None
        value = _coerce_scalar_float(row.get(column))
        if value is None:
            LOGGER.warning("%s spec %r has missing/non-numeric value in column %r", role, spec, column)
        return value
    LOGGER.warning("unsupported %s spec type: %r", role, type(spec))
    return None


def _coerce_scalar_float(value: object) -> float | None:
    """Best-effort cast of a CSV cell (possibly numpy scalar) to a finite float."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        f = float(value)
        return f if np.isfinite(f) else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            f = float(text)
        except ValueError:
            return None
        return f if np.isfinite(f) else None
    return None


def _make_synthetic_result(
    name: str,
    category: MetricCategory,
    direction: MetricDirection,
    level: MetricLevel,
    *,
    raw_value: float,
) -> MetricResult:
    return MetricResult(
        name=name,
        category=category,
        direction=direction,
        level=level,
        raw_value=raw_value,
        normalized_value=None,
        metadata={"source": "sheet_injection"},
        value_range=(None, None),
        source_data_type=None,
    )


def load_report(path: str | Path) -> EvaluationReport:
    """Load an eval_report.json into an EvaluationReport.

    An unknown category string would make ``EvaluationReport.from_dict`` raise
    ``ValueError``; ``_report_from_dict_graceful`` retries after sanitizing such
    categories so the load degrades instead of aborting.
    """
    with open(path) as f:
        data = json.load(f)
    return _report_from_dict_graceful(data)


def _load_reference_split(path: str | Path) -> tuple[EvaluationReport, EvaluationReport]:
    """Split a poisson_baseline.json (EvalArtifact) into (Poisson floor, perfect ceiling)."""
    with open(path) as f:
        artifact = json.load(f)
    aggregated = artifact.get("aggregated")
    perfect = artifact.get("perfect_aggregated")
    if not isinstance(aggregated, dict) or not isinstance(perfect, dict):
        raise ValueError(
            f"reference at {path} does not have both 'aggregated' (Poisson floor) and "
            "'perfect_aggregated' (ceiling) keys"
        )
    return _report_from_dict_graceful(aggregated), _report_from_dict_graceful(perfect)


# Catalog source labels we recognize as a slash-prefix in ``CategorySpec``
# entries. When a spec name like ``"recon/acg_correlation"`` doesn't match the
# bundle directly, we strip the leading ``"<known_source>/"`` and retry — so
# the catalog's slash-form selectors can be pasted straight into a spec for
# eval-report metrics. CSV-side selectors still need ``inject_sheet_metrics``;
# we emit a clear log line directing the user there.
_KNOWN_CATALOG_SOURCES: tuple[str, ...] = (
    "recon",
    "encoding",
    "sheet_decoding",
    "sheet",
    "latent_input_corruption",
    "latent_latent_corruption",
    "latent_snr",
)


def _resolve_metric_name(
    raw_name: str,
    comparisons: Mapping[str, MetricComparison],
) -> tuple[str, MetricComparison] | None:
    """Look ``raw_name`` up in ``comparisons``; fall back to stripping a known source prefix."""
    cmp_ = comparisons.get(raw_name)
    if cmp_ is not None:
        return raw_name, cmp_
    if "/" not in raw_name:
        return None
    prefix, _, suffix = raw_name.partition("/")
    if prefix not in _KNOWN_CATALOG_SOURCES:
        return None
    cmp_ = comparisons.get(suffix)
    if cmp_ is None:
        return None
    return suffix, cmp_


def category_scores_for(
    model: EvaluationReport,
    baseline: EvaluationReport,
    ceiling: EvaluationReport,
    spec: CategorySpec,
    *,
    aggregator: str = "mean",
    normalize: bool = True,
) -> dict[str, float]:
    """Compute per-category scores for one tokenizer.

    ``aggregator`` is ``"mean"`` (equal-weighted) or ``"median"``. Categories
    with no usable metrics return ``np.nan``.

    ``normalize=True`` (default) returns Poisson-normalized scores: ``0`` is
    the Poisson floor and ``1`` is the perfect-reconstruction ceiling, sign-
    oriented by ``MetricDirection``. ``normalize=False`` returns the average
    of the raw metric values (``MetricResult.raw_value``) — useful when you
    want to inspect units rather than relative position. The two modes are
    intentionally NOT comparable on the same plot; mark the mode explicitly
    when you display them.

    Metric names in ``spec`` may use either form:

    * Bare metric name (``"acg_correlation"``) — looked up directly in the
      tokenizer's report.
    * Catalog ``"<source>/<metric>"`` form (``"recon/acg_correlation"``).
      The leading source prefix is stripped and the bare name is retried.

    CSV-side selectors (``"encoding/...", "sheet_decoding/...", "recon/<mlflow_eval_*>"``)
    only resolve after ``inject_sheet_metrics`` has registered them on the
    bundle. When a name can't be resolved either way, the metric is recorded as
    missing and the category falls back to ``np.nan`` if no metric remains.
    """
    comparisons = compute_normalized_comparisons(model, baseline, ceiling)
    scores: dict[str, float] = {}
    for label, metric_names in spec.categories.items():
        values: list[float] = []
        missing: list[str] = []
        for name in metric_names:
            resolved = _resolve_metric_name(name, comparisons)
            if resolved is None:
                missing.append(name)
                continue
            resolved_name, cmp_ = resolved
            if normalize:
                if cmp_.normalized_score is None or cmp_.status != "ok":
                    missing.append(name)
                    continue
                values.append(float(cmp_.normalized_score))
            else:
                # Raw mode: pull the metric's raw_value from the model report.
                model_result = model.results.get(resolved_name)
                if model_result is None:
                    missing.append(name)
                    continue
                raw = _to_scalar_or_nan(model_result.raw_value)
                if not np.isfinite(raw):
                    missing.append(name)
                    continue
                values.append(raw)
        if missing:
            LOGGER.debug("category %r: %d metrics unavailable (%s)", label, len(missing), ", ".join(missing))
        if not values:
            scores[label] = float("nan")
            continue
        scores[label] = float(np.mean(values) if aggregator == "mean" else np.median(values))
    return scores


def _to_scalar_or_nan(value: object) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, np.ndarray):
        return float(np.nanmean(value))
    return float("nan")


def collect_category_scores(
    bundle: RadarBundle,
    spec: CategorySpec,
    *,
    aggregator: str = "mean",
    normalize: bool = True,
) -> pd.DataFrame:
    """Return a (tokenizer, category) -> score table for every tokenizer in ``bundle``.

    Pass ``normalize=False`` to get raw averaged metric values instead of
    Poisson-normalized scores. The shape of the returned DataFrame is identical
    in both modes; only the values differ. Tag the mode explicitly when you
    display either (the radar's ``mode_tag`` parameter renders ``(NORMED)`` or
    ``(RAW)`` in the title for you).
    """
    rows: dict[str, dict[str, float]] = {}
    for tokenizer, report in bundle.tokenizers.items():
        rows[tokenizer] = category_scores_for(
            report,
            bundle.baseline,
            bundle.ceiling,
            spec,
            aggregator=aggregator,
            normalize=normalize,
        )
    df = pd.DataFrame(rows).T  # rows = tokenizers, cols = categories
    return df.reindex(columns=spec.labels())


# Wide-CSV column prefix → source label. Order matters: longest prefix first so
# 'latent_latent_corruption' wins over a hypothetical 'latent' on the same column.
# Mirrors the metric-source vocabulary used by the encoding analysis tables.
#
# Source labels intentionally collapse CSV-side and eval-report-side rows under
# the same name (e.g., ``recon`` covers both ``recon_acg_correlation`` from the
# CSV and the cached recon ``eval_report.json``). The ``where`` column on the
# catalog distinguishes the two; the dedup step in ``list_available_metrics``
# keeps only the eval-report row when both expose the same metric.
_CSV_SOURCE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("latent_input_corruption_", "latent_input_corruption"),
    ("latent_latent_corruption_", "latent_latent_corruption"),
    ("latent_snr_", "latent_snr"),
    ("recon_", "recon"),
    ("encoding_", "encoding"),
    # Longer ``sheet_decoding_`` prefix wins; ``sheet_`` is the catch-all for
    # generic pass-through sheet columns (robustness scores, manual encoding
    # entries, etc.).
    ("sheet_decoding_", "sheet_decoding"),
    ("sheet_", "sheet"),
)

# Aggregator-flavor suffixes that *describe* a metric rather than carry its
# value. ``_mean`` is treated specially below as the canonical value column.
_CSV_AUX_VALUE_SUFFIXES = (
    "_min",
    "_max",
    "_n_sessions",
    "_n",
    "_se",
    "_std",
    "_normalized",
    "_aggregation_strategy",
)

# CSV columns we never want to surface as metrics (run ids, error blobs, etc.).
# NOTE: do NOT add "_error" here — it would silently drop legitimate metrics
# whose names end in ``_error`` (firing_rate_error, fano_factor_error_10ms,
# participation_ratio_error_5ms, isi_cv_error, isi_cv2_error). The table's
# error blob is the bare ``<prefix>_error`` column with stem ``error``, handled
# specifically inside ``_csv_metric_base``.
_CSV_METRIC_BLOCKLIST_SUFFIXES = (
    "_run_id",
    "_run_name",
    "_experiment_id",
    "_aggregation_strategy",
)

# Substrings that mark a CSV column as an MLflow-side reference scalar (logged
# alongside ``perfect_aggregated`` / Poisson runs). They appear as columns like
# ``recon_mlflow_perfect_acg_correlation`` and only have a value for the single
# reference tokenizer — not useful as user-facing radar metrics.
_CSV_METRIC_BLOCKLIST_SUBSTRINGS = (
    "_mlflow_perfect_",
    "_mlflow_poisson_",
    "mlflow_perfect_",
    "mlflow_poisson_",
)


def _csv_source_for(column: str) -> tuple[str, str] | None:
    """Return ``(source_label, prefix)`` for ``column``, or ``None`` if unknown."""
    for prefix, source in _CSV_SOURCE_PREFIXES:
        if column.startswith(prefix):
            return source, prefix
    return None


def _csv_metric_base(column: str, prefix: str) -> tuple[str, str] | None:
    """Map a CSV column to ``(metric_name, value_column)`` or ``None`` if auxiliary.

    For ``*_mean`` columns the metric name is the stem and the value column is
    the original ``*_mean``. For columns like ``sheet_decoding_poyo`` the column
    is itself the value. Auxiliary suffixes (``_se``, ``_n_sessions``,
    ``_normalized``, …) collapse to ``None`` so the catalog stays one row per
    canonical metric.
    """
    if column.endswith(_CSV_METRIC_BLOCKLIST_SUFFIXES):
        return None
    if any(s in column for s in _CSV_METRIC_BLOCKLIST_SUBSTRINGS):
        return None
    stem = column[len(prefix) :]
    if stem == "error":
        # The table can carry a ``<prefix>_error`` column holding the stringified exception
        # whenever a per-tokenizer pull fails. That's the only column whose stem
        # is literally ``error`` — keep this guard tight so that legitimate
        # metrics ending in ``_error`` (firing_rate_error, fano_factor_error,
        # ...) still come through.
        return None
    if any(stem.endswith(s) for s in _CSV_AUX_VALUE_SUFFIXES):
        return None
    if stem.endswith("_mean"):
        return stem.removesuffix("_mean"), column
    return stem, column


def _looks_like_csv_metric(column: str, df: pd.DataFrame) -> bool:
    if column.endswith(_CSV_METRIC_BLOCKLIST_SUFFIXES):
        return False
    if column not in df.columns:
        return False
    return bool(pd.api.types.is_numeric_dtype(df[column]))


def list_available_metrics(
    cache_dir: str | Path | None = None,
    df: pd.DataFrame | None = None,
    *,
    include_eval_reports: bool = True,
) -> pd.DataFrame:
    """Catalog every metric the radar can plot, organized by source.

    Two complementary sources are scanned:

    1. **Cached eval reports** (``include_eval_reports=True``). For each cached
       ``eval_report.json`` (recon + any latent corruption / SNR runs), enumerate
       the metric names with their declared category, direction, and metric level.
       These are usable as raw recon-report metrics in a ``CategorySpec``.
    2. **Wide per-tokenizer table** (``df``). Walks every column whose prefix matches
       the convention described in the module docstring — ``recon_``,
       ``latent_input_corruption_``, ``latent_latent_corruption_``, ``latent_snr_``,
       ``encoding_``, ``sheet_decoding_`` — and reports how many tokenizers have
       a finite value. These are usable via ``SheetMetric`` + ``inject_sheet_metrics``.

    Returns a long DataFrame with columns: ``source``, ``metric``, ``column``,
    ``category``, ``direction``, ``level``, ``n_tokenizers_with_value``,
    ``where``. ``column`` is the wide-CSV column name when applicable (else
    empty); ``where`` is ``"eval_report"`` or ``"csv"``.
    """
    rows: list[dict[str, object]] = []

    if include_eval_reports and cache_dir is not None:
        cache = Path(cache_dir)
        manifest_path = cache / "manifest.csv"
        if manifest_path.exists():
            manifest = pd.read_csv(manifest_path)
            # Group the report files we know about. Source labels mirror the
            # CSV prefixes (``recon``, ``latent_input_corruption``, ...) so the
            # ``where`` column on the catalog is the only thing that
            # distinguishes eval-report-side from CSV-side rows.
            report_columns: list[tuple[str, str]] = [("recon", "eval_report_path")]
            for col in manifest.columns:
                if col.endswith("_eval_report_path") and col != "eval_report_path":
                    source = col.removesuffix("_eval_report_path")
                    report_columns.append((source, col))

            for source_label, path_col in report_columns:
                seen: dict[str, dict[str, object]] = {}
                tokenizers_with_metric: dict[str, set[str]] = {}
                for _, row in manifest.iterrows():
                    rel = row.get(path_col)
                    if not isinstance(rel, str) or not rel:
                        continue
                    full = cache / rel
                    if not full.exists():
                        continue
                    try:
                        report = load_report(full)
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.debug("skip %s for %s: %r", path_col, row.get("tokenizer"), exc)
                        continue
                    tokenizer_name = str(row["tokenizer"])
                    for metric_name, result in report.results.items():
                        seen.setdefault(
                            metric_name,
                            {
                                "category": result.category.value,
                                "direction": result.direction.value,
                                "level": result.level.value,
                            },
                        )
                        tokenizers_with_metric.setdefault(metric_name, set()).add(tokenizer_name)
                for metric_name in sorted(seen):
                    rows.append(
                        {
                            "source": source_label,
                            "metric": metric_name,
                            "column": "",
                            "category": seen[metric_name]["category"],
                            "direction": seen[metric_name]["direction"],
                            "level": seen[metric_name]["level"],
                            "n_tokenizers_with_value": len(tokenizers_with_metric[metric_name]),
                            "where": "eval_report",
                        }
                    )

    if df is not None:
        # Collapse (source, metric) duplicates: a metric like
        # ``encoding_rate_pearson_psth`` may be exposed by both
        # ``encoding_rate_pearson_psth`` and ``encoding_rate_pearson_psth_mean`` —
        # keep the ``_mean`` version when present.
        seen_csv: dict[tuple[str, str], dict[str, object]] = {}
        for col in df.columns:
            if not isinstance(col, str):
                continue
            src = _csv_source_for(col)
            if src is None:
                continue
            source_label, prefix = src
            base = _csv_metric_base(col, prefix)
            if base is None:
                continue
            metric_name, value_col = base
            if not _looks_like_csv_metric(value_col, df):
                continue
            n = int(df[value_col].notna().sum())
            key = (source_label, metric_name)
            existing = seen_csv.get(key)
            if existing is not None:
                # Prefer ``_mean`` over the bare column when both are present.
                if value_col.endswith("_mean") and not str(existing["column"]).endswith("_mean"):
                    seen_csv[key] = {"column": value_col, "n": n}
                continue
            seen_csv[key] = {"column": value_col, "n": n}

        for (source_label, metric_name), info in seen_csv.items():
            rows.append(
                {
                    "source": source_label,
                    "metric": metric_name,
                    "column": info["column"],
                    "category": "",
                    "direction": "",
                    "level": "",
                    "n_tokenizers_with_value": info["n"],
                    "where": "csv",
                }
            )

    columns = ["source", "metric", "column", "category", "direction", "level", "n_tokenizers_with_value", "where"]
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})
    # Drop CSV-side rows when the same (source, metric) is already exposed
    # through a cached eval_report (which carries category/direction/level
    # metadata). After this step the catalog has at most one row per
    # (source, metric) pair, with the eval-report-side variant winning.
    eval_pairs = {(r["source"], r["metric"]) for r in rows if r["where"] == "eval_report"}
    deduped = [r for r in rows if not (r["where"] == "csv" and (r["source"], r["metric"]) in eval_pairs)]
    deduped.sort(key=lambda r: (str(r["where"]), str(r["source"]), str(r["metric"])))
    catalog = pd.DataFrame.from_records(deduped)
    catalog = cast(pd.DataFrame, catalog[columns])
    return catalog.reset_index(drop=True)


def print_metric_catalog(
    catalog: pd.DataFrame,
    *,
    show_zero_coverage: bool = False,
) -> None:
    """Pretty-print a catalog produced by ``list_available_metrics``.

    Groups by ``source`` and prints one line per metric with its coverage and,
    where known, category/direction. Pass ``show_zero_coverage=True`` to keep
    metrics that no tokenizer has a finite value for (otherwise they're hidden).
    """
    df = catalog if show_zero_coverage else catalog[catalog["n_tokenizers_with_value"] > 0]
    if df.empty:
        print("(no metrics)")
        return
    for source, group in df.groupby("source", sort=True):
        print(f"== {source} ({len(group)} metrics) ==")
        for _, row in group.iterrows():
            where = str(row["where"] or "")
            where_tag = f" <{where}>" if where else ""
            category = str(row["category"] or "")
            direction = str(row["direction"] or "")
            meta_tag = f" [{category}, {direction}]" if category else ""
            column = str(row["column"] or "")
            col = f"  column={column}" if column else ""
            print(f"  {row['source']}/{row['metric']}{where_tag}{meta_tag}  (n={row['n_tokenizers_with_value']}){col}")
        print()


def _radar_projection(num_vars: int) -> tuple[np.ndarray, str]:
    """Register a radar projection with polygon gridlines and return its angles/name."""
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    projection_name = f"ntt_radar_polygon_{num_vars}"

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return MplPath(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        PolarTransform = RadarTransform
        name = projection_name

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location("N")
            self.set_theta_direction(-1)

        def fill(self, *args, closed: bool = True, **kwargs):
            return super().fill(*args, closed=closed, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines

        def _close_line(self, line) -> None:
            x, y = line.get_data()
            if len(x) and x[0] != x[-1]:
                line.set_data(np.append(x, x[0]), np.append(y, y[0]))

        def set_varlabels(self, labels: list[str]) -> None:
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="none")

        def _gen_axes_spines(self):
            spine = Spine(axes=self, spine_type="circle", path=MplPath.unit_regular_polygon(num_vars))
            spine.set_transform(Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes)
            return {"polar": spine}

    register_projection(RadarAxes)
    return theta, projection_name


def plot_radar(
    scores_by_tokenizer: Mapping[str, Mapping[str, float]] | pd.DataFrame,
    *,
    title: str | None = None,
    mode: str = "normed",
    figsize: tuple[float, float] = (8, 8),
    clip_range: tuple[float, float] | None = None,
    floor_negative: bool = True,
    floor_name: str = "Poisson floor",
    display_floor: bool | None = None,
    show_ceiling_ring: bool | None = None,
    cmap: str = "tab10",
    fill_alpha: float = 0.18,
    legend_loc: str | None = "upper right",
    category_label_pad: float = 18.0,
    ax: Axes | None = None,
) -> Figure:
    """Overlay one polygon per tokenizer on a polar radar.

    Parameters
    ----------
    mode:
        ``"normed"`` (default) — values are Poisson-normalized scores. ``0``
        corresponds to the tokenizer floor and ``1`` to the perfect-
        reconstruction ceiling. The dashed floor / solid ceiling guide rings
        are drawn and the title gets a ``(NORMED)`` suffix.

        ``"raw"`` — values are interpreted as raw metric values. Floor /
        ceiling rings are hidden by default (the canonical 0/1 positions don't
        align with raw units), the radial range is auto-derived from the data,
        and the title gets a ``(RAW)`` suffix.

        Any other string is appended to the title verbatim, in caps —
        e.g., ``mode="z-score"`` becomes ``(Z-SCORE)``.
    clip_range:
        ``(lower, upper)`` for the radial axis. Defaults to ``(-0.5, 1.0)`` in
        ``"normed"`` mode and to a tight envelope around the data in ``"raw"``
        mode. ``floor_negative=False`` clips the lower bound at ``0``.
    display_floor / show_ceiling_ring:
        Override the mode default. ``None`` (default) means follow the mode.
    """
    if isinstance(scores_by_tokenizer, pd.DataFrame):
        df = scores_by_tokenizer
    else:
        df = pd.DataFrame(scores_by_tokenizer).T

    if df.empty:
        raise ValueError("no tokenizers / categories to plot")

    categories = list(df.columns)
    n = len(categories)
    angles, projection_name = _radar_projection(n)
    angles_closed = np.concatenate([angles, angles[:1]])

    is_normed = mode == "normed"

    if clip_range is None:
        if is_normed:
            clip_range = (-0.5, 1.0)
        else:
            data = df.to_numpy(dtype=float)
            finite = data[np.isfinite(data)]
            if finite.size == 0:
                clip_range = (0.0, 1.0)
            else:
                lo = float(np.nanmin(finite))
                hi = float(np.nanmax(finite))
                pad = max((hi - lo) * 0.1, 1e-9)
                clip_range = (lo - pad, hi + pad)

    requested_lower, upper = clip_range
    lower = requested_lower if floor_negative else max(0.0, requested_lower)
    if lower >= upper:
        raise ValueError("clip_range lower bound must be less than upper bound")

    # Resolve mode-dependent defaults for the guide rings.
    if display_floor is None:
        display_floor = is_normed
    if show_ceiling_ring is None:
        show_ceiling_ring = is_normed

    # Append a CAPS mode tag to the title so the active mode is unambiguous.
    if title is not None:
        mode_tag = mode.upper() if mode else ""
        if mode_tag and f"({mode_tag})" not in title:
            title = f"{title}  ({mode_tag})"

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": projection_name})
    else:
        parent_fig = ax.figure
        if not isinstance(parent_fig, Figure):
            raise TypeError("plot_radar requires an Axes attached to a top-level Figure")
        fig = parent_fig
        set_theta_zero_location = getattr(ax, "set_theta_zero_location", None)
        if callable(set_theta_zero_location):
            set_theta_zero_location("N")
        set_theta_direction = getattr(ax, "set_theta_direction", None)
        if callable(set_theta_direction):
            set_theta_direction(-1)

    polar_ax = cast(PolarAxes, ax)

    polar_ax.set_facecolor("white")
    polar_ax.spines["polar"].set_visible(False)
    polar_ax.grid(color="0.82", linewidth=0.8)
    polar_ax.xaxis.grid(True, color="0.82", linewidth=0.8)
    polar_ax.yaxis.grid(True, color="0.82", linewidth=0.8)

    if display_floor:
        polar_ax.plot(
            angles_closed,
            [0.0 - lower] * (n + 1),
            color="0.25",
            linestyle="--",
            linewidth=1,
            label=floor_name,
        )
    if show_ceiling_ring:
        polar_ax.plot(
            angles_closed,
            [1.0 - lower] * (n + 1),
            color="black",
            linestyle="-",
            linewidth=1.2,
            label="Perfect ceiling",
        )

    colors = plt.get_cmap(cmap)
    for idx, (tokenizer, row) in enumerate(df.iterrows()):
        clipped = np.clip(row.to_numpy(dtype=float), lower, upper) - lower
        color = colors(idx % colors.N)
        polar_ax.plot(angles, clipped, marker="o", markersize=4, linewidth=2, label=tokenizer, color=color)
        polar_ax.fill(angles, clipped, alpha=fill_alpha, color=color, label="_nolegend_")

    polar_ax.set_xticks(angles)
    polar_ax.set_xticklabels(categories, size=9)
    polar_ax.tick_params(axis="x", pad=category_label_pad)
    polar_ax.set_ylim(0, upper - lower)
    if is_normed:
        tick_values = list(dict.fromkeys(t for t in [lower, 0.0, 0.5, 1.0] if lower <= t <= upper))
        tick_fmt = "{:.1f}"
    else:
        tick_values = list(np.linspace(lower, upper, 5))
        tick_fmt = "{:.2g}"
    polar_ax.set_yticks([t - lower for t in tick_values])
    polar_ax.set_yticklabels([tick_fmt.format(t) for t in tick_values], size=8)
    polar_ax.set_rlabel_position(90)
    if title:
        polar_ax.set_title(title, pad=24, weight="bold")
    if legend_loc:
        polar_ax.legend(loc=cast(Any, legend_loc), bbox_to_anchor=(1.25, 1.05), fontsize=9, frameon=False)
    fig.tight_layout()
    return fig


__all__ = [
    "CategorySpec",
    "RadarBundle",
    "MetricComparison",
    "SheetMetric",
    "build_sheet_metrics",
    "category_scores_for",
    "collect_category_scores",
    "inject_sheet_metrics",
    "list_available_metrics",
    "load_report",
    "plot_radar",
    "print_metric_catalog",
    "tokenizers_with_data",
]
