"""Parse encoding ``callback_metrics`` snapshots into a shared ``MetricBlock``.

Recognizes the keys logged by ``EncodingLitModule`` / ``AudioEncodingLitModule``:
``val/rate_corr/{session|overall}``, ``val/corr/{session|overall}`` (temporal),
``val_nc_r2/{session|overall|median}``, ``val/rate_corr_trial_avg/overall``,
``val/corr_trial_avg/overall``, ``val/latent_*``, ``val/{rate_corr,corr}_subject/*``,
and ``train/*`` diagnostics. Missing keys are simply omitted from the block.
"""

from __future__ import annotations

from ntt_downstream.common.results.schema import MetricBlock, MetricEntry

_CORR_RANGE = (-1.0, 1.0)
_R2_RANGE = (None, 1.0)


def parse_encoding_block(
    metrics: dict[str, float],
    *,
    session_info: dict | None = None,
) -> MetricBlock:
    block = MetricBlock()

    # --- aggregates ---
    def add_agg(name: str, key: str, value_range, median_key: str | None = None) -> None:
        if key in metrics:
            entry = MetricEntry(value=metrics[key], direction="higher", value_range=value_range)
            if median_key is not None and median_key in metrics:
                entry.median = metrics[median_key]
            block.aggregated[name] = entry

    add_agg("rate_corr", "val/rate_corr/overall", _CORR_RANGE)
    add_agg("temporal_corr", "val/corr/overall", _CORR_RANGE)
    add_agg("nc_r2", "val_nc_r2/overall", _R2_RANGE, median_key="val_nc_r2/median")
    add_agg("rate_corr_trial_avg", "val/rate_corr_trial_avg/overall", _CORR_RANGE)
    add_agg("temporal_corr_trial_avg", "val/corr_trial_avg/overall", _CORR_RANGE)
    add_agg("latent_cosine_sim", "val/latent_cosine_sim", _CORR_RANGE)
    add_agg("latent_r2", "val/latent_r2", _R2_RANGE)

    # --- per session ---
    session_ids = list(session_info.keys()) if session_info else _infer_sessions(metrics)
    per_session: dict[str, dict[str, float | None]] = {}
    for sess in session_ids:
        entry: dict[str, float | None] = {}
        for out_name, prefix in (
            ("rate_corr", "val/rate_corr/"),
            ("temporal_corr", "val/corr/"),
            ("nc_r2", "val_nc_r2/"),
        ):
            key = f"{prefix}{sess}"
            if key in metrics:
                entry[out_name] = metrics[key]
        if entry:
            per_session[sess] = entry
    block.per_session = per_session

    # --- per subject ---
    per_subject: dict[str, dict[str, float]] = {}
    for key, val in metrics.items():
        for out_name, prefix in (
            ("rate_corr", "val/rate_corr_subject/"),
            ("temporal_corr", "val/corr_subject/"),
        ):
            if key.startswith(prefix):
                subj = key[len(prefix) :]
                per_subject.setdefault(subj, {})[out_name] = val
    if per_subject:
        block.groups["per_subject"] = per_subject

    # --- train diagnostics ---
    train: dict[str, float] = {}
    for out_name, key in (
        ("rate_corr", "train/rate_corr/overall"),
        ("temporal_corr", "train/corr/overall"),
        ("loss", "train/loss"),
    ):
        if key in metrics:
            train[out_name] = metrics[key]
    if train:
        block.groups["train"] = train

    return block


def _infer_sessions(metrics: dict[str, float]) -> list[str]:
    """Derive session ids from ``val/rate_corr/<session>`` keys (excluding overall)."""
    prefix = "val/rate_corr/"
    sessions = set()
    for key in metrics:
        if key.startswith(prefix):
            rest = key[len(prefix) :]
            if rest != "overall":
                sessions.add(rest)
    return sorted(sessions)
