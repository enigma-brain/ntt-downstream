"""End-of-training final evaluation.

Produces a per-session/overall metric table over single-trial val + repeat-averaged
val ("PSTH") loaders. The loss-mode-aware rate prediction lives in the shared
``prediction.predict_and_align`` helper (also used by the periodic Lightning val),
so the two cannot drift; final_eval just walks the loaders and accumulates metrics.
Loss-mode behavior (see ``prediction.predict_and_align``):

    poisson, mse_rate     -> direct rate from model(images, key)
    mse, smooth_l1        -> 1x sample_tokens + decode_differentiable (deterministic)
    flow_matching         -> MC mean of N samples via sample_tokens(flow_solver=heun)
    diffusion, energy     -> MC mean of N samples via sample_tokens
    bce (BSQ)             -> MC mean of N samples via sample_tokens
    cross_entropy, Q=1    -> exact per-position marginal expectation (codebook table einsum)
    cross_entropy, Q>1    -> MC mean of N samples via sample_tokens
    image_encoding_mar    -> MC mean of N model.generate() calls (deterministic for mse/smooth_l1)

Per-session row reports mean +/- std across neurons for each metric. The "overall"
row reports mean-of-session-means and SE = std(session_means)/sqrt(n_sessions).

Splits. The checkpoint is always selected on val. When test loaders are passed,
the same checkpoint is also scored on the test split (single-trial test +
repeat-averaged test). ``run`` returns the val table with the test table nested
under ``test`` (same shape); each carries a ``split`` marker ("valid" / "test").
``encoding/train.py`` turns them into the record's ``val_final`` and ``test``
blocks via ``table_to_metric_block``.
"""

from __future__ import annotations

import csv
import logging
import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch

from ntt_downstream.common.results.schema import MetricBlock, MetricEntry
from ntt_downstream.encoding.models.enc.hiera.evals import (
    OnlinePearsonAccumulator,
    TemporalCorrAccumulator,
    batch_temporal_correlation,
)
from ntt_downstream.encoding.models.enc.hiera.noise_corrected_r2 import noise_corrected_r2
from ntt_downstream.encoding.training.prediction import predict_and_align

logger = logging.getLogger(__name__)


# Metrics computed on each loader. These are the dict keys in returned tables.
METRIC_KEYS_SINGLE = ("rate_pearson", "temp_pearson", "nc_r2")
METRIC_KEYS_PSTH = ("rate_pearson_psth", "temp_pearson_psth")
METRIC_KEYS_ALL = METRIC_KEYS_SINGLE + METRIC_KEYS_PSTH


# ---------------------------------------------------------------------------
# Loss-mode-aware rate prediction
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_rates(
    lit,
    images: torch.Tensor,
    responses: torch.Tensor,
    session_key: str,
    *,
    n_samples: int,
    flow_solver: str,
    n_flow_steps: int,
    flow_shift: float = 1.0,
    ce_codebook_rates_cache: dict[str, list[torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return aligned ``(pred, target)`` as ``(B, T', N)`` via the shared helper.

    Thin wrapper around ``prediction.predict_and_align`` that pins final_eval's
    MC settings (``n_samples``/``flow_solver``/``n_flow_steps``/``flow_shift``)
    and the CE-Q1 codebook-rate cache (mutated in place). The shared helper owns
    the loss-mode dispatch, the LDNS/Wavelet boundary slice, and the rebin —
    keeping final_eval and the periodic Lightning val on one code path so their
    metrics agree.

    Sampling knobs are resolved HERE (caller-owned), not inside the helper.
    final_eval pins its own ``run()`` args for the sample_tokens path
    (``flow_solver``/``n_flow_steps``/``flow_shift``, ``temperature=1.0``,
    ``cfg_scale=1.0``) and, for MAR, mixes ``mar_eval``
    (``n_mar_steps``/``temperature``) with its run args
    (``n_flow_steps``/``flow_solver``/``flow_shift``) — the same shift is applied
    to both flow paths. This differs from the periodic path, which reads
    everything from ``cfg.trainer.tokenizer_eval`` / ``mar_eval``; hence the
    helper must not read cfg itself or one caller's defaults would clobber the
    other's.
    """
    cfg = lit.cfg
    mar_eval = cfg.trainer.get("mar_eval", {})
    return predict_and_align(
        lit,
        images,
        responses,
        session_key,
        loss_mode=str(cfg.model.get("loss_mode", "poisson")),
        model_type=str(cfg.model.get("model_type", "image_encoding")),
        n_samples=n_samples,
        flow_solver=flow_solver,
        n_flow_steps=n_flow_steps,
        sample_temperature=1.0,
        cfg_scale=1.0,
        flow_shift=flow_shift,
        mar_n_mar_steps=int(mar_eval.get("n_mar_steps", 8)),
        mar_n_flow_steps=n_flow_steps,
        mar_temperature=float(mar_eval.get("temperature", 1.0)),
        mar_flow_solver=flow_solver,
        mar_flow_shift=flow_shift,
        ce_cache=ce_codebook_rates_cache,
    )


# ---------------------------------------------------------------------------
# Loader walks
# ---------------------------------------------------------------------------


@torch.no_grad()
def _walk_loader(
    lit,
    loader_fn,
    *,
    n_samples: int,
    flow_solver: str,
    n_flow_steps: int,
    flow_shift: float,
    compute_nc_r2: bool,
    ce_cache: dict[str, list[torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    """Walk a loader once, return per-session per-metric (N,) tensors on CPU.

    Output shape: ``{session: {metric_name: (N,) tensor}}``. Metric names:
    ``rate_pearson``, ``temp_pearson``, optionally ``nc_r2``.
    """
    rate_acc = OnlinePearsonAccumulator()
    temp_acc = TemporalCorrAccumulator()
    has_temporal = False
    session_n_units: dict[str, int] = {}

    # nc_r2 grouping (single-trial only): per-session (image_id -> {pred:[N], resp:[N]})
    nc_data: dict[str, dict[int, dict[str, list[torch.Tensor]]]] = defaultdict(
        lambda: defaultdict(lambda: {"pred": [], "resp": []})
    )

    model_dtype = next(lit.model.parameters()).dtype

    for key, batch_dict in loader_fn():
        images = batch_dict["image"].to(lit.device, model_dtype)
        responses = batch_dict["response"].to(lit.device, model_dtype)
        n_units = batch_dict["n_units"]
        session_n_units[key] = n_units

        # The shared helper returns the fully aligned (pred, target): loss-mode
        # rate estimate, LDNS/wavelet-tokenizer boundary slice, and rebin all applied. No
        # extra slicing is done here: prediction.predict_and_align owns it so the
        # final evaluation matches the periodic validation path.
        pred_btn, target_btn = predict_rates(
            lit,
            images,
            responses,
            key,
            n_samples=n_samples,
            flow_solver=flow_solver,
            n_flow_steps=n_flow_steps,
            flow_shift=flow_shift,
            ce_codebook_rates_cache=ce_cache,
        )

        pred_f = pred_btn.float()
        target_f = target_btn.float()

        rate_acc.update(key, pred_f.sum(dim=1), target_f.sum(dim=1), n_units)

        if pred_btn.shape[1] > 1:
            has_temporal = True
            corr = batch_temporal_correlation(pred_f, target_f)
            temp_acc.update(key, corr, n_units)

        if compute_nc_r2:
            image_ids = batch_dict["image_id"]
            pred_scalar = pred_f.sum(dim=1)  # (B, N)
            resp_scalar = target_f.sum(dim=1)
            for i in range(images.shape[0]):
                iid = int(image_ids[i].item())
                nc_data[key][iid]["pred"].append(pred_scalar[i].clone())
                nc_data[key][iid]["resp"].append(resp_scalar[i].clone())

    rate_corrs = rate_acc.compute()  # {key: (N,)}
    temp_corrs = temp_acc.compute() if has_temporal else {}

    nc_r2_corrs: dict[str, torch.Tensor] = {}
    if compute_nc_r2:
        for sid, image_dict in nc_data.items():
            n_units = session_n_units[sid]
            iids_sorted = sorted(image_dict.keys())
            n_images = len(iids_sorted)
            max_repeats = max(len(image_dict[iid]["resp"]) for iid in iids_sorted)
            if n_images < 2 or max_repeats < 2:
                logger.info(
                    "final_eval: skipping nc_r2 for %s (n_images=%d, max_repeats=%d)",
                    sid,
                    n_images,
                    max_repeats,
                )
                continue
            resp_all = torch.full(
                (n_units, n_images, max_repeats),
                float("nan"),
                device=lit.device,
            )
            pred_mean = torch.zeros(n_units, n_images, device=lit.device)
            for j, iid in enumerate(iids_sorted):
                resps = torch.stack(image_dict[iid]["resp"])  # (n_reps, N)
                preds = torch.stack(image_dict[iid]["pred"])  # (n_reps, N)
                n_reps = resps.shape[0]
                resp_all[:, j, :n_reps] = resps.T
                pred_mean[:, j] = preds.mean(dim=0)
            nc_r2_corrs[sid] = noise_corrected_r2(resp_all, pred_mean)

    out: dict[str, dict[str, torch.Tensor]] = {}
    sessions = set(rate_corrs) | set(temp_corrs) | set(nc_r2_corrs)
    for sid in sessions:
        out[sid] = {}
        if sid in rate_corrs:
            out[sid]["rate_pearson"] = rate_corrs[sid].detach().cpu()
        if sid in temp_corrs:
            out[sid]["temp_pearson"] = temp_corrs[sid].detach().cpu()
        if sid in nc_r2_corrs:
            out[sid]["nc_r2"] = nc_r2_corrs[sid].detach().cpu()
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _per_session_stat(values: torch.Tensor) -> dict[str, float]:
    """Return mean/std (across neurons) of a (N,) tensor; ignore NaN."""
    v = values.float()
    mask = torch.isfinite(v)
    n_valid = int(mask.sum().item())
    if n_valid == 0:
        return {"mean": float("nan"), "std": float("nan"), "n_valid": 0, "n_total": int(v.numel())}
    vv = v[mask]
    mean = float(vv.mean().item())
    std = float(vv.std(unbiased=True).item()) if vv.numel() > 1 else 0.0
    return {"mean": mean, "std": std, "n_valid": n_valid, "n_total": int(v.numel())}


def aggregate(
    val_metrics: dict[str, dict[str, torch.Tensor]],
    val_avg_metrics: dict[str, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    """Build the final-eval table.

    Per-session row: mean +/- std across neurons for each metric.
    Across-session row: mean of session means; SE = std(means)/sqrt(n_sessions).
    """
    sessions = sorted(set(val_metrics) | set(val_avg_metrics))

    # Map raw metrics into the unified namespace (PSTH metrics get _psth suffix).
    per_session: dict[str, dict[str, dict[str, float]]] = {}
    n_units_by_session: dict[str, int] = {}
    for sid in sessions:
        row: dict[str, dict[str, float]] = {}
        if sid in val_metrics:
            for m in METRIC_KEYS_SINGLE:
                if m in val_metrics[sid]:
                    row[m] = _per_session_stat(val_metrics[sid][m])
                    n_units_by_session.setdefault(sid, val_metrics[sid][m].numel())
        if sid in val_avg_metrics:
            mapping = {"rate_pearson": "rate_pearson_psth", "temp_pearson": "temp_pearson_psth"}
            for raw_key, out_key in mapping.items():
                if raw_key in val_avg_metrics[sid]:
                    row[out_key] = _per_session_stat(val_avg_metrics[sid][raw_key])
                    n_units_by_session.setdefault(sid, val_avg_metrics[sid][raw_key].numel())
        per_session[sid] = row

    # Across-session aggregate: mean-of-means, SE = std(means)/sqrt(K).
    overall: dict[str, dict[str, float]] = {}
    for metric in METRIC_KEYS_ALL:
        means = [
            per_session[sid][metric]["mean"]
            for sid in sessions
            if metric in per_session[sid] and not math.isnan(per_session[sid][metric]["mean"])
        ]
        if not means:
            overall[metric] = {"mean": float("nan"), "se": float("nan"), "n_sessions": 0}
            continue
        t = torch.tensor(means, dtype=torch.float64)
        n_s = int(t.numel())
        mean_of_means = float(t.mean().item())
        if n_s > 1:
            std_of_means = float(t.std(unbiased=True).item())
            se = std_of_means / math.sqrt(n_s)
        else:
            se = float("nan")
        overall[metric] = {"mean": mean_of_means, "se": se, "n_sessions": n_s}

    return {
        "sessions": sessions,
        "n_units_by_session": n_units_by_session,
        "per_session": per_session,
        "overall": overall,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def format_table(table: dict[str, Any]) -> str:
    """Pretty-print the per-session and overall rows as a fixed-width table."""
    sessions = table["sessions"]
    per_session = table["per_session"]
    overall = table["overall"]
    n_units = table["n_units_by_session"]

    metrics = METRIC_KEYS_ALL
    headers = ["session", "n"] + [f"{m}" for m in metrics]
    col_widths = [max(20, max((len(s) for s in sessions), default=0))]
    col_widths.append(6)
    col_widths.extend([22] * len(metrics))

    def _fmt_cell(stat: dict[str, float] | None) -> str:
        if stat is None or math.isnan(stat.get("mean", float("nan"))):
            return "      —      "
        return f"{stat['mean']:+.4f} ± {stat['std']:.4f}"

    def _fmt_overall_cell(stat: dict[str, float]) -> str:
        if math.isnan(stat.get("mean", float("nan"))):
            return "      —      "
        return f"{stat['mean']:+.4f} ± {stat['se']:.4f} (SE)"

    lines = []
    lines.append(" | ".join(h.ljust(w) for h, w in zip(headers, col_widths, strict=False)))
    lines.append("-+-".join("-" * w for w in col_widths))
    for sid in sessions:
        row_cells = [sid.ljust(col_widths[0]), str(n_units.get(sid, "—")).ljust(col_widths[1])]
        for m, w in zip(metrics, col_widths[2:], strict=True):
            cell = _fmt_cell(per_session.get(sid, {}).get(m))
            row_cells.append(cell.ljust(w))
        lines.append(" | ".join(row_cells))
    lines.append("-+-".join("-" * w for w in col_widths))
    n_sessions = overall[metrics[0]].get("n_sessions", 0)
    overall_row = ["overall".ljust(col_widths[0]), f"K={n_sessions}".ljust(col_widths[1])]
    for m, w in zip(metrics, col_widths[2:], strict=False):
        overall_row.append(_fmt_overall_cell(overall[m]).ljust(w))
    lines.append(" | ".join(overall_row))
    return "\n".join(lines)


# Every final-eval metric is a correlation or noise-corrected R^2 — higher is
# better — so the schema ``direction`` is uniformly "higher".
_FINAL_METRIC_DIRECTION = "higher"
# Same ranges as the periodic-val blocks (results_parser), so one metric never
# carries two different ranges within a record.
_FINAL_METRIC_RANGES = {
    "rate_pearson": (-1.0, 1.0),
    "temp_pearson": (-1.0, 1.0),
    "rate_pearson_psth": (-1.0, 1.0),
    "temp_pearson_psth": (-1.0, 1.0),
    "nc_r2": (None, 1.0),
}


def table_to_metric_block(table: dict[str, Any]) -> MetricBlock:
    """Convert a final-eval ``table`` into the shared ``MetricBlock`` schema.

    ``aggregated`` <- the table's ``overall`` row (mean -> ``value``, SE ->
    ``std``, n_sessions -> ``n``); ``per_session`` <- the table's per-session
    means (one float per metric, matching the schema's ``per_session`` shape).
    Lets the final-eval block be persisted with the same machine-readable schema
    as the run's best/last blocks instead of a bespoke layout.
    """
    aggregated: dict[str, MetricEntry] = {}
    for metric, stat in table.get("overall", {}).items():
        mean = stat.get("mean", float("nan"))
        if isinstance(mean, float) and math.isnan(mean):
            continue
        aggregated[metric] = MetricEntry(
            value=mean,
            direction=_FINAL_METRIC_DIRECTION,
            value_range=_FINAL_METRIC_RANGES.get(metric, (None, None)),
            std=stat.get("se"),
            n=stat.get("n_sessions"),
            aggregated_over="sessions:mean_of_means",
        )

    per_session: dict[str, dict[str, float | None]] = {}
    for sid, row in table.get("per_session", {}).items():
        entry: dict[str, float | None] = {}
        for metric, stat in row.items():
            mean = stat.get("mean", float("nan"))
            if not (isinstance(mean, float) and math.isnan(mean)):
                entry[metric] = mean
        if entry:
            per_session[sid] = entry

    return MetricBlock(aggregated=aggregated, per_session=per_session)


def write_outputs(
    table: dict[str, Any],
    output_dir: str | Path,
    *,
    prefix: str = "final_eval",
) -> dict[str, Path]:
    """Write the human-readable CSV report(s) for a final-eval table.

    The machine-readable record is the run's ``downstream_encoding_results.json``
    (``val_final`` / ``test`` blocks, written by ``encoding/train.py``); these
    CSVs are local reports only and are not uploaded. A nested test table gets
    its own ``<prefix>_test.csv``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"csv": _write_csv(table, out / f"{prefix}.csv")}
    if "test" in table:
        paths["test_csv"] = _write_csv(table["test"], out / f"{prefix}_test.csv")
    logger.info("final_eval: wrote %s", ", ".join(str(p) for p in paths.values()))
    return paths


def _write_csv(table: dict[str, Any], csv_path: Path) -> Path:
    """Write the wide-format human-readable report for one split's table."""
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["scope", "session", "n_neurons"]
        for m in METRIC_KEYS_ALL:
            header.extend([f"{m}_mean", f"{m}_dispersion"])
        writer.writerow(header)
        for sid in table["sessions"]:
            row = ["session", sid, table["n_units_by_session"].get(sid, "")]
            for m in METRIC_KEYS_ALL:
                stat = table["per_session"].get(sid, {}).get(m)
                if stat is None:
                    row.extend(["", ""])
                else:
                    row.extend([f"{stat['mean']:.6f}", f"{stat['std']:.6f}"])
            writer.writerow(row)
        n_sessions = table["overall"][METRIC_KEYS_ALL[0]].get("n_sessions", 0)
        row = ["overall", "", f"K={n_sessions}"]
        for m in METRIC_KEYS_ALL:
            stat = table["overall"][m]
            row.extend([f"{stat['mean']:.6f}", f"{stat['se']:.6f}"])
        writer.writerow(row)
    return csv_path


def _scalar_metrics(table: dict[str, Any], prefix: str) -> dict[str, float]:
    """Flatten one split's table into ``<prefix>/<metric>/...`` MLflow scalars."""
    metrics: dict[str, float] = {}
    for m, stat in table["overall"].items():
        if not math.isnan(stat["mean"]):
            metrics[f"{prefix}/{m}/overall_mean"] = stat["mean"]
        if not math.isnan(stat.get("se", float("nan"))):
            metrics[f"{prefix}/{m}/overall_se"] = stat["se"]
    for sid in table["sessions"]:
        for m in METRIC_KEYS_ALL:
            stat = table["per_session"].get(sid, {}).get(m)
            if stat and not math.isnan(stat["mean"]):
                metrics[f"{prefix}/{m}/session_mean/{sid}"] = stat["mean"]
                metrics[f"{prefix}/{m}/session_std/{sid}"] = stat["std"]
    return metrics


def log_to_mlflow(table: dict[str, Any], mlflow_logger) -> None:
    """Log overall mean/se and per-session means as MLflow scalar metrics.

    Val goes under ``final/`` (unchanged), a nested test table under
    ``final_test/`` so the two splits never overwrite each other.
    """
    if mlflow_logger is None:
        return
    metrics = _scalar_metrics(table, "final")
    if "test" in table:
        metrics.update(_scalar_metrics(table["test"], "final_test"))
    if metrics:
        try:
            mlflow_logger.log_metrics(metrics)
        except Exception as e:
            logger.warning("final_eval: MLflow logging failed: %r", e)


# ---------------------------------------------------------------------------
# Tokenizer cache warm-up + top-level orchestrator
# ---------------------------------------------------------------------------


def _warmup_tokenizer_caches(lit) -> None:
    """Populate per-session metadata in the tokenizer wrapper before decode-only calls.

    For tokenizer-based models, ``decode_differentiable`` reads cached
    ``_compress_t_session_meta``. The cache is populated during training as a
    side effect of model forward passes; for end-of-training final eval the
    cache is already warm, but we re-populate defensively so the function also
    works after a fresh checkpoint load.
    """
    if lit.tokenizer_wrapper is None or lit._val_dataloader_fn is None:
        return
    seen: set[str] = set()
    model_dtype = next(lit.model.parameters()).dtype
    for key, batch_dict in lit._val_dataloader_fn():
        if key in seen:
            continue
        seen.add(key)
        responses = batch_dict["response"].to(lit.device, model_dtype)
        try:
            lit.tokenizer_wrapper.encode(responses.transpose(1, 2).float(), key)
        except Exception as e:
            logger.warning("final_eval: tokenizer warm-up failed for %s: %r", key, e)


def _seed_mc(seed: int) -> None:
    """Pin the global RNG (CPU + CUDA) that MC sampling draws from."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate_split(
    lit,
    split: str,
    per_trial_fn: Callable[[], Iterable],
    avg_fn: Callable[[], Iterable],
    *,
    seed: int,
    ce_cache: dict[str, list[torch.Tensor]],
    **walk_kw: Any,
) -> dict[str, Any]:
    """Walk one split's single-trial + repeat-averaged loaders into a table.

    Reseeds first, so each split's MC numbers depend only on (checkpoint, seed)
    and not on how much sampling an earlier split consumed. ``ce_cache`` holds
    per-session decoder tables, which depend on the model and session only, so
    it is safely shared across splits.
    """
    _seed_mc(seed)
    logger.info("final_eval: walking %s loader", split)
    metrics = _walk_loader(lit, per_trial_fn, compute_nc_r2=True, ce_cache=ce_cache, **walk_kw)
    logger.info("final_eval: walking %s_average loader", split)
    avg_metrics = _walk_loader(lit, avg_fn, compute_nc_r2=False, ce_cache=ce_cache, **walk_kw)
    table = aggregate(metrics, avg_metrics)
    table["split"] = split
    return table


@torch.no_grad()
def run(
    lit,
    *,
    n_samples: int = 32,
    flow_solver: str = "heun",
    n_flow_steps: int | None = None,
    flow_shift: float | None = None,
    output_dir: str | Path | None = None,
    mlflow_logger=None,
    test_loader_fns: tuple[Callable[[], Iterable], Callable[[], Iterable]] | None = None,
) -> dict[str, Any]:
    """Top-level: walk val (+ optionally test) loaders, build table, write/log, return.

    Args:
        lit: a configured ``EncodingLitModule`` (model + tokenizer + cfg loaded).
        n_samples: MC samples for stochastic generative loss modes.
        flow_solver: flow ODE solver ("euler" or "heun") for flow_matching/MAR.
        n_flow_steps: integration steps; defaults to ``cfg.trainer.tokenizer_eval.n_flow_steps`` or 5.
        flow_shift: inference time-grid shift for flow heads (1.0 = uniform);
            defaults to ``cfg.trainer.tokenizer_eval.flow_shift`` or 1.0, so a
            configured shift is honored at final eval, not just periodic val.
        output_dir: where to write the CSV report(s). ``None`` skips file output.
        mlflow_logger: optional Lightning ``MLFlowLogger`` for scalar metric logging.
        test_loader_fns: optional ``(single_trial_fn, repeat_averaged_fn)`` for
            the test split. When given, the same (val-selected) weights are also
            scored on test and the result is nested under ``table["test"]``.
    """
    cfg = lit.cfg
    if n_flow_steps is None:
        n_flow_steps = int(cfg.trainer.get("tokenizer_eval", {}).get("n_flow_steps", 5))
    if flow_shift is None:
        flow_shift = float(cfg.trainer.get("tokenizer_eval", {}).get("flow_shift", 1.0))

    # Seed the global RNG for reproducible MC numbers (reseeded per split in
    # _evaluate_split). The stochastic loss modes (flow/diffusion/energy, bce,
    # CE-Q>1, MAR) MC-average samples drawn via `model.sample_tokens`/`generate`
    # -> `torch.multinomial`, which has NO generator hook (base_model.py) — so
    # the only way to make two final_eval passes on the same checkpoint produce
    # identical numbers is to pin the global RNG before walking any loader. We
    # deliberately do NOT seed the periodic-val sampling path: that runs mid-fit
    # and reseeding the global RNG there would perturb the training data order /
    # dropout / init draws and silently change the optimization trajectory.
    # Reproducibility of the *reported* final metrics is the goal; it is owned by
    # final_eval alone. If `cfg.seed` is absent we fall back to a fixed constant
    # so behavior is still deterministic across runs (and log which seed we used).
    seed = int(cfg.get("seed", 0))
    logger.info("final_eval: seeding MC sampling with seed=%d", seed)
    _seed_mc(seed)

    lit.model.eval()
    lit._ensure_tokenizer_device()
    _warmup_tokenizer_caches(lit)

    ce_cache: dict[str, list[torch.Tensor]] = {}
    walk_kw = {
        "n_samples": n_samples,
        "flow_solver": flow_solver,
        "n_flow_steps": n_flow_steps,
        "flow_shift": flow_shift,
    }
    logger.info(
        "final_eval: n_samples=%d, flow_solver=%s, n_flow_steps=%d, flow_shift=%s",
        n_samples,
        flow_solver,
        n_flow_steps,
        flow_shift,
    )

    table = _evaluate_split(
        lit, "valid", lit._val_dataloader_fn, lit._val_avg_dataloader_fn, seed=seed, ce_cache=ce_cache, **walk_kw
    )
    if test_loader_fns is not None:
        # A test-split failure (dm.setup("test"), the test walk) must not throw
        # away the already-computed val table: record it and carry on.
        try:
            table["test"] = _evaluate_split(lit, "test", *test_loader_fns, seed=seed, ce_cache=ce_cache, **walk_kw)
        except Exception as e:  # noqa: BLE001
            logger.warning("final_eval: test split failed: %r", e, exc_info=True)
            table["test_error"] = repr(e)

    table["config"] = {
        "loss_mode": str(cfg.model.get("loss_mode", "poisson")),
        "model_type": str(cfg.model.get("model_type", "image_encoding")),
        "n_samples": int(n_samples),
        "flow_solver": str(flow_solver),
        "n_flow_steps": int(n_flow_steps),
        "flow_shift": float(flow_shift),
    }
    # Diffusion: surface the schedule's step count for reporting only.
    diff = getattr(getattr(lit.model, "diffloss", None), "num_sampling_steps", None)
    if diff is not None:
        table["config"]["diffusion_num_sampling_steps"] = int(diff) if str(diff).isdigit() else str(diff)

    for split_table in [table] + ([table["test"]] if "test" in table else []):
        print(
            "\n=== final_eval {} (loss_mode={}, n_samples={}) ===".format(
                split_table["split"], table["config"]["loss_mode"], table["config"]["n_samples"]
            )
        )
        print(format_table(split_table))
        print()

    if output_dir is not None:
        write_outputs(table, output_dir)
    if mlflow_logger is not None:
        log_to_mlflow(table, mlflow_logger)

    return table
