"""Evaluation utilities for neural encoding models.

Provides reusable metric primitives and temporal rebinning, used by the
Lightning training loop (``encoding_module.py``) and the final evaluation
(``training/final_eval.py``).
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Temporal rebinning
# ---------------------------------------------------------------------------


def rebin_time_axis(
    x: torch.Tensor,
    from_hz: float,
    to_hz: float | None,
    method: str = "sum_preserve",
) -> torch.Tensor:
    """Rebin a (B, T, N) tensor from from_hz to to_hz.

    Uses overlap-weighted aggregation that preserves total count mass.
    """
    if to_hz is None:
        return x
    if from_hz <= 0 or to_hz <= 0:
        raise ValueError(f"Sampling rates must be positive. Got from_hz={from_hz}, to_hz={to_hz}")
    if x.ndim != 3:
        raise ValueError(f"Expected tensor with shape (B, T, N), got {tuple(x.shape)}")
    if abs(from_hz - to_hz) < 1e-12:
        return x
    target_t = max(1, int(round(x.shape[1] * (to_hz / from_hz))))

    if method != "sum_preserve":
        raise ValueError(f"Unknown rebin method '{method}'. Only 'sum_preserve' is supported.")

    batch_size, source_t, n_neurons = x.shape
    source_bin_width = 1.0 / float(from_hz)
    target_bin_width = (float(source_t) / float(from_hz)) / float(target_t)

    src_edges = torch.arange(source_t + 1, dtype=torch.float64) * source_bin_width
    tgt_edges = torch.arange(target_t + 1, dtype=torch.float64) * target_bin_width

    tgt_start = tgt_edges[:-1].unsqueeze(1)
    tgt_end = tgt_edges[1:].unsqueeze(1)
    src_start = src_edges[:-1].unsqueeze(0)
    src_end = src_edges[1:].unsqueeze(0)

    overlap = (torch.minimum(tgt_end, src_end) - torch.maximum(tgt_start, src_start)).clamp(min=0)
    weights = (overlap / source_bin_width).to(device=x.device, dtype=x.dtype)

    return torch.einsum("bsn,ts->btn", x, weights)


# ---------------------------------------------------------------------------
# Correlation primitives
# ---------------------------------------------------------------------------


def batch_temporal_correlation(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-image temporal Pearson correlation (corr_stop3d style).

    Correlates predicted and observed time courses independently for each
    (image, neuron) pair, then returns the per-pair values.

    Args:
        pred: (B, T, N) predicted responses.
        target: (B, T, N) observed responses.
        eps: numerical stability constant.

    Returns:
        (B, N) Pearson correlations.  NaN where either signal is constant.
    """
    pred_c = pred - pred.mean(dim=1, keepdim=True)
    target_c = target - target.mean(dim=1, keepdim=True)

    numer = (pred_c * target_c).sum(dim=1)  # (B, N)
    denom = pred_c.pow(2).sum(dim=1).sqrt() * target_c.pow(2).sum(dim=1).sqrt()  # (B, N)

    corr = torch.where(denom > eps, numer / denom, torch.tensor(float("nan"), device=pred.device))
    return corr


# ---------------------------------------------------------------------------
# Metric accumulators
# ---------------------------------------------------------------------------


class OnlinePearsonAccumulator:
    """Incrementally accumulates sufficient statistics for per-neuron Pearson r.

    For each session, tracks ``(sum_x, sum_y, sum_xx, sum_yy, sum_xy, n)``
    per neuron.  After all batches, ``compute()`` returns the exact Pearson
    correlation across the accumulated samples.

    Typical usage::

        acc = OnlinePearsonAccumulator()
        for pred_rate, target_rate in batches:
            acc.update(session_key, pred_rate, target_rate, n_units)
        per_session_corr = acc.compute()   # {session: (N,)}
    """

    def __init__(self) -> None:
        self._stats: dict[str, dict[str, torch.Tensor]] = {}

    def update(self, key: str, x: torch.Tensor, y: torch.Tensor, n_units: int) -> None:
        """Accumulate a batch.  ``x, y``: (B, N)."""
        if key not in self._stats:
            self._stats[key] = {
                "sx": torch.zeros(n_units, device=x.device),
                "sy": torch.zeros(n_units, device=x.device),
                "sxx": torch.zeros(n_units, device=x.device),
                "syy": torch.zeros(n_units, device=x.device),
                "sxy": torch.zeros(n_units, device=x.device),
                "n": torch.zeros(n_units, device=x.device),
            }
        s = self._stats[key]
        s["sx"] += x.sum(dim=0)
        s["sy"] += y.sum(dim=0)
        s["sxx"] += (x * x).sum(dim=0)
        s["syy"] += (y * y).sum(dim=0)
        s["sxy"] += (x * y).sum(dim=0)
        s["n"] += x.shape[0]

    def compute(self) -> dict[str, torch.Tensor]:
        """Return ``{session: (N,)}`` Pearson correlations. NaN where variance is zero."""
        out: dict[str, torch.Tensor] = {}
        for key, s in self._stats.items():
            n = s["n"]
            numer = n * s["sxy"] - s["sx"] * s["sy"]
            denom = (n * s["sxx"] - s["sx"] ** 2).sqrt() * (n * s["syy"] - s["sy"] ** 2).sqrt()
            corr = torch.where(denom > 1e-12, numer / denom, torch.tensor(float("nan"), device=n.device))
            out[key] = corr
        return out

    def reset(self) -> None:
        self._stats.clear()


class TemporalCorrAccumulator:
    """Accumulates per-image temporal correlations across batches.

    Each ``update`` adds (B, N) correlations from ``batch_temporal_correlation``.
    ``compute()`` returns the image-weighted mean per neuron (NaN excluded).
    """

    def __init__(self) -> None:
        self._sums: dict[str, torch.Tensor] = {}
        self._counts: dict[str, torch.Tensor] = {}

    def update(self, key: str, corr_bn: torch.Tensor, n_units: int) -> None:
        """Accumulate.  ``corr_bn``: (B, N), may contain NaN."""
        if key not in self._sums:
            self._sums[key] = torch.zeros(n_units, device=corr_bn.device)
            self._counts[key] = torch.zeros(n_units, device=corr_bn.device)
        valid = ~torch.isnan(corr_bn)
        self._sums[key] += corr_bn.nan_to_num(0.0).sum(dim=0)
        self._counts[key] += valid.float().sum(dim=0)

    def compute(self) -> dict[str, torch.Tensor]:
        """Return ``{session: (N,)}`` mean temporal correlations."""
        out: dict[str, torch.Tensor] = {}
        for key in self._sums:
            counts = self._counts[key]
            mean = self._sums[key] / counts.clamp(min=1)
            mean = torch.where(counts > 0, mean, torch.tensor(float("nan"), device=mean.device))
            out[key] = mean
        return out

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()
