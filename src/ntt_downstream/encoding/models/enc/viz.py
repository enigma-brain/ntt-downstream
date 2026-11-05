"""Eval-time plotting helpers for neuron trace comparisons.

Used by the Lightning modules to log original-vs-reconstructed neural response
figures to MLflow during validation. All functions here are pure (no I/O,
no Lightning/MLflow coupling) so they can be unit-tested in isolation.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def pick_random_neurons(n_total: int, k: int, seed: int) -> list[int]:
    """Return k unique neuron indices sampled from [0, n_total).

    Deterministic given (n_total, k, seed). If k >= n_total, returns all
    indices in ascending order.
    """
    if k >= n_total:
        return list(range(n_total))
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_total), k=k))


def pick_diverse_neurons(rate_corrs: torch.Tensor, k: int = 10) -> list[int]:
    """Return up to k neuron indices spanning top, middle, and bottom performers.

    Args:
        rate_corrs: (N,) per-neuron rate correlation. NaNs are treated as -inf.
        k: target number of indices (default 10 = 3 top + 4 mid + 3 bottom).

    If N <= k, returns all indices. Otherwise returns up to k unique indices
    concatenated as top(3) + mid(4) + bottom(3), dedup-preserving-order.
    """
    n = rate_corrs.shape[0]
    if n <= k:
        return list(range(n))

    # NaN-safe sort: replace NaN with -inf so they rank at the bottom.
    r = rate_corrs.clone()
    r[torch.isnan(r)] = float("-inf")
    order = torch.argsort(r, descending=True).tolist()  # best → worst

    top = order[:3]
    bot = order[-3:]
    # Four contiguous ranks centred at the median.
    mid_start = n // 2 - 2
    mid = order[mid_start : mid_start + 4]

    # Dedup preserving top → mid → bottom order.
    seen: set[int] = set()
    out: list[int] = []
    for i in top + mid + bot:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out[:k]


def plot_neuron_traces(
    pred: torch.Tensor,
    target: torch.Tensor,
    neuron_idx: Sequence[int],
    *,
    session_id: str,
    global_step: int,
    kind: str,
    pearson_r: torch.Tensor | None = None,
    time_axis_hz: float | None = None,
    pred_psth: torch.Tensor | None = None,
    target_psth: torch.Tensor | None = None,
) -> Figure:
    """Plot target vs. predicted neural responses for ``neuron_idx`` in a 2x5 grid.

    Args:
        pred: (T, N) predicted responses (single trial).
        target: (T, N) ground-truth responses (single trial).
        neuron_idx: indices into N; up to 10 entries populate the grid.
        session_id: shown in suptitle.
        global_step: shown in suptitle.
        kind: "random" or "diverse" (shown in suptitle and artifact path).
        pearson_r: optional (N,) per-neuron rate-corr; shown in panel titles.
        time_axis_hz: if set, x-axis is shown in seconds rather than bins.
        pred_psth: optional (T, N) trial-averaged prediction for the same stimulus
            identity as ``pred``.
        target_psth: optional (T, N) trial-averaged response for the same stimulus
            identity as ``target``.

    Returns:
        matplotlib.figure.Figure. Caller owns closing it.
    """
    import matplotlib.pyplot as plt

    pred_np = pred.detach().float().cpu().numpy()
    target_np = target.detach().float().cpu().numpy()
    pred_psth_np = pred_psth.detach().float().cpu().numpy() if pred_psth is not None else None
    target_psth_np = target_psth.detach().float().cpu().numpy() if target_psth is not None else None
    T = pred_np.shape[0]

    if time_axis_hz is not None and time_axis_hz > 0:
        x = torch.arange(T, dtype=torch.float32).numpy() / float(time_axis_hz)
        xlabel = "time (s)"
    else:
        x = torch.arange(T, dtype=torch.float32).numpy()
        xlabel = "time bin"

    fig, axes = plt.subplots(2, 5, figsize=(18, 6), squeeze=False)
    axes_flat = axes.flatten()

    for ax_i, ax in enumerate(axes_flat):
        if ax_i < len(neuron_idx):
            n = int(neuron_idx[ax_i])
            ax.plot(x, target_np[:, n], color="0.5", linewidth=0.8, label="target")
            ax.plot(x, pred_np[:, n], color="tab:blue", linewidth=1.5, label="pred")
            if target_psth_np is not None:
                ax.plot(x, target_psth_np[:, n], color="0.1", linewidth=1.6, label="target PSTH")
            if pred_psth_np is not None:
                ax.plot(x, pred_psth_np[:, n], color="tab:orange", linewidth=1.6, label="pred PSTH")
            title = f"n{n}"
            if pearson_r is not None:
                r_val = float(pearson_r[n].item())
                title += f" r={r_val:+.2f}"
            ax.set_title(title, fontsize=9)
            if ax_i == 0:
                ax.legend(loc="upper right", fontsize=7)
            if ax_i >= 5:
                ax.set_xlabel(xlabel, fontsize=8)
        else:
            ax.set_visible(False)

    fig.suptitle(f"{session_id} | step={global_step} | {kind}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig
