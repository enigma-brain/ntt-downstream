"""Startup tokenizer roundtrip pre-fit check.

Runs a short encode → decode_differentiable roundtrip on the validation loader
before training begins, returning aggregate (non-per-session) metrics and two
diagnostic figures suitable for MLflow upload. The tokenizer is session-agnostic,
so per-session breakdowns add noise without signal — we aggregate over all
processed batches.

The roundtrip mirrors the *training* decode path so the reported ceiling
reflects what the model can actually achieve. That path differs by tokenizer
family:

* **Discrete** tokenizers (transformer-vq / BSQ / FSQ / VQPT): encode with
  ``raw_discrete=True`` to get integer codebook indices, then decode straight
  from indices (``decode_differentiable`` routes ints through
  ``indices_to_codes``). This matches the cross-entropy training target in
  ``base_model.py``.
* **Continuous** tokenizers (VAE / flow / diffusion / mse): ``raw_discrete`` is
  meaningless; the training decode path is
  ``_target_to_tokens(z.float()) → decode_differentiable`` (see
  ``base_model.py:1210-1212``). We replicate exactly that.

The family is selected by ``wrapper.produces_discrete_tokens`` — the same
``spec.latent_type`` discreteness signal the wrapper itself uses.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from ntt_downstream.encoding.models.enc.hiera.evals import OnlinePearsonAccumulator

if TYPE_CHECKING:  # pragma: no cover
    import matplotlib.figure

logger = logging.getLogger(__name__)


def _roundtrip_recon(wrapper, responses_btn: torch.Tensor, key: str) -> torch.Tensor:
    """Encode → decode one batch, matching the training decode path for the family.

    ``responses_btn`` is ``(B, T, N)``. Returns differentiable rates ``(B, T, N)``.

    Discrete tokenizers decode straight from integer indices; continuous ones
    re-derive raw latents via ``_target_to_tokens`` before decoding, exactly as
    the loss code in ``base_model.py`` does.
    """
    if wrapper.produces_discrete_tokens:
        enc = wrapper.encode(responses_btn, key, raw_discrete=True)
        return wrapper.decode_differentiable(enc["target"], session_key=key).float()

    # Continuous family: encode produces a scaled float target; invert the
    # scaling with _target_to_tokens to recover raw latents the decoder expects.
    enc = wrapper.encode(responses_btn, key)
    raw_latents = wrapper._target_to_tokens(enc["target"].float())
    return wrapper.decode_differentiable(raw_latents, session_key=key).float()


def _check_recon_health(recon: torch.Tensor, *, key: str, strict: bool) -> None:
    """Fail loud on a non-finite or degenerate (zero-variance) reconstruction.

    A broken tokenizer otherwise produces all-zeros / NaN rates that silently
    poison every downstream metric. We emit a prominent warning by default and
    raise ``RuntimeError`` when ``strict`` is set so a CI/launch can hard-fail.
    """
    n_total = recon.numel()
    if n_total == 0:
        return
    finite = torch.isfinite(recon)
    frac_finite = float(finite.float().mean().item())
    finite_vals = recon[finite]
    variance = float(finite_vals.var().item()) if finite_vals.numel() > 1 else 0.0

    problems = []
    if frac_finite < 1.0:
        problems.append(f"non-finite recon (finite fraction={frac_finite:.4f})")
    if variance <= 1e-12:
        problems.append(f"degenerate recon (variance={variance:.3e}, near-constant output)")

    if not problems:
        return

    msg = (
        f"[startup_roundtrip] BROKEN TOKENIZER for session {key!r}: "
        + "; ".join(problems)
        + ". The tokenizer ceiling diagnostic cannot be trusted."
    )
    if strict:
        raise RuntimeError(msg)
    logger.warning(msg)


@torch.no_grad()
def run_startup_roundtrip(
    wrapper,
    dataloader,
    *,
    native_hz: float,
    eval_rebin_hz: float | None,
    device: torch.device | str,
    max_batches: int = 20,
    strict: bool = False,
) -> tuple[dict[str, float], matplotlib.figure.Figure, matplotlib.figure.Figure]:
    """Encode → decode_differentiable, return aggregate metrics + two figures.

    Args:
        strict: if True, raise ``RuntimeError`` on a non-finite/degenerate recon
            instead of merely warning. Threaded from
            ``cfg.trainer.startup_roundtrip.strict``.

    Returns:
        metrics: dict with ``tokenizer_ceiling/{rate_corr,temporal_corr,n_batches,n_neurons_total}``.
        fig_traces: 3-row low/mid/high firing-rate trace plot from the first batch.
        fig_scatter: per-neuron rate-corr vs mean firing rate, pooled across all batches.
    """
    from ntt_downstream.encoding.models.enc.hiera.evals import batch_temporal_correlation, rebin_time_axis

    recon_hz = wrapper.reconstruction_rate_hz

    # Per-session online accumulator for the rate Pearson r. We also keep the
    # raw target rates per session so the scatter plot can pair (mean_rate, r)
    # per neuron after the accumulator finalizes.
    rate_acc = OnlinePearsonAccumulator()
    per_session_target_rate: dict[str, list[torch.Tensor]] = {}
    all_temporal_corrs: list[torch.Tensor] = []
    first_batch: dict | None = None

    # True neuron count per session is constant across that session's batches,
    # so record it once (not summed) to avoid the n_batches*N double-count.
    per_session_n_neurons: dict[str, int] = {}
    n_batches = 0

    for key, batch_dict in dataloader:
        if n_batches >= max_batches:
            break

        responses = batch_dict["response"].to(device, torch.float32)  # (B, N, T)
        responses_btn = responses.transpose(1, 2).contiguous()  # (B, T, N)

        # Tokenizer ops in float32 — bottleneck quantization / softplus rate
        # decoding are numerically sensitive, so we disable autocast here to
        # match how the training modules run the tokenizer.
        with torch.autocast(device_type="cuda", enabled=False):
            recon = _roundtrip_recon(wrapper, responses_btn, key)

        _check_recon_health(recon, key=key, strict=strict)

        original = responses_btn

        if eval_rebin_hz is not None:
            recon = rebin_time_axis(recon, from_hz=recon_hz, to_hz=eval_rebin_hz)
            original = rebin_time_axis(original, from_hz=native_hz, to_hz=eval_rebin_hz)

        min_t = min(recon.shape[1], original.shape[1])
        recon = recon[:, :min_t, :]
        original = original[:, :min_t, :]

        pred_rate = recon.sum(dim=1).cpu()  # (B, N)
        target_rate = original.sum(dim=1).cpu()  # (B, N)
        n_units = int(target_rate.shape[1])
        rate_acc.update(key, pred_rate, target_rate, n_units)
        per_session_target_rate.setdefault(key, []).append(target_rate)
        per_session_n_neurons[key] = n_units

        if min_t > 1:
            corr = batch_temporal_correlation(recon, original)  # (B, N)
            all_temporal_corrs.append(corr.cpu())

        if first_batch is None:
            first_batch = {
                "session": key,
                "original": original[:8].cpu(),
                "recon": recon[:8].cpu(),
                "bin_size_s": 1.0 / (eval_rebin_hz if eval_rebin_hz is not None else native_hz),
            }

        n_batches += 1

    scatter_points, rate_corr_agg = _aggregate_rate_corrs(rate_acc, per_session_target_rate)

    if all_temporal_corrs:
        flat_temp = torch.cat([c.flatten() for c in all_temporal_corrs])
        temporal_corr_agg = float(torch.nanmean(flat_temp).item())
    else:
        temporal_corr_agg = float("nan")

    # True neuron count: sum the per-session constants, not n_batches*N.
    n_neurons_total = sum(per_session_n_neurons.values())

    metrics = {
        "tokenizer_ceiling/rate_corr": rate_corr_agg,
        "tokenizer_ceiling/temporal_corr": temporal_corr_agg,
        "tokenizer_ceiling/n_batches": float(n_batches),
        "tokenizer_ceiling/n_neurons_total": float(n_neurons_total),
    }

    fig_traces = _plot_traces(first_batch)
    fig_scatter = _plot_scatter(scatter_points, rate_corr_agg, n_batches)

    return metrics, fig_traces, fig_scatter


def _aggregate_rate_corrs(
    rate_acc: OnlinePearsonAccumulator,
    per_session_target: dict[str, list[torch.Tensor]],
) -> tuple[list[tuple[float, float]], float]:
    """Finalize per-neuron rate Pearson r from the accumulator and aggregate.

    Returns the scatter-point list ``(mean_rate, r)`` per neuron and the
    aggregate nanmean over all per-neuron r values. ``mean_rate`` is the mean
    target rate per neuron across that session's trials.
    """
    per_session_corr = rate_acc.compute()  # {session: (N,)}
    rate_corrs: list[torch.Tensor] = []
    scatter: list[tuple[float, float]] = []

    for sess, r in per_session_corr.items():
        chunks = per_session_target.get(sess)
        if not chunks:
            continue
        targ = torch.cat(chunks, dim=0)  # (total_B, N)
        mean_rate = targ.mean(dim=0)
        rate_corrs.append(r)
        for n in range(r.shape[0]):
            scatter.append((mean_rate[n].item(), r[n].item()))

    if rate_corrs:
        agg = float(torch.nanmean(torch.cat(rate_corrs)).item())
    else:
        agg = float("nan")
    return scatter, agg


def _plot_traces(first_batch: dict | None):
    """3-row trace plot at low / mid / high firing-rate quantiles."""
    import matplotlib.pyplot as plt
    import numpy as np

    if first_batch is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        ax.text(0.5, 0.5, "no batches processed", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    original: torch.Tensor = first_batch["original"]  # (B, T, N)
    recon: torch.Tensor = first_batch["recon"]
    session: str = first_batch["session"]
    bin_size_s: float = first_batch["bin_size_s"]
    T = int(original.shape[1])
    time_ms = np.arange(T) * bin_size_s * 1000.0

    mean_rates = original.mean(dim=(0, 1))  # (N,)
    quantiles = [0.1, 0.5, 0.9]
    quantile_labels = ["Low firing", "Mid firing", "High firing"]
    neuron_idx = []
    for q in quantiles:
        target_rate = torch.quantile(mean_rates, q).item()
        closest = (mean_rates - target_rate).abs().argmin().item()
        neuron_idx.append(int(closest))

    total_per_image = original.sum(dim=(1, 2))
    img_idx = int(total_per_image.argsort()[len(total_per_image) // 2].item())

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f"Tokenizer roundtrip — {session} (image {img_idx})", fontsize=13)

    for row, (n_idx, qlabel) in enumerate(zip(neuron_idx, quantile_labels, strict=False)):
        ax = axes[row]
        orig_trace = original[img_idx, :, n_idx].numpy()
        recon_trace = recon[img_idx, :, n_idx].numpy()
        rate = mean_rates[n_idx].item()
        ax.plot(time_ms, orig_trace, color="black", linewidth=1.0, label="Original", alpha=0.85)
        ax.plot(time_ms, recon_trace, color="tab:red", linewidth=1.0, label="Reconstruction", alpha=0.85)
        ax.set_ylabel("Rate")
        ax.set_title(f"{qlabel} (neuron {n_idx}, mean={rate:.4f})", fontsize=10)
        if row == 0:
            ax.legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("Time (ms)")
    fig.tight_layout()
    return fig


def _plot_scatter(scatter_points: list[tuple[float, float]], agg: float, n_batches: int):
    """Per-neuron rate-corr vs mean firing rate, with horizontal aggregate line."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    if not scatter_points:
        ax.text(0.5, 0.5, "no scatter points", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    arr = np.asarray(scatter_points, dtype=np.float64)
    x = arr[:, 0]
    y = arr[:, 1]
    ax.scatter(x, y, s=10, alpha=0.5, color="tab:blue", edgecolors="none")
    if not np.isnan(agg):
        ax.axhline(agg, linestyle="--", color="black", linewidth=1.0, label=f"aggregate r = {agg:.3f}")
        ax.legend(loc="lower right", fontsize=9)
    ax.set_xlabel("Mean firing rate (sum over T per trial, averaged across trials)")
    ax.set_ylabel("Per-neuron rate Pearson r (recon vs original)")
    ax.set_title(f"Tokenizer ceiling — n_neurons={len(scatter_points)}, n_batches={n_batches}")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
