"""Tokenizer roundtrip evaluation.

Encodes single-trial responses through the tokenizer wrapper and decodes back,
then computes eval metrics to establish the tokenizer's reconstruction ceiling.

For trial-averaged metrics: reconstructs each single trial individually, averages
the reconstructions, then compares to the averaged original. This is the correct
approach since tokenizers are trained on spikes, not smooth averages.

Usage:
    python scripts/encoding/tokenizer_roundtrip.py --config-name cowley_v4_wanneural_vq
"""

import logging
from collections import defaultdict
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch
from omegaconf import DictConfig

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
logger = logging.getLogger(__name__)


def run_roundtrip(wrapper, dataloader, native_hz, eval_rebin_hz, device, max_batches=200, decode_mode="differentiable"):
    """Encode → decode roundtrip on single-trial data.

    Args:
        decode_mode: One of:
            "differentiable" — wrapper.encode + wrapper.decode_differentiable (tokenizer.decode)
            "reconstruct" — wrapper.encode + wrapper.decode (tokenizer.reconstruct)

    Returns (rate_corrs, temporal_corrs, examples) for single-trial metrics, plus
    per-image accumulators for trial-averaged metrics.
    """
    from ntt_downstream.encoding.models.enc.hiera.evals import (
        OnlinePearsonAccumulator,
        TemporalCorrAccumulator,
        batch_temporal_correlation,
        rebin_time_axis,
    )

    rate_acc = OnlinePearsonAccumulator()
    temporal_acc = TemporalCorrAccumulator()
    recon_hz = wrapper.reconstruction_rate_hz
    n_batches = 0

    per_image: dict[tuple[str, int], dict[str, list]] = defaultdict(lambda: {"orig": [], "recon": []})
    examples = []
    seen_sessions: set[str] = set()

    with torch.no_grad():
        for key, batch_dict in dataloader:
            if n_batches >= max_batches:
                break

            responses = batch_dict["response"].to(device, torch.float32)  # (B, N, T)
            n_units = batch_dict["n_units"]
            image_ids = batch_dict["image_id"]  # (B,)

            # Dataset returns (B, N, T); wrapper expects (B, T, N)
            responses_btn = responses.transpose(1, 2)

            # raw_discrete=True returns raw tokenizer latents without target conversion,
            # avoiding lossy int conversion for discrete tokenizers (e.g. BSQ).
            enc = wrapper.encode(responses_btn, key, raw_discrete=True)
            if decode_mode == "reconstruct":
                recon = wrapper.decode(enc["target"], session_key=key).to(torch.float32)
            else:
                recon = wrapper.decode_differentiable(enc["target"], session_key=key)

            original = responses_btn  # (B, T, N)

            if n_batches == 0:
                logger.info(
                    f"session={key} | input={tuple(original.shape)} | "
                    f"tokens={tuple(enc['target'].shape)} | recon={tuple(recon.shape)}"
                )

            # Rebin if requested
            if eval_rebin_hz is not None:
                recon = rebin_time_axis(recon, from_hz=recon_hz, to_hz=eval_rebin_hz)
                original = rebin_time_axis(original, from_hz=native_hz, to_hz=eval_rebin_hz)

            # Align temporal dimensions
            min_t = min(recon.shape[1], original.shape[1])
            recon = recon[:, :min_t, :]
            original = original[:, :min_t, :]

            # --- Single-trial metrics ---
            pred_rate = recon.sum(dim=1)
            target_rate = original.sum(dim=1)
            rate_acc.update(key, pred_rate, target_rate, n_units)

            if min_t > 1:
                corr = batch_temporal_correlation(recon, original)
                temporal_acc.update(key, corr, n_units)

            # --- Accumulate per image for trial averaging ---
            for i in range(original.shape[0]):
                iid = int(image_ids[i].item())
                per_image[(key, iid)]["orig"].append(original[i].cpu())
                per_image[(key, iid)]["recon"].append(recon[i].cpu())

            # Collect plot examples (first batch per session, up to 4 sessions)
            if key not in seen_sessions and len(examples) < 4:
                seen_sessions.add(key)
                effective_hz = eval_rebin_hz if eval_rebin_hz is not None else native_hz
                examples.append(
                    {
                        "session": key,
                        "original": original[:8].cpu().float(),
                        "recon": recon[:8].cpu().float(),
                        "bin_size_s": 1.0 / effective_hz,
                    }
                )

            n_batches += 1

    logger.info(f"Processed {n_batches} batches, {len(per_image)} unique images")
    return rate_acc.compute(), temporal_acc.compute(), examples, per_image


def compute_trial_averaged_metrics(per_image):
    """Average reconstructions per image, then compute metrics.

    For each unique image, averages the single-trial originals and the single-trial
    reconstructions separately, then measures correlation on those averages.
    """
    from ntt_downstream.encoding.models.enc.hiera.evals import (
        OnlinePearsonAccumulator,
        TemporalCorrAccumulator,
        batch_temporal_correlation,
    )

    rate_acc = OnlinePearsonAccumulator()
    temporal_acc = TemporalCorrAccumulator()

    # Group images by session for batch metrics
    session_items: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
    for (session, _iid), data in per_image.items():
        if len(data["orig"]) < 2:
            continue  # need multiple trials
        avg_orig = torch.stack(data["orig"]).mean(dim=0)  # (T, N)
        avg_recon = torch.stack(data["recon"]).mean(dim=0)  # (T, N)
        session_items[session].append((avg_orig, avg_recon))

    examples = []
    for session, pairs in session_items.items():
        if not pairs:
            continue
        origs = torch.stack([p[0] for p in pairs])  # (n_images, T, N)
        recons = torch.stack([p[1] for p in pairs])
        n_units = origs.shape[2]

        pred_rate = recons.sum(dim=1)
        target_rate = origs.sum(dim=1)
        rate_acc.update(session, pred_rate, target_rate, n_units)

        T = origs.shape[1]
        if T > 1:
            corr = batch_temporal_correlation(recons, origs)
            temporal_acc.update(session, corr, n_units)

        # Collect plot examples
        if len(examples) < 4:
            bin_size_s = 0.001  # will be overridden by caller
            examples.append(
                {
                    "session": session,
                    "original": origs[:8],
                    "recon": recons[:8],
                    "bin_size_s": bin_size_s,
                }
            )

    return rate_acc.compute(), temporal_acc.compute(), examples


def plot_roundtrip_examples(examples, label, save_dir):
    """Plot original vs reconstruction for low/mid/high firing rate neurons."""
    quantiles = [0.1, 0.5, 0.9]
    quantile_labels = ["Low firing", "Mid firing", "High firing"]

    for ex in examples:
        session = ex["session"]
        original = ex["original"]  # (n_images, T, N)
        recon = ex["recon"]
        bin_size_s = ex["bin_size_s"]
        T = original.shape[1]
        time_ms = np.arange(T) * bin_size_s * 1000

        # Mean firing rate per neuron
        mean_rates = original.mean(dim=(0, 1))

        # Select neurons at rate quantiles
        neuron_indices = []
        for q in quantiles:
            target_rate = torch.quantile(mean_rates, q).item()
            closest = (mean_rates - target_rate).abs().argmin().item()
            neuron_indices.append(closest)

        # Pick image with median total activity
        total_per_image = original.sum(dim=(1, 2))
        img_idx = total_per_image.argsort()[len(total_per_image) // 2].item()

        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        fig.suptitle(f"{label} — {session}", fontsize=13)

        for row, (neuron_idx, qlabel) in enumerate(zip(neuron_indices, quantile_labels, strict=False)):
            ax = axes[row]
            orig_trace = original[img_idx, :, neuron_idx].numpy()
            recon_trace = recon[img_idx, :, neuron_idx].numpy()
            rate = mean_rates[neuron_idx].item()

            ax.plot(time_ms, orig_trace, color="black", linewidth=1.0, label="Original", alpha=0.8)
            ax.plot(time_ms, recon_trace, color="tab:red", linewidth=1.0, label="Reconstruction", alpha=0.8)
            ax.set_ylabel("Firing rate")
            ax.set_title(f"{qlabel} (neuron {neuron_idx}, mean rate={rate:.4f})", fontsize=10)
            if row == 0:
                ax.legend(loc="upper right", fontsize=9)

        axes[-1].set_xlabel("Time (ms)")
        fig.tight_layout()

        fname = save_dir / f"roundtrip_{label}_{session}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {fname}")


def print_tokenizer_info(wrapper, n_time_bins: int, native_hz: float):
    """Print a compact summary of tokenizer properties and latent geometry."""
    tok = wrapper.tokenizer
    spec = getattr(tok, "spec", None)
    if spec is None:
        print("  (no spec available)")
        return

    # --- Identity ---
    print(f"\n{'=' * 60}")
    print("  Tokenizer Info")
    print(f"{'=' * 60}")
    print(f"  KIND:            {getattr(spec, 'KIND', '?')}")
    print(f"  bottleneck:      {getattr(spec, 'bottleneck_type', '?')}")
    print(f"  input_type:      {getattr(spec, 'input_type', '?')}")
    print(f"  output_type:     {getattr(spec, 'output_type', '?')}")
    print(f"  latent_type:     {getattr(spec, 'latent_type', '?')}")

    # --- Compression axes ---
    compress_n = bool(getattr(spec, "compress_n", False))
    compress_t = bool(getattr(spec, "compress_t", False))
    mode = "compress_n" if compress_n else ("compress_t" if compress_t else "unknown")
    print(f"  mode:            {mode}")

    # --- Spec dimensions ---
    print(f"\n  {'— Spec dimensions —':^40}")
    print(f"  n_neurons:       {getattr(spec, 'n_neurons', '?')}")
    bin_size_ms = getattr(spec, "bin_size_ms", None)
    print(f"  bin_size_ms:     {bin_size_ms}")
    z_dim = getattr(spec, "z_dim", None)
    print(f"  z_dim:           {z_dim}")

    # --- Latent shape (from spec flags) ---
    has_t = bool(getattr(spec, "has_time_dim", False))
    has_u = bool(getattr(spec, "has_unit_dim", False))
    has_d = bool(getattr(spec, "has_latent_dim", False))
    print(f"\n  {'— Latent axes [B, T, U, D] —':^40}")
    print(f"  has_time_dim:    {has_t}")
    print(f"  has_unit_dim:    {has_u}")
    print(f"  has_latent_dim:  {has_d}")

    # --- Probe actual encode shape ---
    t_comp = wrapper.get_t_comp(n_time_bins)
    print(f"\n  {'— Temporal coverage —':^40}")
    t_comp_spec = getattr(spec, "temporal_compression", None)
    if t_comp_spec is not None:
        print(f"  temporal_compression (spec): {t_comp_spec}x")
    window_ms = n_time_bins / native_hz * 1000.0
    print(f"  input window:    {n_time_bins} bins = {window_ms:.0f}ms @ {1000.0 / native_hz:.1f}ms bins")
    print(f"  latent T:        {t_comp} token(s)")
    if t_comp > 0:
        ms_per_token = window_ms / t_comp
        print(f"  ms / token:      {ms_per_token:.1f}ms  (={window_ms:.0f}/{t_comp})")
    if t_comp_spec is not None and t_comp_spec > 0:
        expected_t = n_time_bins / t_comp_spec
        if expected_t != t_comp:
            print(f"  NOTE: spec predicts {expected_t:.1f} tokens but encode produces {t_comp}")
            print(f"        (internal padding to temporal_patch_size={getattr(spec, 'temporal_patch_size', '?')})")

    # --- Wrapper settings ---
    print(f"\n  {'— Wrapper settings —':^40}")
    print(f"  neuron_batch_size: {wrapper.neuron_batch_size}")
    print(f"  latent_scale:    {wrapper.latent_scale}")
    print(f"  recon_rate_hz:   {wrapper.reconstruction_rate_hz:.1f}")
    print(f"{'=' * 60}\n")


def print_results(rate_corrs, temporal_corrs, label):
    """Print per-session and overall metrics."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    all_rate = []
    for key, corr in sorted(rate_corrs.items()):
        print(f"  rate_corr/{key}: {corr.nanmean().item():.4f}")
        all_rate.append(corr)
    if all_rate:
        print(f"  rate_corr/overall: {torch.hstack(all_rate).nanmean().item():.4f}")

    if temporal_corrs:
        all_temp = []
        for key, corr in sorted(temporal_corrs.items()):
            print(f"  temporal_corr/{key}: {corr.nanmean().item():.4f}")
            all_temp.append(corr)
        if all_temp:
            print(f"  temporal_corr/overall: {torch.hstack(all_temp).nanmean().item():.4f}")
    else:
        print("  temporal_corr: N/A (T=1)")

    print(f"{'=' * 60}")


@hydra.main(config_path=f"{root}/configs/encoding", config_name="cowley_v4_wanneural_vq", version_base=None)
def main(cfg: DictConfig):
    from ntt_downstream.encoding.tokenizer_wrappers.encoding_wrapper import build_tokenizer_wrapper
    from ntt_downstream.encoding.training.data_module import CowleyV4DataModule

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # --- Data ---
    dm = CowleyV4DataModule(cfg)
    dm.setup("fit")

    session_info = dm.get_session_info("train")
    for sid, n in session_info.items():
        logger.info(f"Session {sid}: {n} neurons")

    bin_size = float(cfg.data.get("bin_size", 0.001))
    n_time_bins = int(round(float(cfg.data.get("response_window", 0.200)) / bin_size))
    native_hz = 1.0 / bin_size
    eval_rebin_hz = cfg.trainer.get("eval_rebin_hz", None)
    eval_rebin_hz = float(eval_rebin_hz) if eval_rebin_hz is not None else None
    effective_hz = eval_rebin_hz if eval_rebin_hz is not None else native_hz
    logger.info(f"bin_size={bin_size}s | n_time_bins={n_time_bins} | native_hz={native_hz:.1f}")

    # --- Tokenizer ---
    wrapper = build_tokenizer_wrapper(cfg, device=device)
    if wrapper is None:
        raise ValueError("Tokenizer not enabled in config. Set tokenizer.enabled=true.")
    print_tokenizer_info(wrapper, n_time_bins, native_hz)

    # --- Output directory: results/tokenizer_roundtrip/{tokenizer_name}/ ---
    ckpt_path = str(cfg.tokenizer.checkpoint_path)
    if ckpt_path.startswith(("mlflow-artifacts:", "runs:", "models:")):
        tokenizer_name = cfg.get("tags", {}).get("model", Path(ckpt_path).stem)
    else:
        tokenizer_name = Path(ckpt_path).parent.name
    base_dir = Path(root) / "results" / "tokenizer_roundtrip" / tokenizer_name

    val_loader = dm._make_loader(dm.val_dataset, shuffle=False, drop_last=False)

    # --- Run both decode modes, each into its own subfolder ---
    for mode in ["differentiable", "reconstruct"]:
        save_dir = base_dir / mode
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"\n{'=' * 60}\n  Running decode_mode={mode} → {save_dir}\n{'=' * 60}")

        rate_corrs, temporal_corrs, st_examples, per_image = run_roundtrip(
            wrapper, val_loader, native_hz, eval_rebin_hz, device, decode_mode=mode
        )
        print_results(rate_corrs, temporal_corrs, f"Tokenizer Roundtrip — Val (single trial, {mode})")
        plot_roundtrip_examples(st_examples, "single_trial", save_dir)

        ta_rate_corrs, ta_temporal_corrs, ta_examples = compute_trial_averaged_metrics(per_image)
        for ex in ta_examples:
            ex["bin_size_s"] = 1.0 / effective_hz
        print_results(ta_rate_corrs, ta_temporal_corrs, f"Tokenizer Roundtrip — Val (trial averaged, {mode})")
        plot_roundtrip_examples(ta_examples, "trial_avg", save_dir)


if __name__ == "__main__":
    main()
