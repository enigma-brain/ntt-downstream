"""Per-position marginal-rate APPROXIMATION for single-codebook (Q=1) CE eval.

Shared between the periodic Lightning ``validation_step`` and the end-of-fit
``final_eval`` so both produce identical, deterministic-given-``z`` rate
estimates (no Monte-Carlo token sampling) for single-codebook CE tokenizers
(transformer-vq, FSQ, single-codebook VQ).

This is an APPROXIMATION of the true posterior marginal ``E[decode(idx)]``, not
an exact expectation. It equals the true marginal ONLY when the decoder is both
(a) affine in the per-position code embedding and (b) temporally local — i.e.
each token position drives only its own output window. Real conv/attention
decoders with a softplus rate head violate both, so the estimate is biased. See
``compute_ce_q1_marginal_rate`` for the precise form of the bias.

For Q>1 (residual VQ, joint K^Q intractable), both eval paths route through the
shared ``sample_tokens`` path rather than this marginal — this module is Q=1 only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_ce_q1_marginal_rate(
    model,
    wrapper,
    z: torch.Tensor,
    session_key: str,
    cache: dict[str, list[torch.Tensor]],
) -> torch.Tensor:
    """Per-position marginal expected-rate APPROXIMATION for single-codebook CE.

    For each token position ``pos``, decodes every vocab index once (with all
    other positions forced to codebook index 0), keeps that position's response
    window, then returns
    ``rate(b, t, n) = sum_k probs[b, n, pos, k] * decode(idx=k at pos)[t in window, n]``.
    Cached per session via the caller-supplied ``cache`` dict.

    Bias / exactness conditions. This equals the true posterior marginal
    ``E[decode(idx)]`` over the (per-position independent) categorical posterior
    ONLY if the decoder is both:
      (a) affine in the per-position code embedding, and
      (b) temporally local — position ``pos`` affects only its own output window.
    Real conv/attention decoders with a softplus rate head satisfy neither, so
    the estimate is biased. Two sources:
      * The softplus / convolutional nonlinearity makes ``E[f(idx)] != f(E[idx])``
        — averaging the per-index decoded windows is not the decode of the mean.
      * All OTHER positions are forced to codebook index 0, which is a real
        codebook entry (not a neutral / zero element). Through cross-position
        context (conv receptive field, attention) this biases every window
        toward "what the rest of the sequence looks like when it is all token 0".
    The bias is shared by both call sites (val and test use this same path), so
    the two stay internally consistent even though both are biased estimates.

    Determinism. Given ``z`` the result is fully deterministic — no Monte-Carlo
    token sampling — which is why val/test agree exactly. Deterministic does not
    imply unbiased: see above.

    Scale. The returned rate is UNNORMALIZED (no ``1/N`` over the marginalized
    positions / vocab). This is harmless here because every downstream metric
    (rate-Pearson, temporal-Pearson on PSTHs, noise-corrected R²) is invariant to
    a global scale of the prediction; the estimate is only ever compared against
    targets through scale-invariant statistics.

    Output shape ``(B, T_pred, S)`` matches ``wrapper.decode_differentiable``.
    """
    B, S, T_tok, D = z.shape
    V = int(model.ce_vocab_size)
    logits = model.ce_head(z.reshape(B * S * T_tok, D))  # (B*S*T_tok, V) for Q=1
    # Force float32 — bf16-mixed checkpoints produce bf16 logits, but the
    # decoded codebook rates are float32 and the einsum below requires matched
    # dtypes. Float32 is also more numerically robust for the marginal.
    probs = F.softmax(logits, dim=-1).reshape(B, S, T_tok, V).float()

    rates_by_pos = cache.get(session_key)
    if rates_by_pos is None:
        rates_by_pos = build_ce_codebook_rates_table(
            model,
            wrapper,
            session_key,
            n_units=S,
            n_token_times=T_tok,
            device=z.device,
        )
        cache[session_key] = rates_by_pos

    pieces = []
    for pos, rates in enumerate(rates_by_pos):
        rates_dev = rates.to(z.device, dtype=torch.float32)  # (V, S, T_pos)
        # einsum: probs[b, s, V] x rates[V, s, T_pos] -> (B, T_pos, S)
        pieces.append(torch.einsum("bsk,kst->bts", probs[:, :, pos, :], rates_dev))
    return torch.cat(pieces, dim=1)


@torch.no_grad()
def build_ce_codebook_rates_table(
    model,
    wrapper,
    session_key: str,
    *,
    n_units: int,
    n_token_times: int,
    device,
    chunk_size: int = 128,
) -> list[torch.Tensor]:
    """Decode every vocab index at every token position, return per-position
    ``(V, S, T_pos)`` tables (CPU-resident; moved to device on use).

    Each position ``pos`` is decoded with ALL other positions held at codebook
    index 0 (``idx4`` is zeros except column ``pos``). Index 0 is a real codebook
    entry, not a neutral element, so the decoded window for ``pos`` is contaminated
    by the cross-position context of an all-zero-token surround — this is the
    source of the marginal approximation's bias documented on
    ``compute_ce_q1_marginal_rate``. The table itself is exact for the inputs it is
    handed; the approximation lives in how the caller combines these tables.

    Stores tables on CPU because for K=2048, S=80, T_pos~20 this is ~14MB per
    position which is fine on CPU and avoids hogging GPU memory across sessions.
    """
    V = int(model.ce_vocab_size)

    # Prefer the wrapper's per-session compress_t metadata; fall back to a one-off
    # probe decode if metadata isn't populated (e.g. fresh checkpoint load).
    meta = getattr(wrapper, "_compress_t_session_meta", {}).get(session_key, {}) or {}
    window_bins = int(meta.get("window_bins", 0))
    if window_bins <= 0:
        probe = torch.zeros((1, n_units, n_token_times, 1), device=device, dtype=torch.long)
        probe_decoded = wrapper.decode_differentiable(probe, session_key=session_key)
        window_bins = int(probe_decoded.shape[1]) // n_token_times

    pos_tables: list[torch.Tensor] = []
    for pos in range(n_token_times):
        chunks: list[torch.Tensor] = []
        for start in range(0, V, chunk_size):
            end = min(start + chunk_size, V)
            chunk = end - start
            idx4 = torch.zeros(
                (chunk, n_units, n_token_times, 1),
                device=device,
                dtype=torch.long,
            )
            idx4[:, :, pos, 0] = torch.arange(start, end, device=device).view(-1, 1)
            decoded = wrapper.decode_differentiable(idx4, session_key=session_key)  # (chunk, T_raw, N)
            t0 = pos * window_bins
            t1 = min(t0 + window_bins, int(decoded.shape[1]))
            chunks.append(decoded[:, t0:t1].permute(0, 2, 1).contiguous().cpu())
        pos_tables.append(torch.cat(chunks, dim=0))  # (V, N, T_pos)
    return pos_tables
