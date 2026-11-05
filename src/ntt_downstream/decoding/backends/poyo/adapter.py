"""Adapter: bridge between tokenizer output and POYOWithEmbeddingBypass.

Three unit-embedding modes on content tokens; sentinels as independent tokens.
"""

from __future__ import annotations

from typing import Literal, NamedTuple, cast

import torch
from torch import Tensor, nn

from ntt_downstream.decoding.backends.poyo.tokenizer_output import TokenizerOutput


class PreEmbeddedTokens(NamedTuple):
    """Ready-to-feed inputs for POYOWithEmbeddingBypass."""

    embeddings: Tensor  # [B, N_total, D_model]
    timestamps: Tensor  # [B, N_total]
    mask: Tensor  # [B, N_total]


def adapt(
    tok_out: TokenizerOutput,
    unit_emb: nn.Module,
    token_type_emb: nn.Module,
    mode: Literal["none", "add", "concat"],
    sentinels: bool,
    target_dim: int,
    input_proj: nn.Module | None = None,
    code_emb: nn.ModuleList | None = None,
    window_start: float = 0.0,
    window_end: float = 1.0,
) -> PreEmbeddedTokens:
    """Convert tokenizer output to POYO input_embeddings.

    Two paths:
      * Float ``tok_out.tokens``: continuous-embedding path. Optionally
        projected via ``input_proj``, then combined with ``unit_emb`` per
        ``mode``.
      * Int64 ``tok_out.tokens`` shape ``[B, N, Q]``: codebook-index path.
        Looked up via ``code_emb`` (a ``nn.ModuleList`` of ``Q``
        ``nn.Embedding(K, target_dim)`` tables), summed across codebooks.
        ``input_proj`` is bypassed in this branch.

    Args:
        tok_out: output from any tokenizer.
        unit_emb: POYO's unit embedding module (InfiniteVocabEmbedding).
        token_type_emb: POYO's token-type embedding module.
        mode: how to incorporate unit identity into content tokens.
            - "none":   x = (input_proj(x) if input_proj else x)
            - "add":    x = (input_proj(x) if input_proj else x) + unit_emb(unit_id)
            - "concat": x = input_proj(cat(x, unit_emb(unit_id)))   (float branch only)
        sentinels: whether to prepend sentinel tokens (one per unit in the window).
        target_dim: POYO's D_model; all tokens must be this width.
        input_proj: optional learnable projection to target_dim (float branch).
        code_emb: per-codebook embedding tables (int branch). Required when
            ``tok_out.tokens`` is int.
        window_start: start time of the POYO window (for sentinel timestamps).
            POYO's `tokenize` normalises timestamps to [0, sequence_length],
            so callers should pass 0.0 here. Caller-supplied; not inferred
            from the tokenizer output, since dropping all spikes from a
            window would leave us with no observable bound.
        window_end: end time of the POYO window. Currently accepted for
            forward compatibility with end-of-window sentinels (POYO's
            original START + END pair); the adapter only emits START
            sentinels today, so this value is unused at runtime.

    Returns:
        PreEmbeddedTokens with shape [B, N_total, target_dim].
    """
    is_discrete_tokens = not torch.is_floating_point(tok_out.tokens)

    if is_discrete_tokens:
        if code_emb is None:
            raise ValueError("int-dtype tokens require code_emb (one nn.Embedding per codebook).")
        Q = tok_out.tokens.shape[-1]
        if Q != len(code_emb):
            raise ValueError(f"tokens has {Q} codebooks but code_emb has {len(code_emb)} tables.")
        if mode == "concat":
            raise ValueError("mode='concat' is not supported with discrete-input embedding mode.")

        # Sum_q E_q[idx_q]: [B, N, target_dim]
        # sum() of Tensors starts from int 0, so pyright types it `Tensor | int`.
        x = cast(torch.Tensor, sum(code_emb[q](tok_out.tokens[..., q]) for q in range(Q)))
        ts = tok_out.timestamps
        mask = tok_out.mask

        if mode == "add":
            # Checked here (not before the discrete/float fork) so a discrete
            # concat misconfig still reports the accurate "concat not supported
            # with discrete" error above, rather than a misleading unit_ids one.
            if tok_out.unit_ids is None:
                raise ValueError("mode='add' requires per-token unit_ids, but tokenizer returned unit_ids=None.")
            x = x + unit_emb(tok_out.unit_ids)
        # mode == "none": x stays as-is.

        if sentinels:
            x, ts, mask = _prepend_sentinels(
                x,
                ts,
                mask,
                tok_out=tok_out,
                unit_emb=unit_emb,
                token_type_emb=token_type_emb,
                target_dim=target_dim,
                window_start=window_start,
            )
        return PreEmbeddedTokens(embeddings=x, timestamps=ts, mask=mask)

    # Float path — existing behavior preserved verbatim.
    # combines_units=True is allowed with mode in {add, concat} when the
    # tokenizer adapter emits per-slot proxy unit_ids (e.g. perceiver: M latent
    # slots → first-M input unit_ids per session). Without unit_ids the unit_emb
    # lookup is undefined, so the combo is rejected by this guard.
    if mode in ("add", "concat") and tok_out.unit_ids is None:
        raise ValueError(f"mode='{mode}' requires per-token unit_ids, but tokenizer returned unit_ids=None.")
    if mode == "concat":
        assert input_proj is not None, (
            "mode='concat' requires input_proj (D_tok + D_unit -> target_dim); no identity path exists for concat."
        )

    x = tok_out.tokens  # [B, N, D_tok]
    ts = tok_out.timestamps  # [B, N]
    mask = tok_out.mask  # [B, N]

    # Sanity check: tokenizer timestamps are expected to lie inside
    # [window_start, window_end]. POYO's tokenize() pins start=0, so the
    # invariant holds for the current code path; this guard catches future
    # drift (e.g. anyone moving to absolute timestamps) before it silently
    # places sentinels off-window.
    if sentinels and mask.any():
        valid_ts = ts.masked_select(mask)
        t_min = float(valid_ts.min().item())
        t_max = float(valid_ts.max().item())
        # 1e-3 s slack to absorb float32 timestamp rounding; tighter than
        # any reasonable sentinel offset.
        if t_min < window_start - 1e-3 or t_max > window_end + 1e-3:
            raise ValueError(
                f"Tokenizer emitted timestamps in [{t_min:.4f}, {t_max:.4f}] "
                f"outside adapter window [{window_start:.4f}, {window_end:.4f}]; "
                f"sentinels would land off-window."
            )

    if mode == "none":
        if input_proj is None:
            assert x.shape[-1] == target_dim, (
                f"mode='none' without input_proj requires D_tok ({x.shape[-1]}) == target_dim ({target_dim})."
            )
        else:
            x = input_proj(x)
    elif mode == "add":
        u = unit_emb(tok_out.unit_ids)  # [B, N, target_dim]
        if input_proj is None:
            assert x.shape[-1] == target_dim, (
                f"mode='add' without input_proj requires D_tok ({x.shape[-1]}) == target_dim ({target_dim})."
            )
            x = x + u
        else:
            x = input_proj(x) + u
    elif mode == "concat":
        u = unit_emb(tok_out.unit_ids)  # [B, N, target_dim]
        assert input_proj is not None, "mode='concat' requires input_proj"
        x = input_proj(torch.cat([x, u], dim=-1))
    else:
        raise ValueError(f"Unknown mode='{mode}'. Expected one of: none, add, concat.")

    if x.shape[-1] != target_dim:
        raise ValueError(f"adapter output dim {x.shape[-1]} does not match target_dim {target_dim}.")

    if sentinels:
        x, ts, mask = _prepend_sentinels(
            x,
            ts,
            mask,
            tok_out=tok_out,
            unit_emb=unit_emb,
            token_type_emb=token_type_emb,
            target_dim=target_dim,
            window_start=window_start,
        )

    return PreEmbeddedTokens(embeddings=x, timestamps=ts, mask=mask)


def _prepend_sentinels(
    x: Tensor,
    ts: Tensor,
    mask: Tensor,
    *,
    tok_out: TokenizerOutput,
    unit_emb: nn.Module,
    token_type_emb: nn.Module,
    target_dim: int,
    window_start: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build sentinel tokens and prepend them to the sequence.

    Sentinels are independent tokens — one per unique unit in each sample's
    window. Their content is unit_emb(unit_id) + token_type_emb(SENTINEL_TYPE),
    placed at the window start time.

    Note: architectural difference from original POYO — original creates
    START (type=1) + END (type=2) sentinels for ALL session units (including
    silent ones). This adapter creates one START sentinel per unique *spiking*
    unit, because the adapter only sees tokenizer output, not the full session
    unit list.
    """
    B, N, D = x.shape

    # Determine unique unit IDs per sample and build sentinel embeddings.
    # For population tokenizers, no sentinels (no unit_ids).
    if tok_out.unit_ids is None:
        return x, ts, mask

    # START_OF_SEQUENCE = 1, matching original POYO sentinel token type
    # (see torch_brain/utils/tokenizers.py).
    SENTINEL_TYPE = 1

    sentinel_list = []
    sentinel_ts_list = []
    sentinel_mask_list = []

    for b in range(B):
        valid_mask = tok_out.mask[b]  # [N]
        valid_uids = tok_out.unit_ids[b][valid_mask]  # [K]
        unique_uids = valid_uids.unique()
        n_sent = unique_uids.shape[0]

        if n_sent == 0:
            sentinel_list.append(x.new_zeros(0, target_dim))
            sentinel_ts_list.append(ts.new_zeros(0))
            sentinel_mask_list.append(mask.new_zeros(0, dtype=torch.bool))
            continue

        s_emb = unit_emb(unique_uids)  # [n_sent, target_dim]
        s_type = token_type_emb(unique_uids.new_full((n_sent,), SENTINEL_TYPE))  # [n_sent, target_dim]
        s_content = s_emb + s_type  # [n_sent, target_dim]

        # Sentinel timestamps: use the actual window start, not min spike time.
        s_ts = ts.new_full((n_sent,), window_start)

        sentinel_list.append(s_content)
        sentinel_ts_list.append(s_ts)
        sentinel_mask_list.append(mask.new_ones(n_sent, dtype=torch.bool))

    # Pad sentinels to common length across batch.
    max_sent = max(s.shape[0] for s in sentinel_list)
    if max_sent == 0:
        return x, ts, mask

    # Pad sentinel sequences and concatenate with content.
    all_emb = []
    all_ts = []
    all_mask = []
    for b in range(B):
        n_sent = sentinel_list[b].shape[0]
        pad_n = max_sent - n_sent

        s_emb = sentinel_list[b]
        s_ts = sentinel_ts_list[b]
        s_m = sentinel_mask_list[b]

        if pad_n > 0:
            s_emb = torch.cat([s_emb, s_emb.new_zeros(pad_n, target_dim)], dim=0)
            s_ts = torch.cat([s_ts, s_ts.new_zeros(pad_n)], dim=0)
            s_m = torch.cat([s_m, s_m.new_zeros(pad_n, dtype=torch.bool)], dim=0)

        all_emb.append(torch.cat([s_emb, x[b]], dim=0))
        all_ts.append(torch.cat([s_ts, ts[b]], dim=0))
        all_mask.append(torch.cat([s_m, mask[b]], dim=0))

    return torch.stack(all_emb), torch.stack(all_ts), torch.stack(all_mask)
