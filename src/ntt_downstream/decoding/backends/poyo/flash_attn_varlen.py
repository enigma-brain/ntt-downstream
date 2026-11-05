"""FlashAttention drop-in for torch_brain's pytorch attention path.

`rotary_attn_flash_func` matches the signature of
`torch_brain.nn.rotary_attention.rotary_attn_pytorch_func` and is intended
to be monkey-patched in its place when `force_flash_attn=true`.

Routing:
  - attn_mask is None              → flash_attn.flash_attn_func (regular flash).
  - attn_mask is 2-D bool, N_q==N_k → self-attention path. The same mask
                                      applies to both Q and K (q, k, v are
                                      projections of the same input), so we
                                      pack both sides via the mask.
  - attn_mask is 2-D bool, N_q!=N_k → cross-attention path. Q has no
                                      padding (the latent grid is fixed
                                      width), only K/V are repacked.
  - attn_mask is 3-D                → NotImplementedError.

For self-attention, `RotarySelfAttention.forward` documents `x_mask` as
(B, N, N) but the POYO encoder always passes a 2-D `(B, N)` key-padding
mask; the varlen path therefore packs both the Q and the K side, so padded
queries cannot attend to valid keys and pollute valid-token outputs that the
encoder later consumes.

Partly adapted from torch_brain's ``torch_brain/nn/rotary_attention.py``
(https://github.com/neuro-galaxy/torch_brain), Apache-2.0, Copyright 2025
NeuroGalaxy. Modified for ntt-downstream: the attention call is replaced by
flash_attn (BSD-3-Clause) varlen kernels.

Rotary embeddings are applied here in (B, N, H, D) layout (xformers/flash
convention, unsqueeze_dim=2) — semantically identical to the (B, H, N, D)
+ unsqueeze_dim=1 layout used by the SDPA path.

Hot path is host-sync free: `max_seqlen_*` for `flash_attn_varlen_func`
is passed as the padded width (an upper bound), since flash uses it only
for kernel block sizing — per-sample exact lengths come from
`cu_seqlens_*`.
"""

from typing import cast

import torch
import torch.nn.functional as F
from einops import rearrange
from torch_brain.nn import RotaryTimeEmbedding


def rotary_attn_flash_func(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_pos_emb: torch.Tensor,
    kv_pos_emb: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    num_heads: int,
    dropout_p: float,
    rotate_value: bool,
) -> torch.Tensor:
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    q = rearrange(query, "b n (h d) -> b n h d", h=num_heads)
    k = rearrange(key, "b n (h d) -> b n h d", h=num_heads)
    v = rearrange(value, "b n (h d) -> b n h d", h=num_heads)

    q = RotaryTimeEmbedding.rotate(x=q, rotary_emb=q_pos_emb, unsqueeze_dim=2)
    k = RotaryTimeEmbedding.rotate(x=k, rotary_emb=kv_pos_emb, unsqueeze_dim=2)
    if rotate_value:
        v = RotaryTimeEmbedding.rotate(x=v, rotary_emb=kv_pos_emb, unsqueeze_dim=2)

    if attn_mask is None:
        # flash_attn's stub types the return as Optional (softmax-lse variants); this call returns a Tensor.
        out = cast(torch.Tensor, flash_attn_func(q, k, v, dropout_p=dropout_p, causal=False))
    else:
        if attn_mask.dim() != 2:
            raise NotImplementedError(
                "rotary_attn_flash_func only supports 2-D bool key-padding "
                f"masks [B, N_kv]; got mask with shape {tuple(attn_mask.shape)}."
            )
        if attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(torch.bool)
        # Self vs. cross: in self-attention all of q, k, v come from the same
        # input, so they share the sequence dim and the same mask applies to
        # both ends. In cross-attention Q is the (unpadded) latent grid and
        # K/V is the padded input, so N_q != N_kv. Discriminating on shape
        # is robust because q, k, v are produced by separate Linear
        # projections (`to_q` vs `to_kv`), so `q is k` is always False.
        if q.shape[1] == k.shape[1] and attn_mask.shape[1] == q.shape[1]:
            out = _flash_attn_varlen_qkpad(
                q=q,
                k=k,
                v=v,
                mask=attn_mask,
                dropout_p=dropout_p,
                varlen_func=flash_attn_varlen_func,
            )
        else:
            out = _flash_attn_varlen_kvpad(
                q=q,
                k=k,
                v=v,
                kv_mask=attn_mask,
                dropout_p=dropout_p,
                varlen_func=flash_attn_varlen_func,
            )

    if rotate_value:
        out = RotaryTimeEmbedding.rotate(
            x=out,
            rotary_emb=RotaryTimeEmbedding.invert(q_pos_emb),
            unsqueeze_dim=2,
        )

    return rearrange(out, "b n h d -> b n (h d)")


def _flash_attn_varlen_kvpad(
    *,
    q: torch.Tensor,  # [B, N_q, H, D]
    k: torch.Tensor,  # [B, N_kv, H, D]
    v: torch.Tensor,  # [B, N_kv, H, D]
    kv_mask: torch.Tensor,  # [B, N_kv] bool
    dropout_p: float,
    varlen_func,
) -> torch.Tensor:
    """Cross-attention varlen path. Q has no padding (uniform N_q per
    sample); K, V are packed by dropping False positions in `kv_mask`.

    NOTE: when a sample has zero valid keys, FlashAttention emits NaN for
    that sample's queries — same behavior as SDPA with an all-False mask
    (softmax over all -inf). This matches the existing path's semantics.
    """
    B, N_q, H, D = q.shape
    N_kv = kv_mask.shape[1]
    device = q.device

    seqlens_k = kv_mask.sum(dim=1).to(torch.int32)  # [B]
    cu_k = F.pad(seqlens_k.cumsum(0, dtype=torch.int32), (1, 0))  # [B+1]
    # Upper bound is fine: flash_attn_varlen_func uses max_seqlen_k only for
    # kernel block sizing — per-sample exact lengths come from cu_seqlens_k.
    # Using N_kv (the padded width) avoids a per-step CPU sync at the cost
    # of slightly larger kernel tiles when most samples are heavily padded.
    max_k = N_kv

    k_packed = k[kv_mask]  # [sum_k, H, D]
    v_packed = v[kv_mask]  # [sum_k, H, D]

    q_packed = q.reshape(B * N_q, H, D)
    cu_q = torch.arange(
        0,
        (B + 1) * N_q,
        N_q,
        dtype=torch.int32,
        device=device,
    )

    out_packed = varlen_func(
        q_packed,
        k_packed,
        v_packed,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=N_q,
        max_seqlen_k=max_k,  # = N_kv (upper bound; no host sync)
        dropout_p=dropout_p,
        causal=False,
    )
    return out_packed.view(B, N_q, H, D)


def _flash_attn_varlen_qkpad(
    *,
    q: torch.Tensor,  # [B, N, H, D]
    k: torch.Tensor,  # [B, N, H, D]
    v: torch.Tensor,  # [B, N, H, D]
    mask: torch.Tensor,  # [B, N] bool, applies to both Q and K
    dropout_p: float,
    varlen_func,
) -> torch.Tensor:
    """Self-attention varlen path: same mask for Q and K.

    For self-attention `q.shape == k.shape == v.shape`, and a key-padding
    mask is also a query-padding mask — padded queries should not attend
    to anything, and their outputs should not pollute valid downstream
    tokens. We pack both sides via `mask` and scatter the variable-length
    output back into a `[B, N, H, D]` tensor with zeros at padded
    positions. Padded outputs are typically masked again by callers via
    the same mask, so the zero is a safe sentinel.

    NOTE: a sample whose mask is all-False would produce NaNs from
    flash's softmax (no valid keys); the data pipeline rejects empty samples
    upstream.
    """
    B, N, H, D = q.shape

    seqlens = mask.sum(dim=1).to(torch.int32)  # [B]
    cu = F.pad(seqlens.cumsum(0, dtype=torch.int32), (1, 0))  # [B+1]
    # Upper bound; per-sample exact lengths come from `cu`.
    max_n = N

    q_packed = q[mask]  # [sum, H, D]
    k_packed = k[mask]
    v_packed = v[mask]

    out_packed = varlen_func(
        q_packed,
        k_packed,
        v_packed,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=max_n,
        max_seqlen_k=max_n,
        dropout_p=dropout_p,
        causal=False,
    )

    out = q.new_zeros(B, N, H, D)
    out[mask] = out_packed
    return out
