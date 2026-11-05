"""Parity tests for `rotary_attn_flash_func` vs torch_brain's pytorch path.

These compare the flash-varlen wrapper used when `force_flash_attn=true`
against `torch_brain.nn.rotary_attention.rotary_attn_pytorch_func` (the
SDPA-based reference) on the same inputs. Both paths apply the same
rotary embedding — the only difference is the attention kernel.

Skipped when CUDA or the `flash_attn` package is unavailable. Both paths
run in bf16 (flash requires fp16/bf16) and tolerances are sized for the
mantissa width of bf16 (8 bits ≈ 3 sig digits).
"""

from __future__ import annotations

import pytest
import torch

flash_attn = pytest.importorskip("flash_attn")
if not torch.cuda.is_available():
    pytest.skip("flash_attn parity tests require CUDA", allow_module_level=True)


from torch_brain.nn.rotary_attention import rotary_attn_pytorch_func  # noqa: E402

from ntt_downstream.decoding.backends.poyo.flash_attn_varlen import (  # noqa: E402
    rotary_attn_flash_func,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
ATOL = 5e-2  # bf16 mantissa is 8 bits → ~3 sig figs; flash vs SDPA reductions
RTOL = 5e-2  # accumulate in different orders, so allow ~5% relative slack.


def _rand_qkv(B, N_q, N_kv, num_heads, head_dim, seed: int):
    """Build random Q/K/V plus rotary pos-emb tensors.

    `RotaryTimeEmbedding.rotate` chunks the pos-emb in half along the last
    dim into (cos, sin), each of which broadcasts against a head-dim slice
    of x. So the pos-emb's last dim must be `2 * head_dim`, not head_dim.
    Values are random — parity holds regardless of whether the rotation is
    "physical" because both paths apply the same op.
    """
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    inner = num_heads * head_dim
    pos_dim = 2 * head_dim
    q = torch.randn(B, N_q, inner, generator=g, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, N_kv, inner, generator=g, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, N_kv, inner, generator=g, device=DEVICE, dtype=DTYPE)
    q_pos = torch.randn(B, N_q, pos_dim, generator=g, device=DEVICE, dtype=DTYPE)
    kv_pos = torch.randn(B, N_kv, pos_dim, generator=g, device=DEVICE, dtype=DTYPE)
    return q, k, v, q_pos, kv_pos


def _run_both(q, k, v, q_pos, kv_pos, *, mask, num_heads, rotate_value):
    """Run reference and flash paths on identical inputs, dropout off."""
    common = {
        "query": q,
        "key": k,
        "value": v,
        "q_pos_emb": q_pos,
        "kv_pos_emb": kv_pos,
        "attn_mask": mask,
        "num_heads": num_heads,
        "dropout_p": 0.0,
        "rotate_value": rotate_value,
    }
    ref = rotary_attn_pytorch_func(**common)
    new = rotary_attn_flash_func(**common)
    return ref, new


def _assert_close_ignore_nan(ref, new, *, tag: str = ""):
    """Assert that NaN positions agree and finite positions match within tol.

    Both paths produce NaN for queries that attend to all-padded keys
    (softmax over all -inf). We treat that as parity: equal NaN masks,
    finite values close.
    """
    assert ref.shape == new.shape, f"{tag}: shape mismatch {ref.shape} vs {new.shape}"
    nan_ref = torch.isnan(ref)
    nan_new = torch.isnan(new)
    assert torch.equal(nan_ref, nan_new), (
        f"{tag}: NaN locations differ (ref={int(nan_ref.sum())}, new={int(nan_new.sum())})"
    )
    if nan_ref.all():
        return  # everything NaN; nothing finite to compare.
    finite = ~nan_ref
    torch.testing.assert_close(
        ref[finite].float(),
        new[finite].float(),
        atol=ATOL,
        rtol=RTOL,
        msg=lambda m: f"{tag}: {m}",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rotate_value", [False, True])
def test_no_mask_parity(rotate_value):
    """Self-attention-style call: no attn_mask, uniform sequences."""
    B, N_q, N_kv, H, D = 2, 64, 64, 4, 32
    q, k, v, q_pos, kv_pos = _rand_qkv(B, N_q, N_kv, H, D, seed=0)
    ref, new = _run_both(
        q,
        k,
        v,
        q_pos,
        kv_pos,
        mask=None,
        num_heads=H,
        rotate_value=rotate_value,
    )
    _assert_close_ignore_nan(ref, new, tag=f"no_mask rv={rotate_value}")


@pytest.mark.parametrize("rotate_value", [False, True])
def test_keypad_mask_parity(rotate_value):
    """Encoder cross-attention: per-sample varying number of valid keys."""
    B, N_q, N_kv, H, D = 3, 32, 96, 4, 32
    q, k, v, q_pos, kv_pos = _rand_qkv(B, N_q, N_kv, H, D, seed=1)
    # Per-sample seqlens: 96, 50, 13.
    seqlens = torch.tensor([96, 50, 13], dtype=torch.int64, device=DEVICE)
    arange = torch.arange(N_kv, device=DEVICE)[None, :]
    mask = arange < seqlens[:, None]
    ref, new = _run_both(
        q,
        k,
        v,
        q_pos,
        kv_pos,
        mask=mask,
        num_heads=H,
        rotate_value=rotate_value,
    )
    _assert_close_ignore_nan(ref, new, tag=f"keypad rv={rotate_value}")


def test_keypad_mask_all_true_matches_no_mask():
    """All-True mask is equivalent to passing no mask — for both paths."""
    B, N_q, N_kv, H, D = 2, 32, 64, 4, 32
    q, k, v, q_pos, kv_pos = _rand_qkv(B, N_q, N_kv, H, D, seed=2)
    full = torch.ones(B, N_kv, dtype=torch.bool, device=DEVICE)

    ref_full = rotary_attn_pytorch_func(
        query=q,
        key=k,
        value=v,
        q_pos_emb=q_pos,
        kv_pos_emb=kv_pos,
        attn_mask=full,
        num_heads=H,
        dropout_p=0.0,
        rotate_value=False,
    )
    new_full = rotary_attn_flash_func(
        query=q,
        key=k,
        value=v,
        q_pos_emb=q_pos,
        kv_pos_emb=kv_pos,
        attn_mask=full,
        num_heads=H,
        dropout_p=0.0,
        rotate_value=False,
    )
    new_none = rotary_attn_flash_func(
        query=q,
        key=k,
        value=v,
        q_pos_emb=q_pos,
        kv_pos_emb=kv_pos,
        attn_mask=None,
        num_heads=H,
        dropout_p=0.0,
        rotate_value=False,
    )
    _assert_close_ignore_nan(ref_full, new_full, tag="all_true vs ref")
    _assert_close_ignore_nan(new_full, new_none, tag="all_true vs no_mask")


def test_single_kv_per_sample():
    """Each sample has exactly one valid key; flash varlen handles len-1 packs."""
    B, N_q, N_kv, H, D = 4, 16, 32, 4, 32
    q, k, v, q_pos, kv_pos = _rand_qkv(B, N_q, N_kv, H, D, seed=3)
    mask = torch.zeros(B, N_kv, dtype=torch.bool, device=DEVICE)
    # Pick a different valid index per sample to avoid trivial degeneracy.
    for b in range(B):
        mask[b, b * 3] = True
    ref, new = _run_both(
        q,
        k,
        v,
        q_pos,
        kv_pos,
        mask=mask,
        num_heads=H,
        rotate_value=False,
    )
    _assert_close_ignore_nan(ref, new, tag="single_kv")


def test_batch_size_one():
    """B=1 path — cu_seqlens degenerate to length 2."""
    B, N_q, N_kv, H, D = 1, 32, 64, 4, 32
    q, k, v, q_pos, kv_pos = _rand_qkv(B, N_q, N_kv, H, D, seed=4)
    mask = torch.ones(B, N_kv, dtype=torch.bool, device=DEVICE)
    mask[0, 30:] = False
    ref, new = _run_both(
        q,
        k,
        v,
        q_pos,
        kv_pos,
        mask=mask,
        num_heads=H,
        rotate_value=False,
    )
    _assert_close_ignore_nan(ref, new, tag="B=1")


def test_zero_valid_keys_parity():
    """One sample has zero valid keys; both paths must agree on the result.

    Behavior on this edge depends on the SDPA backend (fp32 SDPA softmaxes
    over all -inf → NaN; modern bf16 mem_efficient and flash_attn_varlen
    short-circuit to zeros). We don't assert which one; we assert that
    whatever the reference produces, the flash path matches.
    """
    B, N_q, N_kv, H, D = 2, 16, 32, 4, 32
    q, k, v, q_pos, kv_pos = _rand_qkv(B, N_q, N_kv, H, D, seed=5)
    mask = torch.zeros(B, N_kv, dtype=torch.bool, device=DEVICE)
    mask[1, :10] = True  # second sample has valid keys; first is all-False.
    ref, new = _run_both(
        q,
        k,
        v,
        q_pos,
        kv_pos,
        mask=mask,
        num_heads=H,
        rotate_value=False,
    )
    _assert_close_ignore_nan(ref, new, tag="zero_kv parity")


def test_3d_mask_raises():
    """3-D attention mask is not supported — must raise loudly."""
    B, N_q, N_kv, H, D = 2, 16, 32, 4, 32
    q, k, v, q_pos, kv_pos = _rand_qkv(B, N_q, N_kv, H, D, seed=6)
    mask3d = torch.ones(B, N_q, N_kv, dtype=torch.bool, device=DEVICE)
    with pytest.raises(NotImplementedError, match="2-D"):
        rotary_attn_flash_func(
            query=q,
            key=k,
            value=v,
            q_pos_emb=q_pos,
            kv_pos_emb=kv_pos,
            attn_mask=mask3d,
            num_heads=H,
            dropout_p=0.0,
            rotate_value=False,
        )
