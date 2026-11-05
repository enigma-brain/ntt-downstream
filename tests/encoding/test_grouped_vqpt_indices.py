import torch
from ntt.contrib.tokenizer.wanneural.bottleneck import VQPTBottleneck

from ntt_downstream.encoding.tokenizer_wrappers.encoding_wrapper import TokenizerWrapper


class _DecodeOnlyTokenizer:
    def __init__(self, bottleneck):
        self.bottleneck = bottleneck

    def decode(self, z_4d):
        return z_4d.sum(dim=-1)


def _unpack_unified_to_grouped(unified: torch.Tensor, groups: int) -> torch.Tensor:
    """Reproduce the index-layer transform inside ``_decode_int_chunk``.

    ``_decode_int_chunk`` flattens the unified ``(B, T, N, Q)`` tokens to
    ``(B, T*N, Q)``, splits the trailing axis into ``(groups, q_per_group)``
    and permutes the group axis to the front before handing
    ``(G, B, T*N, q_per_group)`` to ``bottleneck.indices_to_codes``. We mirror
    only that pure permute/reshape step here so the test asserts on the layer
    that is an exact bijection — the subsequent ``indices_to_codes`` codebook
    lookup is *not* guaranteed to be an identity (FSQ projection, dead codes),
    so the index pack/unpack is the bijection we round-trip against.
    """
    B, T, N, Q = unified.shape
    flat = unified.reshape(B, T * N, Q)
    assert Q % groups == 0
    q_per_group = Q // groups
    flat = flat.reshape(B, T * N, groups, q_per_group)
    return flat.permute(2, 0, 1, 3).contiguous()


def test_grouped_vqpt_indices_pack_groups_for_ce_and_unpack_for_decode():
    bn = VQPTBottleneck(
        z_dim=16,
        vq_type="grouped_residual_fsq",
        vq_kwargs={"levels": [4, 4, 4, 4], "num_quantizers": 1, "groups": 2},
    )
    h = torch.randn(3, 16, 2, 5)

    grouped = bn.get_indices(h)
    unified = TokenizerWrapper._unflatten_tn_indices(grouped, T=2, N=5)

    assert grouped.shape == (2, 3, 10, 1)
    assert unified.shape == (3, 2, 5, 2)

    wrapper = object.__new__(TokenizerWrapper)
    wrapper.tokenizer = _DecodeOnlyTokenizer(bn)

    decoded = TokenizerWrapper._decode_int_chunk(
        wrapper,
        unified,
        t_comp=2,
        n_pad=5,
        differentiable=True,
    )

    assert decoded.shape == (3, 2, 5)


def test_grouped_vqpt_decode_restores_lfq_group_axis():
    bn = VQPTBottleneck(
        z_dim=16,
        vq_type="grouped_residual_lfq",
        vq_kwargs={"codebook_size": 16, "num_quantizers": 1, "groups": 2},
    )
    h = torch.randn(2, 16, 2, 4)

    grouped = bn.get_indices(h)
    unified = TokenizerWrapper._unflatten_tn_indices(grouped, T=2, N=4)

    wrapper = object.__new__(TokenizerWrapper)
    wrapper.tokenizer = _DecodeOnlyTokenizer(bn)

    decoded = TokenizerWrapper._decode_int_chunk(
        wrapper,
        unified,
        t_comp=2,
        n_pad=4,
        differentiable=True,
    )

    assert decoded.shape == (2, 2, 4)


def test_grouped_vqpt_num_codebooks_infers_groups_without_probe():
    bn = VQPTBottleneck(
        z_dim=16,
        vq_type="grouped_residual_fsq",
        vq_kwargs={"levels": [4, 4, 4, 4], "num_quantizers": 1, "groups": 2},
    )

    wrapper = object.__new__(TokenizerWrapper)
    wrapper.tokenizer = _DecodeOnlyTokenizer(bn)

    assert TokenizerWrapper._infer_bottleneck_num_codebooks(wrapper) == 2


# ---------------------------------------------------------------------------
# Value-level guards
#
# The shape-only tests above pass even if the group-major vs quantizer-major
# packing order or the leading group-axis permute were wrong. The tests below
# assert the actual index VALUES so a mis-ordered permute is caught directly.
#
# The true bijection here is the *index* pack (``_unflatten_tn_indices``) /
# unpack (the reshape+permute inside ``_decode_int_chunk``): both are pure
# tensor reshapes/permutes with no codebook lookup, so ``unpack(pack(x)) == x``
# exactly. The code round-trip (indices -> codes via ``indices_to_codes``) is
# NOT asserted as an identity, because FSQ/LFQ codebook lookup need not be one.
# Instead, where we touch codes, we assert that the packed-then-unpacked path
# lands on the *same* codes the bottleneck produces from its own raw grouped
# indices — i.e. the wrapper's regrouping agrees with the bottleneck's.
# ---------------------------------------------------------------------------


def test_unflatten_packs_groups_in_group_major_order_with_known_values():
    """Pack uses group-major layout: trailing slot ``g*q_per_group + qp``.

    Built from synthetic indices with a unique value per (group, position,
    quantizer) cell so a quantizer-major permute (or a wrong group-axis move)
    produces a different tensor and fails the element check — not just a
    shape check.
    """
    G, B, T, N, QP = 2, 1, 2, 2, 2
    S = T * N
    grouped = torch.zeros(G, B, S, QP, dtype=torch.long)
    for g in range(G):
        for s in range(S):
            for qp in range(QP):
                # Distinct per (g, s, qp) so the layout is fully identifiable.
                grouped[g, 0, s, qp] = 1000 * g + 10 * s + qp

    packed = TokenizerWrapper._unflatten_tn_indices(grouped, T=T, N=N)
    assert packed.shape == (B, T, N, G * QP)

    # group-major: packed[b, t, n, g*QP + qp] == grouped[g, b, t*N + n, qp]
    for t in range(T):
        for n in range(N):
            s = t * N + n
            for g in range(G):
                for qp in range(QP):
                    assert packed[0, t, n, g * QP + qp].item() == grouped[g, 0, s, qp].item()

    # Spot-check the first token's full slot vector against the explicit
    # group-major expectation, so a self-inverse-but-wrong permute is caught.
    assert packed[0, 0, 0].tolist() == [0, 1, 1000, 1001]

    # And the round-trip back to grouped layout is exact.
    recovered = _unpack_unified_to_grouped(packed, groups=G)
    assert torch.equal(recovered, grouped)


def test_grouped_vqpt_index_pack_unpack_is_exact_inverse():
    """Real-bottleneck value-level round-trip for groups=2, q_per_group=1.

    The existing test only checks shapes; this asserts the VALUES survive
    pack -> unpack unchanged, which fails if the group ordering were wrong.
    """
    bn = VQPTBottleneck(
        z_dim=16,
        vq_type="grouped_residual_fsq",
        vq_kwargs={"levels": [4, 4, 4, 4], "num_quantizers": 1, "groups": 2},
    )
    h = torch.randn(3, 16, 2, 5)

    grouped = bn.get_indices(h).to(torch.long)
    assert grouped.shape == (2, 3, 10, 1)  # (G, B, T*N, q_per_group)

    unified = TokenizerWrapper._unflatten_tn_indices(grouped, T=2, N=5)
    recovered = _unpack_unified_to_grouped(unified, groups=int(bn.quantizer.groups))

    assert torch.equal(recovered, grouped)


def test_grouped_vqpt_index_pack_unpack_exact_inverse_q_per_group_gt_1():
    """groups=2 with q_per_group=2 — the case most likely to confuse
    group-major vs quantizer-major ordering.

    Round-trips the real bottleneck indices at the (exactly invertible) index
    layer, then checks the decode path's regrouped indices reproduce the same
    codes the bottleneck emits from its own raw grouped indices.
    """
    bn = VQPTBottleneck(
        z_dim=16,
        vq_type="grouped_residual_fsq",
        vq_kwargs={"levels": [4, 4, 4, 4], "num_quantizers": 2, "groups": 2},
    )
    h = torch.randn(3, 16, 2, 5)

    grouped = bn.get_indices(h).to(torch.long)
    assert grouped.shape == (2, 3, 10, 2)  # (G, B, T*N, q_per_group)

    unified = TokenizerWrapper._unflatten_tn_indices(grouped, T=2, N=5)
    assert unified.shape == (3, 2, 5, 4)  # Q = groups * q_per_group

    groups = int(bn.quantizer.groups)
    recovered = _unpack_unified_to_grouped(unified, groups=groups)
    assert torch.equal(recovered, grouped)

    # The decode path feeds the regrouped indices to indices_to_codes. Those
    # codes must match what the bottleneck produces from its own raw grouped
    # indices — i.e. the wrapper's regrouping is the bottleneck's grouping.
    codes_from_raw = bn.indices_to_codes(grouped, T=2)
    codes_from_packed = bn.indices_to_codes(recovered, T=2)
    assert torch.equal(codes_from_raw, codes_from_packed)

    # Full production decode path still produces the right shape end-to-end.
    wrapper = object.__new__(TokenizerWrapper)
    wrapper.tokenizer = _DecodeOnlyTokenizer(bn)
    decoded = TokenizerWrapper._decode_int_chunk(
        wrapper,
        unified,
        t_comp=2,
        n_pad=5,
        differentiable=True,
    )
    assert decoded.shape == (3, 2, 5)


def test_grouped_vqpt_groups_eq_1_matches_ungrouped_and_round_trips():
    """groups=1 path: the leading group axis is size 1, so
    packing must leave the per-position indices untouched (Q == q_per_group).

    Value-level round-trip plus a check that the unified indices equal the
    bottleneck's indices squeezed of the singleton group axis (the non-grouped
    behavior).
    """
    bn = VQPTBottleneck(
        z_dim=16,
        vq_type="grouped_residual_fsq",
        vq_kwargs={"levels": [4, 4, 4, 4], "num_quantizers": 2, "groups": 1},
    )
    h = torch.randn(3, 16, 2, 5)

    grouped = bn.get_indices(h).to(torch.long)
    assert grouped.shape == (1, 3, 10, 2)  # singleton group axis
    assert int(bn.quantizer.groups) == 1

    unified = TokenizerWrapper._unflatten_tn_indices(grouped, T=2, N=5)
    assert unified.shape == (3, 2, 5, 2)  # Q == q_per_group (no group inflation)

    # With one group, the unified tokens are just the squeezed grouped indices
    # reshaped to (B, T, N, q) — identical to the non-grouped layout.
    ungrouped = grouped[0].reshape(3, 2, 5, 2)
    assert torch.equal(unified, ungrouped)

    recovered = _unpack_unified_to_grouped(unified, groups=1)
    assert torch.equal(recovered, grouped)
