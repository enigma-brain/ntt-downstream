"""Brute-force tests for the Q=1 cross-entropy marginal-rate approximation.

``compute_ce_q1_marginal_rate`` estimates the per-position expected decoded
rate over the categorical posterior by decoding each vocab index in isolation
(all *other* positions forced to codebook index 0) and weighting by the
posterior probabilities. These tests prove that:

1. The estimator is EXACT iff the decoder is affine in the per-position code
   embedding AND temporally local (each position drives only its own output
   window, index 0 contributes nothing). ``test_marginal_exact_for_linear_local_decoder``.
2. For a realistic decoder (nonlinearity and/or cross-position coupling) the
   estimator is BIASED relative to the true marginal computed by exhaustive
   enumeration over all ``V**T_tok`` joint index combinations.
   ``test_marginal_biased_for_nonlinear_decoder``.

The fakes use ``window_bins == 1`` so ``T_raw == T_tok`` and per-position window
slicing is one time bin per position. The decoder honours the integer indices it
is handed in the ``(chunk, S, T_tok, 1)`` long tensor.
"""

from __future__ import annotations

import itertools

import torch

from ntt_downstream.encoding.training.ce_marginal import compute_ce_q1_marginal_rate

V = 4  # vocab size
S = 2  # units
T_TOK = 2  # token positions
B = 3  # batch


class FakeCEHead(torch.nn.Module):
    """Maps per-token embeddings to fixed logits so the posterior is controllable.

    The embedding ``z`` here is just a one-hot-ish carrier; we ignore its content
    and emit deterministic logits keyed by a learned-free weight so the posterior
    is a non-trivial, position-dependent categorical.
    """

    def __init__(self, d: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(d, V), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, d) -> (N, V)
        return x @ self.weight


class FakeModel:
    def __init__(self, d: int):
        self.ce_vocab_size = V
        self.ce_head = FakeCEHead(d)


def _decode_linear_local(idx4: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Affine + temporally-local decoder.

    Output time bin ``t`` depends only on the index at position ``t``:
    ``out[b, t, s] = w[s, t] * idx[b, s, t]``. Index 0 -> 0, so forcing other
    positions to 0 contributes nothing. ``idx4`` is ``(chunk, S, T_tok, 1)`` long.
    """
    idx = idx4[..., 0].float()  # (chunk, S, T_tok)
    out = idx * w.unsqueeze(0)  # (chunk, S, T_tok)
    return out.permute(0, 2, 1).contiguous()  # (chunk, T_tok, S)


def _decode_nonlinear(idx4: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Nonlinear, cross-position-coupled decoder (a realistic stand-in).

    Output bin ``t`` is a softplus of a linear combination that mixes the index
    at ``t`` with the index at every *other* position. Forcing other positions
    to 0 (as the estimator does) therefore drops real cross-position context,
    and the softplus makes it non-affine -- both sources of bias.
    """
    idx = idx4[..., 0].float()  # (chunk, S, T_tok)
    # Cross-position coupling: each bin sees a weighted sum of all positions.
    coupling = torch.tensor([[1.0, 0.5], [0.3, 1.0]], dtype=torch.float32)  # (T_tok, T_tok), [target_bin, source_pos]
    # mixed[b, s, t] = sum_p coupling[t, p] * idx[b, s, p]
    mixed = torch.einsum("bsp,tp->bst", idx, coupling)
    out = torch.nn.functional.softplus(w.unsqueeze(0) * mixed)  # (chunk, S, T_tok)
    return out.permute(0, 2, 1).contiguous()  # (chunk, T_tok, S)


class FakeWrapper:
    """Wrapper exposing ``decode_differentiable`` and ``_compress_t_session_meta``.

    ``window_bins == 1`` keeps the per-position window one time bin wide, so the
    table-build slicing ``decoded[:, pos:pos+1]`` lines up with position ``pos``.
    """

    def __init__(self, decode_fn, w: torch.Tensor, session_key: str):
        self._decode_fn = decode_fn
        self._w = w
        self._compress_t_session_meta = {session_key: {"window_bins": 1}}

    def decode_differentiable(self, idx4: torch.Tensor, *, session_key: str) -> torch.Tensor:
        return self._decode_fn(idx4, self._w)


def _posterior(model: FakeModel, z: torch.Tensor) -> torch.Tensor:
    """Replicate the estimator's posterior: softmax over CE-head logits.

    Returns ``(B, S, T_tok, V)`` float32 probabilities.
    """
    Bz, Sz, Tz, Dz = z.shape
    logits = model.ce_head(z.reshape(Bz * Sz * Tz, Dz))
    return torch.softmax(logits, dim=-1).reshape(Bz, Sz, Tz, V).float()


def _true_marginal_per_unit(decode_fn, w: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """Exhaustive true marginal E[decode(idx)] over the joint per-unit posterior.

    The posterior factorises across positions (independent categoricals) and is
    independent per unit, so for unit ``s`` the joint probability of an index
    sequence ``combo`` is ``prod_pos probs[b, s, pos, combo[pos]]``. Because
    ``decode_differentiable`` accepts a *distinct* index per unit in its
    ``(chunk, S, T_tok, 1)`` argument, we can enumerate all ``V**T_tok`` joint
    combinations once and apply each unit's own posterior weight, decoding via the
    same fake decoder the estimator uses. This is the honest brute-force marginal
    against which the real ``compute_ce_q1_marginal_rate`` is biased.

    Returns ``(B, T_tok, S)`` to match the estimator output layout.
    """
    Bp = probs.shape[0]
    out = torch.zeros((Bp, T_TOK, S), dtype=torch.float32)
    for combo in itertools.product(range(V), repeat=T_TOK):
        idx4 = torch.zeros((1, S, T_TOK, 1), dtype=torch.long)
        for pos, k in enumerate(combo):
            idx4[:, :, pos, 0] = k
        decoded = decode_fn(idx4, w)[0]  # (T_tok, S)
        for b in range(Bp):
            # Per-unit weight: product over positions of that unit's posterior.
            w_units = torch.ones(S, dtype=torch.float32)
            for pos, k in enumerate(combo):
                w_units = w_units * probs[b, :, pos, k]  # (S,)
            # decoded is (T_tok, S); broadcast the per-unit weight over time bins.
            out[b] += w_units.unsqueeze(0) * decoded
    return out


def test_marginal_exact_for_linear_local_decoder():
    """Affine + temporally-local decoder => estimator equals the true marginal."""
    torch.manual_seed(0)
    d = 5
    session_key = "s1"
    z = torch.randn(B, S, T_TOK, d)
    model = FakeModel(d)
    w = torch.randn(S, T_TOK)  # per-(unit, position) linear gain
    wrapper = FakeWrapper(_decode_linear_local, w, session_key)

    cache: dict[str, list[torch.Tensor]] = {}
    est = compute_ce_q1_marginal_rate(model, wrapper, z, session_key, cache)
    assert est.shape == (B, T_TOK, S)

    # Closed-form per-unit true marginal: for a linear-local decoder, bin pos for
    # unit s is w[s, pos] * E[idx_pos], and E[idx_pos] uses that unit's posterior.
    probs = _posterior(model, z)  # (B, S, T_tok, V)
    ks = torch.arange(V, dtype=torch.float32)
    e_idx = torch.einsum("bstv,v->bst", probs, ks)  # (B, S, T_tok)
    true = (e_idx * w.unsqueeze(0)).permute(0, 2, 1).contiguous()  # (B, T_tok, S)

    assert torch.allclose(est, true, atol=1e-5, rtol=1e-5), (
        f"linear-local estimator should be exact; max abs diff {(est - true).abs().max().item():.3e}"
    )


def test_marginal_biased_for_nonlinear_decoder():
    """Nonlinear + cross-position-coupled decoder => estimator is biased.

    Calls the REAL ``compute_ce_q1_marginal_rate`` and compares it against the
    exhaustive true marginal over all V**T_tok joint index combinations, enumerated
    per unit under each unit's own posterior. They must differ beyond fp tolerance.
    """
    torch.manual_seed(1)
    d = 5
    session_key = "s1"
    z = torch.randn(B, S, T_TOK, d)
    model = FakeModel(d)
    w = torch.randn(S, T_TOK).abs() + 0.5  # keep softplus argument well-scaled
    wrapper = FakeWrapper(_decode_nonlinear, w, session_key)

    cache: dict[str, list[torch.Tensor]] = {}
    est = compute_ce_q1_marginal_rate(model, wrapper, z, session_key, cache)
    assert est.shape == (B, T_TOK, S)
    assert torch.isfinite(est).all()

    # Honest brute force: enumerate all V**T_tok joint index combinations per unit,
    # each weighted by that unit's own independent-per-position posterior, decoded
    # via the same fake decoder. This is the true marginal E[decode(idx)] that the
    # real estimator approximates -- and is biased against -- because the estimator
    # forces every *other* position to index 0 and is affine in the table entries.
    probs = _posterior(model, z)  # (B, S, T_tok, V)
    true = _true_marginal_per_unit(_decode_nonlinear, w, probs)  # (B, T_tok, S)

    diff = (est - true).abs().max().item()
    assert diff > 1e-3, (
        f"nonlinear decoder estimator should be biased vs true marginal, but max abs diff was only {diff:.3e}"
    )
