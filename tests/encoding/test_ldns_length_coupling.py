"""Length-coupling guards for the LDNS encode-long / predict-short slice.

The encode TARGET is sliced to
``compress_n_target_slice`` for LDNS tokenizers, but the PREDICTION latent
``z`` (token_prediction, ``(B, S, T, D)``) is not. If ``z``'s token-time
length disagrees with the sliced target length ``K``, the cross-entropy path
raises a size mismatch while the continuous (flow/diffusion/bce/mse) paths can
broadcast or misalign SILENTLY and corrupt the loss. The forward must instead
FAIL LOUD with a message naming the offending config keys.

These tests exercise the asserted code with a minimal stub object that binds
the real model methods (``_training_loss``, ``_maybe_ldns_slice``,
``_is_ldns_tokenizer``) and stubs out the heavy encode/token-prediction path so
we can drive ``z`` and the target to chosen shapes. Full model instantiation is
deliberately avoided — the invariant under test is purely about tensor shapes.
"""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch

from ntt_downstream.encoding.models.enc.hiera.base_model import ImageEncodingHiera

# ---------------------------------------------------------------------------
# Minimal stub harness
# ---------------------------------------------------------------------------


class _StubTokenizerWrapper:
    """Returns a fixed-length encode target so we can force a z/target mismatch.

    ``target_t`` is the token-time length of the *unsliced* encode target. For
    LDNS this is the full-window length; the forward slices it down to ``K``.
    """

    def __init__(self, target_t, latent_dim=4, vocab=8, tokenizer_kind="ldns"):
        self.target_t = target_t
        self.latent_dim = latent_dim
        self.vocab = vocab
        self.tokenizer_kind = tokenizer_kind

    def encode(self, responses, key, raw_discrete=False, skip_target_scaling=False):
        B = responses.shape[0]
        S, T, D = 1, self.target_t, self.latent_dim
        if raw_discrete:
            target = torch.randint(0, self.vocab, (B, S, T, 1), dtype=torch.long)
        else:
            target = torch.randn(B, S, T, D)
        return {"target": target}


def _make_stub_model(
    *,
    loss_mode,
    z_t,
    target_t,
    compress_n_target_slice,
    tokenizer_kind="ldns",
    latent_dim=4,
    vocab=8,
):
    """Build a stub exposing exactly the attributes ``_training_loss`` reads.

    ``z_t`` is the token-time length of the prediction latent; ``target_t`` the
    token-time length of the (unsliced) encode target. We bind the real
    ``_training_loss``, ``_maybe_ldns_slice`` and ``_is_ldns_tokenizer`` onto a
    SimpleNamespace and stub the encode/attend + token-prediction path so the
    branch under test runs without a real backbone or readout.
    """
    stub = SimpleNamespace()
    stub.loss_mode = loss_mode
    stub.compress_n_target_slice = compress_n_target_slice
    stub.has_temporal_readout = True
    stub.tokenizer_wrapper = _StubTokenizerWrapper(
        target_t=target_t, latent_dim=latent_dim, vocab=vocab, tokenizer_kind=tokenizer_kind
    )
    stub.ce_vocab_size = vocab

    # Batch must match the responses passed in _run_forward (B=2) so the encode
    # target and z share a batch dim; only token-time (dim 2) is varied per test.
    z = torch.randn(2, 1, z_t, latent_dim, requires_grad=True)

    # Stub the encode/attend + token-prediction path. _encode_and_attend returns
    # (o, b, film_cond); we ignore o/b downstream because we override the
    # token-prediction methods to return our chosen z.
    stub._encode_and_attend = lambda images, key: (None, images.shape[0], None)
    stub.readout = {"s0": SimpleNamespace(unique_ids=1)}
    stub._predict_tokens_compress_n = lambda o, num_chunks: {"token_prediction": z}
    stub._predict_tokens = lambda o, film_cond=None: {"token_prediction": z}

    # CE head: map latent_dim -> vocab logits (Q=1).
    ce_head = torch.nn.Linear(latent_dim, vocab)
    stub.ce_head = ce_head
    # CE imbalance knobs at their constructor defaults (plain F.cross_entropy).
    stub.ce_class_weight = None
    stub.ce_focal_gamma = 0.0

    # Continuous-mode plumbing read by _training_loss before/at the assertion.
    stub.use_full_grid_denoiser = False
    stub.use_ldns_denoiser = False
    stub.diffusion_batch_mul = 1

    # Pointwise generative head stub for flow_matching (only reached when shapes match).
    stub.flow_head = SimpleNamespace(compute_loss=lambda t, z_: ((t - z_) ** 2).mean())

    # Tail-of-forward plumbing read after the loss in the continuous branch.
    stub.tokenizer_aux_loss_weight = 0.0
    stub.decoder_recon_loss_weight = 0.0

    # Bind the real methods under test.
    stub._maybe_ldns_slice = MethodType(ImageEncodingHiera._maybe_ldns_slice, stub)
    stub._is_ldns_tokenizer = MethodType(ImageEncodingHiera._is_ldns_tokenizer, stub)
    stub._assert_token_time_match = MethodType(ImageEncodingHiera._assert_token_time_match, stub)
    stub._cross_entropy_loss = MethodType(ImageEncodingHiera._cross_entropy_loss, stub)
    stub._training_loss = MethodType(ImageEncodingHiera._training_loss, stub)
    return stub


def _run_forward(stub, B=2, N=3, T_in=10):
    images = torch.randn(B, 3, 8, 8)
    responses = torch.randn(B, T_in, N)
    return stub._training_loss(images, "s0", responses)


# ---------------------------------------------------------------------------
# _maybe_ldns_slice unit tests
# ---------------------------------------------------------------------------


def _maybe_ldns_slice_stub(compress_n_target_slice, tokenizer_kind):
    stub = SimpleNamespace()
    stub.compress_n_target_slice = compress_n_target_slice
    stub.tokenizer_wrapper = SimpleNamespace(tokenizer_kind=tokenizer_kind)
    stub._maybe_ldns_slice = MethodType(ImageEncodingHiera._maybe_ldns_slice, stub)
    stub._is_ldns_tokenizer = MethodType(ImageEncodingHiera._is_ldns_tokenizer, stub)
    return stub


def test_maybe_ldns_slice_slices_when_active():
    stub = _maybe_ldns_slice_stub(compress_n_target_slice=3, tokenizer_kind="ldns")
    t = torch.randn(2, 1, 10, 4)
    out = stub._maybe_ldns_slice(t)
    assert out.shape == (2, 1, 3, 4)
    assert torch.equal(out, t[:, :, :3, :])


def test_maybe_ldns_slice_identity_when_slice_none():
    stub = _maybe_ldns_slice_stub(compress_n_target_slice=None, tokenizer_kind="ldns")
    t = torch.randn(2, 1, 10, 4)
    out = stub._maybe_ldns_slice(t)
    assert out is t


def test_maybe_ldns_slice_identity_when_not_ldns():
    stub = _maybe_ldns_slice_stub(compress_n_target_slice=3, tokenizer_kind="cosmos")
    t = torch.randn(2, 1, 10, 4)
    out = stub._maybe_ldns_slice(t)
    assert out is t


# ---------------------------------------------------------------------------
# Forward assertion: CE branch
# ---------------------------------------------------------------------------


def test_ce_branch_raises_on_length_mismatch():
    # encode target has T=10, sliced to K=3, but z emits T=5 -> mismatch.
    stub = _make_stub_model(loss_mode="cross_entropy", z_t=5, target_t=10, compress_n_target_slice=3)
    with pytest.raises((ValueError, AssertionError)) as exc:
        _run_forward(stub)
    msg = str(exc.value)
    assert "compress_n_target_slice" in msg
    assert "5" in msg and "3" in msg


def test_ce_branch_proceeds_when_lengths_match():
    # z emits T=3 == sliced K=3 -> assertion is a no-op, loss computes.
    stub = _make_stub_model(loss_mode="cross_entropy", z_t=3, target_t=10, compress_n_target_slice=3)
    out = _run_forward(stub)
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])


# ---------------------------------------------------------------------------
# Forward assertion: continuous branch
# ---------------------------------------------------------------------------


def test_continuous_branch_raises_on_length_mismatch():
    # The dangerous case: continuous paths broadcast silently without a guard.
    stub = _make_stub_model(loss_mode="flow_matching", z_t=5, target_t=10, compress_n_target_slice=3)
    with pytest.raises((ValueError, AssertionError)) as exc:
        _run_forward(stub)
    msg = str(exc.value)
    assert "compress_n_target_slice" in msg
    assert "5" in msg and "3" in msg


def test_continuous_branch_proceeds_when_lengths_match():
    stub = _make_stub_model(loss_mode="flow_matching", z_t=3, target_t=10, compress_n_target_slice=3)
    out = _run_forward(stub)
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])
