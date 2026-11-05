"""Robustness guards for the encoding eval paths.

Two concerns are covered here:

1. **OOM guard scoping** (``encoding_module``). The validation/eval prediction
   passes wrap ``_predict_and_align`` in an OOM-only guard so a genuine CUDA
   out-of-memory event skips one batch's metric update instead of aborting the
   whole training run. Crucially the guard must be *narrow*: a device-side
   assert / illegal-index ``RuntimeError`` (which manifests as "CUDA error: ...")
   is a real bug and must propagate loudly, not be silently swallowed. These
   tests pin both directions.

2. **final_eval MC reproducibility** (``final_eval``). The stochastic loss
   modes MC-average samples drawn via ``model.sample_tokens`` ->
   ``torch.multinomial``, which has no generator hook. ``final_eval.run`` seeds
   the global RNG once up front so two passes on the same checkpoint produce
   bit-identical numbers; a different seed must produce different numbers.

No GPU. Everything runs on CPU with monkeypatched fakes; ``torch.cuda``
attributes (``OutOfMemoryError``, ``empty_cache``) exist regardless of device,
so the guard logic is fully exercisable on CPU.
"""

from __future__ import annotations

import types

import pytest
import torch
from omegaconf import OmegaConf

from ntt_downstream.encoding.training.encoding_module import EncodingLitModule

# ---------------------------------------------------------------------------
# OOM guard
# ---------------------------------------------------------------------------


class _GuardLit:
    """Minimal stand-in carrying just the OOM-guard surface.

    We bind the real ``_guarded_predict_and_align`` (unbound) onto an instance
    of this class so the guard runs against a controllable ``_predict_and_align``
    without constructing a full Lightning module.
    """

    def __init__(self, raiser):
        # raiser(images, responses, key) is invoked in place of the real
        # prediction; tests make it raise to probe the guard.
        self._predict_and_align = raiser


def _make_guard_lit(raiser) -> _GuardLit:
    lit = _GuardLit(raiser)
    lit._guarded_predict_and_align = types.MethodType(EncodingLitModule._guarded_predict_and_align, lit)
    return lit


def test_runtime_error_propagates():
    """A device-side-assert RuntimeError must NOT be swallowed.

    "CUDA error: device-side assert triggered" is a genuine bug (e.g. an
    out-of-range index into an embedding/codebook). Swallowing it would hide
    data/model corruption behind a skipped batch. The guard matches only
    out-of-memory, so this must re-raise.
    """

    def raiser(images, responses, key):
        raise RuntimeError("CUDA error: device-side assert triggered")

    lit = _make_guard_lit(raiser)
    with pytest.raises(RuntimeError, match="device-side assert"):
        lit._guarded_predict_and_align(torch.zeros(1), torch.zeros(1), "sess", context="test")


def test_oom_swallowed(monkeypatch):
    """A RuntimeError whose message contains "out of memory" is swallowed.

    Returns None (caller skips the batch) and frees the CUDA cache.
    """
    called = {"empty_cache": 0}
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: called.__setitem__("empty_cache", called["empty_cache"] + 1))

    def raiser(images, responses, key):
        raise RuntimeError("CUDA out of memory. Tried to allocate ...")

    lit = _make_guard_lit(raiser)
    out = lit._guarded_predict_and_align(torch.zeros(1), torch.zeros(1), "sess", context="test")
    assert out is None
    assert called["empty_cache"] == 1


def test_oom_error_class_swallowed(monkeypatch):
    """A torch.cuda.OutOfMemoryError instance is swallowed (explicit class branch)."""
    called = {"empty_cache": 0}
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: called.__setitem__("empty_cache", called["empty_cache"] + 1))

    def raiser(images, responses, key):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    lit = _make_guard_lit(raiser)
    out = lit._guarded_predict_and_align(torch.zeros(1), torch.zeros(1), "sess", context="test")
    assert out is None
    assert called["empty_cache"] == 1


# ---------------------------------------------------------------------------
# Latent-quality OOM guard (_compute_latent_quality)
# ---------------------------------------------------------------------------
#
# `_compute_latent_quality` calls `self.model(...)` and `self.model.sample_tokens(...)`
# DIRECTLY (not via `_guarded_predict_and_align`), so it needs its own per-batch
# OOM guard. It is the flow/diffusion MC-sampling path that runs every val epoch
# on the primary generative recipes, and it is called BEFORE the train-side DDP
# collectives + barrier — so an unguarded asymmetric OOM there would abort one
# rank or hang NCCL. The guard must stay narrow: OOM skips the batch, a
# device-side-assert RuntimeError propagates.


class _LQModel(torch.nn.Module):
    """Generative-head stand-in whose ``sample_tokens`` can be made to raise.

    ``forward`` returns a token prediction; ``sample_tokens`` either returns a
    deterministic latent or raises whatever the test injected, to exercise the
    OOM guard around the heavy sample/encode block.
    """

    def __init__(self, *, raiser=None, n_units=3, t=4, d=2):
        super().__init__()
        self._raiser = raiser
        self._p = torch.nn.Parameter(torch.zeros(1))
        self.n_units = n_units
        self.t = t
        self.d = d

    def forward(self, images, key, return_token_prediction=False):
        b = images.shape[0]
        return {"token_prediction": torch.zeros(b, 1, self.t, self.d)}

    def sample_tokens(self, token_prediction, **kw):
        if self._raiser is not None:
            self._raiser()
        b = token_prediction.shape[0]
        return torch.ones(b, 1, self.t, self.d)


class _LQTokenizerWrapper:
    """Identity-ish encode wrapper returning a latent target shaped like ``sampled``."""

    def __init__(self, *, t=4, d=2):
        self.tokenizer = _Movable()
        self._t = t
        self._d = d

    def encode(self, responses_btn, key):
        b = responses_btn.shape[0]
        return {"target": torch.zeros(b, 1, self._t, self._d)}


class _LQLit:
    """Minimal stand-in carrying just the ``_compute_latent_quality`` surface."""

    def __init__(self, *, raiser=None, n_units=3, t=4, d=2, n_batch=2):
        self.tokenizer_wrapper = _LQTokenizerWrapper(t=t, d=d)
        self.model = _LQModel(raiser=raiser, n_units=n_units, t=t, d=d)
        self.device = torch.device("cpu")
        self._n_units = n_units
        self._t = t
        self._d = d
        self._n_batch = n_batch
        self.cfg = OmegaConf.create(
            {
                "model": {"model_type": "image_encoding", "loss_mode": "flow_matching"},
                "trainer": {"tokenizer_eval": {}},
            }
        )
        self._val_dataloader_fn = self._loader

    def _loader(self):
        # Yield `_n_batch` separate batches so the per-batch guard's skip/continue
        # behaviour across batches is exercised (each batch has 2 windows).
        for _ in range(self._n_batch):
            images = torch.zeros(2, 1)
            responses = torch.zeros(2, self._n_units, self._t)
            yield ("sess", {"image": images, "response": responses})

    def _ensure_tokenizer_device(self):
        self.tokenizer_wrapper.tokenizer.to(self.device)

    def _ldns_splice_active(self):
        return False


def _make_lq_lit(**kw) -> _LQLit:
    lit = _LQLit(**kw)
    lit._compute_latent_quality = types.MethodType(EncodingLitModule._compute_latent_quality, lit)
    return lit


def test_latent_quality_oom_skips_batch(monkeypatch):
    """An OOM raised inside the sample/encode block skips the batch gracefully.

    With every batch OOM-ing there are no accumulated cosines, so the method
    returns None (rather than aborting the run), and frees the CUDA cache."""
    called = {"empty_cache": 0}
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: called.__setitem__("empty_cache", called["empty_cache"] + 1),
    )

    def oom():
        raise RuntimeError("CUDA out of memory. Tried to allocate ...")

    lit = _make_lq_lit(raiser=oom)
    out = lit._compute_latent_quality(max_batches=20)
    assert out is None
    # One empty_cache per skipped batch.
    assert called["empty_cache"] == lit._n_batch


def test_latent_quality_oom_class_skips_then_returns(monkeypatch):
    """A first-batch OOM is skipped; a subsequent good batch still yields a result."""
    called = {"n": 0, "empty_cache": 0}
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: called.__setitem__("empty_cache", called["empty_cache"] + 1),
    )

    def first_batch_ooms():
        called["n"] += 1
        if called["n"] == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    lit = _make_lq_lit(raiser=first_batch_ooms, n_batch=2)
    out = lit._compute_latent_quality(max_batches=20)
    assert out is not None
    assert "cosine_sim" in out and "r2" in out
    assert called["empty_cache"] == 1


def test_latent_quality_runtime_error_propagates():
    """A device-side-assert RuntimeError must NOT be swallowed by the guard."""

    def raiser():
        raise RuntimeError("CUDA error: device-side assert triggered")

    lit = _make_lq_lit(raiser=raiser)
    with pytest.raises(RuntimeError, match="device-side assert"):
        lit._compute_latent_quality(max_batches=20)


# ---------------------------------------------------------------------------
# final_eval MC reproducibility
# ---------------------------------------------------------------------------


class _Movable:
    def to(self, device):
        return self


class _StochasticWrapper:
    """Identity decode wrapper for the sampled tokenizer path.

    ``decode_differentiable`` collapses (B, S, T, D) -> (B, T, S*D), so the
    stochastic per-position draw produced by ``sample_tokens`` flows straight
    through to the predicted rate. That makes the seeded RNG the only source of
    variation, which is exactly what the determinism test needs.
    """

    reconstruction_rate_hz = 1000.0
    is_cosmos = False
    cosmos_t_safe = None
    tokenizer_kind = "calm"
    is_compress_n = False

    def __init__(self):
        self.tokenizer = _Movable()

    def decode_differentiable(self, sampled, session_key=None):
        b, s, t, d = sampled.shape
        return sampled.reshape(b, t, s * d).float()

    def _target_to_tokens(self, x):
        return x


class _StochasticModel(torch.nn.Module):
    """Model whose ``sample_tokens`` draws from the global torch RNG.

    ``forward(return_token_prediction=True)`` returns deterministic logits;
    ``sample_tokens`` adds RNG-driven noise via ``torch.rand_like`` (a stand-in
    for the ``torch.multinomial`` draw the real samplers use — both read the
    global RNG and have no generator hook). Seeding the global RNG before a pass
    therefore fully determines the output.
    """

    num_codebooks = 2  # >1 so cross_entropy takes the sampled (MC) path

    def __init__(self, *, n_units=3, t=6):
        super().__init__()
        self.n_units = n_units
        self.t = t
        self._p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, images, key, return_token_prediction=False, responses=None):
        b = images.shape[0]
        z = torch.zeros(b, 1, self.t, self.n_units)
        if return_token_prediction:
            return {"token_prediction": z}
        return z.reshape(b, self.n_units, self.t)

    def sample_tokens(self, z, **kw):
        # Reads the global RNG — seeding it upstream makes this reproducible.
        return z + torch.rand_like(z)


class _FinalEvalLit:
    """Lightweight EncodingLitModule stand-in for final_eval.run.

    Exposes the dispatch surface plus the two loader callables and ``cfg.seed``.
    Both loaders yield a single deterministic batch.
    """

    def __init__(self, cfg, *, n_units=3, t=6, n_batch=2):
        self.cfg = cfg
        self.model = _StochasticModel(n_units=n_units, t=t)
        self.tokenizer_wrapper = _StochasticWrapper()
        self.device = torch.device("cpu")
        self._ce_codebook_rates_cache = {}
        self._ldns_bg_bank = {}
        self._n_units = n_units
        self._t = t
        self._n_batch = n_batch
        self._val_dataloader_fn = self._loader
        self._val_avg_dataloader_fn = self._loader

    def _loader(self):
        images = torch.zeros(self._n_batch, 1)
        responses = torch.arange(self._n_batch * self._n_units * self._t, dtype=torch.float32).reshape(
            self._n_batch, self._n_units, self._t
        )
        image_ids = torch.arange(self._n_batch)
        yield (
            "sess",
            {
                "image": images,
                "response": responses,
                "n_units": self._n_units,
                "image_id": image_ids,
            },
        )

    # --- dispatch helpers (behaviorally borrowed from EncodingLitModule) ---

    def _get_native_hz(self):
        return 1.0 / float(self.cfg.data.get("bin_size", 0.001))

    def _get_eval_rebin_hz(self):
        val = self.cfg.trainer.get("eval_rebin_hz", None)
        return float(val) if val is not None else None

    def _get_eval_skip_bins(self, sampling_rate_hz=None):
        return 0

    def _rebin_and_trim(self, pred, target, pred_hz, target_hz):
        min_t = min(pred.shape[1], target.shape[1])
        return pred[:, :min_t, :], target[:, :min_t, :]

    def _ldns_splice_active(self):
        return False

    def _cosmos_active(self):
        return False

    def _ensure_tokenizer_device(self):
        self.tokenizer_wrapper.tokenizer.to(self.device)


def _make_final_eval_cfg(seed=None):
    cfg = {
        "model": {"loss_mode": "cross_entropy", "model_type": "image_encoding"},
        "data": {"bin_size": 0.001},
        "trainer": {
            "eval_rebin_hz": None,
            "eval_skip_seconds": 0.0,
            "val_micro_batch": 0,
            "tokenizer_eval": {},
            "mar_eval": {},
        },
    }
    if seed is not None:
        cfg["seed"] = seed
    return OmegaConf.create(cfg)


def _run_final_eval(seed):
    """Run final_eval.run with no file/MLflow output; return the val table."""
    from ntt_downstream.encoding.training import final_eval

    lit = _FinalEvalLit(_make_final_eval_cfg(seed=seed))
    table = final_eval.run(lit, n_samples=4, flow_solver="heun", n_flow_steps=3)
    # Pull the per-session rate_pearson mean out of the rendered table — it is a
    # deterministic function of the MC-averaged prediction, hence of the seed.
    return table["per_session"]["sess"]["rate_pearson"]["mean"]


def test_final_eval_mc_deterministic():
    """Two final_eval.run passes with the same seed are bit-identical; a
    different seed differs.

    The MC path draws stochastic samples through the global RNG, so without
    seeding two passes would differ. final_eval.run seeds once up front; this
    pins that the seeding actually takes effect (same seed -> equal) and is
    load-bearing (different seed -> not equal)."""
    a = _run_final_eval(seed=123)
    b = _run_final_eval(seed=123)
    assert torch.equal(torch.tensor(a), torch.tensor(b))

    c = _run_final_eval(seed=999)
    # Different seed should move the MC estimate. (Astronomically unlikely to
    # collide for distinct seeds over a 4-sample MC average of rand draws.)
    assert not torch.equal(torch.tensor(a), torch.tensor(c))
