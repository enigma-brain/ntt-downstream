"""Tests for the shared loss-mode -> rate-prediction dispatch helper.

The periodic validation (``encoding_module._predict_and_align``) and the final
evaluation (``final_eval.predict_rates`` / ``_walk_loader``) share one dispatch.
These tests pin that both apply the wavelet/LDNS boundary slice uniformly,
including on the CE-Q1 path, so the checkpoint-selection metric and the
reported metric are computed on the same time support.

Fixture approach
----------------
We build a lightweight stand-in ``FakeLit`` exposing only the attributes the
helper reads: ``model``, ``tokenizer_wrapper``, ``cfg``, ``_get_native_hz``,
``_rebin_and_trim``, ``_ldns_splice_active``, ``_cosmos_active``,
``_get_ldns_bg``, ``_ensure_tokenizer_device``, ``_ce_codebook_rates_cache``,
and ``device``. The real ``EncodingLitModule`` carries a full Lightning +
data-module stack and a real tokenizer, none of which the dispatch logic
needs. ``_rebin_and_trim`` here borrows the real implementation's shape
contract (rebin off, skip 0) so the slice lengths we assert on are the same
ones the production module would produce with ``eval_rebin_hz=None`` and
``eval_skip_seconds=0``.

No GPU, no real tokenizer, no flash-attn. Everything runs on CPU with tiny
hand-built tensors.
"""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from ntt_downstream.encoding.training import prediction

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeWrapper:
    """Minimal stand-in for the tokenizer wrapper.

    ``decode_differentiable`` is the identity on the time axis (returns its
    input unchanged in (B, T, N) layout) so that slice lengths flow through
    deterministically. ``_target_to_tokens`` is also identity.
    """

    def __init__(
        self,
        *,
        reconstruction_rate_hz=1000.0,
        is_cosmos=False,
        cosmos_t_safe=None,
        tokenizer_kind="calm",
        is_compress_n=False,
    ):
        self.reconstruction_rate_hz = reconstruction_rate_hz
        self.is_cosmos = is_cosmos
        self.cosmos_t_safe = cosmos_t_safe
        self.tokenizer_kind = tokenizer_kind
        self.is_compress_n = is_compress_n
        # `.tokenizer.to(device)` is called by _ensure_tokenizer_device.
        self.tokenizer = _Movable()

    def decode_differentiable(self, sampled, session_key=None):
        # `sampled` is (B, S, T, D); collapse S/D and return (B, T, N).
        # For our fakes S==1 and D maps to a single channel; we just produce a
        # (B, T, N) tensor whose time length equals T.
        if sampled.ndim == 4:
            b, s, t, d = sampled.shape
            return sampled.reshape(b, t, s * d).float()
        return sampled.float()

    def _target_to_tokens(self, x):
        return x

    def encode(self, responses_btn, key):
        # Used only by the LDNS background path; not exercised here.
        return {"target": responses_btn.unsqueeze(1)}


class _Movable:
    def to(self, device):
        return self


class FakeModel(torch.nn.Module):
    """Stand-in model covering the dispatch branches under test.

    - ``poisson``/``mse_rate``: ``forward`` returns (B, N, T).
    - tokenizer modes: ``forward(return_token_prediction=True)`` returns a
      dict with a (B, S, T_tok, D) ``token_prediction`` and ``sample_tokens``
      returns it unchanged (deterministic), so MC averaging is a no-op we can
      still assert ran.
    """

    def __init__(self, *, n_units=3, t=10, num_codebooks=1, direct=True, ce_vocab_size=4):
        super().__init__()
        self.n_units = n_units
        self.t = t
        self.num_codebooks = num_codebooks
        self.direct = direct
        self.ce_vocab_size = ce_vocab_size
        # A real parameter so `next(model.parameters())` works if needed.
        self._p = torch.nn.Parameter(torch.zeros(1))
        # Records kwargs from the last sample_tokens call so tests can assert
        # which solver actually reached the sampler.
        self.last_sample_kwargs: dict = {}

    def ce_head(self, flat):
        # (M, D) -> (M, V). Deterministic logits so the marginal is reproducible.
        m = flat.shape[0]
        cols = torch.arange(self.ce_vocab_size, dtype=torch.float32)
        return flat.sum(dim=-1, keepdim=True) * 0.0 + cols.unsqueeze(0).expand(m, -1)

    def forward(self, images, key, return_token_prediction=False, responses=None):
        b = images.shape[0]
        if return_token_prediction:
            # (B, S=1, T_tok, D=n_units) — decode identity yields (B, T, n_units).
            z = torch.arange(b * self.t * self.n_units, dtype=torch.float32)
            z = z.reshape(b, 1, self.t, self.n_units)
            return {"token_prediction": z}
        # Direct rate path: (B, N, T).
        out = torch.arange(b * self.n_units * self.t, dtype=torch.float32)
        return out.reshape(b, self.n_units, self.t)

    def sample_tokens(self, z, **kw):
        self.last_sample_kwargs = dict(kw)
        return z


class FakeLit:
    """Lightweight stand-in exposing only what the dispatch helper reads."""

    def __init__(self, model, wrapper, cfg):
        self.model = model
        self.tokenizer_wrapper = wrapper
        self.cfg = cfg
        self.device = torch.device("cpu")
        self._ce_codebook_rates_cache: dict = {}
        self._ldns_bg_bank: dict = {}

    # --- helpers borrowed verbatim (behaviorally) from EncodingLitModule ---

    def _get_native_hz(self) -> float:
        return 1.0 / float(self.cfg.data.get("bin_size", 0.001))

    def _get_eval_rebin_hz(self):
        val = self.cfg.trainer.get("eval_rebin_hz", None)
        return float(val) if val is not None else None

    def _get_eval_skip_bins(self, sampling_rate_hz=None) -> int:
        skip_sec = float(self.cfg.trainer.get("eval_skip_seconds", 0.0))
        if skip_sec <= 0:
            return 0
        if sampling_rate_hz is None:
            sampling_rate_hz = 1.0 / float(self.cfg.data.get("bin_size", 0.001))
        return int(round(skip_sec * sampling_rate_hz))

    def _rebin_and_trim(self, pred, target, pred_hz, target_hz):
        # With eval_rebin_hz=None and eval_skip_seconds=0 (our cfg), this is a
        # min-time trim only — exactly the production contract for that config.
        rebin_hz = self._get_eval_rebin_hz()
        if rebin_hz is not None:
            from ntt_downstream.encoding.models.enc.hiera.rebin import rebin_time_axis

            pred = rebin_time_axis(pred, from_hz=pred_hz, to_hz=rebin_hz)
            target = rebin_time_axis(target, from_hz=target_hz, to_hz=rebin_hz)
        skip = self._get_eval_skip_bins(rebin_hz)
        min_t = min(pred.shape[1], target.shape[1])
        return pred[:, skip:min_t, :], target[:, skip:min_t, :]

    def _ldns_splice_active(self) -> bool:
        if self.tokenizer_wrapper is None:
            return False
        if (self.tokenizer_wrapper.tokenizer_kind or "").lower() != "ldns":
            return False
        return self.cfg.model.get("compress_n_target_slice") is not None

    def _cosmos_active(self) -> bool:
        if self.tokenizer_wrapper is None:
            return False
        return bool(self.tokenizer_wrapper.is_cosmos) and self.tokenizer_wrapper.cosmos_t_safe is not None

    def _ensure_tokenizer_device(self):
        if self.tokenizer_wrapper is not None:
            self.tokenizer_wrapper.tokenizer.to(self.device)

    def _get_ldns_bg(self, key, responses_btn):
        cached = self._ldns_bg_bank.get(key)
        if cached is not None:
            return cached.to(responses_btn.device)
        enc = self.tokenizer_wrapper.encode(responses_btn[:1].float(), key)
        bg = enc["target"]
        self._ldns_bg_bank[key] = bg
        return bg


def _make_cfg(loss_mode, *, model_type="image_encoding", bin_size=0.001, val_micro_batch=0):
    return OmegaConf.create(
        {
            "model": {"loss_mode": loss_mode, "model_type": model_type},
            "data": {"bin_size": bin_size},
            "trainer": {
                "eval_rebin_hz": None,
                "eval_skip_seconds": 0.0,
                "val_micro_batch": val_micro_batch,
                "tokenizer_eval": {},
                "mar_eval": {},
            },
        }
    )


def _periodic_predict(lit, images, responses, key):
    """Drive the periodic-validation path.

    ``EncodingLitModule._predict_and_align`` recurses through
    ``self._predict_and_align`` for the micro-batch chunking, so we bind the
    real method onto the fake instance to let that recursion resolve.
    """
    import types

    from ntt_downstream.encoding.training.encoding_module import EncodingLitModule

    if not hasattr(lit, "_predict_and_align"):
        lit._predict_and_align = types.MethodType(EncodingLitModule._predict_and_align, lit)
    return lit._predict_and_align(images, responses, key)


def _final_eval_predict(lit, images, responses, key):
    """Drive the final_eval path (predict_rates + the alignment done in
    _walk_loader). The helper owns the slice, so we mirror _walk_loader's
    post-predict alignment by calling predict_and_align directly with the same
    MC settings the periodic path uses (n_samples=1)."""
    return prediction.predict_and_align(
        lit,
        images,
        responses,
        key,
        loss_mode=str(lit.cfg.model.get("loss_mode", "poisson")),
        model_type=str(lit.cfg.model.get("model_type", "image_encoding")),
        n_samples=1,
        ce_cache=lit._ce_codebook_rates_cache,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_consistency_cosmos_ce_q1_same_slice():
    """Wavelet tokenizer + CE-Q1: periodic and final_eval paths agree on time length.

    Both go through the shared helper, so the safe-horizon slice
    (cosmos_t_safe) is applied uniformly rather than only on final_eval.
    """
    K = 6  # cosmos_t_safe
    T = 10
    wrapper = FakeWrapper(is_cosmos=True, cosmos_t_safe=K, tokenizer_kind="calm")
    model = FakeModel(n_units=3, t=T, num_codebooks=1)
    cfg = _make_cfg("cross_entropy")
    lit = FakeLit(model, wrapper, cfg)

    images = torch.zeros(4, 1)
    responses = torch.arange(4 * 3 * T, dtype=torch.float32).reshape(4, 3, T)

    pred_p, target_p = _periodic_predict(lit, images, responses, "sess")
    pred_f, target_f = _final_eval_predict(lit, images, responses, "sess")

    assert pred_p.shape[1] == K, f"periodic CE-Q1 must be sliced to cosmos_t_safe={K}, got {pred_p.shape[1]}"
    assert pred_p.shape[1] == pred_f.shape[1]
    assert target_p.shape[1] == target_f.shape[1]
    torch.testing.assert_close(pred_p, pred_f)
    torch.testing.assert_close(target_p, target_f)


def test_equivalence_poisson_and_continuous():
    """Poisson and a continuous tokenizer mode produce expected shapes/slicing.

    Poisson is the direct-rate path (B, N, T) -> (B, T, N), no slice. The
    continuous mode (flow_matching here, deterministic via identity
    sample_tokens) goes through sample + decode and, with the wavelet tokenizer active, is
    sliced to cosmos_t_safe. Both should match between the two entry points.
    """
    T = 10

    # --- poisson: direct rate, no tokenizer slice ---
    model_p = FakeModel(n_units=3, t=T)
    lit_p = FakeLit(model_p, None, _make_cfg("poisson"))
    images = torch.zeros(2, 1)
    responses = torch.arange(2 * 3 * T, dtype=torch.float32).reshape(2, 3, T)

    pred_p, target_p = _periodic_predict(lit_p, images, responses, "sess")
    pred_pf, target_pf = _final_eval_predict(lit_p, images, responses, "sess")
    assert pred_p.shape == (2, T, 3)
    torch.testing.assert_close(pred_p, pred_pf)
    torch.testing.assert_close(target_p, target_pf)

    # --- continuous (flow_matching) with wavelet-tokenizer slice ---
    K = 7
    wrapper = FakeWrapper(is_cosmos=True, cosmos_t_safe=K, tokenizer_kind="calm")
    model_c = FakeModel(n_units=3, t=T, num_codebooks=1)
    lit_c = FakeLit(model_c, wrapper, _make_cfg("flow_matching"))

    pred_c, target_c = _periodic_predict(lit_c, images, responses, "sess")
    pred_cf, target_cf = _final_eval_predict(lit_c, images, responses, "sess")
    assert pred_c.shape[1] == K
    torch.testing.assert_close(pred_c, pred_cf)
    torch.testing.assert_close(target_c, target_cf)


def test_micro_batch_invariance():
    """val_micro_batch=2 vs 0 on an ODD batch (B=5) yields identical tensors.

    The micro-batch chunking lives in encoding_module._predict_and_align and
    must be a pure batching optimization — it concatenates per-chunk
    predictions and must reproduce the un-chunked result bit-for-bit.
    """
    T = 8
    B = 5  # odd: 2 + 2 + 1 under micro_batch=2
    wrapper = FakeWrapper(is_cosmos=True, cosmos_t_safe=5, tokenizer_kind="calm")
    images = torch.zeros(B, 1)
    responses = torch.arange(B * 3 * T, dtype=torch.float32).reshape(B, 3, T)

    lit_off = FakeLit(FakeModel(n_units=3, t=T), wrapper, _make_cfg("cross_entropy", val_micro_batch=0))
    lit_on = FakeLit(FakeModel(n_units=3, t=T), wrapper, _make_cfg("cross_entropy", val_micro_batch=2))

    pred_off, target_off = _periodic_predict(lit_off, images, responses, "sess")
    pred_on, target_on = _periodic_predict(lit_on, images, responses, "sess")

    assert pred_off.shape[0] == B
    torch.testing.assert_close(pred_off, pred_on)
    torch.testing.assert_close(target_off, target_on)


def test_final_eval_flow_solver_not_overridden_by_tokenizer_eval():
    """final_eval's heun must reach sample_tokens even when tokenizer_eval=euler.

    The sampling knobs are caller-resolved: the shared helper must not read
    cfg.trainer.tokenizer_eval, otherwise the default
    tokenizer_eval.flow_solver="euler" would clobber final_eval's explicit
    flow_solver="heun". With euler pinned in tokenizer_eval, a final_eval-style
    call passing flow_solver="heun" must actually sample with heun.
    """
    from ntt_downstream.encoding.training import final_eval

    T = 10
    wrapper = FakeWrapper(is_cosmos=False, tokenizer_kind="calm")
    model = FakeModel(n_units=3, t=T, num_codebooks=1)
    cfg = _make_cfg("flow_matching")
    # Periodic config pins euler — the value final_eval must NOT pick up.
    cfg.trainer.tokenizer_eval = OmegaConf.create({"flow_solver": "euler", "n_flow_steps": 5})
    lit = FakeLit(model, wrapper, cfg)

    images = torch.zeros(2, 1)
    responses = torch.arange(2 * 3 * T, dtype=torch.float32).reshape(2, 3, T)

    # final_eval's predict_rates resolves knobs from its own run args, here heun.
    final_eval.predict_rates(
        lit,
        images,
        responses,
        "sess",
        n_samples=1,
        flow_solver="heun",
        n_flow_steps=12,
        ce_codebook_rates_cache=lit._ce_codebook_rates_cache,
    )
    assert model.last_sample_kwargs.get("flow_solver") == "heun", (
        f"final_eval flow_solver must reach sample_tokens, got {model.last_sample_kwargs.get('flow_solver')!r}"
    )
    assert model.last_sample_kwargs.get("n_flow_steps") == 12

    # And the periodic path on the SAME lit must still pick up euler from cfg.
    model.last_sample_kwargs = {}
    _periodic_predict(lit, images, responses, "sess")
    assert model.last_sample_kwargs.get("flow_solver") == "euler"


def test_final_eval_flow_shift_reaches_sampler():
    """final_eval's flow_shift must reach sample_tokens.

    predict_rates forwards flow_shift to both the sample_tokens and MAR
    generate paths; otherwise every final-eval flow run would sample on the
    uniform grid (shift=1.0) regardless of config.
    """
    from ntt_downstream.encoding.training import final_eval

    T = 10
    wrapper = FakeWrapper(is_cosmos=False, tokenizer_kind="calm")
    model = FakeModel(n_units=3, t=T, num_codebooks=1)
    cfg = _make_cfg("flow_matching")
    lit = FakeLit(model, wrapper, cfg)

    images = torch.zeros(2, 1)
    responses = torch.arange(2 * 3 * T, dtype=torch.float32).reshape(2, 3, T)

    final_eval.predict_rates(
        lit,
        images,
        responses,
        "sess",
        n_samples=1,
        flow_solver="heun",
        n_flow_steps=12,
        flow_shift=2.5,
        ce_codebook_rates_cache=lit._ce_codebook_rates_cache,
    )
    assert model.last_sample_kwargs.get("flow_shift") == 2.5, (
        f"final_eval flow_shift must reach sample_tokens, got {model.last_sample_kwargs.get('flow_shift')!r}"
    )
