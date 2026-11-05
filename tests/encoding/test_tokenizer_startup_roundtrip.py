"""Tests for the startup tokenizer roundtrip pre-fit check.

The helper takes an ``EncodingWrapper`` and a ``(key, batch_dict)``-yielding
dataloader. We stub both with the smallest possible interface that still
exercises the real metric / figure code paths.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")  # headless figure creation in tests
import matplotlib.figure  # noqa: E402
import torch  # noqa: E402

from ntt_downstream.encoding.training.tokenizer_startup_roundtrip import run_startup_roundtrip  # noqa: E402


class _IdentityWrapper:
    """Smallest stub: encode returns ``{"target": x}``, decode_differentiable returns x.

    Acts like a perfect (continuous) tokenizer — recon == original — so
    rate_corr should be 1.0. ``produces_discrete_tokens=False`` makes the
    roundtrip take the continuous path (``_target_to_tokens`` → decode), and
    ``_target_to_tokens`` is an identity passthrough here.
    """

    produces_discrete_tokens = False

    def __init__(self, recon_hz: float):
        self.reconstruction_rate_hz = recon_hz

    def encode(self, x, key, raw_discrete=False):  # noqa: ARG002
        return {"target": x}

    def _target_to_tokens(self, x):
        return x

    def decode_differentiable(self, x, session_key=None):  # noqa: ARG002
        return x


class _NoisyWrapper:
    """Continuous stub that adds independent noise on decode; rate_corr should be < 1."""

    produces_discrete_tokens = False

    def __init__(self, recon_hz: float, noise_std: float, seed: int = 0):
        self.reconstruction_rate_hz = recon_hz
        self._g = torch.Generator().manual_seed(seed)
        self._noise_std = noise_std

    def encode(self, x, key, raw_discrete=False):  # noqa: ARG002
        return {"target": x}

    def _target_to_tokens(self, x):
        return x

    def decode_differentiable(self, x, session_key=None):  # noqa: ARG002
        return x + self._noise_std * torch.randn(x.shape, generator=self._g, device=x.device)


class _DiscreteWrapper:
    """Discrete tokenizer stub: ``encode(raw_discrete=True)`` yields int indices.

    ``decode_differentiable`` ASSERTS it receives an integer tensor — mirroring
    the real ``_decode_int_chunk`` path that routes int indices through
    ``indices_to_codes``. It would error on float latents. The roundtrip must
    pick the ``raw_discrete=True`` encode for this family.
    """

    produces_discrete_tokens = True

    def __init__(self, recon_hz: float):
        self.reconstruction_rate_hz = recon_hz

    def encode(self, x, key, raw_discrete=False):  # noqa: ARG002
        assert raw_discrete, "discrete family must be encoded with raw_discrete=True"
        # Emulate integer codebook indices derived from the input.
        return {"target": (x * 4).round().long()}

    def _target_to_tokens(self, x):  # pragma: no cover - must not be called for discrete
        raise AssertionError("discrete family must not route through _target_to_tokens")

    def decode_differentiable(self, x, session_key=None):  # noqa: ARG002
        assert not x.is_floating_point(), "discrete decode must receive int indices"
        return x.to(torch.float32) / 4


class _ContinuousWrapper:
    """Continuous tokenizer stub whose decode REQUIRES float latents.

    ``encode(raw_discrete=True)`` is meaningless for a continuous bottleneck; if
    the roundtrip still passes ``raw_discrete=True``, this stub returns an int
    tensor (mimicking how the real wrapper's discrete-only metadata would be
    misused) and the float-only decode then raises. ``decode_differentiable``
    asserts it receives the ``_target_to_tokens`` output, which it tags.
    """

    produces_discrete_tokens = False

    def __init__(self, recon_hz: float):
        self.reconstruction_rate_hz = recon_hz
        self._saw_target_to_tokens = False

    def encode(self, x, key, raw_discrete=False):
        if raw_discrete:
            # Wrong path for a continuous tokenizer: hand back ints so the
            # float-only decoder below blows up, exactly as the current
            # always-raw_discrete code would on a real continuous tokenizer.
            return {"target": (x * 4).round().long()}
        return {"target": x}

    def _target_to_tokens(self, x):
        self._saw_target_to_tokens = True
        return x

    def decode_differentiable(self, x, session_key=None):  # noqa: ARG002
        if not x.is_floating_point():
            raise TypeError("continuous decode requires float latents, got int indices")
        assert self._saw_target_to_tokens, "continuous decode must receive _target_to_tokens output"
        return x


class _DegenerateWrapper:
    """Continuous stub whose decode produces an all-zeros (degenerate) recon."""

    produces_discrete_tokens = False

    def __init__(self, recon_hz: float, mode: str = "zeros"):
        self.reconstruction_rate_hz = recon_hz
        self._mode = mode

    def encode(self, x, key, raw_discrete=False):  # noqa: ARG002
        return {"target": x}

    def _target_to_tokens(self, x):
        return x

    def decode_differentiable(self, x, session_key=None):  # noqa: ARG002
        if self._mode == "nan":
            return torch.full_like(x, float("nan"))
        return torch.zeros_like(x)


def _make_loader(batches):
    """Yield ``(session_key, {"response": (B, N, T)})`` tuples to mimic Cowley loader."""
    return iter(batches)


def _batch(B, N, T, seed: int, session: str = "sess0"):
    g = torch.Generator().manual_seed(seed)
    response = torch.rand(B, N, T, generator=g) * 0.5  # firing rates in [0, 0.5]
    return session, {"response": response}


def test_metrics_have_expected_keys_and_perfect_recon():
    """Identity wrapper → rate_corr ≈ 1.0; all four metric keys present."""
    wrapper = _IdentityWrapper(recon_hz=1000.0)
    batches = [_batch(B=8, N=12, T=20, seed=i, session="sess0") for i in range(4)]

    metrics, fig_traces, fig_scatter = run_startup_roundtrip(
        wrapper,
        _make_loader(batches),
        native_hz=1000.0,
        eval_rebin_hz=None,
        device="cpu",
        max_batches=10,
    )

    expected = {
        "tokenizer_ceiling/rate_corr",
        "tokenizer_ceiling/temporal_corr",
        "tokenizer_ceiling/n_batches",
        "tokenizer_ceiling/n_neurons_total",
    }
    assert set(metrics.keys()) == expected
    assert metrics["tokenizer_ceiling/rate_corr"] == 1.0 or math.isclose(
        metrics["tokenizer_ceiling/rate_corr"], 1.0, rel_tol=1e-5, abs_tol=1e-5
    )
    assert metrics["tokenizer_ceiling/temporal_corr"] == 1.0 or math.isclose(
        metrics["tokenizer_ceiling/temporal_corr"], 1.0, rel_tol=1e-5, abs_tol=1e-5
    )
    assert metrics["tokenizer_ceiling/n_batches"] == 4.0
    # n_neurons_total is the true per-session neuron count (constant across
    # batches), NOT n_batches * N. All four batches share session "sess0" with
    # N=12 neurons, so the count is 12 — not 4*12.
    assert metrics["tokenizer_ceiling/n_neurons_total"] == 12

    # Cleanup
    import matplotlib.pyplot as plt

    plt.close(fig_traces)
    plt.close(fig_scatter)


def test_returns_two_figures_with_axes():
    wrapper = _IdentityWrapper(recon_hz=1000.0)
    batches = [_batch(B=8, N=12, T=20, seed=i, session="sess0") for i in range(2)]

    _, fig_traces, fig_scatter = run_startup_roundtrip(
        wrapper,
        _make_loader(batches),
        native_hz=1000.0,
        eval_rebin_hz=None,
        device="cpu",
        max_batches=10,
    )
    assert isinstance(fig_traces, matplotlib.figure.Figure)
    assert isinstance(fig_scatter, matplotlib.figure.Figure)
    assert len(fig_traces.axes) >= 1
    assert len(fig_scatter.axes) >= 1

    import matplotlib.pyplot as plt

    plt.close(fig_traces)
    plt.close(fig_scatter)


def test_skips_when_loader_empty():
    """Empty loader → n_batches=0, NaN metrics, no crash."""
    wrapper = _IdentityWrapper(recon_hz=1000.0)
    metrics, fig_traces, fig_scatter = run_startup_roundtrip(
        wrapper,
        _make_loader([]),
        native_hz=1000.0,
        eval_rebin_hz=None,
        device="cpu",
        max_batches=10,
    )
    assert metrics["tokenizer_ceiling/n_batches"] == 0.0
    assert metrics["tokenizer_ceiling/n_neurons_total"] == 0
    assert math.isnan(metrics["tokenizer_ceiling/rate_corr"])
    assert math.isnan(metrics["tokenizer_ceiling/temporal_corr"])
    assert isinstance(fig_traces, matplotlib.figure.Figure)
    assert isinstance(fig_scatter, matplotlib.figure.Figure)

    import matplotlib.pyplot as plt

    plt.close(fig_traces)
    plt.close(fig_scatter)


def test_noisy_wrapper_lowers_rate_corr():
    """Sanity check: when decode adds noise, rate_corr < 1."""
    wrapper = _NoisyWrapper(recon_hz=1000.0, noise_std=0.5)
    # Need multiple sessions worth of trials so per-neuron Pearson-r across-trials is meaningful
    batches = [_batch(B=16, N=8, T=30, seed=i, session="sess0") for i in range(3)]

    metrics, fig_traces, fig_scatter = run_startup_roundtrip(
        wrapper,
        _make_loader(batches),
        native_hz=1000.0,
        eval_rebin_hz=None,
        device="cpu",
        max_batches=10,
    )
    rc = metrics["tokenizer_ceiling/rate_corr"]
    assert not math.isnan(rc)
    assert rc < 1.0

    import matplotlib.pyplot as plt

    plt.close(fig_traces)
    plt.close(fig_scatter)


def test_eval_rebin_changes_temporal_metric_definition():
    """With T=1 after rebin, temporal_corr should be NaN (no time axis to correlate)."""
    wrapper = _IdentityWrapper(recon_hz=1000.0)
    # native_hz=1000, T=20 → 20ms window. Rebin to 50hz → 1 bin. Identity recon still gives rate_corr=1.
    batches = [_batch(B=8, N=8, T=20, seed=i, session="sess0") for i in range(2)]

    metrics, fig_traces, fig_scatter = run_startup_roundtrip(
        wrapper,
        _make_loader(batches),
        native_hz=1000.0,
        eval_rebin_hz=50.0,
        device="cpu",
        max_batches=10,
    )
    assert math.isnan(metrics["tokenizer_ceiling/temporal_corr"])
    # rate_corr unchanged because identity recon still equals original
    assert math.isclose(metrics["tokenizer_ceiling/rate_corr"], 1.0, abs_tol=1e-5)

    import matplotlib.pyplot as plt

    plt.close(fig_traces)
    plt.close(fig_scatter)


# ---------------------------------------------------------------------------
# C2 — decode the right representation per tokenizer family
# ---------------------------------------------------------------------------


def _run(wrapper, batches, **kw):
    """Convenience runner returning only metrics, closing figures."""
    import matplotlib.pyplot as plt

    metrics, ft, fs = run_startup_roundtrip(
        wrapper,
        _make_loader(batches),
        native_hz=1000.0,
        eval_rebin_hz=None,
        device="cpu",
        max_batches=10,
        **kw,
    )
    plt.close(ft)
    plt.close(fs)
    return metrics


def test_discrete_family_uses_raw_discrete_path():
    """Discrete tokenizer: encode(raw_discrete=True) → int decode roundtrips cleanly."""
    wrapper = _DiscreteWrapper(recon_hz=1000.0)
    batches = [_batch(B=8, N=6, T=20, seed=i, session="sess0") for i in range(3)]
    metrics = _run(wrapper, batches)
    # round(x*4)/4 is a near-identity quantization of x in [0, 0.5]; rate_corr high.
    assert not math.isnan(metrics["tokenizer_ceiling/rate_corr"])
    assert metrics["tokenizer_ceiling/rate_corr"] > 0.9


def test_continuous_family_uses_target_to_tokens_path():
    """Continuous tokenizer: roundtrip must NOT pass raw_discrete=True.

    The continuous stub's float-only decoder raises if handed int indices, so
    the family branch must route continuous tokenizers through
    ``_target_to_tokens`` → ``decode_differentiable``.
    """
    wrapper = _ContinuousWrapper(recon_hz=1000.0)
    batches = [_batch(B=8, N=6, T=20, seed=i, session="sess0") for i in range(3)]
    metrics = _run(wrapper, batches)
    assert wrapper._saw_target_to_tokens
    assert math.isclose(metrics["tokenizer_ceiling/rate_corr"], 1.0, abs_tol=1e-5)


# ---------------------------------------------------------------------------
# Fail loud on a broken tokenizer
# ---------------------------------------------------------------------------


def test_degenerate_recon_warns_by_default(caplog):
    """All-zeros recon → loud warning, but no raise when strict=False (default)."""
    import logging

    wrapper = _DegenerateWrapper(recon_hz=1000.0, mode="zeros")
    batches = [_batch(B=8, N=6, T=20, seed=i, session="sess0") for i in range(2)]
    with caplog.at_level(logging.WARNING):
        metrics = _run(wrapper, batches)
    assert any("degenerate" in r.message.lower() or "non-finite" in r.message.lower() for r in caplog.records)
    # Did not raise; metrics still returned.
    assert "tokenizer_ceiling/rate_corr" in metrics


def test_nan_recon_warns_by_default(caplog):
    """NaN recon → loud warning, not a silent NaN, when strict=False."""
    import logging

    wrapper = _DegenerateWrapper(recon_hz=1000.0, mode="nan")
    batches = [_batch(B=8, N=6, T=20, seed=i, session="sess0") for i in range(2)]
    with caplog.at_level(logging.WARNING):
        _run(wrapper, batches)
    assert any("non-finite" in r.message.lower() or "degenerate" in r.message.lower() for r in caplog.records)


def test_degenerate_recon_raises_under_strict():
    """All-zeros recon → RuntimeError when strict=True."""
    import pytest

    wrapper = _DegenerateWrapper(recon_hz=1000.0, mode="zeros")
    batches = [_batch(B=8, N=6, T=20, seed=i, session="sess0") for i in range(2)]
    with pytest.raises(RuntimeError):
        _run(wrapper, batches, strict=True)


# ---------------------------------------------------------------------------
# Aggregate r matches a hand-rolled per-neuron Pearson r
# ---------------------------------------------------------------------------


def test_accumulator_matches_handrolled_aggregate():
    """OnlinePearsonAccumulator aggregate equals a hand-rolled value.

    Recompute the per-session per-neuron Pearson r on the same deterministic
    pred/target rates and compare against the metric the roundtrip reports.
    """
    import torch as _torch

    wrapper = _NoisyWrapper(recon_hz=1000.0, noise_std=0.3, seed=7)
    batches = [_batch(B=16, N=8, T=25, seed=i, session="sess0") for i in range(3)]
    metrics = _run(wrapper, batches)
    agg_new = metrics["tokenizer_ceiling/rate_corr"]

    # Reconstruct pred/target rates exactly as the roundtrip does (continuous
    # identity-plus-noise, no rebin), then apply the hand-rolled formula.
    g = _torch.Generator().manual_seed(7)
    preds, targs = [], []
    for i in range(3):
        _, bd = _batch(B=16, N=8, T=25, seed=i, session="sess0")
        resp = bd["response"].to(_torch.float32)  # (B, N, T)
        original = resp.transpose(1, 2)  # (B, T, N)
        recon = original + 0.3 * _torch.randn(original.shape, generator=g)
        preds.append(recon.sum(dim=1))  # (B, N)
        targs.append(original.sum(dim=1))
    pred = _torch.cat(preds, dim=0)
    targ = _torch.cat(targs, dim=0)
    pred_c = pred - pred.mean(dim=0, keepdim=True)
    targ_c = targ - targ.mean(dim=0, keepdim=True)
    numer = (pred_c * targ_c).sum(dim=0)
    denom = pred_c.pow(2).sum(dim=0).sqrt() * targ_c.pow(2).sum(dim=0).sqrt()
    nan_t = _torch.tensor(float("nan"))
    r_old = _torch.where(denom > 1e-8, numer / denom, nan_t)
    agg_old = float(_torch.nanmean(r_old).item())

    assert math.isclose(agg_new, agg_old, rel_tol=1e-4, abs_tol=1e-4)
