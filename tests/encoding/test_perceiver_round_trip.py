"""Real-tokenizer round trip for the encoding wrapper's Perceiver path.

``TokenizerWrapper.decode`` / ``decode_differentiable`` on a Perceiver must
reproduce ``PerceiverTokenizer.reconstruct_from_input`` when fed the wrapper's
own ``encode`` output. A stub tokenizer can't check this: calling a
``tokenizer.decode(z, e)`` signature would be happily accepted by a stub with
that signature while the real ``PerceiverTokenizer.decode(tokens)`` always
raises.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

perceiver_mod = pytest.importorskip("ntt.contrib.tokenizer.perceiver.perceiver_tokenizer")

from ntt.core.components.types import BinnedSpikesInput  # noqa: E402

from ntt_downstream.encoding.tokenizer_wrappers.encoding_wrapper import TokenizerWrapper  # noqa: E402

D_EMBED = 8
N_UNITS = 16
SAMPLING_RATE_HZ = 100.0  # = tokenizer rate (bin_size_ms=10) -> rebin factor 1


def _write_embeddings_file(tmp_path: Path, n_sessions: int = 2) -> Path:
    """Synthetic NuCLR-style embeddings table (mirrors ntt's extraction tests)."""
    gen = torch.Generator().manual_seed(1234)
    embs = torch.randn(n_sessions * N_UNITS, D_EMBED, generator=gen)
    session_ids = [f"session_{s}" for s in range(n_sessions) for _ in range(N_UNITS)]
    unit_ids = [f"session_{s}/{u}" for s in range(n_sessions) for u in range(N_UNITS)]
    path = tmp_path / "synthetic_embeddings.pt"
    torch.save(
        {"embs": embs, "unit_ids": np.asarray(unit_ids), "session_ids": np.asarray(session_ids)},
        path,
    )
    return path


def _make_tokenizer(embeddings_path: Path):
    spec = perceiver_mod.PerceiverTokenizerSpec(
        hidden_dim=16,
        latent_tokens=4,
        latent_dim=4,
        d_embed=D_EMBED,
        num_heads=4,
        mlp_ratio=1,
        num_encoder_N_blocks=1,
        num_encoder_M_blocks=1,
        num_decoder_blocks=1,
        use_s4=False,
        grad_checkpointing=False,
        use_layernorm=True,
        window_length_seconds=0.08,
        bin_size_ms=10.0,
        embeddings_path=str(embeddings_path),
    )
    return perceiver_mod.PerceiverTokenizer(spec).eval()


def _make_wrapper(tokenizer, latent_layout: str) -> TokenizerWrapper:
    return TokenizerWrapper(
        tokenizer=tokenizer,
        mode="frozen",
        sampling_rate=SAMPLING_RATE_HZ,
        reconstruction_rate_hz=SAMPLING_RATE_HZ,
        subject_id="subject_0",
        species="mouse",
        tokenizer_kind=str(tokenizer.spec.KIND),
        discrete_target_mode="float_index",
        discrete_target_vocab_size=None,
        neuron_batch_size=N_UNITS,
        latent_scale=1.0,
        latent_layout=latent_layout,
    )


def _reference_rates(tokenizer, responses: torch.Tensor, session: str) -> torch.Tensor:
    """``reconstruct_from_input`` on the same input the wrapper builds: ``(B, T, N)``."""
    B, T, N = responses.shape
    dur = T / SAMPLING_RATE_HZ
    inp = BinnedSpikesInput(
        spike_matrix=responses,
        pad_and_gap_mask=torch.ones(B, T, N, dtype=torch.bool),
        sampling_rate=SAMPLING_RATE_HZ,
        unit_ids=torch.arange(N).expand(B, -1).contiguous(),
        unit_ids_pad_mask=torch.ones(B, N, dtype=torch.bool),
        duration_sec=torch.full((B,), dur),
        session_id=(session,) * B,
        subject_id=("subject_0",) * B,
        species=("mouse",) * B,
        absolute_start=torch.zeros(B),
        absolute_end=torch.full((B,), dur),
    )
    with torch.no_grad():
        return tokenizer.reconstruct_from_input(inp).reconstruction


@pytest.mark.parametrize("latent_layout", ["m_as_spatial", "m_flat"])
def test_perceiver_decode_round_trips_real_tokenizer(tmp_path: Path, latent_layout: str) -> None:
    tokenizer = _make_tokenizer(_write_embeddings_file(tmp_path))
    wrapper = _make_wrapper(tokenizer, latent_layout)
    assert wrapper.is_perceiver

    # T deliberately differs from the spec's 8-bin window: the extracted
    # Perceiver is length-agnostic, so any T must round-trip.
    B, T, session = 3, 12, "session_1"
    responses = torch.poisson(torch.full((B, T, N_UNITS), 0.4), generator=torch.Generator().manual_seed(0))

    with torch.no_grad():
        enc = wrapper.encode(responses, session)
    M, C = tokenizer.spec.latent_tokens, tokenizer.spec.latent_dim
    expected_target = (B, M, T, C) if latent_layout == "m_as_spatial" else (B, 1, T, M * C)
    assert tuple(enc["target"].shape) == expected_target

    tokens = wrapper._target_to_tokens(enc["target"])
    reference = _reference_rates(tokenizer, responses, session)
    assert reference.shape == (B, T, N_UNITS)

    rates = wrapper.decode(tokens, session)
    torch.testing.assert_close(rates, reference)

    # The differentiable path (what the encoding losses/metrics use) returns
    # the same rates and lets gradients reach the predicted latents.
    tokens_grad = tokens.detach().clone().requires_grad_(True)
    rates_diff = wrapper.decode_differentiable(tokens_grad, session)
    torch.testing.assert_close(rates_diff.detach(), reference)
    rates_diff.sum().backward()
    assert tokens_grad.grad is not None and tokens_grad.grad.abs().sum() > 0


def test_perceiver_decode_uses_the_requested_session_embeddings(tmp_path: Path) -> None:
    """Same latents decoded under two sessions must differ (per-session NuCLR lookup)."""
    tokenizer = _make_tokenizer(_write_embeddings_file(tmp_path))
    wrapper = _make_wrapper(tokenizer, "m_as_spatial")
    responses = torch.poisson(torch.full((2, 8, N_UNITS), 0.4), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        tokens = wrapper._target_to_tokens(wrapper.encode(responses, "session_0")["target"])
        wrapper.get_or_create_plan("session_1", N_UNITS)
        r0 = wrapper.decode(tokens, "session_0")
        r1 = wrapper.decode(tokens, "session_1")
    assert not torch.allclose(r0, r1)
