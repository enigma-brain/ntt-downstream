"""Tests for the MAR (Masked Autoregressive) model architecture.

Verifies forward pass, generation, loss modes, diagnostics, and gradient flow.
Uses tiny models on CPU — no GPU required.
"""

import pytest
import torch
from torch import nn

from ntt_downstream.encoding.models.enc.hiera.mar_model import ImageEncodingHieraMAR

DIM = 64
MAR_DIM = 32
TOKEN_DIM = 8
BATCH = 2
N_NEURONS_S1 = 10
N_NEURONS_S2 = 8
SESSION_INFO = {"s1": N_NEURONS_S1, "s2": N_NEURONS_S2}


class TinyBackbone(nn.Module):
    def __init__(self, dim=DIM):
        super().__init__()
        self.proj = nn.Linear(48, dim)

    def forward(self, x):
        b = x.shape[0]
        return self.proj(x.reshape(b, -1)[:, :48]).view(b, 1, 1, -1)


class MockTokenizerWrapper:
    def __init__(self, token_dim=TOKEN_DIM):
        self.token_dim = token_dim
        self.latent_scale = 1.0
        self.latent_channel_means = None
        self.latent_channel_stds = None
        self.is_compress_n = False
        self.reconstruction_rate_hz = 1000.0
        self.tokenizer = type(
            "Tok",
            (),
            {
                "spec": type("S", (), {"latent_dim": None, "z_dim": token_dim})(),
                "parameters": lambda self: iter([]),
                "buffers": lambda self: iter([]),
            },
        )()

    def encode(self, responses, key, raw_discrete=False):
        B, T, N = responses.shape
        return {"target": torch.randn(B, N, self.token_dim), "aux_loss": None}

    def _tokens_to_target(self, t):
        return t

    def _target_to_tokens(self, t):
        return t

    def decode(self, tokens, session_key=None, raw_discrete=False):
        B = tokens.shape[0]
        N = tokens.shape[1]
        return torch.randn(B, 5, N).abs()

    def decode_differentiable(self, tokens, session_key=None):
        B = tokens.shape[0]
        N = tokens.shape[1]
        return torch.randn(B, 5, N).abs()


@pytest.fixture
def backbone():
    return TinyBackbone()


@pytest.fixture
def images():
    return torch.randn(BATCH, 3, 4, 4)


@pytest.fixture
def responses_s1():
    return torch.randn(BATCH, 5, N_NEURONS_S1).abs()


def _make_mar(backbone, loss_mode="mse", **kwargs):
    defaults = {
        "backbone": backbone,
        "tokenizer_wrapper": MockTokenizerWrapper(),
        "session_info": SESSION_INFO,
        "dim": DIM,
        "backbone_dim": DIM,
        "mar_dim": MAR_DIM,
        "mar_depth": 2,
        "mar_num_heads": 4,
    }
    defaults.update(kwargs)
    return ImageEncodingHieraMAR(loss_mode=loss_mode, **defaults)


# ---------------------------------------------------------------------------
# MSE loss mode
# ---------------------------------------------------------------------------


class TestMARMSE:
    def test_forward_returns_loss(self, backbone, images, responses_s1):
        model = _make_mar(backbone, "mse")
        out = model(images, "s1", responses=responses_s1)
        assert "loss" in out
        assert out["loss"].ndim == 0
        assert out["loss"].requires_grad

    def test_smooth_l1_forward(self, backbone, images, responses_s1):
        model = _make_mar(backbone, "smooth_l1")
        out = model(images, "s1", responses=responses_s1)
        assert "loss" in out
        assert out["loss"].requires_grad


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


class TestMARDiagnostics:
    def test_mse_returns_latent_diagnostics(self, backbone, images, responses_s1):
        model = _make_mar(backbone, "mse")
        out = model(images, "s1", responses=responses_s1)
        assert "latent_cosine_sim" in out
        assert "latent_r2" in out
        assert out["latent_cosine_sim"].ndim == 0
        assert out["latent_r2"].ndim == 0

    def test_flow_matching_no_training_diagnostics(self, backbone, images, responses_s1):
        model = _make_mar(backbone, "flow_matching", flow_head_width=32, flow_head_depth=2)
        out = model(images, "s1", responses=responses_s1)
        assert "latent_cosine_sim" not in out


# ---------------------------------------------------------------------------
# Flow matching
# ---------------------------------------------------------------------------


class TestMARFlowMatching:
    def test_forward_returns_loss(self, backbone, images, responses_s1):
        model = _make_mar(backbone, "flow_matching", flow_head_width=32, flow_head_depth=2)
        out = model(images, "s1", responses=responses_s1)
        assert "loss" in out
        assert out["loss"].requires_grad


# ---------------------------------------------------------------------------
# Poisson rates
# ---------------------------------------------------------------------------


class TestMARPoisson:
    def test_forward_returns_loss_without_tokenizer(self, backbone, images, responses_s1):
        model = _make_mar(
            backbone,
            "poisson",
            tokenizer_wrapper=None,
            n_temporal_tokens=responses_s1.shape[1],
        )
        out = model(images, "s1", responses=responses_s1)
        assert "loss" in out
        assert out["loss"].ndim == 0
        assert out["loss"].requires_grad

    def test_generate_shape_without_tokenizer(self, backbone, images, responses_s1):
        model = _make_mar(
            backbone,
            "poisson",
            tokenizer_wrapper=None,
            n_temporal_tokens=responses_s1.shape[1],
        )
        pred = model.generate(images, "s1", n_mar_steps=2)
        assert pred.shape == (BATCH, responses_s1.shape[1], N_NEURONS_S1)


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


class TestMARGenerate:
    def test_mse_generate_shape(self, backbone, images):
        model = _make_mar(backbone, "mse")
        pred = model.generate(images, "s1", n_mar_steps=2)
        assert pred.shape[0] == BATCH
        assert pred.shape[2] == N_NEURONS_S1

    def test_flow_generate_shape(self, backbone, images):
        model = _make_mar(backbone, "flow_matching", flow_head_width=32, flow_head_depth=2)
        pred = model.generate(images, "s1", n_mar_steps=2, n_flow_steps=2)
        assert pred.shape[0] == BATCH
        assert pred.shape[2] == N_NEURONS_S1

    def test_smooth_l1_generate_shape(self, backbone, images):
        model = _make_mar(backbone, "smooth_l1")
        pred = model.generate(images, "s1", n_mar_steps=2)
        assert pred.shape[0] == BATCH
        assert pred.shape[2] == N_NEURONS_S1

    def test_cross_entropy_generate_decodes_indices_with_unified_shape(self, backbone, images):
        class StrictCEWrapper(MockTokenizerWrapper):
            def __init__(self):
                super().__init__(token_dim=1)
                self.seen_decode_shape = None

            def decode(self, tokens, session_key=None):
                raise AssertionError("CE generation should use decode_differentiable")

            def decode_differentiable(self, tokens, session_key=None):
                self.seen_decode_shape = tuple(tokens.shape)
                B = tokens.shape[0]
                N = tokens.shape[1]
                return torch.randn(B, 5, N).abs()

        wrapper = StrictCEWrapper()
        model = _make_mar(
            backbone,
            "cross_entropy",
            tokenizer_wrapper=wrapper,
            ce_vocab_size=32,
        )

        pred = model.generate(images, "s1", n_mar_steps=2)

        assert pred.shape == (BATCH, 5, N_NEURONS_S1)
        assert wrapper.seen_decode_shape == (BATCH, N_NEURONS_S1, 1, 1)


# ---------------------------------------------------------------------------
# Multi-session
# ---------------------------------------------------------------------------


class TestMARMultiSession:
    def test_different_neuron_counts(self, backbone, images):
        model = _make_mar(backbone, "mse")
        resp_s1 = torch.randn(BATCH, 5, N_NEURONS_S1).abs()
        resp_s2 = torch.randn(BATCH, 5, N_NEURONS_S2).abs()
        out_s1 = model(images, "s1", responses=resp_s1)
        out_s2 = model(images, "s2", responses=resp_s2)
        assert out_s1["loss"].ndim == 0
        assert out_s2["loss"].ndim == 0


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestMARGradientFlow:
    def test_gradients_reach_all_components(self, backbone, images, responses_s1):
        model = _make_mar(backbone, "mse")
        out = model(images, "s1", responses=responses_s1)
        out["loss"].backward()

        assert model.mse_head[0].weight.grad is not None, "No grad for mse_head"
        assert model.mar_decoder.layers[0].attn.in_proj_weight.grad is not None, "No grad for transformer"
        assert model.video_proj.weight.grad is not None, "No grad for video_proj"
        assert model.readout["s1"].neuron_embed.grad is not None, "No grad for neuron_embed"

    def test_flow_gradients(self, backbone, images, responses_s1):
        model = _make_mar(backbone, "flow_matching", flow_head_width=32, flow_head_depth=2)
        out = model(images, "s1", responses=responses_s1)
        out["loss"].backward()

        assert model.flow_head.input_proj.weight.grad is not None, "No grad for flow_head"
        assert model.mar_decoder.layers[0].attn.in_proj_weight.grad is not None, "No grad for transformer"


# ---------------------------------------------------------------------------
# Energy loss construction
# ---------------------------------------------------------------------------


class TestMAREnergyConstruction:
    def test_energy_mode_constructs(self, backbone):
        model = _make_mar(
            backbone,
            "energy",
            energy_head_width=32,
            energy_head_depth=2,
            energy_noise_channels=16,
            energy_alpha=1.5,
        )
        assert hasattr(model, "energy_loss")
        assert model.energy_loss.alpha == 1.5
        assert model.energy_loss.noise_channels == 16
        # Attribute name must be ``energy_loss`` (not ``energy_head``) so the
        # optimizer partition in encoding_module.py finds it.
        assert getattr(model, "energy_loss", None) is not None

    def test_energy_mode_alpha_out_of_range_raises(self, backbone):
        # EnergyLoss raises ValueError if alpha not in [1, 2).
        import pytest as _pytest

        with _pytest.raises(ValueError, match="alpha must be in"):
            _make_mar(backbone, "energy", energy_alpha=0.5)


# ---------------------------------------------------------------------------
# Energy loss training
# ---------------------------------------------------------------------------


class TestMAREnergyTraining:
    def test_forward_returns_loss_and_diversity(self, backbone, images, responses_s1):
        model = _make_mar(
            backbone,
            "energy",
            energy_head_width=32,
            energy_head_depth=2,
            energy_noise_channels=16,
            energy_alpha=1.5,
        )
        out = model(images, "s1", responses=responses_s1)
        assert "loss" in out
        assert out["loss"].ndim == 0
        assert out["loss"].requires_grad
        assert "energy_diversity_abs" in out
        assert "energy_diversity_ratio" in out

    def test_backward_populates_grads(self, backbone, images, responses_s1):
        model = _make_mar(
            backbone,
            "energy",
            energy_head_width=32,
            energy_head_depth=2,
            energy_noise_channels=16,
            energy_alpha=1.5,
        )
        out = model(images, "s1", responses=responses_s1)
        out["loss"].backward()
        # Energy head must receive grad signal.
        assert model.energy_loss.net.cond_embed.weight.grad is not None
        # Transformer must receive grad signal too (conditioning path).
        assert model.mar_decoder.layers[0].attn.in_proj_weight.grad is not None


# ---------------------------------------------------------------------------
# Energy loss generate
# ---------------------------------------------------------------------------


class TestMAREnergyGenerate:
    def test_energy_generate_shape(self, backbone, images):
        model = _make_mar(
            backbone,
            "energy",
            energy_head_width=32,
            energy_head_depth=2,
            energy_noise_channels=16,
            energy_alpha=1.5,
        )
        pred = model.generate(images, "s1", n_mar_steps=2)
        assert pred.shape[0] == BATCH
        assert pred.shape[2] == N_NEURONS_S1
