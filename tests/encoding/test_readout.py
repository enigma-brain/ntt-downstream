"""Tests for the multi-layer readout architecture.

Verifies forward pass shapes and loss computation for all readout configurations
and loss modes. Uses tiny models on CPU — no GPU required.
"""

import pytest
import torch
from torch import nn

from ntt_downstream.encoding.models.enc.hiera.base_model import (
    ImageEncodingHiera,
    ReadoutCrossAttentionLayer,
    ReadoutSelfAttentionLayer,
    SessionReadout,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DIM = 64
DIM_HEAD = 16
NUM_HEADS = 4
N_TIME_BINS = 5
SESSION_INFO = {"s1": 10, "s2": 8}
BATCH = 2


class TinyBackbone(nn.Module):
    """Minimal backbone for testing: project flattened input to (B, 1, 1, dim)."""

    def __init__(self, dim=DIM):
        super().__init__()
        self.proj = nn.Linear(48, dim)

    def forward(self, x):
        b = x.shape[0]
        return self.proj(x.reshape(b, -1)[:, :48]).view(b, 1, 1, -1)


class MockTokenizerWrapper:
    """Minimal tokenizer wrapper that returns random latent targets."""

    def __init__(self, latent_dim=8):
        self.latent_dim = latent_dim
        self.latent_scale = 1.0
        self.latent_channel_means = None
        self.latent_channel_stds = None
        self.bsq_codebook_scale = 1.0
        self.is_compress_n = False
        self.reconstruction_rate_hz = 1000.0
        self.tokenizer = nn.Linear(1, 1)  # dummy

    def encode(self, responses, key, raw_discrete=False, **kwargs):
        B, T, N = responses.shape
        return {"target": torch.randn(B, N, 1, self.latent_dim), "aux_loss": None}

    def _tokens_to_target(self, t):
        return t

    def _target_to_tokens(self, t):
        return t

    def decode_differentiable(self, t, session_key=None):
        B = t.shape[0]
        return torch.randn(B, N_TIME_BINS, 10).abs()  # non-negative rates


@pytest.fixture
def backbone():
    return TinyBackbone()


@pytest.fixture
def images():
    return torch.randn(BATCH, 3, 4, 4)


@pytest.fixture
def responses_s1():
    """(B, T, N) for session s1 with 10 neurons."""
    return torch.randn(BATCH, N_TIME_BINS, 10).abs()


# Patch SessionReadout to default to CPU
@pytest.fixture(autouse=True)
def _patch_session_readout(monkeypatch):
    orig = SessionReadout.__init__

    def patched(self, n_neurons, dim_head=32, num_heads=4, bias=True, device="cpu", dtype=torch.float32):
        orig(self, n_neurons, dim_head, num_heads, bias, device, dtype)

    monkeypatch.setattr(SessionReadout, "__init__", patched)


# ---------------------------------------------------------------------------
# Readout layer unit tests
# ---------------------------------------------------------------------------


class TestReadoutLayers:
    def test_cross_attention_layer_shape(self):
        layer = ReadoutCrossAttentionLayer(DIM, DIM, NUM_HEADS, DIM_HEAD)
        q = torch.randn(BATCH, 10, DIM)
        kv = torch.randn(BATCH, 16, DIM)
        out = layer(q, kv)
        assert out.shape == (BATCH, 10, DIM)

    def test_self_attention_layer_shape(self):
        layer = ReadoutSelfAttentionLayer(DIM, NUM_HEADS, DIM_HEAD)
        q = torch.randn(BATCH, 10, DIM)
        out = layer(q)
        assert out.shape == (BATCH, 10, DIM)

    def test_cross_attention_different_kv_dim(self):
        layer = ReadoutCrossAttentionLayer(DIM, 128, NUM_HEADS, DIM_HEAD)
        q = torch.randn(BATCH, 10, DIM)
        kv = torch.randn(BATCH, 16, 128)
        out = layer(q, kv)
        assert out.shape == (BATCH, 10, DIM)

    def test_layer_init_weights(self):
        cross = ReadoutCrossAttentionLayer(DIM, DIM, NUM_HEADS, DIM_HEAD)
        self_attn = ReadoutSelfAttentionLayer(DIM, NUM_HEADS, DIM_HEAD)
        # Should not raise
        cross.init_weights()
        self_attn.init_weights()


# ---------------------------------------------------------------------------
# Model integration tests
# ---------------------------------------------------------------------------


class TestSingleLayerReadout:
    """Default config (single cross-attention layer) works with a single cross-attention layer."""

    def test_poisson_forward(self, backbone, images):
        model = ImageEncodingHiera(
            backbone=backbone,
            session_info=SESSION_INFO,
            n_time_bins=N_TIME_BINS,
            dim=DIM,
            backbone_dim=DIM,
            dim_head=DIM_HEAD,
            num_heads=NUM_HEADS,
            loss_mode="poisson",
        )
        out = model(images, "s1")
        assert out.shape == (BATCH, 10, N_TIME_BINS)

    def test_poisson_training_loss(self, backbone, images, responses_s1):
        model = ImageEncodingHiera(
            backbone=backbone,
            session_info=SESSION_INFO,
            n_time_bins=N_TIME_BINS,
            dim=DIM,
            backbone_dim=DIM,
            dim_head=DIM_HEAD,
            num_heads=NUM_HEADS,
            loss_mode="poisson",
        )
        out = model(images, "s1", responses=responses_s1)
        assert "loss" in out
        assert out["loss"].ndim == 0
        assert out["loss"].requires_grad

    def test_default_is_single_cross(self, backbone):
        model = ImageEncodingHiera(
            backbone=backbone,
            session_info=SESSION_INFO,
            n_time_bins=N_TIME_BINS,
            dim=DIM,
            backbone_dim=DIM,
            dim_head=DIM_HEAD,
            num_heads=NUM_HEADS,
            loss_mode="poisson",
        )
        assert model.readout_layer_types == ["cross"]
        assert len(model.readout_layers) == 1
        assert isinstance(model.readout_layers[0], ReadoutCrossAttentionLayer)


class TestDeepReadout:
    """Multi-layer readout with cross and self attention."""

    DEEP_LAYERS = ["cross", "cross", "self", "cross"]

    def test_poisson_deep_readout(self, backbone, images):
        model = ImageEncodingHiera(
            backbone=backbone,
            session_info=SESSION_INFO,
            n_time_bins=N_TIME_BINS,
            dim=DIM,
            backbone_dim=DIM,
            dim_head=DIM_HEAD,
            num_heads=NUM_HEADS,
            loss_mode="poisson",
            readout_layer_types=self.DEEP_LAYERS,
        )
        assert len(model.readout_layers) == 4
        out = model(images, "s1")
        assert out.shape == (BATCH, 10, N_TIME_BINS)

    def test_mse_deep_readout_train(self, backbone, images, responses_s1):
        model = ImageEncodingHiera(
            backbone=backbone,
            session_info=SESSION_INFO,
            n_time_bins=N_TIME_BINS,
            dim=DIM,
            backbone_dim=DIM,
            dim_head=DIM_HEAD,
            num_heads=NUM_HEADS,
            loss_mode="mse",
            tokenizer_wrapper=MockTokenizerWrapper(),
            token_prediction_dim=8,
            readout_layer_types=self.DEEP_LAYERS,
        )
        out = model(images, "s1", responses=responses_s1)
        assert "loss" in out
        assert "latent_cosine_sim" in out
        assert "latent_r2" in out
        assert out["loss"].requires_grad

    def test_mse_deep_readout_eval(self, backbone, images):
        model = ImageEncodingHiera(
            backbone=backbone,
            session_info=SESSION_INFO,
            n_time_bins=N_TIME_BINS,
            dim=DIM,
            backbone_dim=DIM,
            dim_head=DIM_HEAD,
            num_heads=NUM_HEADS,
            loss_mode="mse",
            tokenizer_wrapper=MockTokenizerWrapper(),
            token_prediction_dim=8,
            readout_layer_types=self.DEEP_LAYERS,
        )
        out = model(images, "s1", return_token_prediction=True)
        assert out["token_prediction"].shape == (BATCH, 10, 1, 8)

    def test_multiple_sessions(self, backbone, images):
        model = ImageEncodingHiera(
            backbone=backbone,
            session_info=SESSION_INFO,
            n_time_bins=N_TIME_BINS,
            dim=DIM,
            backbone_dim=DIM,
            dim_head=DIM_HEAD,
            num_heads=NUM_HEADS,
            loss_mode="poisson",
            readout_layer_types=self.DEEP_LAYERS,
        )
        out_s1 = model(images, "s1")
        out_s2 = model(images, "s2")
        assert out_s1.shape == (BATCH, 10, N_TIME_BINS)
        assert out_s2.shape == (BATCH, 8, N_TIME_BINS)

    def test_cross_only_deep(self, backbone, images):
        model = ImageEncodingHiera(
            backbone=backbone,
            session_info=SESSION_INFO,
            n_time_bins=N_TIME_BINS,
            dim=DIM,
            backbone_dim=DIM,
            dim_head=DIM_HEAD,
            num_heads=NUM_HEADS,
            loss_mode="poisson",
            readout_layer_types=["cross", "cross"],
        )
        assert len(model.readout_layers) == 2
        out = model(images, "s1")
        assert out.shape == (BATCH, 10, N_TIME_BINS)

    def test_more_params_with_more_layers(self, backbone):
        model1 = ImageEncodingHiera(
            backbone=backbone,
            session_info=SESSION_INFO,
            n_time_bins=N_TIME_BINS,
            dim=DIM,
            backbone_dim=DIM,
            dim_head=DIM_HEAD,
            num_heads=NUM_HEADS,
            loss_mode="poisson",
        )
        model4 = ImageEncodingHiera(
            backbone=backbone,
            session_info=SESSION_INFO,
            n_time_bins=N_TIME_BINS,
            dim=DIM,
            backbone_dim=DIM,
            dim_head=DIM_HEAD,
            num_heads=NUM_HEADS,
            loss_mode="poisson",
            readout_layer_types=self.DEEP_LAYERS,
        )
        p1 = sum(p.numel() for p in model1.readout_layers.parameters())
        p4 = sum(p.numel() for p in model4.readout_layers.parameters())
        assert p4 > p1


class TestInvalidConfig:
    def test_invalid_layer_type(self, backbone):
        with pytest.raises(ValueError, match="Unknown readout layer type"):
            ImageEncodingHiera(
                backbone=backbone,
                session_info=SESSION_INFO,
                n_time_bins=N_TIME_BINS,
                dim=DIM,
                backbone_dim=DIM,
                dim_head=DIM_HEAD,
                num_heads=NUM_HEADS,
                loss_mode="poisson",
                readout_layer_types=["cross", "invalid"],
            )


class TestGradientFlow:
    """Verify gradients flow through all readout layers."""

    def test_gradients_reach_all_layers(self, backbone, images, responses_s1):
        model = ImageEncodingHiera(
            backbone=backbone,
            session_info=SESSION_INFO,
            n_time_bins=N_TIME_BINS,
            dim=DIM,
            backbone_dim=DIM,
            dim_head=DIM_HEAD,
            num_heads=NUM_HEADS,
            loss_mode="poisson",
            readout_layer_types=["cross", "cross", "self", "cross"],
        )
        out = model(images, "s1", responses=responses_s1)
        out["loss"].backward()

        for i, layer in enumerate(model.readout_layers):
            for name, param in layer.named_parameters():
                assert param.grad is not None, f"No gradient for readout_layers[{i}].{name}"
                assert param.grad.abs().sum() > 0, f"Zero gradient for readout_layers[{i}].{name}"
