"""Tests for EAR energy-score loss module.

Reference: Shao et al., "Continuous Visual Autoregressive Generation via Score
Maximization" (ICML 2025, arXiv:2505.07812). We port the non-MAR path only.
"""

from __future__ import annotations

import math

import pytest
import torch

from ntt_downstream.encoding.models.enc.losses import EnergyLoss
from ntt_downstream.encoding.models.enc.losses.energy_loss import EnergyGenerator

TARGET_D = 16
Z_D = 16
WIDTH = 32
DEPTH = 2
NOISE_D = 8
BATCH = 4


class TestEnergyLossImport:
    def test_module_importable(self):
        loss = EnergyLoss(
            target_channels=TARGET_D,
            z_channels=Z_D,
            width=WIDTH,
            depth=DEPTH,
            noise_channels=NOISE_D,
            alpha=1.5,
        )
        assert isinstance(loss, torch.nn.Module)
        assert hasattr(loss, "net")
        assert isinstance(loss.net, EnergyGenerator)

    def test_generator_importable(self):
        gen = EnergyGenerator(
            noise_channels=NOISE_D,
            target_channels=TARGET_D,
            z_channels=Z_D,
            width=WIDTH,
            depth=DEPTH,
        )
        assert isinstance(gen, torch.nn.Module)


class TestEnergyGenerator:
    def test_forward_shape(self):
        gen = EnergyGenerator(
            noise_channels=NOISE_D,
            target_channels=TARGET_D,
            z_channels=Z_D,
            width=WIDTH,
            depth=DEPTH,
        )
        noise = torch.randn(BATCH, NOISE_D)
        z = torch.randn(BATCH, Z_D)
        out = gen(noise, z)
        assert out.shape == (BATCH, TARGET_D)

    def test_output_near_zero_at_init(self):
        """Zero-init of final_layer.linear + AdaLN modulations ensures the
        generator starts near the origin — important for training stability."""
        gen = EnergyGenerator(
            noise_channels=NOISE_D,
            target_channels=TARGET_D,
            z_channels=Z_D,
            width=WIDTH,
            depth=DEPTH,
        )
        noise = torch.randn(BATCH, NOISE_D)
        z = torch.randn(BATCH, Z_D)
        with torch.no_grad():
            out = gen(noise, z)
        assert out.abs().max().item() == pytest.approx(0.0, abs=1e-6)

    def test_forward_gradients_flow(self):
        gen = EnergyGenerator(
            noise_channels=NOISE_D,
            target_channels=TARGET_D,
            z_channels=Z_D,
            width=WIDTH,
            depth=DEPTH,
        )
        noise = torch.randn(BATCH, NOISE_D, requires_grad=False)
        z = torch.randn(BATCH, Z_D, requires_grad=True)
        out = gen(noise, z)
        out.sum().backward()
        # With zero-init final layer, grads on input/res_blocks may be zero at step 0;
        # but grads on cond_embed / noise_embed should flow via AdaLN.
        assert gen.cond_embed.weight.grad is not None
        assert gen.noise_embed.weight.grad is not None


class TestEnergyLossMath:
    """Verify the exact energy-score loss formula by patching the generator
    to return deterministic samples."""

    def test_loss_matches_hand_computed_value(self):
        loss_mod = EnergyLoss(
            target_channels=2,
            z_channels=2,
            width=4,
            depth=1,
            noise_channels=2,
            alpha=1.0,
            train_temperature=1.0,
        )

        # Patch the generator to produce deterministic outputs:
        # For the 2B-batched noise input, return fixed samples so we can
        # compute the expected loss by hand.
        # Input ordering from EnergyLoss.forward: noise is (2B, noise_d),
        # with x1 = samples[:B], x2 = samples[B:].
        class _FixedNet:
            def __init__(self):
                pass

            def __call__(self, noise, c):
                B2 = noise.shape[0]
                B = B2 // 2
                D = 2
                x1 = torch.tensor([[1.0, 0.0]]).expand(B, D)
                x2 = torch.tensor([[0.0, 1.0]]).expand(B, D)
                return torch.cat([x1, x2], dim=0)

            @property
            def input_proj(self):
                # mimic attribute lookup for weight_dtype
                return type("P", (), {"weight": torch.zeros(1, 1)})()

        object.__setattr__(loss_mod, "net", _FixedNet())  # bypass nn.Module setattr guard

        target = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        z = torch.zeros(2, 2)
        loss = loss_mod(target, z)
        # x1=[1,0], x2=[0,1], y=[1,1]
        # ||x1-y||  = ||[0,-1]|| = 1
        # ||x2-y||  = ||[-1,0]|| = 1
        # ||x1-x2|| = ||[1,-1]|| = sqrt(2)
        # L = 1 + 1 - sqrt(2) ≈ 0.5858
        expected = 1.0 + 1.0 - math.sqrt(2.0)
        assert loss.item() == pytest.approx(expected, abs=1e-5)

    def test_loss_with_alpha_1_5(self):
        loss_mod = EnergyLoss(
            target_channels=2,
            z_channels=2,
            width=4,
            depth=1,
            noise_channels=2,
            alpha=1.5,
            train_temperature=1.0,
        )

        class _FixedNet:
            def __call__(self, noise, c):
                B = noise.shape[0] // 2
                x1 = torch.tensor([[2.0, 0.0]]).expand(B, 2)
                x2 = torch.tensor([[0.0, 2.0]]).expand(B, 2)
                return torch.cat([x1, x2], dim=0)

            @property
            def input_proj(self):
                return type("P", (), {"weight": torch.zeros(1, 1)})()

        object.__setattr__(loss_mod, "net", _FixedNet())  # bypass nn.Module setattr guard

        target = torch.zeros(2, 2)
        z = torch.zeros(2, 2)
        loss = loss_mod(target, z)
        # ||x1|| = 2, ||x2|| = 2, ||x1-x2|| = 2*sqrt(2)
        # L = 2^1.5 + 2^1.5 - (2*sqrt(2))^1.5
        expected = 2 * (2**1.5) - (2 * math.sqrt(2)) ** 1.5
        assert loss.item() == pytest.approx(expected, abs=1e-4)

    def test_mask_averages_only_valid_positions(self):
        loss_mod = EnergyLoss(
            target_channels=2,
            z_channels=2,
            width=4,
            depth=1,
            noise_channels=2,
            alpha=1.0,
            train_temperature=1.0,
        )

        class _FixedNet:
            def __call__(self, noise, c):
                B = noise.shape[0] // 2
                x1 = torch.tensor([[1.0, 0.0]]).expand(B, 2)
                x2 = torch.tensor([[0.0, 1.0]]).expand(B, 2)
                return torch.cat([x1, x2], dim=0)

            @property
            def input_proj(self):
                return type("P", (), {"weight": torch.zeros(1, 1)})()

        object.__setattr__(loss_mod, "net", _FixedNet())  # bypass nn.Module setattr guard

        target = torch.tensor([[1.0, 1.0], [5.0, 5.0]])  # 2nd row is far off
        z = torch.zeros(2, 2)
        mask = torch.tensor([1.0, 0.0])
        loss = loss_mod(target, z, mask=mask)
        # Only row 0 is counted
        expected = 1.0 + 1.0 - math.sqrt(2.0)
        assert loss.item() == pytest.approx(expected, abs=1e-5)


class TestEnergyLossValidation:
    def test_alpha_below_1_raises(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            EnergyLoss(target_channels=8, z_channels=8, alpha=0.5)

    def test_alpha_equal_2_raises(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            EnergyLoss(target_channels=8, z_channels=8, alpha=2.0)

    def test_alpha_above_2_raises(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            EnergyLoss(target_channels=8, z_channels=8, alpha=3.0)

    def test_alpha_1_emits_warning(self):
        with pytest.warns(UserWarning, match="boundary of strict propriety"):
            EnergyLoss(target_channels=8, z_channels=8, alpha=1.0)

    def test_train_temperature_zero_raises(self):
        with pytest.raises(ValueError, match="train_temperature must be"):
            EnergyLoss(target_channels=8, z_channels=8, alpha=1.5, train_temperature=0.0)


class TestEnergyLossGradientFlow:
    def test_gradients_reach_generator(self):
        loss_mod = EnergyLoss(
            target_channels=TARGET_D,
            z_channels=Z_D,
            width=WIDTH,
            depth=DEPTH,
            noise_channels=NOISE_D,
            alpha=1.5,
        )
        target = torch.randn(BATCH, TARGET_D)
        z = torch.randn(BATCH, Z_D, requires_grad=True)
        loss = loss_mod(target, z)
        loss.backward()

        assert loss.requires_grad
        assert z.grad is not None
        assert loss_mod.net.cond_embed.weight.grad is not None
        assert loss_mod.net.noise_embed.weight.grad is not None
        # At step 0, final_layer.linear is zero-init but adaLN_modulation inside
        # each ResBlock still receives gradient via the final-layer AdaLN path.
        assert loss_mod.net.final_layer.adaLN_modulation[-1].weight.grad is not None

    def test_loss_is_scalar(self):
        loss_mod = EnergyLoss(target_channels=TARGET_D, z_channels=Z_D, alpha=1.5)
        target = torch.randn(BATCH, TARGET_D)
        z = torch.randn(BATCH, Z_D)
        loss = loss_mod(target, z)
        assert loss.ndim == 0


class TestEnergyLossSample:
    def test_sample_shape_single(self):
        loss_mod = EnergyLoss(
            target_channels=TARGET_D,
            z_channels=Z_D,
            width=WIDTH,
            depth=DEPTH,
            noise_channels=NOISE_D,
            alpha=1.5,
        )
        z = torch.randn(BATCH, Z_D)
        out = loss_mod.sample(z)
        assert out.shape == (BATCH, TARGET_D)

    def test_sample_shape_multi(self):
        loss_mod = EnergyLoss(
            target_channels=TARGET_D,
            z_channels=Z_D,
            width=WIDTH,
            depth=DEPTH,
            noise_channels=NOISE_D,
            alpha=1.5,
        )
        z = torch.randn(BATCH, Z_D)
        out = loss_mod.sample(z, n_samples=3)
        assert out.shape == (BATCH * 3, TARGET_D)

    def test_sample_temperature_scales_variance(self):
        torch.manual_seed(0)
        loss_mod = EnergyLoss(
            target_channels=TARGET_D,
            z_channels=Z_D,
            width=WIDTH,
            depth=DEPTH,
            noise_channels=NOISE_D,
            alpha=1.5,
        )
        # After some training, sample variance at temperature 2.0 should exceed
        # variance at 0.5. We approximate this post-init by training briefly.
        z = torch.randn(16, Z_D)
        target = torch.randn(16, TARGET_D)
        opt = torch.optim.AdamW(loss_mod.parameters(), lr=1e-2)
        for _ in range(30):
            opt.zero_grad()
            loss_mod(target, z).backward()
            opt.step()
        torch.manual_seed(123)
        low = loss_mod.sample(z, n_samples=8, temperature=0.5)
        torch.manual_seed(123)
        high = loss_mod.sample(z, n_samples=8, temperature=2.0)
        assert high.var().item() > low.var().item()

    def test_sample_cfg_ne_1_emits_warning(self):
        loss_mod = EnergyLoss(target_channels=TARGET_D, z_channels=Z_D, alpha=1.5)
        z = torch.randn(BATCH, Z_D)
        with pytest.warns(UserWarning, match="cfg .* not implemented"):
            _ = loss_mod.sample(z, temperature=1.0, cfg=1.5)


class TestEnergyLossDiagnostics:
    def test_return_diagnostics_structure(self):
        loss_mod = EnergyLoss(
            target_channels=TARGET_D,
            z_channels=Z_D,
            width=WIDTH,
            depth=DEPTH,
            noise_channels=NOISE_D,
            alpha=1.5,
        )
        target = torch.randn(BATCH, TARGET_D)
        z = torch.randn(BATCH, Z_D)
        loss, diag = loss_mod(target, z, return_diagnostics=True)
        assert loss.ndim == 0
        assert "diversity_abs" in diag
        assert "diversity_ratio" in diag
        assert diag["diversity_abs"].ndim == 0
        assert diag["diversity_ratio"].ndim == 0
        # Diagnostics must be detached — the loss graph should not depend on them.
        assert not diag["diversity_abs"].requires_grad
        assert not diag["diversity_ratio"].requires_grad

    def test_diversity_ratio_nonzero_post_init(self):
        """After a few optimizer steps, a healthy generator should have
        diversity_ratio well above zero (>~0.1). Pure-collapse would give ~0."""
        torch.manual_seed(0)
        loss_mod = EnergyLoss(
            target_channels=TARGET_D,
            z_channels=Z_D,
            width=WIDTH,
            depth=DEPTH,
            noise_channels=NOISE_D,
            alpha=1.5,
        )
        z = torch.randn(16, Z_D)
        target = torch.randn(16, TARGET_D)
        opt = torch.optim.AdamW(loss_mod.parameters(), lr=1e-2)
        for _ in range(20):
            opt.zero_grad()
            loss_mod(target, z).backward()
            opt.step()
        loss_mod.eval()
        _, diag = loss_mod(target, z, return_diagnostics=True)
        assert diag["diversity_ratio"].item() > 0.1


# ---------------------------------------------------------------------------
# Integration test helpers (module-level, not inside any class)
# ---------------------------------------------------------------------------

from torch import nn as _nn


class _TinyHieraBackbone(_nn.Module):
    """Minimal stand-in backbone for integration tests — returns (B, 1, 1, DIM)."""

    def __init__(self, dim: int = 32):
        super().__init__()
        self.proj = _nn.Linear(3 * 4 * 4, dim)
        self._dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return self.proj(x.reshape(B, -1)).view(B, 1, 1, self._dim)


class _MockTokenizerWrapper:
    """Minimal tokenizer wrapper for integration tests."""

    def __init__(self, token_dim: int = 16):
        self.token_dim = token_dim
        self.latent_scale = 1.0
        self.latent_channel_means = None
        self.latent_channel_stds = None
        self.is_compress_n = False
        self.bsq_codebook_scale = None
        self.reconstruction_rate_hz = 1000.0

    def encode(self, responses, key, raw_discrete=False, skip_target_scaling=False):
        B, T, N = responses.shape
        # Unified (B, S=N, T=1, D) format
        return {"target": torch.randn(B, N, 1, self.token_dim), "aux_loss": None}


class TestEnergyLossIntegration:
    @pytest.fixture(autouse=True)
    def _patch_session_readout_cpu(self, monkeypatch):
        """Force SessionReadout to default to CPU so tests run without a GPU."""
        from ntt_downstream.encoding.models.enc.hiera.base_model import SessionReadout

        orig = SessionReadout.__init__

        def patched(self, n_neurons, dim_head=32, num_heads=4, bias=True, device="cpu", dtype=torch.float32):
            orig(self, n_neurons, dim_head, num_heads, bias, device, dtype)

        monkeypatch.setattr(SessionReadout, "__init__", patched)

    def test_image_encoding_hiera_forward_backward_energy(self):
        from ntt_downstream.encoding.models.enc.hiera.base_model import ImageEncodingHiera
        from ntt_downstream.encoding.models.enc.losses import EnergyLoss

        DIM = 32
        N_NEURONS = 5
        TOKEN_DIM = 16
        BATCH = 2

        backbone = _TinyHieraBackbone(dim=DIM)
        tokenizer = _MockTokenizerWrapper(token_dim=TOKEN_DIM)
        energy = EnergyLoss(
            target_channels=TOKEN_DIM,
            z_channels=TOKEN_DIM,
            width=32,
            depth=2,
            noise_channels=16,
            alpha=1.5,
        )

        model = ImageEncodingHiera(
            backbone=backbone,
            session_info={"s1": N_NEURONS},
            n_time_bins=10,
            dim=DIM,
            backbone_dim=DIM,
            dim_head=8,
            num_heads=4,
            mlp_ratio=2,
            token_prediction_dim=TOKEN_DIM,
            energy_loss=energy,
            loss_mode="energy",
            tokenizer_wrapper=tokenizer,
            diffusion_batch_mul=1,
        )

        images = torch.randn(BATCH, 3, 4, 4)
        # (B, T, N) format — responses are dummy for mock tokenizer
        responses = torch.randn(BATCH, 10, N_NEURONS).abs()

        out = model(images, "s1", responses=responses)
        assert "loss" in out
        assert out["loss"].ndim == 0
        assert out["loss"].requires_grad
        assert "energy_diversity_abs" in out
        assert "energy_diversity_ratio" in out

        out["loss"].backward()
        assert model.energy_loss.net.cond_embed.weight.grad is not None
        assert model.energy_conditioning_norm.weight.grad is not None

    def test_image_encoding_hiera_sample_tokens_energy(self):
        from ntt_downstream.encoding.models.enc.hiera.base_model import ImageEncodingHiera
        from ntt_downstream.encoding.models.enc.losses import EnergyLoss

        DIM = 32
        N_NEURONS = 5
        TOKEN_DIM = 16
        BATCH = 2

        backbone = _TinyHieraBackbone(dim=DIM)
        tokenizer = _MockTokenizerWrapper(token_dim=TOKEN_DIM)
        energy = EnergyLoss(
            target_channels=TOKEN_DIM,
            z_channels=TOKEN_DIM,
            width=32,
            depth=2,
            noise_channels=16,
            alpha=1.5,
        )

        model = ImageEncodingHiera(
            backbone=backbone,
            session_info={"s1": N_NEURONS},
            n_time_bins=10,
            dim=DIM,
            backbone_dim=DIM,
            dim_head=8,
            num_heads=4,
            mlp_ratio=2,
            token_prediction_dim=TOKEN_DIM,
            energy_loss=energy,
            loss_mode="energy",
            tokenizer_wrapper=tokenizer,
            diffusion_batch_mul=1,
        )

        images = torch.randn(BATCH, 3, 4, 4)
        with torch.no_grad():
            token_pred = model(images, "s1", return_token_prediction=True)
            sampled = model.sample_tokens(token_pred["token_prediction"])
        assert sampled.shape == (BATCH, N_NEURONS, 1, TOKEN_DIM)
