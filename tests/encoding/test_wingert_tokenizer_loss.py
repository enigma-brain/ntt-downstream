"""Tests for AudioEncodingModel tokenizer loss integration."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf


def _make_mock_tokenizer_wrapper(latent_dim=16, tokens_per_window=1):
    """Mock that returns deterministic encode_windowed() results."""
    wrapper = MagicMock()
    wrapper.is_compress_n = False
    wrapper.latent_dim = latent_dim

    def mock_encode_windowed(responses, session_key, raw_discrete=False, **kwargs):
        B, T, N = responses.shape
        W = T // 10  # assume M=10
        S = N
        T_tok = tokens_per_window
        target = torch.randn(B, W * T_tok, S, latent_dim)
        return {"target": target, "n_windows": W, "tokens_per_window": T_tok, "n_spatial": S, "n_temporal": W * T_tok}

    wrapper.encode_windowed = mock_encode_windowed
    return wrapper


class TestAudioEncodingModelTokenizerLoss:
    def _make_model(self, loss_mode="mse_latent", token_prediction_dim=16, session_info=None):
        from ntt_downstream.encoding.models.enc.audio.base_model import AudioEncodingModel

        if session_info is None:
            session_info = {"sess_A": 40}
        return AudioEncodingModel(
            session_info=session_info,
            n_freq_channels=32,
            conv_rank=32,
            conv_kernel_size=7,
            conv_dilations=[1],
            window_size=10,
            dim_q=32,
            num_heads=4,
            n_self_attn_layers=1,
            loss_mode=loss_mode,
            token_prediction_dim=token_prediction_dim,
        )

    def test_mse_latent_loss(self):
        model = self._make_model(loss_mode="mse_latent", token_prediction_dim=16)
        model.tokenizer_wrapper = _make_mock_tokenizer_wrapper(latent_dim=16)
        spec = torch.randn(2, 100, 32)
        resp = torch.randn(2, 100, 40)
        out = model(spec, "sess_A", responses=resp)
        assert "loss" in out
        assert out["loss"].ndim == 0
        assert out["loss"].requires_grad

    def test_smooth_l1_latent_loss(self):
        model = self._make_model(loss_mode="smooth_l1_latent", token_prediction_dim=16)
        model.tokenizer_wrapper = _make_mock_tokenizer_wrapper(latent_dim=16)
        out = model(torch.randn(2, 100, 32), "sess_A", responses=torch.randn(2, 100, 40))
        assert "loss" in out
        assert out["loss"].ndim == 0

    def test_readout_d_out_is_token_dim(self):
        model = self._make_model(token_prediction_dim=16)
        assert model.readout["sess_A"].d_out == 16

    def test_readout_d_out_is_window_size_in_rate_mode(self):
        from ntt_downstream.encoding.models.enc.audio.base_model import AudioEncodingModel

        model = AudioEncodingModel(
            session_info={"sess_A": 40},
            n_freq_channels=32,
            conv_rank=32,
            conv_kernel_size=7,
            conv_dilations=[1],
            window_size=10,
            dim_q=32,
            num_heads=4,
            n_self_attn_layers=1,
            loss_mode="poisson",
        )
        assert model.readout["sess_A"].d_out == 10  # window_size

    def test_rate_mode_still_works(self):
        from ntt_downstream.encoding.models.enc.audio.base_model import AudioEncodingModel

        model = AudioEncodingModel(
            session_info={"sess_A": 40},
            n_freq_channels=32,
            conv_rank=32,
            conv_kernel_size=7,
            conv_dilations=[1],
            window_size=10,
            dim_q=32,
            num_heads=4,
            n_self_attn_layers=1,
            loss_mode="poisson",
        )
        out = model(torch.randn(2, 100, 32), "sess_A", responses=torch.rand(2, 100, 40).abs())
        assert "loss" in out

    def test_inference_tokenizer_mode_returns_tensor(self):
        model = self._make_model(loss_mode="mse_latent", token_prediction_dim=16)
        model.eval()
        with torch.no_grad():
            pred = model(torch.randn(1, 100, 32), "sess_A")
        assert isinstance(pred, torch.Tensor)

    def test_raises_without_tokenizer_wrapper(self):
        model = self._make_model(loss_mode="mse_latent", token_prediction_dim=16)
        # tokenizer_wrapper is None by default
        with pytest.raises(RuntimeError, match="tokenizer_wrapper"):
            model(torch.randn(1, 100, 32), "sess_A", responses=torch.randn(1, 100, 40))

    def test_mse_loss_is_fp32_under_autocast(self):
        """MSE loss must be computed in fp32 even when readout outputs bf16.

        Without the ``.float()`` cast in ``_tokenizer_loss``, ``F.mse_loss``
        on bf16 tensors loses precision / overflows on large latents — the
        same NaN-on-large-activations issue the visual path guards against.
        """
        model = self._make_model(loss_mode="mse_latent", token_prediction_dim=16)
        model.tokenizer_wrapper = _make_mock_tokenizer_wrapper(latent_dim=16)
        spec = torch.randn(2, 100, 32)
        resp = torch.randn(2, 100, 40)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
            out = model(spec, "sess_A", responses=resp)
        assert out["loss"].dtype == torch.float32, f"MSE loss must be fp32 under autocast, got {out['loss'].dtype}"

    def test_smooth_l1_loss_is_fp32_under_autocast(self):
        model = self._make_model(loss_mode="smooth_l1_latent", token_prediction_dim=16)
        model.tokenizer_wrapper = _make_mock_tokenizer_wrapper(latent_dim=16)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
            out = model(torch.randn(2, 100, 32), "sess_A", responses=torch.randn(2, 100, 40))
        assert out["loss"].dtype == torch.float32

    def test_precomputed_token_targets_skip_online_encoding(self):
        from ntt_downstream.encoding.models.enc.audio.base_model import AudioEncodingModel

        model = AudioEncodingModel(
            session_info={"sess_A": 40},
            n_freq_channels=32,
            conv_rank=32,
            conv_kernel_size=7,
            conv_dilations=[1],
            window_size=10,
            dim_q=32,
            num_heads=4,
            n_self_attn_layers=1,
            loss_mode="cross_entropy",
            token_prediction_dim=16,
            ce_vocab_size=32,
        )
        model.tokenizer_wrapper = MagicMock()
        spec = torch.randn(2, 100, 32)
        resp = torch.randn(2, 100, 40)
        precomputed = {
            "target": torch.randint(0, 32, (2, 20, 40, 1), dtype=torch.long),
            "tokens_per_window": 2,
            "n_windows": 10,
        }
        out = model(spec, "sess_A", responses=resp, tokenizer_targets=precomputed)
        model.tokenizer_wrapper.encode_windowed.assert_not_called()
        assert out["loss"].ndim == 0


class TestAudioTokenizerIntegration:
    """End-to-end: DataModule -> model with mock tokenizer -> train step."""

    def _make_cfg(self, data_dir: str) -> OmegaConf:
        return OmegaConf.create(
            {
                "data": {
                    "dataset_path": data_dir,
                    "bin_size": 0.01,
                    "spectrogram_fs": 100.0,
                    "n_freq_channels": 32,
                    "crop_length": 50,
                    "session_ids": None,
                    "subjects": None,
                },
                "dataloader": {
                    "batch_size": 4,
                    "num_workers": 0,
                    "prefetch_factor": None,
                    "persistent_workers": False,
                },
                "model": {
                    "model_type": "audio_encoding",
                    "conv_rank": 16,
                    "conv_kernel_size": 5,
                    "conv_dilations": [1],
                    "window_size": 5,
                    "n_self_attn_layers": 1,
                    "n_self_attn_heads": 2,
                    "dim_head": 16,
                    "num_heads": 2,
                    "readout_mlp_ratio": 2,
                    "readout_use_subject_embedding": False,
                    "learnable_softplus": False,
                    "softplus_beta": 0.1,
                    "loss_mode": "mse_latent",
                    "token_prediction_dim": 16,
                },
                "tokenizer": {
                    "enabled": False,
                },
                "trainer": {
                    "lr": 1e-3,
                    "backbone_lr_multiplier": 1.0,
                    "weight_decay": 0.01,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "max_steps": 10,
                    "warmup_loss_mode": None,
                    "warmup_steps": 0,
                    "tokenizer_mode": "frozen",
                    "scheduler": {
                        "warmup_steps": 2,
                        "eta_min": 1e-5,
                    },
                },
            }
        )

    def test_tokenizer_training_step(self, wingert_data_dir):
        """Full pipeline with mock tokenizer wrapper injected after model creation."""
        from ntt_downstream.encoding.training.audio_encoding_module import AudioEncodingLitModule
        from ntt_downstream.encoding.training.wingert_data_module import WingertAuditoryDataModule

        cfg = self._make_cfg(str(wingert_data_dir))
        dm = WingertAuditoryDataModule(cfg)
        dm.setup("fit")

        session_info = dm.get_session_info("train")
        model = AudioEncodingLitModule(cfg, session_info=session_info)
        model.configure_model()

        # Inject mock tokenizer wrapper with matching window_size=5
        wrapper = _make_mock_tokenizer_wrapper(latent_dim=16)

        # Override the mock to use window_size=5 (matching model config)
        def mock_encode_windowed_5(responses, session_key, raw_discrete=False, **kwargs):
            B, T, N = responses.shape
            W = T // 5  # window_size=5
            S = N
            target = torch.randn(B, W, S, 16)
            return {"target": target, "n_windows": W, "tokens_per_window": 1, "n_spatial": S, "n_temporal": W}

        wrapper.encode_windowed = mock_encode_windowed_5
        model.model.tokenizer_wrapper = wrapper

        train_dl = dm.train_dataloader()
        batch = next(iter(train_dl))
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0
        assert loss.requires_grad


class TestRepeatAverageEval:
    """Exercises the _compute_loader_correlation helper and repeat-avg wiring."""

    def _make_cfg(self, data_dir: str) -> OmegaConf:
        return OmegaConf.create(
            {
                "data": {
                    "dataset_path": data_dir,
                    "bin_size": 0.01,
                    "spectrogram_fs": 100.0,
                    "n_freq_channels": 32,
                    "crop_length": 50,
                    "session_ids": None,
                    "subjects": None,
                },
                "dataloader": {
                    "batch_size": 4,
                    "num_workers": 0,
                    "prefetch_factor": None,
                    "persistent_workers": False,
                },
                "model": {
                    "model_type": "audio_encoding",
                    "conv_rank": 16,
                    "conv_kernel_size": 5,
                    "conv_dilations": [1],
                    "window_size": 5,
                    "n_self_attn_layers": 1,
                    "n_self_attn_heads": 2,
                    "dim_head": 16,
                    "num_heads": 2,
                    "readout_mlp_ratio": 2,
                    "readout_use_subject_embedding": False,
                    "learnable_softplus": False,
                    "softplus_beta": 0.1,
                    "loss_mode": "poisson",
                },
                "tokenizer": {"enabled": False},
                "trainer": {
                    "lr": 1e-3,
                    "backbone_lr_multiplier": 1.0,
                    "weight_decay": 0.01,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "max_steps": 10,
                    "warmup_loss_mode": None,
                    "warmup_steps": 0,
                    "tokenizer_mode": "frozen",
                    "scheduler": {"warmup_steps": 2, "eta_min": 1e-5},
                },
            }
        )

    def test_compute_loader_correlation_returns_rate_corr(self, wingert_data_dir):
        """_compute_loader_correlation runs the model over val_average and returns rate_corr."""
        from ntt_downstream.encoding.training.audio_encoding_module import AudioEncodingLitModule
        from ntt_downstream.encoding.training.wingert_data_module import WingertAuditoryDataModule

        cfg = self._make_cfg(str(wingert_data_dir))
        dm = WingertAuditoryDataModule(cfg)
        dm.setup("fit")

        module = AudioEncodingLitModule(cfg, session_info=dm.get_session_info("train"))
        module.configure_model()

        result = module._compute_loader_correlation(dm.val_average_dataloader)
        assert result is not None, "repeat-avg eval returned None"
        assert "rate_corr" in result
        assert isinstance(result["rate_corr"], float)
        # temporal_corr may be None (single timestep) or a float
        assert "temporal_corr" in result

    def test_compute_loader_correlation_respects_max_batches(self, wingert_data_dir):
        """max_batches truncates the eval pass."""
        from ntt_downstream.encoding.training.audio_encoding_module import AudioEncodingLitModule
        from ntt_downstream.encoding.training.wingert_data_module import WingertAuditoryDataModule

        cfg = self._make_cfg(str(wingert_data_dir))
        dm = WingertAuditoryDataModule(cfg)
        dm.setup("fit")

        module = AudioEncodingLitModule(cfg, session_info=dm.get_session_info("train"))
        module.configure_model()

        result = module._compute_loader_correlation(dm.val_average_dataloader, max_batches=1)
        assert result is not None
        assert isinstance(result["rate_corr"], float)

    def test_compute_loader_correlation_handles_empty_loader(self):
        """Gracefully returns None when dataloader_fn raises."""
        from omegaconf import OmegaConf

        from ntt_downstream.encoding.training.audio_encoding_module import AudioEncodingLitModule

        cfg = OmegaConf.create(
            {
                "data": {"bin_size": 0.01, "spectrogram_fs": 100.0, "n_freq_channels": 32},
                "dataloader": {"batch_size": 1, "num_workers": 0},
                "model": {
                    "model_type": "audio_encoding",
                    "conv_rank": 16,
                    "conv_kernel_size": 5,
                    "conv_dilations": [1],
                    "window_size": 5,
                    "n_self_attn_layers": 1,
                    "n_self_attn_heads": 2,
                    "dim_head": 16,
                    "num_heads": 2,
                    "readout_mlp_ratio": 2,
                    "readout_use_subject_embedding": False,
                    "learnable_softplus": False,
                    "softplus_beta": 0.1,
                    "loss_mode": "poisson",
                },
                "tokenizer": {"enabled": False},
                "trainer": {
                    "lr": 1e-3,
                    "backbone_lr_multiplier": 1.0,
                    "weight_decay": 0.01,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "max_steps": 10,
                    "warmup_loss_mode": None,
                    "warmup_steps": 0,
                    "tokenizer_mode": "frozen",
                    "scheduler": {"warmup_steps": 2, "eta_min": 1e-5},
                },
            }
        )
        module = AudioEncodingLitModule(cfg, session_info={"sess_A": 5})
        module.configure_model()

        def broken_loader():
            raise RuntimeError("no loader")

        result = module._compute_loader_correlation(broken_loader)
        assert result is None


def test_flow_matching_head_importable_from_losses_module():
    """FlowMatchingHead must live next to its siblings DiffLoss and EnergyLoss."""
    from ntt_downstream.encoding.models.enc.losses.flow_matching_head import FlowMatchingHead  # noqa: F401


def test_audio_model_accepts_new_tokenizer_loss_modes():
    """All four new modes must pass constructor-time validation."""
    from ntt_downstream.encoding.models.enc.audio.base_model import AudioEncodingModel

    for mode in ("flow_matching", "diffusion", "energy", "cross_entropy"):
        assert mode in AudioEncodingModel.TOKENIZER_LOSS_MODES, f"Mode {mode!r} missing from TOKENIZER_LOSS_MODES"


@pytest.mark.parametrize(
    "mode,extra_kwargs",
    [
        ("flow_matching", {"flow_cfg": {"depth": 2, "width": 64}}),
        ("diffusion", {"diffusion_cfg": {"depth": 2, "width": 64, "num_sampling_steps": "20"}}),
        (
            "energy",
            {"energy_cfg": {"depth": 2, "width": 64, "alpha": 1.5, "train_temperature": 1.0, "noise_channels": 32}},
        ),
        ("cross_entropy", {"ce_vocab_size": 128}),
    ],
)
def test_audio_model_instantiates_tokenizer_head(mode, extra_kwargs):
    """Each new loss mode should build its head at __init__ time."""
    from ntt_downstream.encoding.models.enc.audio.base_model import AudioEncodingModel

    model = AudioEncodingModel(
        session_info={"s0": 8},
        n_freq_channels=16,
        conv_rank=16,
        conv_kernel_size=3,
        conv_dilations=[1, 2],
        window_size=4,
        dim_q=16,
        num_heads=2,
        n_self_attn_layers=1,
        mlp_ratio=2,
        loss_mode=mode,
        token_prediction_dim=16,
        n_tokens_per_neuron=2,
        **extra_kwargs,
    )

    if mode == "flow_matching":
        assert hasattr(model, "flow_head")
    elif mode == "diffusion":
        assert hasattr(model, "diffloss")
    elif mode == "energy":
        assert hasattr(model, "energy_loss")
    elif mode == "cross_entropy":
        assert hasattr(model, "ce_head")
        assert model.ce_head.out_features == 128


def test_audio_model_cross_entropy_requires_ce_vocab_size():
    from ntt_downstream.encoding.models.enc.audio.base_model import AudioEncodingModel

    with pytest.raises(ValueError, match="ce_vocab_size"):
        AudioEncodingModel(
            session_info={"s0": 8},
            n_freq_channels=16,
            conv_rank=16,
            conv_kernel_size=3,
            conv_dilations=[1, 2],
            window_size=4,
            dim_q=16,
            num_heads=2,
            n_self_attn_layers=1,
            mlp_ratio=2,
            loss_mode="cross_entropy",
            token_prediction_dim=16,
            n_tokens_per_neuron=1,
            # ce_vocab_size intentionally omitted
        )


def test_audio_model_generative_modes_require_token_prediction_dim():
    from ntt_downstream.encoding.models.enc.audio.base_model import AudioEncodingModel

    with pytest.raises(ValueError, match="token_prediction_dim"):
        AudioEncodingModel(
            session_info={"s0": 8},
            n_freq_channels=16,
            conv_rank=16,
            conv_kernel_size=3,
            conv_dilations=[1, 2],
            window_size=4,
            dim_q=16,
            num_heads=2,
            n_self_attn_layers=1,
            mlp_ratio=2,
            loss_mode="diffusion",
            token_prediction_dim=None,
            diffusion_cfg={"depth": 2, "width": 64, "num_sampling_steps": "20"},
        )


# ---------------------------------------------------------------------------
# _tokenizer_loss dispatch tests
# ---------------------------------------------------------------------------


class _MockTokenizerWrapper:
    """Minimal mock exposing encode_windowed → {"target", "tokens_per_window", "n_windows", ...}."""

    def __init__(self, latent_dim=16, t_tok=2, raw_discrete_vocab=None, aux_loss=None):
        self.latent_dim = latent_dim
        self._t_tok = t_tok
        self._vocab = raw_discrete_vocab
        self._aux = aux_loss

    def encode_windowed(self, responses, session_key, raw_discrete=False, skip_target_scaling=False):
        B, T, N = responses.shape
        W = max(1, T // 4)  # window_size used in tests is 4
        if raw_discrete:
            assert self._vocab is not None, "raw_discrete requires a vocab"
            target = torch.randint(0, self._vocab, (B, W * self._t_tok, N, 1), dtype=torch.long)
        else:
            target = torch.randn(B, W * self._t_tok, N, self.latent_dim)
        out = {"target": target, "tokens_per_window": self._t_tok, "n_windows": W}
        if self._aux is not None:
            out["aux_loss"] = self._aux
        return out


def _build_audio_model(loss_mode, **extra):
    from ntt_downstream.encoding.models.enc.audio.base_model import AudioEncodingModel

    base_kwargs = {
        "session_info": {"s0": 3},
        "n_freq_channels": 8,
        "conv_rank": 8,
        "conv_kernel_size": 3,
        "conv_dilations": [1, 2],
        "window_size": 4,
        "dim_q": 8,
        "num_heads": 2,
        "n_self_attn_layers": 1,
        "mlp_ratio": 2,
        "loss_mode": loss_mode,
        "token_prediction_dim": 16,
        "n_tokens_per_neuron": 2,
    }
    base_kwargs.update(extra)
    return AudioEncodingModel(**base_kwargs)


@pytest.mark.parametrize(
    "mode,extra",
    [
        ("flow_matching", {"flow_cfg": {"depth": 2, "width": 32}}),
        ("diffusion", {"diffusion_cfg": {"depth": 2, "width": 32, "num_sampling_steps": "10"}}),
        ("energy", {"energy_cfg": {"depth": 2, "width": 32, "alpha": 1.5, "noise_channels": 16}}),
    ],
)
def test_audio_generative_loss_finite_and_backprops(mode, extra):
    model = _build_audio_model(mode, **extra)
    model.tokenizer_wrapper = _MockTokenizerWrapper(latent_dim=16, t_tok=2)

    B, T_in, N, F_mel = 2, 16, 3, 8
    spec = torch.randn(B, T_in, F_mel)
    resp = torch.randn(B, T_in, N)

    out = model(spec, "s0", responses=resp)
    loss = out["loss"]

    assert loss.ndim == 0, f"loss must be scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), f"loss must be finite, got {loss.item()}"
    assert loss.requires_grad, "loss must require grad"

    loss.backward()
    # The generative heads (flow/diffusion/energy) use zero-init AdaLN gating,
    # which structurally blocks gradient flow through the conditioning path at
    # initialization (gate=0 means ResBlock(x,y)=x). Instead, verify that the
    # head's own input_proj / cond_embed parameters received non-zero gradients,
    # confirming that backward() ran and the computation graph is connected.
    if mode == "flow_matching":
        head_params = model.flow_head.parameters()
    elif mode == "diffusion":
        head_params = model.diffloss.parameters()
    else:
        head_params = model.energy_loss.parameters()
    has_head_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in head_params)
    assert has_head_grad, f"gradient did not flow into the {mode} head parameters"


def test_audio_energy_diagnostics_present():
    model = _build_audio_model("energy", energy_cfg={"depth": 2, "width": 32, "alpha": 1.5, "noise_channels": 16})
    model.tokenizer_wrapper = _MockTokenizerWrapper(latent_dim=16, t_tok=2)

    spec = torch.randn(2, 16, 8)
    resp = torch.randn(2, 16, 3)
    out = model(spec, "s0", responses=resp)
    assert "energy_diversity_abs" in out
    assert "energy_diversity_ratio" in out


def test_audio_aux_loss_passthrough():
    aux = torch.tensor(0.5, requires_grad=True)
    model = _build_audio_model("flow_matching", flow_cfg={"depth": 2, "width": 32}, tokenizer_aux_loss_weight=2.0)
    model.tokenizer_wrapper = _MockTokenizerWrapper(latent_dim=16, t_tok=2, aux_loss=aux)

    spec = torch.randn(2, 16, 8)
    resp = torch.randn(2, 16, 3)
    out = model(spec, "s0", responses=resp)

    out["loss"].backward()
    assert aux.grad is not None and aux.grad.item() == pytest.approx(2.0)


def test_audio_cross_entropy_loss_finite_and_backprops():
    model = _build_audio_model("cross_entropy", ce_vocab_size=64)
    model.tokenizer_wrapper = _MockTokenizerWrapper(latent_dim=16, t_tok=2, raw_discrete_vocab=64)

    spec = torch.randn(2, 16, 8)
    resp = torch.randn(2, 16, 3)
    out = model(spec, "s0", responses=resp)

    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])

    out["loss"].backward()
    # CE has no zero-init gate in the head (ce_head is a plain Linear) — backbone gradients should flow.
    has_backbone_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.conv_blocks.parameters())
    assert has_backbone_grad, "gradient did not flow back to the backbone for cross_entropy mode"


# ---------------------------------------------------------------------------
# Latent diagnostics for non-generative modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["mse_latent", "smooth_l1_latent", "bce"])
def test_audio_non_generative_modes_emit_latent_diagnostics(mode):
    model = _build_audio_model(mode)
    model.tokenizer_wrapper = _MockTokenizerWrapper(latent_dim=16, t_tok=2)

    spec = torch.randn(2, 16, 8)
    resp = torch.randn(2, 16, 3)
    out = model(spec, "s0", responses=resp)
    assert "latent_cosine_sim" in out
    assert "latent_r2" in out


@pytest.mark.parametrize(
    "mode,extra",
    [
        ("flow_matching", {"flow_cfg": {"depth": 2, "width": 32}}),
        ("diffusion", {"diffusion_cfg": {"depth": 2, "width": 32, "num_sampling_steps": "10"}}),
        ("energy", {"energy_cfg": {"depth": 2, "width": 32, "alpha": 1.5, "noise_channels": 16}}),
    ],
)
def test_audio_generative_modes_skip_latent_diagnostics(mode, extra):
    model = _build_audio_model(mode, **extra)
    model.tokenizer_wrapper = _MockTokenizerWrapper(latent_dim=16, t_tok=2)

    spec = torch.randn(2, 16, 8)
    resp = torch.randn(2, 16, 3)
    out = model(spec, "s0", responses=resp)
    assert "latent_cosine_sim" not in out
    assert "latent_r2" not in out


# ---------------------------------------------------------------------------
# sample_tokens
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode,extra",
    [
        ("mse_latent", {}),
        ("smooth_l1_latent", {}),
        ("flow_matching", {"flow_cfg": {"depth": 2, "width": 32}}),
        ("diffusion", {"diffusion_cfg": {"depth": 2, "width": 32, "num_sampling_steps": "10"}}),
        ("energy", {"energy_cfg": {"depth": 2, "width": 32, "alpha": 1.5, "noise_channels": 16}}),
    ],
)
def test_audio_sample_tokens_shape_continuous(mode, extra):
    model = _build_audio_model(mode, **extra)
    B, W, N, T_tok, D = 2, 3, 5, 2, 16
    z = torch.randn(B, W, N, T_tok, D)
    sampled = model.sample_tokens(z)
    assert sampled.shape == (B, W, N, T_tok, D)


def test_audio_sample_tokens_cross_entropy_returns_indices():
    model = _build_audio_model("cross_entropy", ce_vocab_size=32)
    B, W, N, T_tok, D = 2, 3, 5, 2, 16
    z = torch.randn(B, W, N, T_tok, D)
    sampled = model.sample_tokens(z)
    assert sampled.shape == (B, W, N, T_tok)
    assert sampled.dtype in (torch.int64, torch.int32, torch.long)
    assert (sampled >= 0).all() and (sampled < 32).all()


def test_audio_ce_eval_decode_uses_unified_n_before_t_order():
    from ntt_downstream.encoding.training.audio_encoding_module import AudioEncodingLitModule

    cfg = OmegaConf.create({"trainer": {}, "data": {}})
    module = AudioEncodingLitModule(cfg, session_info={"s0": 3})

    sampled = torch.arange(2 * 4 * 3 * 2, dtype=torch.long).reshape(2, 4, 3, 2)
    module.model = MagicMock()
    module.model.loss_mode = "cross_entropy"
    module.model.sample_tokens.return_value = sampled

    captured: dict[str, torch.Tensor] = {}
    module.tokenizer_wrapper = MagicMock()
    module.tokenizer_wrapper.tokenizer = MagicMock()

    def _decode(tokens, session_key):
        captured["tokens"] = tokens
        captured["session_key"] = session_key
        B, N, T, _ = tokens.shape
        return torch.zeros(B, T, N)

    module.tokenizer_wrapper.decode_differentiable.side_effect = _decode

    out = module._decode_latent_to_rates(torch.randn(2, 4, 3, 2, 16), "s0")

    assert out.shape == (2, 8, 3)
    assert captured["session_key"] == "s0"
    assert captured["tokens"].shape == (2, 3, 8, 1)
    expected = sampled.permute(0, 2, 1, 3).reshape(2, 3, 8).unsqueeze(-1)
    assert torch.equal(captured["tokens"], expected)


# ---------------------------------------------------------------------------
# Factory wiring
# ---------------------------------------------------------------------------


def test_audio_factory_forwards_head_configs():
    from omegaconf import OmegaConf

    from ntt_downstream.encoding.models.enc.audio.model_factory import create_audio_model

    cfg = OmegaConf.create(
        {
            "data": {"n_freq_channels": 8},
            "model": {
                "conv_rank": 8,
                "conv_kernel_size": 3,
                "conv_dilations": [1, 2],
                "window_size": 4,
                "dim_head": 4,
                "num_heads": 2,
                "n_self_attn_layers": 1,
                "readout_mlp_ratio": 2,
                "loss_mode": "energy",
                "dropout": 0.0,
                "rope_max_seq_len": 64,
                "energy": {"depth": 2, "width": 32, "alpha": 1.5, "noise_channels": 16},
                "diffusion_batch_mul": 4,
                "tokenizer_aux_loss_weight": 0.25,
            },
        }
    )
    model = create_audio_model(
        cfg,
        session_info={"s0": 3},
        token_prediction_dim=16,
        n_tokens_per_neuron=2,
    )
    assert model.energy_loss is not None
    assert model.diffusion_batch_mul == 4
    assert model.tokenizer_aux_loss_weight == pytest.approx(0.25)
