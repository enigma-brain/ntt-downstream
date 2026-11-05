"""Tests for Wingert auditory encoding model components."""

from __future__ import annotations

import pytest
import torch


class TestCausalDWSepBlock:
    def test_output_shape(self):
        from ntt_downstream.encoding.models.enc.audio.base_model import CausalDWSepBlock

        block = CausalDWSepBlock(in_channels=32, rank=48, kernel_size=7, dilation=1, dropout=0.0)
        x = torch.randn(2, 100, 32)
        out = block(x)
        assert out.shape == (2, 100, 48)

    def test_subsequent_block_skips_spectral_proj(self):
        from ntt_downstream.encoding.models.enc.audio.base_model import CausalDWSepBlock

        block = CausalDWSepBlock(in_channels=48, rank=48, kernel_size=7, dilation=2, dropout=0.0)
        assert block.spectral is None
        x = torch.randn(2, 100, 48)
        out = block(x)
        assert out.shape == (2, 100, 48)

    def test_causality(self):
        from ntt_downstream.encoding.models.enc.audio.base_model import CausalDWSepBlock

        block = CausalDWSepBlock(in_channels=32, rank=48, kernel_size=7, dilation=4, dropout=0.0)
        block.eval()
        x = torch.randn(1, 100, 32)
        out1 = block(x)
        x2 = x.clone()
        x2[:, 80:, :] = torch.randn(1, 20, 32)
        out2 = block(x2)
        assert torch.allclose(out1[:, :80, :], out2[:, :80, :], atol=1e-6)

    def test_preserves_time_dim(self):
        from ntt_downstream.encoding.models.enc.audio.base_model import CausalDWSepBlock

        block = CausalDWSepBlock(in_channels=32, rank=48, kernel_size=7, dilation=1, dropout=0.0)
        for T in [50, 100, 200]:
            x = torch.randn(1, T, 32)
            assert block(x).shape[1] == T

    def test_dilation_increases_receptive_field(self):
        from ntt_downstream.encoding.models.enc.audio.base_model import CausalDWSepBlock

        block_d1 = CausalDWSepBlock(in_channels=32, rank=48, kernel_size=7, dilation=1, dropout=0.0)
        block_d8 = CausalDWSepBlock(in_channels=32, rank=48, kernel_size=7, dilation=8, dropout=0.0)
        block_d1.eval()
        block_d8.eval()
        x = torch.randn(1, 100, 32)
        x2 = x.clone()
        x2[:, 0, :] = torch.randn(1, 32)
        out1_d1 = block_d1(x)
        out2_d1 = block_d1(x2)
        assert torch.allclose(out1_d1[:, 7:, :], out2_d1[:, 7:, :], atol=1e-5)
        out1_d8 = block_d8(x)
        out2_d8 = block_d8(x2)
        assert torch.allclose(out1_d8[:, 49:, :], out2_d8[:, 49:, :], atol=1e-5)


class TestCausalTemporalSelfAttention:
    def test_output_shape(self):
        from ntt_downstream.encoding.models.enc.audio.base_model import CausalTemporalSelfAttention

        attn = CausalTemporalSelfAttention(dim=64, num_heads=4, n_layers=2, rope_max_seq_len=512)
        x = torch.randn(2, 20, 64)
        out = attn(x)
        assert out.shape == (2, 20, 64)

    def test_causality(self):
        from ntt_downstream.encoding.models.enc.audio.base_model import CausalTemporalSelfAttention

        attn = CausalTemporalSelfAttention(dim=64, num_heads=4, n_layers=2, rope_max_seq_len=512)
        attn.eval()
        x = torch.randn(1, 20, 64)
        out1 = attn(x)
        x2 = x.clone()
        x2[:, 15:, :] = torch.randn(1, 5, 64)
        out2 = attn(x2)
        assert torch.allclose(out1[:, :15, :], out2[:, :15, :], atol=1e-5)

    def test_different_seq_lengths(self):
        from ntt_downstream.encoding.models.enc.audio.base_model import CausalTemporalSelfAttention

        attn = CausalTemporalSelfAttention(dim=64, num_heads=4, n_layers=1, rope_max_seq_len=512)
        attn.eval()
        for W in [10, 50, 100, 200]:
            x = torch.randn(1, W, 64)
            out = attn(x)
            assert out.shape == (1, W, 64)

    def test_seq_length_beyond_cache_works(self):
        from ntt_downstream.encoding.models.enc.audio.base_model import CausalTemporalSelfAttention

        attn = CausalTemporalSelfAttention(dim=64, num_heads=4, n_layers=1, rope_max_seq_len=16)
        attn.eval()
        x = torch.randn(1, 32, 64)
        out = attn(x)
        assert out.shape == (1, 32, 64)

    def test_rope_makes_position_matter(self):
        from ntt_downstream.encoding.models.enc.audio.base_model import CausalTemporalSelfAttention

        attn = CausalTemporalSelfAttention(dim=64, num_heads=4, n_layers=1, rope_max_seq_len=512)
        attn.eval()
        token = torch.randn(1, 1, 64)
        x1 = token.expand(1, 10, 64).clone()
        x2 = token.expand(1, 10, 64).clone()
        x2[:, :5, :] = torch.randn(1, 5, 64)
        out1 = attn(x1)
        out2 = attn(x2)
        assert not torch.allclose(out1[:, 5, :], out2[:, 5, :], atol=1e-3)


class TestCrossAttentionReadout:
    def test_rate_mode_output_shape(self):
        from ntt_downstream.encoding.models.enc.audio.base_model import CrossAttentionReadout

        readout = CrossAttentionReadout(n_units=40, dim=64, dim_q=48, num_heads=4, d_out=10)
        window_features = torch.randn(2, 20, 10, 64)
        out = readout(window_features)
        assert out.shape == (2, 20, 40, 10)

    def test_tokenizer_mode_output_shape(self):
        from ntt_downstream.encoding.models.enc.audio.base_model import CrossAttentionReadout

        readout = CrossAttentionReadout(n_units=40, dim=64, dim_q=48, num_heads=4, d_out=16, n_tokens_per_neuron=3)
        window_features = torch.randn(2, 20, 10, 64)
        out = readout(window_features)
        assert out.shape == (2, 20, 40, 3, 16)

    def test_tokenizer_mode_t_tok_1_shape(self):
        from ntt_downstream.encoding.models.enc.audio.base_model import CrossAttentionReadout

        readout = CrossAttentionReadout(n_units=40, dim=64, dim_q=48, num_heads=4, d_out=36, n_tokens_per_neuron=1)
        window_features = torch.randn(2, 20, 10, 64)
        out = readout(window_features)
        assert out.shape == (2, 20, 40, 1, 36)

    def test_different_n_units(self):
        from ntt_downstream.encoding.models.enc.audio.base_model import CrossAttentionReadout

        for n_units in [5, 50, 200]:
            readout = CrossAttentionReadout(n_units=n_units, dim=64, dim_q=48, num_heads=4, d_out=10)
            out = readout(torch.randn(1, 10, 10, 64))
            assert out.shape == (1, 10, n_units, 10)

    def test_different_M_values(self):
        from ntt_downstream.encoding.models.enc.audio.base_model import CrossAttentionReadout

        readout = CrossAttentionReadout(n_units=20, dim=64, dim_q=48, num_heads=4, d_out=10)
        for M in [5, 10, 20]:
            out = readout(torch.randn(2, 15, M, 64))
            assert out.shape == (2, 15, 20, 10)

    def test_no_session_embedding_parameter(self):
        from ntt_downstream.encoding.models.enc.audio.base_model import CrossAttentionReadout

        readout = CrossAttentionReadout(n_units=20, dim=64, dim_q=48, num_heads=4, d_out=10)
        param_names = [name for name, _ in readout.named_parameters()]
        assert "session_embedding" not in param_names

    def test_query_bias_changes_output(self):
        from ntt_downstream.encoding.models.enc.audio.base_model import CrossAttentionReadout

        readout = CrossAttentionReadout(n_units=20, dim=64, dim_q=48, num_heads=4, d_out=10)
        readout.eval()
        wf = torch.randn(1, 5, 10, 64)
        out1 = readout(wf)
        bias = torch.randn(1, 1, 48)
        out2 = readout(wf, query_bias=bias)
        assert not torch.allclose(out1, out2, atol=1e-3)


class TestAudioEncodingModel:
    def _make_model(self, session_info=None, loss_mode="poisson", token_prediction_dim=None, n_tokens_per_neuron=None):
        from ntt_downstream.encoding.models.enc.audio.base_model import AudioEncodingModel

        if session_info is None:
            session_info = {"sess_A": 40, "sess_B": 20}
        return AudioEncodingModel(
            session_info=session_info,
            n_freq_channels=32,
            conv_rank=32,
            conv_kernel_size=7,
            conv_dilations=[1, 2, 4],
            window_size=10,
            dim_q=32,
            num_heads=4,
            n_self_attn_layers=1,
            loss_mode=loss_mode,
            dropout=0.0,
            rope_max_seq_len=128,
            token_prediction_dim=token_prediction_dim,
            n_tokens_per_neuron=n_tokens_per_neuron,
        )

    def test_training_forward_poisson(self):
        model = self._make_model(loss_mode="poisson")
        spec = torch.randn(2, 100, 32)
        resp = torch.rand(2, 100, 40).abs()
        out = model(spec, "sess_A", responses=resp)
        assert "loss" in out
        assert out["loss"].ndim == 0
        assert out["loss"].requires_grad

    def test_training_forward_mse(self):
        model = self._make_model(loss_mode="mse")
        spec = torch.randn(2, 100, 32)
        resp = torch.randn(2, 100, 40)
        out = model(spec, "sess_A", responses=resp)
        assert "loss" in out
        assert out["loss"].ndim == 0

    def test_inference_returns_rates(self):
        model = self._make_model(loss_mode="poisson")
        model.eval()
        spec = torch.randn(2, 100, 32)
        with torch.no_grad():
            pred = model(spec, "sess_A")
        assert pred.shape == (2, 100, 40)
        assert (pred >= 0).all()

    def test_inference_tokenizer_mode(self):
        model = self._make_model(loss_mode="mse_latent", token_prediction_dim=16, n_tokens_per_neuron=3)
        model.eval()
        spec = torch.randn(2, 100, 32)
        with torch.no_grad():
            pred = model(spec, "sess_A")
        assert pred.shape == (2, 10, 40, 3, 16)  # W=10, N=40, T_tok=3, D=16

    def test_different_sessions(self):
        model = self._make_model()
        model.eval()
        spec = torch.randn(1, 50, 32)
        with torch.no_grad():
            pred_a = model(spec, "sess_A")
            pred_b = model(spec, "sess_B")
        assert pred_a.shape == (1, 50, 40)
        assert pred_b.shape == (1, 50, 20)

    def test_variable_length_input(self):
        model = self._make_model()
        model.eval()
        for T in [50, 100, 200]:
            spec = torch.randn(1, T, 32)
            with torch.no_grad():
                pred = model(spec, "sess_A")
            expected_T = (T // 10) * 10
            assert pred.shape == (1, expected_T, 40)

    def test_loss_masking_with_lengths(self):
        model = self._make_model(loss_mode="mse")
        spec = torch.randn(2, 100, 32)
        resp = torch.randn(2, 100, 40)
        lengths = torch.tensor([80, 100])
        out = model(spec, "sess_A", responses=resp, lengths=lengths)
        assert "loss" in out

    def test_conv_dilations_determine_layer_count(self):
        from ntt_downstream.encoding.models.enc.audio.base_model import AudioEncodingModel

        model = AudioEncodingModel(
            session_info={"s": 10},
            conv_rank=32,
            conv_kernel_size=7,
            conv_dilations=[1, 2],
            window_size=5,
            dim_q=32,
            num_heads=4,
            n_self_attn_layers=1,
            loss_mode="poisson",
        )
        assert len(model.conv_blocks) == 2


from omegaconf import OmegaConf


class TestAudioEncodingIntegration:
    """End-to-end: DataModule -> model -> train step -> val step."""

    def _make_cfg(self, data_dir: str, loss_mode: str = "poisson") -> OmegaConf:
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
                    "response_normalization": "none",
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
                    "conv_dilations": [1, 2],
                    "window_size": 5,
                    "n_self_attn_layers": 1,
                    "n_self_attn_heads": 2,
                    "dim_head": 16,
                    "num_heads": 2,
                    "readout_mlp_ratio": 2,
                    "readout_use_subject_embedding": False,
                    "learnable_softplus": False,
                    "softplus_beta": 0.1,
                    "loss_mode": loss_mode,
                    "dropout": 0.0,
                    "rope_max_seq_len": 64,
                },
                "trainer": {
                    "lr": 1e-3,
                    "backbone_lr_multiplier": 1.0,
                    "weight_decay": 0.01,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "max_steps": 10,
                    "scheduler": {
                        "warmup_steps": 2,
                        "eta_min": 1e-5,
                    },
                },
            }
        )

    def test_full_pipeline_poisson(self, wingert_data_dir):
        from ntt_downstream.encoding.training.audio_encoding_module import AudioEncodingLitModule
        from ntt_downstream.encoding.training.wingert_data_module import WingertAuditoryDataModule

        cfg = self._make_cfg(str(wingert_data_dir), loss_mode="poisson")
        dm = WingertAuditoryDataModule(cfg)
        dm.setup("fit")

        session_info = dm.get_session_info("train")
        model = AudioEncodingLitModule(cfg, session_info=session_info)
        model.configure_model()

        train_dl = dm.train_dataloader()
        batch = next(iter(train_dl))
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0
        assert loss.requires_grad

        val_dl = dm.val_dataloader()
        val_batch = next(iter(val_dl))
        model.validation_step(val_batch, 0)

    def test_full_pipeline_mse(self, wingert_data_dir):
        from ntt_downstream.encoding.training.audio_encoding_module import AudioEncodingLitModule
        from ntt_downstream.encoding.training.wingert_data_module import WingertAuditoryDataModule

        cfg = self._make_cfg(str(wingert_data_dir), loss_mode="mse")
        dm = WingertAuditoryDataModule(cfg)
        dm.setup("fit")

        session_info = dm.get_session_info("train")
        model = AudioEncodingLitModule(cfg, session_info=session_info)
        model.configure_model()

        train_dl = dm.train_dataloader()
        batch = next(iter(train_dl))
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0
