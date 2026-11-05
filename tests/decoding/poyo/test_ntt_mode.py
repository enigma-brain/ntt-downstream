"""Tests for NTT tokenizer mode in the pipeline.

Core tests:
  1. NTTSpikeTimesTokenizer produces correct TokenizerOutput from synthetic SpikeTimesInput.
  2. The ntt mode path in V2TrainWrapper._ntt_tokenize works end-to-end with a mock batch.
"""

from __future__ import annotations

import pytest
import torch
from ntt.core.components.types import SpikeTimesInput

from ntt_downstream.decoding.backends.poyo.ntt_spike_times_tokenizer import NTTSpikeTimesTokenizer
from ntt_downstream.decoding.backends.poyo.tokenizer_output import TokenizerOutput


class TestNTTSpikeTimesTokenizer:
    """Core tests for NTTSpikeTimesTokenizer."""

    def test_from_padded_spike_times(self):
        """Tokenizer produces correct shapes and values from SpikeTimesInput."""
        B, N, D = 2, 5, 64
        tokenizer = NTTSpikeTimesTokenizer(token_dim=D, token_value=0.0)

        spike_timestamps = torch.rand(B, N)
        spike_unit_index = torch.randint(0, 10, (B, N))
        spike_pad_mask = torch.ones(B, N, dtype=torch.long)
        # Make last 2 positions padding for sample 1
        spike_pad_mask[1, 3:] = 0

        inp = SpikeTimesInput(
            spike_unit_index=spike_unit_index,
            spike_timestamps=spike_timestamps,
            spike_pad_mask=spike_pad_mask,
            unit_ids=torch.arange(10).unsqueeze(0).expand(B, -1),
            unit_ids_pad_mask=torch.ones(B, 10, dtype=torch.long),
            duration_sec=torch.ones(B),
            session_id=("s0", "s1"),
            subject_id=("subj", "subj"),
            species=("monkey", "monkey"),
        )

        out = tokenizer(inp)
        assert isinstance(out, TokenizerOutput)
        assert out.tokens.shape == (B, N, D)
        assert out.timestamps.shape == (B, N)
        assert out.unit_ids.shape == (B, N)
        assert out.mask.shape == (B, N)
        assert out.combines_units is False
        assert out.token_bin_size_s is None

        # Check mask matches pad_mask
        assert out.mask.dtype == torch.bool
        assert out.mask[0].all()  # sample 0: all valid
        assert out.mask[1, :3].all()  # sample 1: first 3 valid
        assert not out.mask[1, 3:].any()  # sample 1: last 2 padding

        # Token content should be constant 0.0
        assert (out.tokens == 0.0).all()

    def test_rejects_wrong_input_type(self):
        """Tokenizer raises TypeError for non-spike inputs."""
        tokenizer = NTTSpikeTimesTokenizer(token_dim=64)
        with pytest.raises(TypeError, match="Expected SpikeTimesInput"):
            tokenizer({"not": "a spike input"})

    def test_rejects_invalid_token_dim(self):
        """Constructor raises on non-positive token_dim."""
        with pytest.raises(AssertionError):
            NTTSpikeTimesTokenizer(token_dim=0)
        with pytest.raises(AssertionError):
            NTTSpikeTimesTokenizer(token_dim=-1)


class TestNTTModeInWrapper:
    """Core test: ntt mode path in V2TrainWrapper._ntt_tokenize."""

    def test_ntt_tokenize_extracts_spikes(self):
        """_ntt_tokenize filters sentinels and produces correct TokenizerOutput."""
        from ntt_downstream.decoding.backends.poyo.train_wrapper import V2TrainWrapper

        D = 64
        tokenizer = NTTSpikeTimesTokenizer(token_dim=D, token_value=0.0)

        # Build a minimal mock batch as POYO's tokenize() would produce.
        # token_type: 0=DEFAULT (spike), 1=START sentinel, 2=END sentinel
        B, N = 2, 8
        # Sample 0: 5 spikes + 1 START + 1 END + 1 pad
        # Sample 1: 3 spikes + 1 START + 1 END + 3 pad
        token_type = torch.zeros(B, N, dtype=torch.long)
        unit_index = torch.zeros(B, N, dtype=torch.long)
        timestamps = torch.zeros(B, N)
        mask = torch.zeros(B, N, dtype=torch.bool)

        # Sample 0: START, 5 spikes, END, pad
        token_type[0] = torch.tensor([1, 0, 0, 0, 0, 0, 2, 0])
        unit_index[0] = torch.tensor([0, 1, 2, 3, 1, 2, 0, 0])
        timestamps[0] = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 0.0])
        mask[0, :7] = True

        # Sample 1: START, 3 spikes, END, pad, pad, pad
        token_type[1] = torch.tensor([1, 0, 0, 0, 2, 0, 0, 0])
        unit_index[1] = torch.tensor([0, 5, 6, 7, 0, 0, 0, 0])
        timestamps[1] = torch.tensor([0.0, 0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])
        mask[1, :5] = True

        batch = {
            "model_inputs": {
                "input_token_type": token_type,
                "input_unit_index": unit_index,
                "input_timestamps": timestamps,
                "input_mask": mask,
            }
        }

        # Create wrapper with ntt mode — we only test _ntt_tokenize, not full
        # training_step, so we use a minimal cfg mock.
        class FakeCfg:
            batch_size = 2

            class optim:
                base_lr = 1e-4
                weight_decay = 1e-4
                warmup_pct = 0.05
                div_factor = 25
                final_div_factor = 1.0e4

        # We can't easily construct a full V2TrainWrapper without a real model,
        # so we build a minimal object that has _ntt_tokenize bound to it.
        # We use a plain nn.Module to get proper __setattr__ handling.
        wrapper = torch.nn.Module()
        wrapper.ntt_tokenizer = tokenizer
        wrapper.tokenizer_mode = "ntt"
        # Bind the method from V2TrainWrapper to our wrapper object.
        import types

        wrapper._ntt_tokenize = types.MethodType(V2TrainWrapper._ntt_tokenize, wrapper)

        tok_out = wrapper._ntt_tokenize(batch)

        assert isinstance(tok_out, TokenizerOutput)
        assert tok_out.tokens.ndim == 3
        assert tok_out.tokens.shape[0] == B
        assert tok_out.tokens.shape[2] == D
        assert tok_out.combines_units is False

        # Sample 0 has 5 spikes, sample 1 has 3 spikes → padded to 5
        assert tok_out.tokens.shape[1] == 5
        assert tok_out.mask[0].sum() == 5
        assert tok_out.mask[1].sum() == 3

        # Check that sentinel tokens were excluded
        # Unit indices in spike tokens: sample 0 = [1,2,3,1,2], sample 1 = [5,6,7]
        assert tok_out.unit_ids[0, :5].tolist() == [1, 2, 3, 1, 2]
        assert tok_out.unit_ids[1, :3].tolist() == [5, 6, 7]

    def test_ntt_mode_requires_tokenizer(self):
        """V2TrainWrapper raises when tokenizer_mode='ntt' without tokenizer."""
        from omegaconf import OmegaConf

        from ntt_downstream.decoding.backends.poyo.train_wrapper import V2TrainWrapper

        cfg = OmegaConf.create(
            {
                "batch_size": 2,
                "optim": {
                    "base_lr": 1e-4,
                    "weight_decay": 1e-4,
                    "warmup_pct": 0.05,
                    "div_factor": 25,
                    "final_div_factor": 1.0e4,
                },
            }
        )

        with pytest.raises(AssertionError, match="ntt_tokenizer must be provided"):
            V2TrainWrapper(
                cfg=cfg,
                model=torch.nn.Linear(1, 1),
                modality_spec=None,
                tokenizer_mode="ntt",
                ntt_tokenizer=None,
            )
