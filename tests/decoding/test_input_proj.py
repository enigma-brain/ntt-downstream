"""Tests for the adapter's input_proj wiring in V2TrainWrapper.

The original spike_times path must NOT build an MLP between the tokenizer and the
model — D_tok already equals model.dim and the original POYO formula is
`unit_emb(uid) + token_type_emb(type)`. Inserting a Linear in that path would
break the equivalence (the projection would have to be exactly identity).
"""

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch_brain.registry import DataType, ModalitySpec

from ntt_downstream.decoding.backends.poyo import (
    NTTSpikeTimesTokenizer,
    POYOWithEmbeddingBypass,
    V2TrainWrapper,
)
from ntt_downstream.decoding.backends.poyo.adapter import adapt
from ntt_downstream.decoding.backends.poyo.tokenizer_output import TokenizerOutput

DIM = 64


def _readout_spec() -> ModalitySpec:
    return ModalitySpec(
        id=0,
        dim=2,
        type=DataType.CONTINUOUS,
        timestamp_key="cursor.timestamps",
        value_key="cursor.vel",
        context_key="cursor.subtask_index",
        loss_fn=lambda a, b, w: (a - b).pow(2).mean(),
    )


def _model() -> POYOWithEmbeddingBypass:
    return POYOWithEmbeddingBypass(
        sequence_length=1.0,
        latent_step=0.125,
        num_latents_per_step=16,
        dim=DIM,
        depth=2,
        dim_head=32,
        cross_heads=2,
        self_heads=2,
        readout_spec=_readout_spec(),
    )


def _cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "model": {"dim": DIM},
            "tokenizer": {
                "mode": "ntt",
                "unit_embedding_mode": "add",
                "sentinels": True,
                "ntt_tokenizer": "spike_times",
            },
            "seed": 0,
            "batch_size": 1,
            "num_workers": 0,
        }
    )


class _Spec:
    loss_fn = staticmethod(lambda a, b, w: (a - b).pow(2).mean())


@pytest.mark.parametrize("mode", ["none", "add"])
def test_spike_times_does_not_build_mlp(mode):
    """spike_times tokenizer with mode in {none, add} must not build a Linear."""
    ntt = NTTSpikeTimesTokenizer(token_dim=DIM)
    wrapper = V2TrainWrapper(
        cfg=_cfg(),
        model=_model(),
        modality_spec=_Spec(),
        tokenizer_mode="ntt",
        unit_embedding_mode=mode,
        sentinels=True,
        ntt_tokenizer=ntt,
    )
    assert wrapper.input_proj is None, (
        f"Expected input_proj=None for spike_times+mode={mode}, got {type(wrapper.input_proj).__name__}"
    )


@pytest.mark.parametrize("mode", ["none", "add"])
def test_fake_tokenizer_does_not_build_mlp(mode):
    """fake tokenizer (D_tok = model.dim) with mode in {none, add} must not build a Linear."""
    wrapper = V2TrainWrapper(
        cfg=_cfg(),
        model=_model(),
        modality_spec=_Spec(),
        tokenizer_mode="fake",
        unit_embedding_mode=mode,
        sentinels=True,
    )
    assert wrapper.input_proj is None


def test_spike_times_concat_builds_projection():
    """concat always needs a projection (no identity path); it's an MLP."""
    ntt = NTTSpikeTimesTokenizer(token_dim=DIM)
    wrapper = V2TrainWrapper(
        cfg=_cfg(),
        model=_model(),
        modality_spec=_Spec(),
        tokenizer_mode="ntt",
        unit_embedding_mode="concat",
        sentinels=True,
        ntt_tokenizer=ntt,
    )
    assert wrapper.input_proj is not None
    # Round-trip a tensor to verify the MLP shapes are correct.
    x = torch.zeros(1, 1, DIM + DIM)
    y = wrapper.input_proj(x)
    assert y.shape == (1, 1, DIM)
    # Confirm a non-linearity is present (1 hidden layer + activation).
    activations = [m for m in wrapper.input_proj.modules() if isinstance(m, (nn.ReLU, nn.GELU, nn.SiLU, nn.Mish))]
    assert len(activations) >= 1, "input_proj must contain a non-linear activation"
    linears = [m for m in wrapper.input_proj.modules() if isinstance(m, nn.Linear)]
    assert len(linears) == 2, "input_proj must have 2 Linear layers (1 hidden)"


def test_adapter_add_no_proj_preserves_original_poyo_formula():
    """mode='add' with input_proj=None must compute exactly x + unit_emb(uid)."""
    B, N = 2, 5
    n_units = 10
    unit_emb = nn.Embedding(n_units, DIM)
    token_type_emb = nn.Embedding(4, DIM)
    tokens = torch.randn(B, N, DIM)
    unit_ids = torch.randint(1, n_units, (B, N))
    tok = TokenizerOutput(
        tokens=tokens,
        timestamps=torch.linspace(0, 1, N).expand(B, N),
        unit_ids=unit_ids,
        mask=torch.ones(B, N, dtype=torch.bool),
        combines_units=False,
        token_bin_size_s=None,
    )

    pre = adapt(
        tok,
        unit_emb=unit_emb,
        token_type_emb=token_type_emb,
        mode="add",
        sentinels=False,
        target_dim=DIM,
        input_proj=None,
    )
    expected = tokens + unit_emb(unit_ids)
    assert torch.allclose(pre.embeddings, expected)


def test_adapter_concat_without_proj_raises():
    B, N = 2, 5
    n_units = 10
    tok = TokenizerOutput(
        tokens=torch.randn(B, N, DIM),
        timestamps=torch.linspace(0, 1, N).expand(B, N),
        unit_ids=torch.randint(1, n_units, (B, N)),
        mask=torch.ones(B, N, dtype=torch.bool),
        combines_units=False,
        token_bin_size_s=None,
    )
    with pytest.raises(AssertionError, match="concat"):
        adapt(
            tok,
            unit_emb=nn.Embedding(n_units, DIM),
            token_type_emb=nn.Embedding(4, DIM),
            mode="concat",
            sentinels=False,
            target_dim=DIM,
            input_proj=None,
        )


def test_adapter_dim_mismatch_without_proj_raises():
    """mode in {none, add}: D_tok != target_dim with no projection must assert."""
    B, N = 2, 5
    n_units = 10
    tok = TokenizerOutput(
        tokens=torch.randn(B, N, DIM - 16),
        timestamps=torch.linspace(0, 1, N).expand(B, N),
        unit_ids=torch.randint(1, n_units, (B, N)),
        mask=torch.ones(B, N, dtype=torch.bool),
        combines_units=False,
        token_bin_size_s=None,
    )
    for mode in ("none", "add"):
        with pytest.raises(AssertionError, match="D_tok"):
            adapt(
                tok,
                unit_emb=nn.Embedding(n_units, DIM),
                token_type_emb=nn.Embedding(4, DIM),
                mode=mode,
                sentinels=False,
                target_dim=DIM,
                input_proj=None,
            )
