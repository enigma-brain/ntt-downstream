"""Discrete-input plumbing on V2TrainWrapper."""

from __future__ import annotations

import os

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from ntt_downstream.decoding.backends.poyo.frozen_ntt_tokenizer import FrozenNTTTokenizer
from ntt_downstream.decoding.backends.poyo.train_wrapper import V2TrainWrapper

# Set RESIDUAL_LFQ_URI to point at a residual-LFQ checkpoint — either a
# local .ckpt path or an MLflow artifact URI (`mlflow-artifacts:` / `mlflow:`).
# Tests that need this URI skip when it's empty.
RESIDUAL_LFQ_URI = os.environ.get("RESIDUAL_LFQ_URI", "")


def _minimal_cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "log_every_n_steps": 1,
            "gpus": 1,
            "num_nodes": 1,
            "batch_size": 4,
            "optim": {
                "base_lr": 1e-6,
                "weight_decay": 0.0,
                "warmup_pct": 0.1,
                "div_factor": 25,
                "final_div_factor": 100,
            },
        }
    )


class _DummyPOYO(nn.Module):
    """Minimal POYO stand-in exposing only the attributes V2TrainWrapper reads."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.dim = dim
        self.unit_emb = nn.Embedding(64, dim)
        self.token_type_emb = nn.Embedding(8, dim)
        self.session_emb = nn.Embedding(8, dim)


class _DummyModalitySpec:
    def loss_fn(self, *args, **kwargs):  # noqa: D401
        return torch.tensor(0.0)


@pytest.mark.skipif(
    not RESIDUAL_LFQ_URI,
    reason="RESIDUAL_LFQ_URI env var required for residual-LFQ checkpoint tests",
)
def test_train_wrapper_builds_code_emb_in_embedding_mode():
    frozen = FrozenNTTTokenizer.from_checkpoint(RESIDUAL_LFQ_URI, device="cpu", discrete_input_mode="embedding")

    wrapper = V2TrainWrapper(
        cfg=_minimal_cfg(),
        model=_DummyPOYO(dim=64),
        modality_spec=_DummyModalitySpec(),
        tokenizer_mode="ntt_frozen",
        unit_embedding_mode="add",
        sentinels=False,
        ntt_tokenizer=frozen,
    )

    assert wrapper.code_emb is not None
    assert len(wrapper.code_emb) == 4
    for emb in wrapper.code_emb:
        assert isinstance(emb, nn.Embedding)
        assert emb.num_embeddings == 1024
        assert emb.embedding_dim == 64
    # input_proj should be bypassed.
    assert wrapper.input_proj is None


@pytest.mark.skipif(
    not RESIDUAL_LFQ_URI,
    reason="RESIDUAL_LFQ_URI env var required for residual-LFQ checkpoint tests",
)
def test_train_wrapper_zq_mode_no_code_emb():
    frozen = FrozenNTTTokenizer.from_checkpoint(RESIDUAL_LFQ_URI, device="cpu")

    wrapper = V2TrainWrapper(
        cfg=_minimal_cfg(),
        model=_DummyPOYO(dim=64),
        modality_spec=_DummyModalitySpec(),
        tokenizer_mode="ntt_frozen",
        unit_embedding_mode="add",
        sentinels=False,
        ntt_tokenizer=frozen,
    )

    assert wrapper.code_emb is None
    # input_proj must exist because z_dim=16 != target_dim=64.
    assert wrapper.input_proj is not None


@pytest.mark.skipif(
    not RESIDUAL_LFQ_URI,
    reason="RESIDUAL_LFQ_URI env var required for residual-LFQ checkpoint tests",
)
def test_end_to_end_gradient_flow_embedding_mode():
    """One forward+backward pass populates gradients on code_emb only."""
    from ntt.core.model.tokenizer.validation import make_dummy_input

    from ntt_downstream.decoding.backends.poyo.adapter import adapt

    frozen = FrozenNTTTokenizer.from_checkpoint(RESIDUAL_LFQ_URI, device="cpu", discrete_input_mode="embedding")

    # Run the tokenizer once on dummy input to derive shapes.
    dummy = make_dummy_input(frozen.tokenizer.spec)
    tok_out = frozen(dummy)
    assert tok_out.tokens.dtype == torch.int64

    # Build the embedding tables and adapter inputs directly — wiring a
    # full POYO + Lightning trainer is too heavy for a unit test.
    D = 64
    unit_emb = nn.Embedding(64, D)
    token_type_emb = nn.Embedding(8, D)
    code_emb = nn.ModuleList([nn.Embedding(frozen.vocab_size, D) for _ in range(frozen.num_codebooks)])
    for emb in code_emb:
        nn.init.normal_(emb.weight, std=0.02)

    pre = adapt(
        tok_out,
        unit_emb=unit_emb,
        token_type_emb=token_type_emb,
        mode="add",
        sentinels=False,
        target_dim=D,
        input_proj=None,
        code_emb=code_emb,
    )
    pre.embeddings.sum().backward()

    # Gradients must populate on the trainable embeddings.
    assert unit_emb.weight.grad is not None
    for q, emb in enumerate(code_emb):
        assert emb.weight.grad is not None, f"code_emb[{q}] received no gradient"

    # Gradients must NOT populate on the frozen tokenizer.
    for name, p in frozen.tokenizer.named_parameters():
        assert p.grad is None, f"frozen tokenizer param {name} unexpectedly received a gradient"


# -- init-time wrapper-construction tests --


def test_wrapper_accepts_combines_units_with_add_mode():
    """combines_units=True + unit_embedding_mode='add' constructs cleanly.

    A population / abstract-latent tokenizer has no per-token unit_ids, so the
    adapter emits per-slot proxy unit_ids (slot k → batch.unit_ids[k]). That
    gives each (session, latent_slot) its own learnable unit_emb entry and
    makes unit_identification fine-tuning possible for these tokenizers.
    """
    frozen = FrozenNTTTokenizer.from_poisson(
        rate_bin_size_ms=50.0,
        input_bin_size_ms=1.0,
        window_length_seconds=1.0,
    )
    # Force combines_units=True (Poisson is normally per-unit). Both the
    # property and the underlying flag are read by the wrapper.
    frozen._combines_units = True

    wrapper = V2TrainWrapper(
        cfg=_minimal_cfg(),
        model=_DummyPOYO(dim=64),
        modality_spec=_DummyModalitySpec(),
        tokenizer_mode="ntt_frozen",
        unit_embedding_mode="add",
        sentinels=False,
        ntt_tokenizer=frozen,
    )
    assert wrapper.unit_embedding_mode == "add"


def test_wrapper_rejects_negative_output_dim_in_zq_mode():
    """output_dim≤0 in z_q mode ⇒ explicit ValueError, not opaque nn.Linear crash."""
    frozen = FrozenNTTTokenizer.from_poisson(
        rate_bin_size_ms=50.0,
        input_bin_size_ms=1.0,
        window_length_seconds=1.0,
    )
    # Inject a -1 sentinel by short-circuiting the cached_property.
    # cached_property writes into instance __dict__ on first access; we set
    # it directly to bypass the probe.
    frozen.__dict__["output_dim"] = -1

    with pytest.raises(ValueError, match=r"output_dim probe failed"):
        V2TrainWrapper(
            cfg=_minimal_cfg(),
            model=_DummyPOYO(dim=64),
            modality_spec=_DummyModalitySpec(),
            tokenizer_mode="ntt_frozen",
            unit_embedding_mode="none",
            sentinels=False,
            ntt_tokenizer=frozen,
        )
