"""Tests for the ``_TOKENIZE_METHODS`` registry + ``_tokenize`` dispatch.

``_tokenize`` is registry-driven (``_TOKENIZE_METHODS``) rather than an
``if/elif`` ladder. The contract that must hold:

  * each ``tokenizer_mode`` routes to the right method;
  * an unknown mode raises a clear ``ValueError``;
  * ``getattr(self, name)`` resolves against the subclass MRO so
    ``FullWindowUnitV2TrainWrapper``'s ``_ntt_frozen_tokenize`` override is
    picked up automatically (the whole point of the registry-of-names design).
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from ntt_downstream.decoding.backends.poyo.full_window_train_wrapper import (
    FullWindowUnitV2TrainWrapper,
)
from ntt_downstream.decoding.backends.poyo.train_wrapper import V2TrainWrapper


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
    def __init__(self, dim: int = 64, sequence_length: float = 1.0) -> None:
        super().__init__()
        self.dim = dim
        self.sequence_length = sequence_length
        self.unit_emb = nn.Embedding(64, dim)
        self.token_type_emb = nn.Embedding(8, dim)
        self.session_emb = nn.Embedding(8, dim)


class _DummyModalitySpec:
    def loss_fn(self, *args, **kwargs):
        return torch.tensor(0.0)


def _wrapper(cls=V2TrainWrapper) -> V2TrainWrapper:
    return cls(
        cfg=_minimal_cfg(),
        model=_DummyPOYO(dim=64),
        modality_spec=_DummyModalitySpec(),
        tokenizer_mode="fake",
        unit_embedding_mode="none",
        sentinels=False,
    )


def test_registry_maps_every_mode():
    assert V2TrainWrapper._TOKENIZE_METHODS == {
        "fake": "_fake_tokenize",
        "ntt": "_ntt_tokenize",
        "ntt_frozen": "_ntt_frozen_tokenize",
    }


def test_subclass_inherits_same_registry_object():
    # No accidental per-subclass copy — the override is via method MRO, not a
    # rewritten registry.
    assert FullWindowUnitV2TrainWrapper._TOKENIZE_METHODS is V2TrainWrapper._TOKENIZE_METHODS


def test_dispatch_routes_to_selected_method():
    w = _wrapper()
    seen = []
    w._fake_tokenize = lambda batch: seen.append("fake") or "FAKE"
    w._ntt_tokenize = lambda batch: seen.append("ntt") or "NTT"
    w._ntt_frozen_tokenize = lambda batch: seen.append("frozen") or "FROZEN"

    assert w._tokenize({}) == "FAKE"  # tokenizer_mode == "fake"
    w.tokenizer_mode = "ntt"
    assert w._tokenize({}) == "NTT"
    w.tokenizer_mode = "ntt_frozen"
    assert w._tokenize({}) == "FROZEN"
    assert seen == ["fake", "ntt", "frozen"]


def test_dispatch_unknown_mode_raises():
    w = _wrapper()
    w.tokenizer_mode = "does_not_exist"
    with pytest.raises(ValueError, match="Unknown tokenizer_mode"):
        w._tokenize({})


def test_mro_resolves_subclass_override():
    base = _wrapper(V2TrainWrapper)
    full = _wrapper(FullWindowUnitV2TrainWrapper)
    name = V2TrainWrapper._TOKENIZE_METHODS["ntt_frozen"]

    # The base resolves to the base method; the subclass instance resolves to
    # the override — purely via getattr/MRO, no registry edit.
    assert getattr(base, name).__func__.__qualname__ == "V2TrainWrapper._ntt_frozen_tokenize"
    assert getattr(full, name).__func__.__qualname__ == "FullWindowUnitV2TrainWrapper._ntt_frozen_tokenize"
