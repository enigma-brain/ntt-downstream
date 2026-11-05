"""Unit tests for the pure helpers extracted from ``decoding/train.py``.

``main()`` delegates to module-level helpers. These three are pure (no
GPU / dataset / Hydra-runtime needed) and govern determinism-critical behavior
— optimizer-step / OneCycleLR cadence (``_resolve_trainer_schedule``), the
eval-only loader collapse (``_maybe_loaders``), and which params train under
fine-tuning (``_apply_freeze_mode``) — so they get explicit tests.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf
from torch import nn

from ntt_downstream.decoding.train import (
    _apply_freeze_mode,
    _maybe_loaders,
    _resolve_trainer_schedule,
)

# --- _resolve_trainer_schedule: epoch- vs step-driven cadence, scaled by W ---


@pytest.mark.parametrize("world_size,exp_epochs,exp_eval", [(1, 10, 2), (2, 5, 1)])
def test_resolve_schedule_epoch_mode(world_size, exp_epochs, exp_eval):
    cfg = OmegaConf.create({"epochs": 10, "eval_epochs": 2})
    s = _resolve_trainer_schedule(cfg, world_size=world_size)
    assert s.use_max_steps is False
    assert s.max_steps == -1
    assert s.max_epochs == exp_epochs
    assert s.check_val_every_n_epoch == exp_eval
    # Checkpoint cadence mirrors the eval cadence in epoch mode.
    assert s.ckpt_every_n_epochs == exp_eval
    assert s.val_check_interval is None
    assert s.ckpt_every_n_train_steps is None


@pytest.mark.parametrize(
    "world_size,exp_steps,exp_val,exp_ckpt",
    [(1, 1000, 100, 500), (2, 500, 50, 250)],
)
def test_resolve_schedule_step_mode(world_size, exp_steps, exp_val, exp_ckpt):
    cfg = OmegaConf.create(
        {
            "max_steps": 1000,
            "eval_every_n_steps": 100,
            "checkpoint_every_n_steps": 500,
        }
    )
    s = _resolve_trainer_schedule(cfg, world_size=world_size)
    assert s.use_max_steps is True
    assert s.max_steps == exp_steps
    assert s.val_check_interval == exp_val
    assert s.ckpt_every_n_train_steps == exp_ckpt
    assert s.max_epochs == -1
    assert s.check_val_every_n_epoch is None
    assert s.ckpt_every_n_epochs is None


def test_resolve_schedule_floors_to_one():
    """With more GPUs than epochs the floor-divide must clamp to 1, never 0."""
    cfg = OmegaConf.create({"epochs": 1, "eval_epochs": 1})
    s = _resolve_trainer_schedule(cfg, world_size=8)
    assert s.max_epochs == 1
    assert s.check_val_every_n_epoch == 1


# --- _maybe_loaders: empty -> None, singleton -> bare, many -> list ---


def test_maybe_loaders_empty_is_none():
    assert _maybe_loaders([]) is None


def test_maybe_loaders_singleton_is_collapsed():
    sentinel = object()
    assert _maybe_loaders([sentinel]) is sentinel


def test_maybe_loaders_multiple_stays_list():
    a, b = object(), object()
    assert _maybe_loaders([a, b]) == [a, b]


# --- _apply_freeze_mode: substring-matched param selection ---


class _FreezeStub(nn.Module):
    """nn.Module exposing the param-name prefixes _apply_freeze_mode matches."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.unit_emb = nn.Embedding(4, 8)
        self.model.session_emb = nn.Embedding(4, 8)
        self.model.encoder = nn.Linear(8, 8)  # body — should freeze
        self.input_proj = nn.Linear(8, 8)


def _trainable_names(module: nn.Module) -> set[str]:
    return {n for n, p in module.named_parameters() if p.requires_grad}


def test_freeze_mode_none_is_noop():
    stub = _FreezeStub()
    _apply_freeze_mode(OmegaConf.create({"freeze_mode": "none"}), stub)
    assert _trainable_names(stub) == {n for n, _ in stub.named_parameters()}


def test_freeze_mode_unit_identification_trains_only_embeddings():
    stub = _FreezeStub()
    _apply_freeze_mode(OmegaConf.create({"freeze_mode": "unit_identification"}), stub)
    trainable = _trainable_names(stub)
    assert trainable == {"model.unit_emb.weight", "model.session_emb.weight"}
    # Body + input_proj are frozen by default.
    assert not stub.model.encoder.weight.requires_grad
    assert not stub.input_proj.weight.requires_grad


def test_freeze_mode_unit_identification_with_input_proj():
    stub = _FreezeStub()
    _apply_freeze_mode(
        OmegaConf.create({"freeze_mode": "unit_identification", "ft_train_input_proj": True}),
        stub,
    )
    trainable = _trainable_names(stub)
    assert "input_proj.weight" in trainable and "input_proj.bias" in trainable
    assert "model.unit_emb.weight" in trainable
    # Body still frozen.
    assert not stub.model.encoder.weight.requires_grad


def test_freeze_mode_unknown_raises():
    with pytest.raises(ValueError, match="Unknown freeze_mode"):
        _apply_freeze_mode(OmegaConf.create({"freeze_mode": "bogus"}), _FreezeStub())
