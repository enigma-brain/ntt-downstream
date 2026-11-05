"""Train-time augmentations for dense binned spike responses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def spike_augmentations_enabled(cfg: Any) -> bool:
    """Return whether any spike augmentation should be applied."""
    if cfg is None:
        return False
    return bool(_cfg_get(cfg, "enabled", True))


def _sample_rows(batch_size: int, p: float, device: torch.device) -> torch.Tensor:
    if p <= 0:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)
    if p >= 1:
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    return torch.rand(batch_size, device=device) < p


def _normalize_to_btn(response: torch.Tensor, time_dim: int) -> tuple[torch.Tensor, int]:
    """Move a 3-D response tensor to (batch, time, units)."""
    if response.ndim != 3:
        raise ValueError(f"Spike augmentations expect a 3-D tensor, got shape {tuple(response.shape)}")
    time_dim = time_dim % response.ndim
    if time_dim == 1:
        return response, time_dim
    if time_dim == 2:
        return response.transpose(1, 2), time_dim
    raise ValueError("time_dim must point to a non-batch response axis")


def _restore_from_btn(response_btn: torch.Tensor, original_time_dim: int) -> torch.Tensor:
    if original_time_dim == 1:
        return response_btn
    return response_btn.transpose(1, 2)


def _cfg_first(cfg: Any, keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        value = _cfg_get(cfg, key, None)
        if value is not None:
            return value
    return default


def _event_coords_from_counts(counts: torch.Tensor) -> torch.Tensor:
    """Expand integer binned counts into one (time, unit) row per spike event."""
    coords = torch.nonzero(counts > 0, as_tuple=False)
    if coords.numel() == 0:
        return coords
    repeats = counts[coords[:, 0], coords[:, 1]]
    return torch.repeat_interleave(coords, repeats, dim=0)


def _cap_count_events(counts: torch.Tensor, max_events: int | None) -> torch.Tensor:
    """Randomly keep at most ``max_events`` events from a dense count tensor."""
    if max_events is None:
        return counts
    max_events = int(max_events)
    if max_events < 0 or int(counts.sum().item()) <= max_events:
        return counts

    events = _event_coords_from_counts(counts)
    if events.shape[0] <= max_events:
        return counts

    keep = torch.randperm(events.shape[0], device=counts.device)[:max_events]
    kept_events = events[keep]
    capped = torch.zeros_like(counts)
    capped.index_put_(
        (kept_events[:, 0], kept_events[:, 1]),
        torch.ones(max_events, dtype=counts.dtype, device=counts.device),
        accumulate=True,
    )
    return capped


def _integer_counts(response: torch.Tensor) -> torch.Tensor:
    # These augmentations operate on raw spike counts. The round keeps them usable
    # if a caller accidentally passes a floating count tensor.
    return response.clamp_min(0).round().to(dtype=torch.long)


def _apply_jitter(response: torch.Tensor, cfg: Any) -> torch.Tensor:
    """Shift a random subset of spike events in time without changing counts.

    Config:
        p: Probability that a batch item is jittered.
        max_bins / strength: Maximum absolute temporal shift in bins. The
            sampled shift is always nonzero.
        fraction: Fraction of spike events to jitter within selected items.
        num_spikes: Absolute number of spike events to jitter per selected item.
            Takes precedence over ``fraction`` when both are set.

    If neither ``fraction`` nor ``num_spikes`` is set, every spike event in a
    selected item is eligible, matching the original behavior.
    """
    p = float(_cfg_get(cfg, "p", 0.0))
    max_bins = int(_cfg_get(cfg, "max_bins", _cfg_get(cfg, "strength", 0)))
    if p <= 0 or max_bins <= 0:
        return response

    selected = _sample_rows(response.shape[0], p, response.device)
    if not selected.any():
        return response

    out = response.clone()
    for b in torch.nonzero(selected, as_tuple=False).flatten().tolist():
        sample = response[b]
        events = _event_coords_from_counts(_integer_counts(sample))
        n_events = events.shape[0]
        if n_events == 0:
            continue

        num_spikes = _cfg_first(cfg, ("num_spikes", "max_spikes"), None)
        fraction = _cfg_first(cfg, ("fraction", "spike_fraction", "jitter_fraction"), None)
        if num_spikes is not None:
            n_jitter = min(n_events, max(0, int(num_spikes)))
        elif fraction is not None:
            n_jitter = min(n_events, max(0, int(round(n_events * float(fraction)))))
        else:
            n_jitter = n_events
        if n_jitter == 0:
            continue

        jittered_events = events[torch.randperm(n_events, device=response.device)[:n_jitter]]
        magnitudes = torch.randint(1, max_bins + 1, (n_jitter,), device=response.device)
        signs = torch.where(
            torch.rand(n_jitter, device=response.device) < 0.5,
            -torch.ones(n_jitter, dtype=magnitudes.dtype, device=response.device),
            torch.ones(n_jitter, dtype=magnitudes.dtype, device=response.device),
        )
        offsets = magnitudes * signs
        old_t = jittered_events[:, 0]
        unit = jittered_events[:, 1]
        new_t = (old_t + offsets).clamp_(0, sample.shape[0] - 1)
        values = torch.ones(n_jitter, dtype=out.dtype, device=response.device)
        out[b].index_put_((old_t, unit), -values, accumulate=True)
        out[b].index_put_((new_t, unit), values, accumulate=True)
    return out


def _apply_dropout(response: torch.Tensor, cfg: Any) -> torch.Tensor:
    """Randomly remove spike events, optionally capped per selected item.

    Config:
        p: Probability that a batch item receives dropout.
        drop_prob / strength: Independent probability of removing each spike.
        max_spikes / max_dropped_spikes: Upper bound on removed spikes per item.
    """
    p = float(_cfg_get(cfg, "p", 0.0))
    drop_prob = float(_cfg_get(cfg, "drop_prob", _cfg_get(cfg, "strength", 0.0)))
    if p <= 0 or drop_prob <= 0:
        return response

    selected = _sample_rows(response.shape[0], p, response.device)
    if not selected.any():
        return response

    out = response.clone()
    drop_prob = max(0.0, min(1.0, drop_prob))
    max_spikes = _cfg_first(cfg, ("max_spikes", "max_dropped_spikes"), None)
    for b in torch.nonzero(selected, as_tuple=False).flatten().tolist():
        counts = response[b].clamp_min(0)
        dropped = torch.distributions.Binomial(total_count=counts, probs=drop_prob).sample()  # pyright: ignore[reportArgumentType] # stub says Tensor; torch accepts float
        dropped = _cap_count_events(dropped.round().to(dtype=torch.long), max_spikes).to(dtype=response.dtype)
        out[b] = (response[b] - dropped).clamp_min(0)
    return out


def _apply_addition(response: torch.Tensor, cfg: Any) -> torch.Tensor:
    """Add sparse synthetic spikes, optionally capped per selected item.

    Config:
        p: Probability that a batch item receives additions.
        add_prob / strength: Independent probability of adding at each bin/unit.
        add_count: Count increment for each accepted synthetic event.
        max_spikes / max_added_spikes: Upper bound on added events per item.
    """
    p = float(_cfg_get(cfg, "p", 0.0))
    add_prob = float(_cfg_get(cfg, "add_prob", _cfg_get(cfg, "strength", 0.0)))
    add_count = float(_cfg_get(cfg, "add_count", 1.0))
    if p <= 0 or add_prob <= 0 or add_count <= 0:
        return response

    selected = _sample_rows(response.shape[0], p, response.device)
    if not selected.any():
        return response

    out = response.clone()
    add_prob = max(0.0, min(1.0, add_prob))
    max_spikes = _cfg_first(cfg, ("max_spikes", "max_added_spikes"), None)
    for b in torch.nonzero(selected, as_tuple=False).flatten().tolist():
        candidates = torch.bernoulli(torch.full_like(response[b], add_prob)).to(dtype=torch.long)
        additions = _cap_count_events(candidates, max_spikes).to(dtype=response.dtype) * add_count
        out[b] = response[b] + additions
    return out


def apply_spike_augmentations(response: torch.Tensor, cfg: Any, *, time_dim: int) -> torch.Tensor:
    """Apply jitter, dropout, and addition to dense binned spike responses.

    Args:
        response: 3-D dense response tensor. The first dimension is batch.
        cfg: Config object/dict with optional ``jitter``, ``dropout``, and
            ``addition`` sections. Each section has an application probability
            ``p`` and augmentation-specific strength/cap parameters documented
            in the private helper docstrings above.
        time_dim: Axis containing time bins in ``response``.

    Returns:
        Augmented response tensor with the same shape and dtype.
    """
    if not spike_augmentations_enabled(cfg):
        return response

    work, original_time_dim = _normalize_to_btn(response, time_dim)
    work = _apply_jitter(work, _cfg_get(cfg, "jitter"))
    work = _apply_dropout(work, _cfg_get(cfg, "dropout"))
    work = _apply_addition(work, _cfg_get(cfg, "addition"))
    return _restore_from_btn(work, original_time_dim)
