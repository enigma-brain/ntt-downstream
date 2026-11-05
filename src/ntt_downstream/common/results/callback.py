"""Lightning callback that snapshots ``callback_metrics`` at best/last/test points."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch


def _to_float(v: Any) -> float | None:
    if isinstance(v, torch.Tensor):
        if v.numel() != 1:
            return None
        return float(v.detach().cpu().item())
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _snapshot(metrics: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in metrics.items():
        fv = _to_float(v)
        if fv is not None:
            out[str(k)] = fv
    return out


class MetricSnapshotCallback(L.Callback):
    """Capture flat snapshots of ``trainer.callback_metrics`` for the results writer.

    Tracks the best (per ``monitor``/``mode``) and last validation snapshots, plus a
    test snapshot. When ``monitor`` is None, ``best_val`` tracks the latest validation
    snapshot; when a configured ``monitor`` is never present in ``callback_metrics``,
    ``best_val`` stays None (the writer then falls back to the last snapshot). Rank-0
    only; snapshots are taken at ``on_validation_end`` / ``on_test_end`` (the same point
    ``ModelCheckpoint`` reads).
    """

    def __init__(self, monitor: str | None = None, mode: str = "max"):
        super().__init__()
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.monitor = monitor
        self.mode = mode
        self.best_val: dict[str, float] | None = None
        self.last_val: dict[str, float] | None = None
        self.test: dict[str, float] | None = None
        self.best_score: float | None = None
        self.best_epoch: int | None = None
        self.best_step: int | None = None

    def _is_better(self, score: float) -> bool:
        if self.best_score is None:
            return True
        return score > self.best_score if self.mode == "max" else score < self.best_score

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        snap = _snapshot(trainer.callback_metrics)
        if not snap:
            return
        self.last_val = snap
        if self.monitor is None:
            # No monitor configured: "best" tracks the latest validation snapshot.
            self.best_val = snap
        elif self.monitor in snap:
            score = snap[self.monitor]
            if self._is_better(score):
                self.best_score = score
                self.best_epoch = trainer.current_epoch
                self.best_step = trainer.global_step
                self.best_val = snap
        # else: a monitor was configured but is absent this epoch — leave best_*
        # untouched. If it is never found, best_val stays None and the writer falls
        # back to the last snapshot.

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        snap = _snapshot(trainer.callback_metrics)
        if snap:
            self.test = snap
