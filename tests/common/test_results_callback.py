import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset

from ntt_downstream.common.results.callback import MetricSnapshotCallback


class _ToyModule(L.LightningModule):
    """Logs a deterministic 'score' per val epoch (and a fixed test metric)."""

    def __init__(self, scores=(0.1, 0.5, 0.3)):
        super().__init__()
        self.layer = torch.nn.Linear(4, 1)
        self._scores = list(scores)

    def training_step(self, batch, _):
        (x,) = batch
        return self.layer(x).pow(2).mean()

    def validation_step(self, batch, _):
        pass

    def on_validation_epoch_end(self):
        score = self._scores[min(self.current_epoch, len(self._scores) - 1)]
        self.log("val/score", score)
        self.log("val/per_session/bs/sess", score - 0.05)

    def test_step(self, batch, _):
        pass

    def on_test_epoch_end(self):
        self.log("average_test_metric", 0.42)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _loader():
    x = torch.randn(8, 4)
    return DataLoader(TensorDataset(x), batch_size=4)


def _trainer(cb, max_epochs=3):
    return L.Trainer(
        max_epochs=max_epochs,
        callbacks=[cb],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        accelerator="cpu",
        devices=1,
    )


def test_snapshot_tracks_best_and_last():
    cb = MetricSnapshotCallback(monitor="val/score", mode="max")
    _trainer(cb).fit(_ToyModule(), _loader(), _loader())

    assert cb.best_val is not None and cb.last_val is not None
    assert abs(cb.best_score - 0.5) < 1e-6
    assert cb.best_epoch == 1
    assert abs(cb.best_val["val/score"] - 0.5) < 1e-6  # best holds the epoch-1 snapshot
    assert abs(cb.last_val["val/score"] - 0.3) < 1e-6  # last holds the final epoch
    assert "val/per_session/bs/sess" in cb.last_val


def test_snapshot_min_mode_selects_minimum():
    cb = MetricSnapshotCallback(monitor="val/score", mode="min")
    _trainer(cb).fit(_ToyModule(scores=(0.5, 0.1, 0.3)), _loader(), _loader())
    assert abs(cb.best_score - 0.1) < 1e-6
    assert cb.best_epoch == 1


def test_snapshot_no_monitor_tracks_latest():
    cb = MetricSnapshotCallback(monitor=None)
    _trainer(cb).fit(_ToyModule(), _loader(), _loader())
    # With no monitor, best tracks the latest snapshot (NOT the first).
    assert cb.best_val == cb.last_val
    assert abs(cb.best_val["val/score"] - 0.3) < 1e-6


def test_snapshot_absent_monitor_leaves_best_none():
    cb = MetricSnapshotCallback(monitor="does/not/exist", mode="max")
    _trainer(cb).fit(_ToyModule(), _loader(), _loader())
    # A configured-but-absent monitor must NOT fabricate a best from epoch 0.
    assert cb.best_val is None
    assert cb.best_score is None
    assert cb.last_val is not None  # last is still captured


def test_snapshot_captures_test_phase():
    cb = MetricSnapshotCallback(monitor="val/score", mode="max")
    model = _ToyModule()
    trainer = _trainer(cb, max_epochs=2)
    trainer.fit(model, _loader(), _loader())
    trainer.test(model, _loader())
    assert cb.test is not None
    assert abs(cb.test["average_test_metric"] - 0.42) < 1e-6
