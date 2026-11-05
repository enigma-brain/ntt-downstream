"""DecodingStitchEvaluator variant that gates val-epoch wandb Table logging.

The upstream ``DecodingStitchEvaluator`` logs a ``wandb.Table`` with per-session
metrics at the end of every validation and test epoch. Each table costs ~2 MB in
wandb media; across 100s of epochs this adds 10+ MB per run and is mostly
redundant with the scalar metrics logged via ``log_dict``.

This subclass lets you disable *validation*-epoch table logging while keeping
the *test*-epoch table (logged once, at the end of training via
``trainer.test()``).
"""

from __future__ import annotations

from torch_brain.utils.callbacks import DecodingStitchEvaluator


class DecodingStitchEvaluatorWithOptionalValTable(DecodingStitchEvaluator):
    """DecodingStitchEvaluator with optional val-epoch wandb-Table suppression.

    Args:
        session_ids: forwarded to base class.
        modality_spec: forwarded.
        metric_factory: forwarded.
        quiet: forwarded.
        log_val_tables: if False (default), wandb Table logging inside
            ``on_validation_epoch_end`` is suppressed. Scalar metrics
            (``self.log_dict(metrics)``) are still logged. Test-epoch table
            logging is always kept (it runs through the base class's test hooks
            which reuse the validation code path; we distinguish by the
            ``prefix`` argument in ``on_validation_epoch_end``).
    """

    def __init__(
        self,
        *args,
        log_val_tables: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.log_val_tables = log_val_tables

    def on_validation_epoch_end(self, trainer, pl_module, prefix: str = "val"):
        """Gate wandb Table logging for val prefix; keep everything else.

        Implementation note: we monkey-patch ``wandb.Table`` on the imported
        module scope of the base class only for the duration of the call when
        ``prefix == "val"`` and ``log_val_tables`` is False. That avoids
        reimplementing the whole method (which is ~70 lines and may change
        upstream).
        """
        if prefix == "val" and not self.log_val_tables:
            import torch_brain.utils.callbacks as _stitcher_mod

            real_wandb = _stitcher_mod.wandb
            if real_wandb is None:
                # wandb not installed; base class already skips Table logging.
                return super().on_validation_epoch_end(trainer, pl_module, prefix=prefix)

            # Temporarily swap wandb to None so the isinstance/`wandb is not None`
            # branch in the base class falls through, skipping Table logging.
            # Scalar metrics (self.log_dict) are logged before that branch.
            _stitcher_mod.wandb = None
            try:
                return super().on_validation_epoch_end(trainer, pl_module, prefix=prefix)
            finally:
                _stitcher_mod.wandb = real_wandb
        else:
            return super().on_validation_epoch_end(trainer, pl_module, prefix=prefix)
