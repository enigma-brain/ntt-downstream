"""Write a ResultsDocument to ``downstream_{task}_results.json`` and upload it to experiment loggers."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ntt_downstream.common.results.schema import ResultsDocument
from ntt_downstream.common.results.serialization import write_json

logger = logging.getLogger(__name__)


def check_config_resolves(cfg: Any, *, task: str) -> None:
    """Raise ``RuntimeError`` unless every interpolation in ``cfg`` resolves.

    The results record stores ``OmegaConf.to_container(cfg, resolve=True)``, so
    an unresolvable value (e.g. an ``${oc.env:VAR}`` with no default and the
    variable unset) would otherwise surface only after training, when the
    record is written. Training entry points call this at startup instead.
    ``task`` only names the config in the error message.
    """
    from omegaconf import OmegaConf
    from omegaconf.errors import InterpolationResolutionError

    try:
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    except InterpolationResolutionError as exc:
        raise RuntimeError(
            f"The {task} config does not resolve, so its results record could not be written: {exc}"
        ) from exc


def results_filename(task: str) -> str:
    """Record filename for ``task``. Task-named so it can't clash with ntt's own ``results.json``."""
    return f"downstream_{task}_results.json"


def write_results(
    doc: ResultsDocument,
    output_dir: str | Path,
    *,
    loggers: Any | None = None,
) -> Path | None:
    """Write ``doc`` to ``<output_dir>/<results_filename(doc.task)>``; upload to each supporting logger.

    ``loggers`` may be a single Lightning logger or an iterable of them (e.g.
    ``trainer.loggers``); MLflow and Weights & Biases loggers are recognized and
    receive the file as an artifact, other logger types are ignored.

    A write failure logs a warning and returns ``None``; an upload failure logs
    and skips that upload. Callers decide what a ``None`` means: both training
    entry points fail the run, since the record is the run's output.
    """
    try:
        path = write_json(Path(output_dir) / results_filename(doc.task), doc.to_dict())
    except Exception:  # noqa: BLE001
        logger.warning("Failed to write %s; skipping results persistence.", results_filename(doc.task), exc_info=True)
        return None

    logger.info("Wrote results to %s", path)
    for lg in _iter_loggers(loggers):
        _upload_to_logger(path, lg)
    return path


def upload_to_loggers(path: Path, loggers: Any | None) -> None:
    """Upload a single file to each supported logger in ``loggers``.

    Public counterpart to the per-logger ``_upload_to_logger`` dispatch, so
    callers persisting a supplementary artifact (e.g. final-eval's canonical
    record) reuse the same upload path as ``write_results`` instead of poking a
    private name or reimplementing the logger-class dispatch. ``loggers`` may be
    ``None``, a single Lightning logger, or an iterable of them. Best-effort: an
    upload failure is logged and skipped, never raised.
    """
    for lg in _iter_loggers(loggers):
        _upload_to_logger(path, lg)


def _iter_loggers(loggers: Any | None) -> list[Any]:
    """Normalize ``None`` / single-logger / iterable-of-loggers to a flat list."""
    if loggers is None:
        return []
    if isinstance(loggers, Iterable) and not isinstance(loggers, (str, bytes)):
        return [lg for lg in loggers if lg is not None]
    return [loggers]


def _upload_to_logger(path: Path, lg: Any) -> None:
    """Dispatch upload by logger class name (keeps this module free of lightning deps)."""
    name = type(lg).__name__
    if name == "MLFlowLogger":
        _upload_mlflow(path, lg)
    elif name == "WandbLogger":
        _upload_wandb(path, lg)
    # else: logger type we don't know how to upload to — nothing to do.


def _upload_mlflow(path: Path, mlflow_logger: Any) -> None:
    run_id = getattr(mlflow_logger, "run_id", None)
    experiment = getattr(mlflow_logger, "experiment", None)
    if run_id is None or experiment is None:
        return
    try:
        experiment.log_artifact(run_id, str(path))
        logger.info("Uploaded %s to MLflow run %s", path.name, run_id)
    except Exception:  # noqa: BLE001
        logger.warning("Failed to upload %s to MLflow.", path.name, exc_info=True)


def _upload_wandb(path: Path, wandb_logger: Any) -> None:
    # ``experiment`` is the live ``wandb.Run``; ``save(..., policy="now")`` copies the
    # file into the run dir and uploads it immediately (works while the run is active).
    run = getattr(wandb_logger, "experiment", None)
    if run is None or not hasattr(run, "save"):
        return
    try:
        run.save(str(path), base_path=str(path.parent), policy="now")
        logger.info("Uploaded %s to W&B run %s", path.name, getattr(run, "id", "?"))
    except Exception:  # noqa: BLE001
        logger.warning("Failed to upload %s to W&B.", path.name, exc_info=True)
