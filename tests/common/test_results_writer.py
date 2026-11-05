import json

from ntt_downstream.common.results.schema import MetricBlock, ResultsDocument, RunMetadata
from ntt_downstream.common.results.writer import results_filename, write_results


def _doc(task="encoding"):
    md = RunMetadata(
        git_sha="abc",
        timestamp="2026-06-06T00:00:00Z",
        hostname="h",
        run_name="r",
        mlflow_run_id=None,
        config={"k": 1},
        checkpoint={},
        sessions=[],
        n_sessions=0,
    )
    return ResultsDocument(task=task, metadata=md, metrics={"last": MetricBlock()})


# Loggers are dispatched by class name, so the fakes must match the real classes.
class MLFlowLogger:  # noqa: N801
    def __init__(self, run_id="RUN123"):
        self.run_id = run_id
        self.experiment = self._Exp()

    class _Exp:
        def __init__(self):
            self.calls = []

        def log_artifact(self, run_id, local_path):
            self.calls.append((run_id, local_path))


class WandbLogger:  # noqa: N801
    def __init__(self):
        self.experiment = self._Run()

    class _Run:
        id = "wb-run"

        def __init__(self):
            self.saved = []

        def save(self, glob_str, base_path=None, policy="live"):
            self.saved.append((glob_str, base_path, policy))


def test_write_results_writes_file(tmp_path):
    path = write_results(_doc(), tmp_path)
    assert path is not None and path.name == results_filename("encoding")
    loaded = json.loads(path.read_text())
    assert loaded["task"] == "encoding"


def test_write_results_uploads_to_mlflow(tmp_path):
    logger = MLFlowLogger()
    path = write_results(_doc(), tmp_path, loggers=[logger])
    assert logger.experiment.calls == [("RUN123", str(path))]


def test_write_results_uploads_to_wandb(tmp_path):
    logger = WandbLogger()
    path = write_results(_doc(), tmp_path, loggers=[logger])
    assert logger.experiment.saved == [(str(path), str(path.parent), "now")]


def test_write_results_uploads_to_all_loggers(tmp_path):
    mlf, wb = MLFlowLogger(), WandbLogger()
    path = write_results(_doc(), tmp_path, loggers=[mlf, wb])
    assert mlf.experiment.calls == [("RUN123", str(path))]
    assert wb.experiment.saved == [(str(path), str(path.parent), "now")]


def test_write_results_accepts_single_logger(tmp_path):
    logger = MLFlowLogger()
    path = write_results(_doc(), tmp_path, loggers=logger)
    assert logger.experiment.calls == [("RUN123", str(path))]


def test_write_results_ignores_unknown_logger(tmp_path):
    class CSVLogger:  # noqa: N801
        save_dir = "x"

    # Unknown logger type must be silently skipped, not raise.
    path = write_results(_doc(), tmp_path, loggers=[CSVLogger()])
    assert path is not None and path.exists()


def test_write_results_never_raises_on_upload_failure(tmp_path):
    class MLFlowLogger:  # noqa: N801
        run_id = "RUN"

        class experiment:  # noqa: N801
            @staticmethod
            def log_artifact(run_id, local_path):
                raise RuntimeError("network down")

    # Should not raise; file is still written.
    path = write_results(_doc(), tmp_path, loggers=[MLFlowLogger()])
    assert path is not None and path.exists()


def test_write_results_never_raises_on_wandb_failure(tmp_path):
    class WandbLogger:  # noqa: N801
        class experiment:  # noqa: N801
            @staticmethod
            def save(glob_str, base_path=None, policy="live"):
                raise RuntimeError("wandb offline error")

    path = write_results(_doc(), tmp_path, loggers=[WandbLogger()])
    assert path is not None and path.exists()


def test_write_results_skips_upload_when_run_id_none(tmp_path):
    class MLFlowLogger:  # noqa: N801
        run_id = None

        class experiment:  # noqa: N801
            @staticmethod
            def log_artifact(run_id, local_path):
                raise AssertionError("must not upload when run_id is None")

    path = write_results(_doc(), tmp_path, loggers=[MLFlowLogger()])
    assert path is not None and path.exists()


def test_write_results_skips_upload_when_experiment_missing(tmp_path):
    class MLFlowLogger:  # noqa: N801
        run_id = "RUN"  # no `experiment` attribute at all

    path = write_results(_doc(), tmp_path, loggers=[MLFlowLogger()])
    assert path is not None and path.exists()


def test_write_results_returns_none_on_write_failure(tmp_path):
    # Parent path is a regular file, so mkdir of a subdir under it fails ->
    # write_results must swallow the error and return None (best-effort).
    a_file = tmp_path / "afile"
    a_file.write_text("x")
    result = write_results(_doc(), a_file / "sub")
    assert result is None


def test_write_results_names_file_by_task(tmp_path):
    assert write_results(_doc("encoding"), tmp_path).name == "downstream_encoding_results.json"
    assert write_results(_doc("decoding"), tmp_path).name == "downstream_decoding_results.json"


# ---------------------------------------------------------------------------
# check_config_resolves: the record stores the resolved config, so both tasks
# resolve it at startup and a broken interpolation stops the run before training.
# ---------------------------------------------------------------------------


def test_check_config_resolves_accepts_a_resolvable_config(monkeypatch):
    from omegaconf import OmegaConf

    from ntt_downstream.common.results.writer import check_config_resolves

    monkeypatch.delenv("NTT_TEST_SURELY_UNSET_VAR", raising=False)
    check_config_resolves(
        OmegaConf.create({"a": 1, "b": "${a}", "c": "${oc.env:NTT_TEST_SURELY_UNSET_VAR,null}"}),
        task="decoding",
    )


def test_check_config_resolves_names_task_and_key(monkeypatch):
    import pytest
    from omegaconf import OmegaConf

    from ntt_downstream.common.results.writer import check_config_resolves

    monkeypatch.delenv("NTT_TEST_SURELY_UNSET_VAR", raising=False)
    cfg = OmegaConf.create({"trainer": {"x": "${oc.env:NTT_TEST_SURELY_UNSET_VAR}"}})
    with pytest.raises(RuntimeError, match=r"(?s)decoding config does not resolve.*trainer\.x"):
        check_config_resolves(cfg, task="decoding")
