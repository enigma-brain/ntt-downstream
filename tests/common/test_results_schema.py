from ntt_downstream.common.results.schema import (
    SCHEMA_VERSION,
    MetricBlock,
    MetricEntry,
    ResultsDocument,
    RunMetadata,
)


def test_metric_entry_to_dict_minimal():
    e = MetricEntry(value=0.5, direction="higher", value_range=(-1.0, 1.0))
    assert e.to_dict() == {"value": 0.5, "direction": "higher", "value_range": [-1.0, 1.0]}


def test_metric_entry_optional_fields():
    e = MetricEntry(value=0.5, value_range=(None, 1.0), median=0.6, std=0.1, n=20)
    d = e.to_dict()
    assert d["value_range"] == [None, 1.0]
    assert d["median"] == 0.6 and d["std"] == 0.1 and d["n"] == 20


def test_metric_block_to_dict_merges_groups():
    block = MetricBlock(
        aggregated={"r2": MetricEntry(value=0.4, value_range=(None, 1.0))},
        per_session={"bs/sess": {"r2": 0.3}},
        groups={"tiers": {"tier1_val": {"r2": 0.35}}, "train": {"loss": 0.1}},
    )
    d = block.to_dict()
    assert d["aggregated"]["r2"]["value"] == 0.4
    assert d["per_session"] == {"bs/sess": {"r2": 0.3}}
    assert d["tiers"] == {"tier1_val": {"r2": 0.35}}
    assert d["train"] == {"loss": 0.1}


def test_results_document_to_dict():
    md = RunMetadata(
        git_sha="abc",
        timestamp="2026-06-06T00:00:00Z",
        hostname="h",
        run_name="r",
        mlflow_run_id="m",
        config={"k": 1},
        checkpoint={"monitor": "x"},
        sessions=["bs/sess"],
        n_sessions=1,
    )
    doc = ResultsDocument(
        task="encoding",
        metadata=md,
        metrics={"last": MetricBlock(), "best": MetricBlock()},
    )
    d = doc.to_dict()
    assert d["schema_version"] == SCHEMA_VERSION
    assert d["task"] == "encoding"
    assert d["metadata"]["git_sha"] == "abc"
    assert set(d["metrics"].keys()) == {"last", "best"}


def test_metric_entry_aggregated_over_emitted_only_when_set():
    e_unset = MetricEntry(value=0.5, value_range=(None, 1.0))
    assert "aggregated_over" not in e_unset.to_dict()

    e_set = MetricEntry(value=0.5, value_range=(None, 1.0), aggregated_over="sessions")
    assert e_set.to_dict()["aggregated_over"] == "sessions"


def test_run_metadata_has_no_headline():
    """``metrics.test`` is the authoritative block; metadata carries no headline pointer."""
    base = {
        "git_sha": "abc",
        "timestamp": "2026-06-22T00:00:00Z",
        "hostname": "h",
        "run_name": "r",
        "mlflow_run_id": None,
        "config": {},
        "checkpoint": {},
        "sessions": [],
        "n_sessions": 0,
    }
    assert "headline" not in RunMetadata(**base).to_dict()


def _md(**over):
    base = {
        "git_sha": "a",
        "timestamp": "t",
        "hostname": "h",
        "run_name": "r",
        "mlflow_run_id": None,
        "config": {},
        "checkpoint": {},
        "sessions": ["s"],
        "n_sessions": 1,
    }
    base.update(over)
    return RunMetadata(**base)


def test_results_document_v2_shape():
    md = _md(tokenizer_checkpoint="/x.ckpt", ntt_version="0.1.1", ntt_path="/n/src/ntt/__init__.py")
    d = ResultsDocument(task="decoding", kind="cohort", metadata=md, metrics={}).to_dict()
    assert d["schema_version"] == 2 and d["kind"] == "cohort"
    assert d["metadata"]["tokenizer_checkpoint"] == "/x.ckpt"
    assert d["metadata"]["ntt_version"] == "0.1.1"
    assert d["metadata"]["ntt_path"] == "/n/src/ntt/__init__.py"


def test_results_document_default_kind_is_run_and_provenance_null():
    d = ResultsDocument(task="encoding", metadata=_md(), metrics={}).to_dict()
    assert d["kind"] == "run"
    assert d["metadata"]["tokenizer_checkpoint"] is None and d["metadata"]["ntt_path"] is None
