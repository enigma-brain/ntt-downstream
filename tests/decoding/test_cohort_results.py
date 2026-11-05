"""Benchmark cohort record: built from local child run dirs, completeness explicit."""

from __future__ import annotations

import json
import math

from ntt_downstream.decoding import cohort_results as cr


def _block(name: str, r2: float) -> dict:
    return {
        "aggregated": {"r2": {"value": r2, "direction": "higher", "value_range": [None, 1.0]}},
        "per_session": {name: {"r2": r2}},
    }


def _child(root, name, version, r2=None, *, record=True, test=True, git="g1", ntt="0.1.1"):
    """A benchmark child dir with ``csv_logs/version_<version>/`` and optionally a v2 record."""
    v = root / name / "csv_logs" / f"version_{version}"
    v.mkdir(parents=True, exist_ok=True)
    (v / "run_result.json").write_text("{}")
    if record:
        metrics = {"val_best": _block(name, 0.9), "val_last": _block(name, 0.8)}
        if test:
            metrics["test"] = _block(name, r2)
        doc = {
            "schema_version": 2,
            "task": "decoding",
            "kind": "run",
            "metrics": metrics,
            "metadata": {"git_sha": git, "ntt_version": ntt, "run_name": name, "mlflow_run_id": f"id-{name}"},
        }
        (v / "downstream_decoding_results.json").write_text(json.dumps(doc))
    return root / name


def _build(dirs, **kw):
    kw.setdefault("tokenizer_checkpoint", "/a.ckpt")
    kw.setdefault("parent_run_id", "P")
    return cr.build_cohort_document(tokenizer_name="tokA", child_dirs=dirs, **kw).to_dict()


def test_complete_cohort(tmp_path):
    d = _build({"s1": _child(tmp_path, "s1", 0, 0.4), "s2": _child(tmp_path, "s2", 0, 0.6)})

    assert (d["schema_version"], d["kind"], d["task"]) == (2, "cohort", "decoding")
    c = d["metadata"]["cohort"]
    assert c["complete"] is True
    assert c["expected"] == ["s1", "s2"] and c["found"] == ["s1", "s2"] and c["missing"] == []
    assert c["mixed_provenance"] is False
    assert c["children"]["s1"]["mlflow_run_id"] == "id-s1"
    agg = d["metrics"]["test"]["aggregated"]["r2"]
    assert math.isclose(agg["value"], 0.5) and agg["n"] == 2 and agg["aggregated_over"] == "sessions:mean"
    assert d["metrics"]["test"]["per_session"] == {"s1": {"r2": 0.4}, "s2": {"r2": 0.6}}
    assert d["metrics"]["val_best"]["aggregated"]["r2"]["value"] == 0.9
    assert d["metadata"]["tokenizer_checkpoint"] == "/a.ckpt"
    assert d["metadata"]["mlflow_run_id"] == "P"
    assert d["metadata"]["sessions"] == ["s1", "s2"]


def test_child_without_record_is_missing(tmp_path):
    d = _build({"s1": _child(tmp_path, "s1", 0, 0.4), "s2": _child(tmp_path, "s2", 0, record=False)})

    c = d["metadata"]["cohort"]
    assert c["complete"] is False and c["missing"] == ["s2"] and c["found"] == ["s1"]
    assert d["metrics"]["test"]["aggregated"]["r2"]["n"] == 1
    assert "s2" not in d["metrics"]["test"]["per_session"]


def test_child_dir_absent_is_missing(tmp_path):
    d = _build({"s1": tmp_path / "never_ran"})
    assert d["metadata"]["cohort"]["missing"] == ["s1"]


def test_newest_version_without_record_is_missing(tmp_path):
    _child(tmp_path, "s1", 0, 0.4)
    _child(tmp_path, "s1", 1, record=False)  # crashed rerun: newest version has no record

    d = _build({"s1": tmp_path / "s1"})

    assert d["metadata"]["cohort"]["missing"] == ["s1"]


def test_newest_version_is_used(tmp_path):
    _child(tmp_path, "s1", 2, 0.1)
    _child(tmp_path, "s1", 10, 0.7)  # version-aware sort: 10 > 2

    d = _build({"s1": tmp_path / "s1"})

    assert d["metrics"]["test"]["per_session"]["s1"]["r2"] == 0.7


def test_child_without_test_block_is_missing(tmp_path):
    d = _build({"s1": _child(tmp_path, "s1", 0, test=False)})

    assert d["metadata"]["cohort"]["missing"] == ["s1"]
    assert d["metrics"]["test"]["aggregated"] == {}


def test_unparseable_child_record_is_missing(tmp_path):
    child = _child(tmp_path, "s1", 0, 0.4)
    (child / "csv_logs" / "version_0" / "downstream_decoding_results.json").write_text("{trunc")

    assert _build({"s1": child})["metadata"]["cohort"]["missing"] == ["s1"]


def test_mixed_provenance_flagged(tmp_path):
    d = _build({"s1": _child(tmp_path, "s1", 0, 0.4, git="g1"), "s2": _child(tmp_path, "s2", 0, 0.6, git="g2")})
    assert d["metadata"]["cohort"]["mixed_provenance"] is True


def test_nonfinite_child_r2_excluded_from_aggregate(tmp_path):
    dirs = {"s1": _child(tmp_path, "s1", 0, 0.4), "s2": _child(tmp_path, "s2", 0, float("nan"))}

    doc = cr.build_cohort_document(
        tokenizer_name="tokA", tokenizer_checkpoint=None, parent_run_id=None, child_dirs=dirs
    )
    out = cr.write_cohort_record(doc, tmp_path / "out")

    written = json.loads(out.read_text())
    assert written["metrics"]["test"]["aggregated"]["r2"]["n"] == 1
    assert math.isclose(written["metrics"]["test"]["aggregated"]["r2"]["value"], 0.4)
    assert written["metrics"]["test"]["per_session"]["s2"]["r2"] is None


def test_write_cohort_record_names_file_and_never_raises(tmp_path):
    doc = cr.build_cohort_document(tokenizer_name="t", tokenizer_checkpoint=None, parent_run_id=None, child_dirs={})
    assert cr.write_cohort_record(doc, tmp_path / "ok").name == "downstream_decoding_results.json"

    blocker = tmp_path / "file"
    blocker.write_text("x")  # out_dir is a regular file -> mkdir fails
    assert cr.write_cohort_record(doc, blocker) is None


def test_version_dirs_sorted_numerically(tmp_path):
    for n in (10, 2, 1):
        (tmp_path / "csv_logs" / f"version_{n}").mkdir(parents=True)
    assert [p.name for p in cr.version_dirs(tmp_path)] == ["version_1", "version_2", "version_10"]


def _set_timestamp(child_dir, version, ts):
    p = child_dir / "csv_logs" / f"version_{version}" / "downstream_decoding_results.json"
    doc = json.loads(p.read_text())
    doc["metadata"]["timestamp"] = ts
    p.write_text(json.dumps(doc))


def test_record_older_than_sweep_start_is_stale_missing(tmp_path):
    """A rerun that crashes before Lightning creates a new version dir leaves the
    previous attempt's record as 'newest'; it predates this sweep, so it must
    count as missing, never as a found (stale) result."""
    old = _child(tmp_path, "s1", 0, 0.42)
    _set_timestamp(old, 0, "2026-10-06T10:00:00Z")
    fresh = _child(tmp_path, "s2", 0, 0.6)
    _set_timestamp(fresh, 0, "2026-10-06T12:00:01Z")

    d = _build({"s1": old, "s2": fresh}, not_before="2026-10-06T12:00:00Z")

    c = d["metadata"]["cohort"]
    assert c["missing"] == ["s1"] and c["stale"] == ["s1"] and c["found"] == ["s2"]
    assert c["complete"] is False and c["not_before"] == "2026-10-06T12:00:00Z"
    assert "s1" not in d["metrics"]["test"]["per_session"]


def test_record_without_timestamp_is_stale_when_gated(tmp_path):
    child = _child(tmp_path, "s1", 0, 0.4)  # helper writes no timestamp
    d = _build({"s1": child}, not_before="2026-10-06T12:00:00Z")
    assert d["metadata"]["cohort"]["stale"] == ["s1"]


def test_malformed_child_only_drops_that_child(tmp_path):
    bad = _child(tmp_path, "s1", 0, 0.4)
    (bad / "csv_logs" / "version_0" / "downstream_decoding_results.json").write_text('{"metrics": "oops"}')
    (tmp_path / "s2" / "csv_logs" / "version_tmp").mkdir(parents=True)  # non-numeric version dir
    good = _child(tmp_path, "s2", 0, 0.6)

    d = _build({"s1": bad, "s2": good})

    assert d["metadata"]["cohort"]["missing"] == ["s1"] and d["metadata"]["cohort"]["found"] == ["s2"]


def test_mixed_ntt_commit_flagged(tmp_path):
    """Same ntt version string, different ntt commits (embedded overlay across a change)."""
    a = _child(tmp_path, "s1", 0, 0.4)
    b = _child(tmp_path, "s2", 0, 0.6)
    for child, commit in ((a, "c1"), (b, "c2")):
        p = child / "csv_logs" / "version_0" / "downstream_decoding_results.json"
        doc = json.loads(p.read_text())
        doc["metadata"]["ntt_commit"] = commit
        p.write_text(json.dumps(doc))

    d = _build({"s1": a, "s2": b})

    assert d["metadata"]["cohort"]["mixed_provenance"] is True
    assert d["metadata"]["cohort"]["children"]["s1"]["ntt_commit"] == "c1"
