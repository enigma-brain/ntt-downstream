"""The checked-in v2 fixtures match what the writers produce today.

Consumers such as ntt's aggregation reader test against copies of these files;
when a writer change makes them stale, regenerate with
``uv run python scripts/fixtures/make_results_v2_fixtures.py``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "fixtures" / "make_results_v2_fixtures.py"
_EXPECTED = {
    "encoding_run.json",
    "encoding_run_final_eval_failed.json",
    "decoding_run.json",
    "decoding_run_multisession_nan.json",
    "decoding_cohort.json",
    "decoding_cohort_incomplete.json",
}


def _load_script():
    spec = importlib.util.spec_from_file_location("make_v2_fixtures", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v2_fixtures_are_current():
    script = _load_script()
    built = script.build_all()

    assert set(built) == _EXPECTED
    for name, doc in built.items():
        on_disk = json.loads((script.FIXTURES_DIR / name).read_text())
        assert on_disk == doc, f"{name} is stale: run python scripts/fixtures/make_results_v2_fixtures.py"


def test_fixtures_cover_the_contract():
    """Spot-check the cases the ntt reader must handle are really in the fixtures."""
    script = _load_script()
    load = lambda n: json.loads((script.FIXTURES_DIR / n).read_text())  # noqa: E731

    assert "test" in load("encoding_run.json")["metrics"]
    assert "test" not in load("encoding_run_final_eval_failed.json")["metrics"]
    assert load("decoding_cohort.json")["metadata"]["cohort"]["complete"] is True
    multi = load("decoding_run_multisession_nan.json")
    assert multi["metrics"]["test"]["per_session"]["t_20130819_center_out_reaching"]["r2"] is None
    assert multi["metrics"]["test"]["aggregated"]["r2"]["n"] == 2
    assert multi["metadata"]["checkpoint"]["best_path"]
    assert load("decoding_cohort_incomplete.json")["metadata"]["cohort"]["missing"] == [
        "j_20160405_center_out_reaching"
    ]
