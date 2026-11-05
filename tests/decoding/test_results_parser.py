from ntt_downstream.decoding.backends.poyo.results_parser import parse_decoding_block


def _val_metrics():
    return {
        "average_val_metric": 0.55,
        "perich_miller_population_2018/c_x": 0.50,
        "odoherty_sabes/indy_y": 0.60,
        "val/tier1_val_r2": 0.58,
        "val/tier1_val_r2_odoherty": 0.61,
        "val/tier1_val_r2_perich_miller_co": 0.49,
        "val/tier2_val_r2": 0.40,
        "train_loss": 0.2,
        "train/grad_norm_total": 1.3,
    }


def test_val_aggregate_and_per_session():
    block = parse_decoding_block(
        _val_metrics(),
        session_ids=["perich_miller_population_2018/c_x", "odoherty_sabes/indy_y"],
        prefix="val",
    )
    assert block.aggregated["r2"].value == 0.55
    assert block.aggregated["r2"].value_range == (None, 1.0)
    assert block.per_session["perich_miller_population_2018/c_x"] == {"r2": 0.50}
    assert block.per_session["odoherty_sabes/indy_y"] == {"r2": 0.60}


def test_tiers_with_brainset_breakdown():
    block = parse_decoding_block(_val_metrics(), session_ids=[], prefix="val")
    tiers = block.groups["tiers"]
    assert tiers["tier1_val"]["r2"] == 0.58
    assert tiers["tier1_val"]["by_brainset"]["odoherty"] == 0.61
    assert tiers["tier1_val"]["by_brainset"]["perich_miller_co"] == 0.49
    assert tiers["tier2_val"]["r2"] == 0.40


def test_train_diagnostics_on_val_only():
    block = parse_decoding_block(_val_metrics(), session_ids=[], prefix="val")
    assert block.groups["train"] == {"loss": 0.2, "grad_norm_total": 1.3}


def test_test_phase_block():
    metrics = {
        "average_test_metric": 0.52,
        "perich_miller_population_2018/c_x": 0.51,
        "test/tier2_test_r2": 0.45,
        "train_loss": 0.2,  # must NOT appear in the test block
    }
    block = parse_decoding_block(metrics, session_ids=["perich_miller_population_2018/c_x"], prefix="test")
    assert block.aggregated["r2"].value == 0.52
    assert block.per_session["perich_miller_population_2018/c_x"] == {"r2": 0.51}
    assert block.groups["tiers"]["tier2_test"]["r2"] == 0.45
    assert "train" not in block.groups


def test_session_ids_inferred_when_absent():
    metrics = {"average_val_metric": 0.5, "bs/sess": 0.4, "val/tier1_val_r2": 0.5}
    block = parse_decoding_block(metrics, prefix="val")
    assert block.per_session == {"bs/sess": {"r2": 0.4}}


def test_tier_key_without_r2_suffix_is_skipped_and_empty_session_ids_yields_no_sessions():
    # A `<prefix>/foo` key with no `_r2` must not create a tier; an explicit empty
    # session_ids list must NOT trigger inference (yields empty per_session).
    metrics = {
        "average_val_metric": 0.5,
        "val/some_other_scalar": 0.9,  # no `_r2` -> not a tier
        "val/tier1_val_r2": 0.58,
        "bs/sess": 0.4,  # would be inferred, but session_ids=[] suppresses inference
    }
    block = parse_decoding_block(metrics, session_ids=[], prefix="val")
    assert block.per_session == {}
    assert "tiers" in block.groups
    assert set(block.groups["tiers"].keys()) == {"tier1_val"}


def test_aggregate_carries_median_std_n_across_sessions():
    metrics = {
        "average_val_metric": 0.50,
        "bs/a": 0.40,
        "bs/b": 0.50,
        "bs/c": 0.60,
    }
    block = parse_decoding_block(metrics, session_ids=["bs/a", "bs/b", "bs/c"], prefix="val")
    e = block.aggregated["r2"]
    assert e.value == 0.50  # authoritative logged mean, unchanged
    assert e.median == 0.50  # middle of [0.40, 0.50, 0.60]
    assert e.n == 3
    assert e.std is not None and e.std > 0
    assert e.aggregated_over == "sessions:mean"


def test_aggregate_single_session_degenerates():
    metrics = {"average_val_metric": 0.42, "bs/only": 0.42}
    block = parse_decoding_block(metrics, session_ids=["bs/only"], prefix="val")
    e = block.aggregated["r2"]
    assert e.value == 0.42
    assert e.median == 0.42
    assert e.n == 1
    assert e.std is None
    assert e.aggregated_over == "sessions:mean"


def test_aggregate_value_only_when_no_per_session():
    # avg present but no session values in hand -> no median/std/n.
    block = parse_decoding_block({"average_val_metric": 0.5}, session_ids=[], prefix="val")
    e = block.aggregated["r2"]
    assert e.value == 0.5
    assert e.median is None and e.std is None and e.n is None


def test_nonfinite_session_r2_excluded_from_aggregate():
    """A NaN/inf session must not crash the block (statistics.stdev raises on NaN) nor
    poison median/n: it stays as a null-able per-session row, excluded from the aggregate."""
    import math

    block = parse_decoding_block(
        {"a": 0.6, "b": float("nan"), "c": 0.8, "d": float("inf")}, session_ids=["a", "b", "c", "d"], prefix="test"
    )

    e = block.aggregated["r2"]
    assert e.n == 2 and math.isclose(e.median, 0.7) and math.isclose(e.value, 0.7) and e.std is not None
    assert block.per_session["b"]["r2"] is None and block.per_session["d"]["r2"] is None


def test_all_sessions_nonfinite_keeps_logged_average_only():
    block = parse_decoding_block({"average_test_metric": 0.5, "a": float("nan")}, session_ids=["a"], prefix="test")
    e = block.aggregated["r2"]
    assert e.value == 0.5 and e.n is None and e.median is None
