from ntt_downstream.encoding.training.results_parser import parse_encoding_block


def _image_metrics():
    return {
        "val/rate_corr/overall": 0.42,
        "val/rate_corr/bs/sessA": 0.40,
        "val/rate_corr/bs/sessB": 0.44,
        "val/corr/overall": 0.31,
        "val/corr/bs/sessA": 0.29,
        "val_nc_r2/overall": 0.55,
        "val_nc_r2/median": 0.58,
        "val_nc_r2/bs/sessA": 0.50,
        "val/rate_corr_trial_avg/overall": 0.60,
        "val/corr_trial_avg/overall": 0.50,
        "val/latent_cosine_sim": 0.7,
        "val/latent_r2": 0.65,
        "val/rate_corr_subject/subjX": 0.41,
        "val/corr_subject/subjX": 0.30,
        "train/rate_corr/overall": 0.7,
        "train/corr/overall": 0.6,
        "train/loss": 0.12,
    }


def test_aggregates_with_ranges_and_median():
    block = parse_encoding_block(
        _image_metrics(),
        session_info={"bs/sessA": 1, "bs/sessB": 1},
    )
    agg = block.aggregated
    assert agg["rate_corr"].value == 0.42
    assert agg["rate_corr"].value_range == (-1.0, 1.0)
    assert agg["temporal_corr"].value == 0.31
    assert agg["nc_r2"].value == 0.55
    assert agg["nc_r2"].median == 0.58
    assert agg["nc_r2"].value_range == (None, 1.0)
    assert agg["rate_corr_trial_avg"].value == 0.60
    assert agg["temporal_corr_trial_avg"].value == 0.50
    assert agg["latent_cosine_sim"].value == 0.7
    assert agg["latent_r2"].value == 0.65


def test_per_session_and_subject_and_train():
    block = parse_encoding_block(
        _image_metrics(),
        session_info={"bs/sessA": 1, "bs/sessB": 1},
    )
    assert block.per_session["bs/sessA"] == {"rate_corr": 0.40, "temporal_corr": 0.29, "nc_r2": 0.50}
    assert block.per_session["bs/sessB"] == {"rate_corr": 0.44}
    assert block.groups["per_subject"]["subjX"] == {"rate_corr": 0.41, "temporal_corr": 0.30}
    assert block.groups["train"] == {"rate_corr": 0.7, "temporal_corr": 0.6, "loss": 0.12}


def test_audio_style_missing_groups_are_omitted():
    # Audio module logs no nc_r2 / latent / train-set pass.
    metrics = {
        "val/rate_corr/overall": 0.3,
        "val/rate_corr/bs/sessA": 0.3,
        "val/corr/overall": 0.2,
    }
    block = parse_encoding_block(metrics, session_info={"bs/sessA": 1})
    assert "nc_r2" not in block.aggregated
    assert "latent_r2" not in block.aggregated
    assert "train" not in block.groups
    assert "per_subject" not in block.groups
    assert block.per_session["bs/sessA"] == {"rate_corr": 0.3}


def test_sessions_inferred_when_session_info_absent():
    metrics = {"val/rate_corr/overall": 0.3, "val/rate_corr/bs/sessA": 0.31}
    block = parse_encoding_block(metrics)
    assert block.per_session == {"bs/sessA": {"rate_corr": 0.31}}


def test_no_prefix_leakage_into_sessions():
    # trial_avg / subject keys must NOT be misclassified as per-session entries,
    # and overall must be excluded. This pins the most correctness-critical property.
    metrics = {
        "val/rate_corr/overall": 0.4,
        "val/rate_corr/bs/sessA": 0.41,
        "val/rate_corr_trial_avg/overall": 0.6,
        "val/rate_corr_subject/subjX": 0.42,
        "val/corr_trial_avg/overall": 0.5,
        "val/corr_subject/subjX": 0.31,
    }
    block = parse_encoding_block(metrics)  # sessions inferred (no session_info)
    assert set(block.per_session.keys()) == {"bs/sessA"}
    assert block.per_session["bs/sessA"] == {"rate_corr": 0.41}
    assert "rate_corr_trial_avg" in block.aggregated
    assert "temporal_corr_trial_avg" in block.aggregated
    assert block.groups["per_subject"]["subjX"] == {"rate_corr": 0.42, "temporal_corr": 0.31}


def test_session_in_info_absent_from_metrics_is_omitted():
    # A session present in session_info but with no logged metrics must be omitted
    # from per_session (no empty/ghost entries).
    metrics = {"val/rate_corr/overall": 0.4, "val/rate_corr/bs/sessA": 0.41}
    block = parse_encoding_block(metrics, session_info={"bs/sessA": 1, "bs/ghost": 1})
    assert "bs/ghost" not in block.per_session
    assert block.per_session["bs/sessA"] == {"rate_corr": 0.41}
