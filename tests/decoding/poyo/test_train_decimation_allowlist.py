"""Verify `train_decimation.brainsets` gates decimation per child dataset.

`train_decimation` is forwarded only to child datasets built by
`build_multi_brainset_co_dataset` whose brainset is in the allowlist;
otherwise multi-brainset runs that set `train_decimation.trials=N` would
silently decimate brainsets (Churchland / Odoherty) the calibration sweeps
don't intend to touch.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from ntt_downstream.decoding.backends.poyo.train_decimation import (
    TrainDecimationConfig,
    applies_to,
)


class TestFromCfg:
    def test_brainsets_default_none(self):
        node = OmegaConf.create({"trials": 4, "seed": 42})
        cfg = TrainDecimationConfig.from_cfg(node)
        assert cfg is not None
        assert cfg.trials == 4
        assert cfg.brainsets is None

    def test_brainsets_list(self):
        node = OmegaConf.create(
            {
                "trials": 4,
                "seed": 42,
                "brainsets": ["perich_miller_population_2018"],
            }
        )
        cfg = TrainDecimationConfig.from_cfg(node)
        assert cfg is not None
        assert cfg.brainsets == frozenset({"perich_miller_population_2018"})

    def test_brainsets_empty_list_is_none(self):
        # Empty list = "no allowlist" so users can clear without dropping the key.
        node = OmegaConf.create({"trials": 4, "seed": 42, "brainsets": []})
        cfg = TrainDecimationConfig.from_cfg(node)
        assert cfg is not None
        assert cfg.brainsets is None


class TestAppliesTo:
    def test_no_cfg(self):
        assert applies_to("perich_miller_population_2018", None) is False

    def test_no_allowlist_means_apply_everywhere(self):
        cfg = TrainDecimationConfig(trials=4, brainsets=None)
        assert applies_to("perich_miller_population_2018", cfg) is True
        assert applies_to("churchland_shenoy_neural_2012", cfg) is True
        assert applies_to("odoherty_sabes_nonhuman_2017", cfg) is True

    def test_allowlist_restricts(self):
        cfg = TrainDecimationConfig(
            trials=4,
            brainsets=frozenset({"perich_miller_population_2018"}),
        )
        assert applies_to("perich_miller_population_2018", cfg) is True
        assert applies_to("churchland_shenoy_neural_2012", cfg) is False
        assert applies_to("odoherty_sabes_nonhuman_2017", cfg) is False


def test_multi_brainset_call_site_uses_applies_to():
    """The call site inside `build_multi_brainset_co_dataset` must gate
    `train_decimation` with the allowlist helper. This catches
    the gate being dropped during a refactor.
    """
    import inspect

    from ntt_downstream.decoding.backends.poyo import multi_brainset_dataset

    src = inspect.getsource(multi_brainset_dataset.build_multi_brainset_co_dataset)
    # The function imports `applies_to` as `_decim_applies_to`; assert
    # both the import and the call land in the function body so future
    # edits trip the test rather than silently un-gating decimation.
    assert "_decim_applies_to(brainset, train_decimation)" in src, (
        "train_decimation must be gated by the brainset allowlist at the child-dataset construction site."
    )
