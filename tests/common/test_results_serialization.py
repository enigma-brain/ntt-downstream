import json

import numpy as np
import pytest
import torch

from ntt_downstream.common.results.serialization import (
    json_default,
    replace_non_finite_floats,
    sanitize_keys,
    write_json,
)


def test_json_default_numpy_and_torch():
    assert json_default(np.int64(3)) == 3
    assert json_default(np.float32(1.5)) == 1.5
    assert json_default(np.array([1, 2])) == [1, 2]
    assert json_default(torch.tensor([1.0, 2.0])) == [1.0, 2.0]


def test_json_default_raises_on_unknown():
    with pytest.raises(TypeError):
        json_default(object())


def test_replace_non_finite_floats():
    out = replace_non_finite_floats({"a": float("nan"), "b": [float("inf"), 1.0], "c": (float("-inf"),)})
    assert out == {"a": None, "b": [None, 1.0], "c": (None,)}


def test_sanitize_keys_stringifies_tuple_keys():
    out = sanitize_keys({(1, 2): "x", "ok": 1})
    assert out == {"(1, 2)": "x", "ok": 1}


def test_write_json_is_rfc8259_conformant(tmp_path):
    path = write_json(
        tmp_path / "sub" / "results.json",
        {"v": float("nan"), "arr": np.array([1.0, 2.0]), "t": torch.tensor(0.5)},
    )
    text = path.read_text()
    assert "NaN" not in text and "Infinity" not in text

    def _reject_constant(token):  # json.loads accepts NaN/Infinity by default; forbid them
        raise AssertionError(f"non-finite JSON token: {token}")

    loaded = json.loads(text, parse_constant=_reject_constant)
    assert loaded["v"] is None
    assert loaded["arr"] == [1.0, 2.0]
    assert loaded["t"] == 0.5


def test_replace_non_finite_floats_only_handles_native_floats():
    # numpy/torch non-finite scalars are NOT python floats, so replace_non_finite_floats
    # passes them through unchanged. The results schema feeds native python floats, so
    # this boundary doesn't affect real output. This test pins the documented contract.
    import numpy as np

    out = replace_non_finite_floats({"np": np.float32("inf"), "py": float("inf")})
    assert out["py"] is None  # native python float -> nulled
    assert np.isinf(out["np"])  # numpy scalar passes through untouched


def test_json_default_omegaconf():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"a": 1, "b": [2, 3]})
    assert json_default(cfg) == {"a": 1, "b": [2, 3]}
    lst = OmegaConf.create([1, 2, 3])
    assert json_default(lst) == [1, 2, 3]


def test_sanitize_keys_recurses_into_lists():
    out = sanitize_keys([{(1, 2): "x"}, {"ok": [{(3, 4): "y"}]}])
    assert out == [{"(1, 2)": "x"}, {"ok": [{"(3, 4)": "y"}]}]
