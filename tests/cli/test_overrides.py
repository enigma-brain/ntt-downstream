import pytest

from ntt_downstream.cli import overrides
from ntt_downstream.cli.errors import ContractError


def test_decode_promoted_flags_map_to_keys():
    ov = overrides.promoted_to_hydra(
        "decode",
        {
            "tokenizer_kind": "frozen",
            "tokenizer_ckpt": "/m/x.ckpt",
            "data_root": "/data",
            "output_root": "/out",
            "seed": 7,
        },
    )
    assert "tokenizer=frozen" in ov
    assert "tokenizer.ntt_checkpoint=/m/x.ckpt" in ov
    assert "data_root=/data" in ov
    assert "log_dir=/out" in ov
    assert "seed=7" in ov


def test_encode_promoted_flags_map_to_keys():
    ov = overrides.promoted_to_hydra(
        "encode",
        {"tokenizer_ckpt": "/m/x.ckpt", "data_root": "/data", "output_root": "/out"},
    )
    # `++` (set-or-create): recipe schemas don't all declare these keys, so the
    # launcher must force-add them to survive Hydra struct mode.
    assert "++tokenizer.checkpoint_path=/m/x.ckpt" in ov
    assert "++tokenizer.enabled=true" in ov
    assert "data.dataset_path=/data" in ov
    assert "hydra.run.dir=/out" in ov


def test_encode_seed_maps_to_top_level_key():
    ov = overrides.promoted_to_hydra("encode", {"seed": 7})
    assert "seed=7" in ov


def test_decode_uri_passed_through_verbatim():
    ov = overrides.promoted_to_hydra("decode", {"tokenizer_ckpt": "mlflow-artifacts:/m/x"})
    assert "tokenizer.ntt_checkpoint=mlflow-artifacts:/m/x" in ov


def test_unset_flags_emit_nothing():
    ov = overrides.promoted_to_hydra("decode", {"data_root": "/d"})
    assert ov == ["data_root=/d"]


def test_unknown_verb_rejected():
    with pytest.raises(ContractError, match="verb"):
        overrides.promoted_to_hydra("nope", {})
