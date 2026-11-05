"""Composition test: encode with --tokenizer-ckpt composes for every recipe.

Encode recipes don't all declare `tokenizer.checkpoint_path` (the default
cowley_v4_base declares only `enabled: false`; full_grid_flow has no tokenizer
block at all), so the launcher must force-add with `++` or Hydra's struct mode
rejects the override. These tests actually drive `main([...])` through the
--print-resolved-config path, which subprocesses `python -m
ntt_downstream.encoding.train --cfg job` in the repo venv (heavy import but
GPU-free), and assert success + that the ckpt path lands in the resolved config.
"""

from __future__ import annotations

from ntt_downstream.cli.app import main

_CKPT = "/tmp/fake.ckpt"


def test_encode_composes_with_tokenizer_ckpt_default_entrypoint(capfd):
    """Default entrypoint (configs/cli/encode.yaml -> cowley_v4_base) composes."""
    rc = main(["encode", "--tokenizer-ckpt", _CKPT, "--print-resolved-config"])
    out = capfd.readouterr().out
    assert rc == 0, out
    assert _CKPT in out
    assert "enabled: true" in out


def test_encode_composes_with_tokenizer_ckpt_no_tokenizer_block_recipe(capfd):
    """cowley_v4_full_grid_flow has NO tokenizer block; `++` must create it."""
    rc = main(
        [
            "encode",
            "--config-name",
            "cowley_v4_full_grid_flow",
            "--tokenizer-ckpt",
            _CKPT,
            "--print-resolved-config",
        ]
    )
    out = capfd.readouterr().out
    assert rc == 0, out
    assert _CKPT in out
