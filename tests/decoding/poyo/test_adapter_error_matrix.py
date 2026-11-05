"""Error-ordering matrix for ``adapter.adapt``.

The ``mode in {add, concat} -> needs unit_ids`` guard sits *after* the
discrete/float fork, so each path reports its own most specific error for
degenerate discrete misconfigs. This test pins raise/no-raise AND message text
across the discrete/float x {none, add, concat} x {unit_ids, None} matrix, plus
the ``code_emb=None`` precedence, so the ordering cannot silently change.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from ntt_downstream.decoding.backends.poyo.adapter import PreEmbeddedTokens, adapt
from ntt_downstream.decoding.backends.poyo.tokenizer_output import TokenizerOutput

TARGET = 16
B, N, Q, K = 2, 4, 2, 32


def _ts_mask(B=B, N=N):
    ts = torch.linspace(0.0, 1.0, N).unsqueeze(0).expand(B, -1).contiguous()
    return ts, torch.ones(B, N, dtype=torch.bool)


def _float_tok(*, with_unit_ids: bool) -> TokenizerOutput:
    ts, mask = _ts_mask()
    return TokenizerOutput(
        tokens=torch.randn(B, N, TARGET),  # D_tok == target_dim
        timestamps=ts,
        unit_ids=torch.randint(0, 10, (B, N)) if with_unit_ids else None,
        mask=mask,
        combines_units=False,
        token_bin_size_s=0.02,
    )


def _discrete_tok(*, with_unit_ids: bool) -> TokenizerOutput:
    ts, mask = _ts_mask()
    return TokenizerOutput(
        tokens=torch.randint(0, K, (B, N, Q), dtype=torch.int64),
        timestamps=ts,
        unit_ids=torch.randint(0, 10, (B, N)) if with_unit_ids else None,
        mask=mask,
        combines_units=False,
        token_bin_size_s=0.1,
        vocab_sizes=(K,) * Q,
    )


def _unit_emb():
    return nn.Embedding(20, TARGET)


def _type_emb():
    return nn.Embedding(4, TARGET)


def _code_emb():
    return nn.ModuleList([nn.Embedding(K, TARGET) for _ in range(Q)])


def _adapt(tok, mode, *, code_emb=None, input_proj=None):
    return adapt(
        tok,
        _unit_emb(),
        _type_emb(),
        mode,
        sentinels=False,
        target_dim=TARGET,
        input_proj=input_proj,
        code_emb=code_emb,
    )


# --- float path ---


@pytest.mark.parametrize("mode", ["none", "add"])
def test_float_with_unit_ids_ok(mode):
    out = _adapt(_float_tok(with_unit_ids=True), mode)
    assert isinstance(out, PreEmbeddedTokens)
    assert out.embeddings.shape == (B, N, TARGET)


def test_float_none_without_unit_ids_ok():
    # mode='none' never needs unit_ids.
    out = _adapt(_float_tok(with_unit_ids=False), "none")
    assert out.embeddings.shape == (B, N, TARGET)


def test_float_add_without_unit_ids_raises_unit_ids():
    with pytest.raises(ValueError, match=r"mode='add' requires per-token unit_ids"):
        _adapt(_float_tok(with_unit_ids=False), "add")


def test_float_concat_without_unit_ids_raises_unit_ids():
    with pytest.raises(ValueError, match=r"mode='concat' requires per-token unit_ids"):
        _adapt(_float_tok(with_unit_ids=False), "concat")


def test_float_concat_with_unit_ids_and_input_proj_ok():
    proj = nn.Linear(TARGET + TARGET, TARGET)
    out = _adapt(_float_tok(with_unit_ids=True), "concat", input_proj=proj)
    assert out.embeddings.shape == (B, N, TARGET)


# --- discrete path ---


@pytest.mark.parametrize("with_uids", [True, False])
def test_discrete_none_ok_regardless_of_unit_ids(with_uids):
    out = _adapt(_discrete_tok(with_unit_ids=with_uids), "none", code_emb=_code_emb())
    assert out.embeddings.shape == (B, N, TARGET)


def test_discrete_add_with_unit_ids_ok():
    out = _adapt(_discrete_tok(with_unit_ids=True), "add", code_emb=_code_emb())
    assert out.embeddings.shape == (B, N, TARGET)


def test_discrete_add_without_unit_ids_raises_unit_ids():
    with pytest.raises(ValueError, match=r"mode='add' requires per-token unit_ids"):
        _adapt(_discrete_tok(with_unit_ids=False), "add", code_emb=_code_emb())


@pytest.mark.parametrize("with_uids", [True, False])
def test_discrete_concat_reports_concat_unsupported_not_unit_ids(with_uids):
    # Key assertion: a discrete concat misconfig must report the accurate
    # "concat not supported with discrete" error, NOT a misleading unit_ids one
    # — even when unit_ids is also None.
    with pytest.raises(ValueError, match=r"concat' is not supported with discrete-input"):
        _adapt(_discrete_tok(with_unit_ids=with_uids), "concat", code_emb=_code_emb())


def test_discrete_missing_code_emb_takes_precedence_over_unit_ids():
    # Key assertion: the code_emb check fires before the add-block unit_ids
    # check, so a doubly-degenerate discrete+add config reports code_emb first.
    with pytest.raises(ValueError, match=r"int-dtype tokens require code_emb"):
        _adapt(_discrete_tok(with_unit_ids=False), "add", code_emb=None)
