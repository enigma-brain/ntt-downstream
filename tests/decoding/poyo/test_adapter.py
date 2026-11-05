"""Core tests for the adapter: unit-embedding modes + sentinels."""

import pytest
import torch
from torch import nn

from ntt_downstream.decoding.backends.poyo.adapter import PreEmbeddedTokens, adapt
from ntt_downstream.decoding.backends.poyo.tokenizer_output import TokenizerOutput


def _make_tok_out(B=2, N=5, D=8, combines_units=False, with_unit_ids=True):
    """Helper: build a synthetic TokenizerOutput."""
    tokens = torch.randn(B, N, D)
    timestamps = torch.linspace(0.0, 1.0, N).unsqueeze(0).expand(B, -1)
    unit_ids = torch.randint(0, 10, (B, N)) if with_unit_ids else None
    mask = torch.ones(B, N, dtype=torch.bool)
    return TokenizerOutput(
        tokens=tokens,
        timestamps=timestamps,
        unit_ids=unit_ids,
        mask=mask,
        combines_units=combines_units,
        token_bin_size_s=0.02,
    )


def _make_unit_emb(D_unit=8, n_vocab=20):
    """Simple embedding standing in for POYO's InfiniteVocabEmbedding."""
    return nn.Embedding(n_vocab, D_unit)


def _make_token_type_emb(D_type=8, n_types=4):
    return nn.Embedding(n_types, D_type)


class TestAdaptModeNone:
    def test_passthrough_shape(self):
        D_tok, target_dim = 8, 16
        tok = _make_tok_out(B=2, N=5, D=D_tok)
        # D_tok != target_dim → adapter requires an input_proj for mode='none'.
        input_proj = nn.Linear(D_tok, target_dim)
        out = adapt(
            tok,
            _make_unit_emb(target_dim),
            _make_token_type_emb(target_dim),
            "none",
            sentinels=False,
            target_dim=target_dim,
            input_proj=input_proj,
        )
        assert isinstance(out, PreEmbeddedTokens)
        assert out.embeddings.shape == (2, 5, target_dim)
        assert out.timestamps.shape == (2, 5)
        assert out.mask.shape == (2, 5)

    def test_population_tokenizer_without_unit_ids_rejects_add(self):
        # Population tokenizers (combines_units=True) MAY use mode='add' when
        # the adapter receives per-slot proxy unit_ids (the merged
        # abstract-latent path). Without unit_ids, however, the unit_emb
        # lookup is undefined and the adapter must still reject.
        tok = _make_tok_out(combines_units=True, with_unit_ids=False)
        with pytest.raises(ValueError, match=r"mode='add' requires per-token unit_ids"):
            adapt(tok, _make_unit_emb(), _make_token_type_emb(), "add", sentinels=False, target_dim=16)


class TestAdaptModeAdd:
    def test_output_shape(self):
        D_tok, target_dim = 8, 16
        tok = _make_tok_out(B=2, N=5, D=D_tok)
        # mode='add' with D_tok != target_dim requires input_proj.
        input_proj = nn.Linear(D_tok, target_dim)
        out = adapt(
            tok,
            _make_unit_emb(target_dim),
            _make_token_type_emb(target_dim),
            "add",
            sentinels=False,
            target_dim=target_dim,
            input_proj=input_proj,
        )
        assert out.embeddings.shape == (2, 5, target_dim)

    def test_unit_ids_required(self):
        tok = _make_tok_out(with_unit_ids=False, combines_units=False)
        with pytest.raises(ValueError, match="unit_ids"):
            adapt(tok, _make_unit_emb(), _make_token_type_emb(), "add", sentinels=False, target_dim=16)


class TestAdaptModeConcat:
    def test_output_shape(self):
        D_tok, target_dim = 8, 14
        tok = _make_tok_out(B=2, N=5, D=D_tok)
        # mode='concat' always requires input_proj (D_tok + target_dim → target_dim).
        input_proj = nn.Linear(D_tok + target_dim, target_dim)
        out = adapt(
            tok,
            _make_unit_emb(target_dim),
            _make_token_type_emb(target_dim),
            "concat",
            sentinels=False,
            target_dim=target_dim,
            input_proj=input_proj,
        )
        assert out.embeddings.shape == (2, 5, target_dim)


class TestSentinels:
    def test_sentinels_prepended(self):
        B, N, D_tok, D_unit, target_dim = 1, 5, 8, 8, 8
        tok = _make_tok_out(B=B, N=N, D=D_tok)
        # Force all unit_ids to be in {0, 1, 2} for predictable sentinel count.
        tok.unit_ids = torch.tensor([[0, 1, 2, 0, 1]])
        out = adapt(
            tok,
            _make_unit_emb(D_unit),
            _make_token_type_emb(D_type=D_unit),
            "none",
            sentinels=True,
            target_dim=target_dim,
        )
        # 3 unique units -> 3 sentinels + 5 content = 8 total.
        assert out.embeddings.shape[1] == 3 + N
        assert out.mask[:, :3].all()  # sentinel mask is True

    def test_sentinel_timestamps_use_window_start(self):
        """Sentinel timestamps should be window_start, not min spike time."""
        B, N, D_tok, D_unit, target_dim = 1, 5, 8, 8, 8
        tok = _make_tok_out(B=B, N=N, D=D_tok)
        tok.unit_ids = torch.tensor([[0, 1, 2, 0, 1]])
        # Timestamps start at 0.2, so min spike != 0.0
        tok.timestamps = torch.tensor([[0.2, 0.4, 0.6, 0.8, 1.0]])

        out = adapt(
            tok,
            _make_unit_emb(D_unit),
            _make_token_type_emb(D_type=D_unit),
            "none",
            sentinels=True,
            target_dim=target_dim,
            window_start=0.0,
        )
        # Sentinel timestamps (first 3) should all be 0.0, not 0.2.
        sentinel_ts = out.timestamps[0, :3]
        assert torch.allclose(sentinel_ts, torch.zeros(3)), (
            f"Expected sentinel timestamps=0.0 (window_start), got {sentinel_ts}"
        )

    def test_sentinel_type_is_start_of_sequence(self):
        """Sentinels should use token_type=1 (START_OF_SEQUENCE), not 0 (DEFAULT)."""
        B, N, D_tok, D_unit, target_dim = 1, 3, 8, 8, 8
        tok = _make_tok_out(B=B, N=N, D=D_tok)
        tok.unit_ids = torch.tensor([[0, 1, 0]])

        unit_emb = _make_unit_emb(D_unit)
        type_emb = _make_token_type_emb(D_type=D_unit)

        out = adapt(tok, unit_emb, type_emb, "none", sentinels=True, target_dim=target_dim)

        # Manually compute expected sentinel embedding with type=1 (START).
        with torch.no_grad():
            uids = torch.tensor([0, 1])
            expected = unit_emb(uids) + type_emb(torch.ones(2, dtype=torch.long))

        sentinel_emb = out.embeddings[0, :2, :D_unit]
        assert torch.allclose(sentinel_emb, expected, atol=1e-5), (
            "Sentinel embeddings should use token_type_emb(1), not token_type_emb(0)"
        )

    def test_no_sentinels_when_population(self):
        D_tok, target_dim = 8, 16
        tok = _make_tok_out(D=D_tok, combines_units=True, with_unit_ids=False)
        input_proj = nn.Linear(D_tok, target_dim)
        out = adapt(
            tok,
            _make_unit_emb(target_dim),
            _make_token_type_emb(target_dim),
            "none",
            sentinels=True,
            target_dim=target_dim,
            input_proj=input_proj,
        )
        # No unit_ids -> no sentinels added, sequence length unchanged.
        assert out.embeddings.shape[1] == tok.tokens.shape[1]


class TestDiscreteAdapter:
    """Int-dtype branch of adapt(): code_emb lookup for discrete tokenizers."""

    def _make_int_tok_out(self, B=2, N=10, Q=4, K=1024) -> TokenizerOutput:
        return TokenizerOutput(
            tokens=torch.randint(0, K, (B, N, Q), dtype=torch.int64),
            timestamps=torch.linspace(0.0, 1.0, N).unsqueeze(0).expand(B, N).contiguous(),
            unit_ids=torch.randint(0, 32, (B, N), dtype=torch.long),
            mask=torch.ones(B, N, dtype=torch.bool),
            combines_units=False,
            token_bin_size_s=0.1,
            vocab_sizes=(K,) * Q,
        )

    def _make_emb(self, K, Q, D) -> nn.ModuleList:
        return nn.ModuleList([nn.Embedding(K, D) for _ in range(Q)])

    def test_int_branch_shape_and_dtype(self):
        D = 64
        Q = 4
        K = 1024
        tok_out = self._make_int_tok_out(B=2, N=10, Q=Q, K=K)
        unit_emb = nn.Embedding(64, D)
        token_type_emb = nn.Embedding(8, D)
        code_emb = self._make_emb(K, Q, D)

        out = adapt(
            tok_out,
            unit_emb=unit_emb,
            token_type_emb=token_type_emb,
            mode="add",
            sentinels=False,
            target_dim=D,
            input_proj=None,
            code_emb=code_emb,
        )
        assert out.embeddings.shape == (2, 10, D)
        assert torch.is_floating_point(out.embeddings)
        assert out.timestamps.shape == (2, 10)
        assert out.mask.shape == (2, 10)

    def test_int_branch_gradients_flow_to_code_emb_and_unit_emb(self):
        D = 64
        Q = 4
        K = 1024
        tok_out = self._make_int_tok_out(B=2, N=10, Q=Q, K=K)
        unit_emb = nn.Embedding(64, D)
        token_type_emb = nn.Embedding(8, D)
        code_emb = self._make_emb(K, Q, D)

        out = adapt(
            tok_out,
            unit_emb=unit_emb,
            token_type_emb=token_type_emb,
            mode="add",
            sentinels=False,
            target_dim=D,
            input_proj=None,
            code_emb=code_emb,
        )
        out.embeddings.sum().backward()
        assert unit_emb.weight.grad is not None
        for q in range(Q):
            assert code_emb[q].weight.grad is not None

    def test_int_branch_requires_code_emb(self):
        D = 64
        tok_out = self._make_int_tok_out(B=1, N=4, Q=2, K=16)
        unit_emb = nn.Embedding(32, D)
        token_type_emb = nn.Embedding(8, D)
        with pytest.raises(ValueError, match=r"code_emb"):
            adapt(
                tok_out,
                unit_emb=unit_emb,
                token_type_emb=token_type_emb,
                mode="add",
                sentinels=False,
                target_dim=D,
                input_proj=None,
                code_emb=None,
            )

    def test_int_branch_q_mismatch_raises(self):
        D = 64
        Q = 4
        tok_out = self._make_int_tok_out(B=1, N=4, Q=Q, K=16)
        unit_emb = nn.Embedding(32, D)
        token_type_emb = nn.Embedding(8, D)
        wrong_emb = nn.ModuleList([nn.Embedding(16, D) for _ in range(Q - 1)])
        with pytest.raises(ValueError, match=r"codebooks"):
            adapt(
                tok_out,
                unit_emb=unit_emb,
                token_type_emb=token_type_emb,
                mode="add",
                sentinels=False,
                target_dim=D,
                input_proj=None,
                code_emb=wrong_emb,
            )

    def test_int_branch_concat_mode_raises(self):
        D = 64
        Q = 4
        K = 1024
        tok_out = self._make_int_tok_out(B=1, N=4, Q=Q, K=K)
        unit_emb = nn.Embedding(32, D)
        token_type_emb = nn.Embedding(8, D)
        code_emb = self._make_emb(K, Q, D)
        proj = nn.Linear(D + D, D)
        with pytest.raises(ValueError, match=r"concat"):
            adapt(
                tok_out,
                unit_emb=unit_emb,
                token_type_emb=token_type_emb,
                mode="concat",
                sentinels=False,
                target_dim=D,
                input_proj=proj,
                code_emb=code_emb,
            )
