"""Tests for TokenizerWrapper.encode_windowed()."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from ntt_downstream.encoding.tokenizer_wrappers.encoding_wrapper import TokenizerWrapper


def _make_mock_tokenizer(
    *,
    compress_t: bool = True,
    compress_n: bool = False,
    n_neurons: int = 32,
    z_dim: int = 8,
    latent_dim: int = 8,
    window_length_seconds: float = 0.01,
    bin_size_ms: float = 1.0,
    token_bin_size_ms: float = 1.0,
    latent_type: str = "continuous",
) -> MagicMock:
    """Create a mock tokenizer with a spec that satisfies TokenizerWrapper init."""
    spec = SimpleNamespace(
        compress_n=compress_n,
        compress_t=compress_t,
        n_neurons=n_neurons,
        window_length_seconds=window_length_seconds,
        bin_size_ms=bin_size_ms,
        token_bin_size_ms=token_bin_size_ms,
        z_dim=z_dim,
        latent_dim=latent_dim,
        latent_type=latent_type,
        variable_n=False,
    )
    tok = MagicMock()
    tok.spec = spec
    tok.parameters.return_value = iter([])
    tok.buffers.return_value = iter([])
    return tok


def _make_wrapper(
    mock_tokenizer: MagicMock,
    sampling_rate: float = 1000.0,
    tokenizer_window_length_seconds: float | None = None,
) -> TokenizerWrapper:
    """Create a TokenizerWrapper around a mock tokenizer."""
    return TokenizerWrapper(
        tokenizer=mock_tokenizer,
        mode="frozen",
        sampling_rate=sampling_rate,
        reconstruction_rate_hz=sampling_rate,
        subject_id="test_subject",
        species="test_species",
        tokenizer_kind="test",
        discrete_target_mode="float_index",
        discrete_target_vocab_size=None,
        neuron_batch_size=32,
        latent_scale=1.0,
        tokenizer_window_length_seconds=tokenizer_window_length_seconds,
    )


def _patch_encode(
    wrapper: TokenizerWrapper,
    *,
    n_spatial: int,
    tokens_per_window: int,
    latent_dim: int,
) -> None:
    """Replace wrapper.encode with a deterministic mock.

    The mock returns (B, S, T_tok, D) tensors filled with ones, where B is
    the batch size of the input (which will be B_orig * W for windowed calls).
    """

    def mock_encode(responses: torch.Tensor, session_key: str, raw_discrete: bool = False, **kwargs) -> dict:
        B_eff = responses.shape[0]
        target = torch.ones(B_eff, n_spatial, tokens_per_window, latent_dim)
        return {
            "target": target,
            "n_spatial": n_spatial,
            "n_temporal": tokens_per_window,
            "aux_loss": None,
        }

    wrapper.encode = mock_encode  # type: ignore[assignment]


class TestEncodeWindowedCompressT:
    """Tests for encode_windowed with compress_t tokenizers."""

    def test_basic_shape(self):
        """(B=2, T=100, N=32) with 10-bin windows -> correct output shape."""
        B, T, N = 2, 100, 32
        S = N  # compress_t: spatial = neurons
        T_tok = 1  # 1 token per 10-bin window
        D = 8

        tok = _make_mock_tokenizer(compress_t=True, token_bin_size_ms=10.0)
        wrapper = _make_wrapper(tok, sampling_rate=1000.0, tokenizer_window_length_seconds=0.01)
        _patch_encode(wrapper, n_spatial=S, tokens_per_window=T_tok, latent_dim=D)

        responses = torch.randn(B, T, N)
        result = wrapper.encode_windowed(responses, "sess_0")

        assert result["target"].shape == (B, 10 * T_tok, S, D)
        assert result["n_windows"] == 10
        assert result["tokens_per_window"] == T_tok
        assert result["n_spatial"] == S
        assert result["n_temporal"] == 10 * T_tok

    def test_multi_token_windows(self):
        """Each window produces multiple latent tokens."""
        B, T, N = 4, 200, 16
        S = N
        T_tok = 5  # 5 tokens per window
        D = 4
        M = 20  # 20 bins per window at 1000 Hz = 0.02s

        tok = _make_mock_tokenizer(compress_t=True, token_bin_size_ms=4.0)
        wrapper = _make_wrapper(tok, sampling_rate=1000.0, tokenizer_window_length_seconds=0.02)
        _patch_encode(wrapper, n_spatial=S, tokens_per_window=T_tok, latent_dim=D)

        responses = torch.randn(B, T, N)
        result = wrapper.encode_windowed(responses, "sess_0")

        W = T // M  # 10 windows
        assert result["target"].shape == (B, W * T_tok, S, D)
        assert result["n_windows"] == W
        assert result["tokens_per_window"] == T_tok


class TestEncodeCompressTNativeWindowing:
    """Tests for image-path encode() with tokenizer windows shorter than responses."""

    def test_encode_splits_long_response_into_tokenizer_windows(self):
        B, T, N = 2, 200, 16
        window_bins = 100
        D = 20

        tok = _make_mock_tokenizer(
            compress_t=True,
            n_neurons=32,
            z_dim=D,
            latent_dim=D,
            window_length_seconds=0.1,
            token_bin_size_ms=100.0,
        )

        def mock_tokenizer_encode(inp):
            assert inp.spike_matrix.shape[1] == window_bins
            b_eff = inp.spike_matrix.shape[0]
            n_units = inp.spike_matrix.shape[2]
            return torch.ones(b_eff, 1, n_units, D)

        tok.encode.side_effect = mock_tokenizer_encode
        wrapper = _make_wrapper(tok, sampling_rate=1000.0, tokenizer_window_length_seconds=None)

        result = wrapper.encode(torch.randn(B, T, N), "sess_0")

        assert result["target"].shape == (B, N, 2, D)
        assert result["n_temporal"] == 2
        assert tok.encode.call_count == 1


class TestEncodeWindowedCompressN:
    """Tests for encode_windowed with compress_n tokenizers."""

    def test_basic_shape(self):
        """compress_n: spatial dim = num_chunks, not neurons."""
        B, T, N = 2, 100, 64
        num_chunks = 2  # e.g., 64 neurons / 32 per chunk
        T_tok = 1
        D = 8

        tok = _make_mock_tokenizer(compress_n=True, compress_t=False, token_bin_size_ms=10.0)
        wrapper = _make_wrapper(tok, sampling_rate=1000.0, tokenizer_window_length_seconds=0.01)
        _patch_encode(wrapper, n_spatial=num_chunks, tokens_per_window=T_tok, latent_dim=D)

        responses = torch.randn(B, T, N)
        result = wrapper.encode_windowed(responses, "sess_0")

        W = 10
        assert result["target"].shape == (B, W * T_tok, num_chunks, D)
        assert result["n_windows"] == W
        assert result["n_spatial"] == num_chunks


class TestTruncation:
    """Tests that responses not divisible by window size are truncated."""

    def test_truncation(self):
        """T=105 with M=10 -> only 100 bins used, n_windows=10."""
        B, T, N = 2, 105, 32
        S = N
        T_tok = 1
        D = 8

        tok = _make_mock_tokenizer(compress_t=True, token_bin_size_ms=10.0)
        wrapper = _make_wrapper(tok, sampling_rate=1000.0, tokenizer_window_length_seconds=0.01)

        # Track what gets passed to encode
        encode_calls = []

        def mock_encode(responses: torch.Tensor, session_key: str, raw_discrete: bool = False, **kwargs) -> dict:
            encode_calls.append(responses.shape)
            B_eff = responses.shape[0]
            return {
                "target": torch.ones(B_eff, S, T_tok, D),
                "n_spatial": S,
                "n_temporal": T_tok,
                "aux_loss": None,
            }

        wrapper.encode = mock_encode  # type: ignore[assignment]

        responses = torch.randn(B, T, N)
        result = wrapper.encode_windowed(responses, "sess_0")

        assert result["n_windows"] == 10
        # encode should have been called with B*W=20 samples, each of length M=10
        assert encode_calls[0] == (20, 10, N)


class TestDefaultWindowFromSpec:
    """Test that latent_window_length_seconds defaults from spec."""

    def test_default_from_spec(self):
        """When not set explicitly, picks up spec.window_length_seconds."""
        tok = _make_mock_tokenizer(compress_t=True, window_length_seconds=0.05, token_bin_size_ms=50.0)
        wrapper = _make_wrapper(tok, sampling_rate=1000.0, tokenizer_window_length_seconds=None)

        assert wrapper.tokenizer_window_length_seconds == 0.05

    def test_explicit_overrides_spec(self):
        """Explicit value overrides spec default."""
        tok = _make_mock_tokenizer(compress_t=True, window_length_seconds=0.05, token_bin_size_ms=10.0)
        wrapper = _make_wrapper(tok, sampling_rate=1000.0, tokenizer_window_length_seconds=0.02)

        assert wrapper.tokenizer_window_length_seconds == 0.02

    def test_dist_window_in_spec_leaves_wrapper_unset(self):
        """Tokenizers trained with a sampling Dist for window_length_seconds must
        not crash __post_init__ (``float(UniformDist)`` would raise TypeError).
        Runtime per-batch override handles the actual value.
        """
        from ntt.core.components.dist import UniformDist

        tok = _make_mock_tokenizer(compress_t=True, token_bin_size_ms=10.0)
        tok.spec.window_length_seconds = UniformDist(low=0.05, high=0.2)
        wrapper = _make_wrapper(tok, sampling_rate=1000.0, tokenizer_window_length_seconds=None)

        assert wrapper.tokenizer_window_length_seconds is None

    def test_dist_window_not_seeded_from_per_batch_spec_mutation(self):
        """The window default is read once, at init. Later per-batch mutation of
        ``spec.window_length_seconds`` (the discrete-metadata probe, every
        encode) must not seed a variable-window tokenizer's window — that would
        chunk responses into probe-length windows in ``_encode_compress_t``.
        """
        from ntt.core.components.dist import UniformDist

        tok = _make_mock_tokenizer(compress_t=True, token_bin_size_ms=10.0)
        tok.spec.window_length_seconds = UniformDist(low=0.05, high=0.2)
        wrapper = _make_wrapper(tok, sampling_rate=1000.0, tokenizer_window_length_seconds=None)

        wrapper._set_window_length_seconds_for_current_batch(10)
        wrapper._align_tokenizer_device(torch.device("cpu"))

        assert wrapper.tokenizer_window_length_seconds is None


class TestOutputKeys:
    """Test that encode_windowed returns all expected keys."""

    def test_returns_expected_keys(self):
        B, T, N = 1, 50, 8
        tok = _make_mock_tokenizer(compress_t=True, token_bin_size_ms=10.0)
        wrapper = _make_wrapper(tok, sampling_rate=1000.0, tokenizer_window_length_seconds=0.01)
        _patch_encode(wrapper, n_spatial=N, tokens_per_window=1, latent_dim=8)

        result = wrapper.encode_windowed(torch.randn(B, T, N), "sess_0")
        expected_keys = {"target", "n_windows", "tokens_per_window", "n_spatial", "n_temporal"}
        assert set(result.keys()) == expected_keys


class TestValidation:
    """Tests for __post_init__ validation of latent_window_length_seconds."""

    def test_rejects_non_multiple(self):
        """Non-multiple of token_bin_size_ms should raise ValueError."""
        tok = _make_mock_tokenizer(compress_t=True, token_bin_size_ms=10.0)
        with pytest.raises(ValueError, match="must be a multiple of token_bin_size_ms"):
            _make_wrapper(tok, sampling_rate=1000.0, tokenizer_window_length_seconds=0.015)

    def test_accepts_valid_multiple(self):
        """Valid multiples should not raise."""
        tok = _make_mock_tokenizer(compress_t=True, token_bin_size_ms=10.0)
        wrapper = _make_wrapper(tok, sampling_rate=1000.0, tokenizer_window_length_seconds=0.03)
        assert wrapper.tokenizer_window_length_seconds == 0.03

    def test_accepts_exact_match(self):
        """Exact match (1x multiple) should not raise."""
        tok = _make_mock_tokenizer(compress_t=True, token_bin_size_ms=10.0)
        wrapper = _make_wrapper(tok, sampling_rate=1000.0, tokenizer_window_length_seconds=0.01)
        assert wrapper.tokenizer_window_length_seconds == 0.01

    def test_no_validation_without_token_bin_size(self):
        """When spec has no token_bin_size_ms, any value is accepted."""
        tok = _make_mock_tokenizer(compress_t=True)
        tok.spec.token_bin_size_ms = None
        wrapper = _make_wrapper(tok, sampling_rate=1000.0, tokenizer_window_length_seconds=0.0137)
        assert wrapper.tokenizer_window_length_seconds == 0.0137


class TestErrorCases:
    """Test error conditions for encode_windowed."""

    def test_raises_if_window_not_set(self):
        """Should raise if tokenizer_window_length_seconds is None and spec has no default."""
        tok = _make_mock_tokenizer(compress_t=True)
        tok.spec.window_length_seconds = None
        wrapper = _make_wrapper(tok, sampling_rate=1000.0, tokenizer_window_length_seconds=None)

        # Manually ensure it's None (spec didn't provide a default)
        assert wrapper.tokenizer_window_length_seconds is None

        with pytest.raises(ValueError, match="tokenizer_window_length_seconds is not set"):
            wrapper.encode_windowed(torch.randn(1, 100, 8), "sess_0")

    def test_raises_if_response_shorter_than_window(self):
        """Should raise if T < M (response shorter than one window)."""
        tok = _make_mock_tokenizer(compress_t=True, token_bin_size_ms=100.0)
        wrapper = _make_wrapper(tok, sampling_rate=1000.0, tokenizer_window_length_seconds=0.1)

        with pytest.raises(ValueError, match="shorter than one window"):
            wrapper.encode_windowed(torch.randn(1, 50, 8), "sess_0")
