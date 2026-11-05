"""Tests for evaluation metrics: temporal correlation, cross-image rate correlation, and noise-corrected R².

Each test constructs synthetic data with analytically known expected values so that
errors in the metric computation logic are caught immediately.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ntt_downstream.encoding.models.enc.hiera.evals import (
    OnlinePearsonAccumulator,
    TemporalCorrAccumulator,
    batch_temporal_correlation,
)
from ntt_downstream.encoding.models.enc.hiera.noise_corrected_r2 import noise_corrected_r2

# ---------------------------------------------------------------------------
# batch_temporal_correlation  (corr_stop3d-style, T > 1 path)
# ---------------------------------------------------------------------------


class TestBatchTemporalCorrelation:
    """Tests for batch_temporal_correlation: per-image Pearson-r along time axis."""

    def test_perfect_positive_correlation(self):
        """pred == target → r = 1.0 for every (image, neuron)."""
        pred = torch.randn(5, 40, 8)
        corr = batch_temporal_correlation(pred, pred)
        assert corr.shape == (5, 8)
        torch.testing.assert_close(corr, torch.ones_like(corr), atol=1e-5, rtol=0)

    def test_perfect_negative_correlation(self):
        """pred == -target → r = -1.0."""
        pred = torch.randn(3, 30, 4)
        corr = batch_temporal_correlation(pred, -pred)
        torch.testing.assert_close(corr, -torch.ones_like(corr), atol=1e-5, rtol=0)

    def test_known_correlation_value(self):
        """Construct a pair with analytically known r ≈ 0.8.

        target = pred + noise, where noise is independent.
        r = std(pred) / sqrt(std(pred)^2 + std(noise)^2)
        With signal_std=2.0, noise_std=1.5:
        r = 2.0 / sqrt(4.0 + 2.25) = 2.0 / 2.5 = 0.8
        """
        torch.manual_seed(42)
        B, T, N = 100, 500, 10  # large T for tight empirical estimate
        signal = torch.randn(B, T, N) * 2.0
        noise = torch.randn(B, T, N) * 1.5
        target = signal + noise
        expected_r = 2.0 / (4.0 + 2.25) ** 0.5  # 0.8

        corr = batch_temporal_correlation(signal, target)
        mean_corr = corr.mean().item()
        assert abs(mean_corr - expected_r) < 0.05, f"Expected ~{expected_r:.2f}, got {mean_corr:.3f}"

    def test_zero_correlation(self):
        """Independent signals → r ≈ 0."""
        torch.manual_seed(0)
        pred = torch.randn(50, 200, 6)
        target = torch.randn(50, 200, 6)
        corr = batch_temporal_correlation(pred, target)
        mean_corr = corr.mean().item()
        assert abs(mean_corr) < 0.1, f"Expected ~0, got {mean_corr:.3f}"

    def test_constant_signal_returns_nan(self):
        """Constant pred or target → NaN (undefined correlation)."""
        const = torch.ones(2, 20, 3)
        varying = torch.randn(2, 20, 3)

        corr1 = batch_temporal_correlation(const, varying)
        assert corr1.isnan().all(), "Constant pred should give NaN"

        corr2 = batch_temporal_correlation(varying, const)
        assert corr2.isnan().all(), "Constant target should give NaN"

    def test_single_time_step_returns_nan(self):
        """T=1 → can't compute temporal correlation, all NaN."""
        pred = torch.randn(4, 1, 5)
        target = torch.randn(4, 1, 5)
        corr = batch_temporal_correlation(pred, target)
        assert corr.shape == (4, 5)
        assert corr.isnan().all()

    def test_output_shape(self):
        """Verify (B, T, N) → (B, N)."""
        for B, T, N in [(1, 10, 1), (8, 50, 20), (3, 3, 100)]:
            corr = batch_temporal_correlation(torch.randn(B, T, N), torch.randn(B, T, N))
            assert corr.shape == (B, N)

    def test_per_image_independence(self):
        """Each image's correlation is computed independently.

        Image 0: perfect correlation (r=1)
        Image 1: perfect anti-correlation (r=-1)
        Image 2: zero signal (r=NaN)
        """
        T, N = 50, 4
        pred = torch.randn(3, T, N)
        target = torch.zeros(3, T, N)
        target[0] = pred[0]  # r = 1
        target[1] = -pred[1]  # r = -1
        target[2] = torch.ones(T, N)  # constant → NaN

        corr = batch_temporal_correlation(pred, target)
        torch.testing.assert_close(corr[0], torch.ones(N), atol=1e-5, rtol=0)
        torch.testing.assert_close(corr[1], -torch.ones(N), atol=1e-5, rtol=0)
        assert corr[2].isnan().all()


# ---------------------------------------------------------------------------
# TemporalCorrAccumulator
# ---------------------------------------------------------------------------


class TestTemporalCorrAccumulator:
    """Test TemporalCorrAccumulator: mean per-image temporal correlation across batches."""

    def test_single_batch_matches_direct(self):
        """Single batch: accumulated result == direct batch mean."""
        torch.manual_seed(7)
        pred = torch.randn(8, 30, 5)
        target = pred + torch.randn_like(pred) * 0.5
        corr = batch_temporal_correlation(pred, target)

        acc = TemporalCorrAccumulator()
        acc.update("s1", corr, 5)
        result = acc.compute()["s1"]

        expected = corr.mean(dim=0)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=0)

    def test_multi_batch_weighted_mean(self):
        """Multiple batches with different sizes: result is image-weighted mean."""
        torch.manual_seed(12)
        N = 4
        # Batch 1: 10 images, perfect correlation
        pred1 = torch.randn(10, 40, N)
        corr1 = batch_temporal_correlation(pred1, pred1)  # all 1.0
        # Batch 2: 5 images, zero correlation
        corr2 = batch_temporal_correlation(torch.randn(5, 40, N), torch.randn(5, 40, N))

        acc = TemporalCorrAccumulator()
        acc.update("s1", corr1, N)
        acc.update("s1", corr2, N)
        result = acc.compute()["s1"]

        # (10 * 1.0 + 5 * ~0.0) / 15 ≈ 0.667
        assert result.mean().item() > 0.5
        assert result.mean().item() < 0.8

    def test_nan_images_excluded(self):
        """Images with constant responses (NaN corr) don't affect the mean."""
        N, T = 3, 30
        pred = torch.randn(6, T, N)
        target = torch.zeros(6, T, N)
        target[:4] = pred[:4]  # perfect corr
        target[4:] = 1.0  # constant → NaN
        corr = batch_temporal_correlation(pred, target)

        acc = TemporalCorrAccumulator()
        acc.update("s1", corr, N)
        result = acc.compute()["s1"]
        torch.testing.assert_close(result, torch.ones(N), atol=1e-5, rtol=0)

    def test_all_nan_returns_nan_not_zero(self):
        """Neurons where ALL images have NaN correlation should report NaN, not 0.0."""
        N = 4
        # All NaN correlations for every image
        corr = torch.full((5, N), float("nan"))

        acc = TemporalCorrAccumulator()
        acc.update("s1", corr, N)
        result = acc.compute()["s1"]

        assert result.isnan().all(), f"Expected all NaN for neurons with no valid data, got {result}"

    def test_partial_nan_neurons(self):
        """Mix of neurons with valid and all-NaN data: valid neurons get means, NaN neurons get NaN."""
        N = 4
        corr = torch.full((10, N), float("nan"))
        # Neurons 0 and 1 get some valid correlations
        corr[:, 0] = 0.8
        corr[:, 1] = 0.6
        # Neurons 2 and 3 stay all NaN

        acc = TemporalCorrAccumulator()
        acc.update("s1", corr, N)
        result = acc.compute()["s1"]

        assert abs(result[0].item() - 0.8) < 1e-6, f"Neuron 0 should be 0.8, got {result[0]}"
        assert abs(result[1].item() - 0.6) < 1e-6, f"Neuron 1 should be 0.6, got {result[1]}"
        assert result[2].isnan(), f"Neuron 2 should be NaN, got {result[2]}"
        assert result[3].isnan(), f"Neuron 3 should be NaN, got {result[3]}"


# ---------------------------------------------------------------------------
# OnlinePearsonAccumulator  (rate correlation / cross-image correlation)
# ---------------------------------------------------------------------------


class TestOnlinePearsonAccumulator:
    """Test OnlinePearsonAccumulator: cross-image Pearson correlation per neuron."""

    def test_perfect_linear_relationship(self):
        """pred = 2*target + 3 → r = 1.0 for each neuron."""
        N = 5
        target = torch.randn(50, N)
        pred = 2.0 * target + 3.0

        acc = OnlinePearsonAccumulator()
        acc.update("s1", pred, target, N)
        corr = acc.compute()["s1"]
        torch.testing.assert_close(corr, torch.ones(N), atol=1e-4, rtol=0)

    def test_known_cross_image_correlation(self):
        """signal + noise construction with expected r = 1/sqrt(2) ≈ 0.707."""
        torch.manual_seed(99)
        N = 6
        n_images = 500
        signal = torch.randn(n_images, N)
        pred = signal
        target = signal + torch.randn(n_images, N)
        expected_r = 1.0 / 2**0.5

        acc = OnlinePearsonAccumulator()
        acc.update("s1", pred, target, N)
        corr = acc.compute()["s1"]
        mean_r = corr.mean().item()
        assert abs(mean_r - expected_r) < 0.05, f"Expected ~{expected_r:.3f}, got {mean_r:.3f}"

    def test_multi_batch_matches_single(self):
        """Splitting data across batches gives the same result as one batch."""
        torch.manual_seed(55)
        N = 4
        pred = torch.randn(40, N)
        target = pred + torch.randn(40, N) * 0.3

        single = OnlinePearsonAccumulator()
        single.update("s1", pred, target, N)

        multi = OnlinePearsonAccumulator()
        multi.update("s1", pred[:20], target[:20], N)
        multi.update("s1", pred[20:], target[20:], N)

        torch.testing.assert_close(single.compute()["s1"], multi.compute()["s1"], atol=1e-5, rtol=0)

    def test_independent_signals_near_zero(self):
        """Independent pred and target → r ≈ 0."""
        torch.manual_seed(77)
        N = 5
        acc = OnlinePearsonAccumulator()
        acc.update("s1", torch.randn(200, N), torch.randn(200, N), N)
        corr = acc.compute()["s1"]
        assert corr.abs().mean().item() < 0.15, f"Expected ~0, got {corr.mean():.3f}"

    def test_constant_signal_returns_nan(self):
        """Constant predictions → NaN correlation (no eps hack needed)."""
        N = 3
        pred = torch.ones(20, N)
        target = torch.randn(20, N)

        acc = OnlinePearsonAccumulator()
        acc.update("s1", pred, target, N)
        corr = acc.compute()["s1"]
        assert corr.isnan().all(), "Constant predictions should give NaN"

    def test_matches_torch_corrcoef(self):
        """Verify against torch.corrcoef on small examples."""
        torch.manual_seed(42)
        N = 3
        pred = torch.randn(30, N)
        target = torch.randn(30, N)

        acc = OnlinePearsonAccumulator()
        acc.update("s1", pred, target, N)
        ours = acc.compute()["s1"]

        # Reference: compute per-neuron correlation via torch.corrcoef
        ref = torch.stack([torch.corrcoef(torch.stack([pred[:, i], target[:, i]]))[0, 1] for i in range(N)])

        torch.testing.assert_close(ours, ref, atol=1e-5, rtol=0)

    def test_multiple_sessions(self):
        """Different sessions are tracked independently."""
        torch.manual_seed(0)
        acc = OnlinePearsonAccumulator()
        # Session A: perfect corr
        x = torch.randn(20, 3)
        acc.update("A", x, x, 3)
        # Session B: anti-corr
        y = torch.randn(20, 3)
        acc.update("B", y, -y, 3)

        result = acc.compute()
        torch.testing.assert_close(result["A"], torch.ones(3), atol=1e-4, rtol=0)
        torch.testing.assert_close(result["B"], -torch.ones(3), atol=1e-4, rtol=0)


# ---------------------------------------------------------------------------
# Rate correlation equivalence at T=1
# ---------------------------------------------------------------------------


class TestRateCorrAtT1:
    """Verify that rate_corr (sum over T=1, then cross-image Pearson) gives
    the same result as direct cross-image Pearson on the scalar values."""

    def test_t1_rate_corr_equals_direct(self):
        """For T=1 data, summing over T is identity, so rate_corr == direct Pearson."""
        torch.manual_seed(10)
        N = 5
        B = 50
        pred_btn = torch.randn(B, 1, N)
        target_btn = torch.randn(B, 1, N)

        # Rate path: sum over T (no-op for T=1), then accumulate
        rate_acc = OnlinePearsonAccumulator()
        rate_acc.update("s1", pred_btn.sum(dim=1), target_btn.sum(dim=1), N)
        rate_corr = rate_acc.compute()["s1"]

        # Direct path: squeeze T, accumulate
        direct_acc = OnlinePearsonAccumulator()
        direct_acc.update("s1", pred_btn[:, 0, :], target_btn[:, 0, :], N)
        direct_corr = direct_acc.compute()["s1"]

        torch.testing.assert_close(rate_corr, direct_corr, atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------
# noise_corrected_r2  (Pospisil & Bair ER estimator)
# ---------------------------------------------------------------------------


class TestNoiseCorrectedR2:
    """Tests for noise_corrected_r2 with analytically tractable cases."""

    def test_perfect_noiseless_prediction(self):
        """Model perfectly predicts mean response (near-zero noise) → R² ≈ 1.0."""
        torch.manual_seed(0)
        N, n_img, R = 5, 30, 10
        true_mean = torch.randn(N, n_img) * 2.0
        noise = torch.randn(N, n_img, R) * 1e-4
        responses = true_mean.unsqueeze(2) + noise
        predictions = true_mean

        r2 = noise_corrected_r2(responses, predictions)
        assert r2.shape == (N,)
        assert r2.mean().item() > 0.95, f"Expected ~1.0, got {r2.mean():.3f}"

    def test_random_prediction_gives_low_r2(self):
        """Predicting random noise uncorrelated with responses → R² ≈ 0."""
        torch.manual_seed(1)
        N, n_img, R = 4, 100, 8
        true_mean = torch.randn(N, n_img)
        noise = torch.randn(N, n_img, R) * 0.5
        responses = true_mean.unsqueeze(2) + noise
        predictions = torch.randn(N, n_img)

        r2 = noise_corrected_r2(responses, predictions)
        valid = ~r2.isnan()
        assert valid.any(), "Should have at least some valid neurons"
        assert r2[valid].abs().mean().item() < 0.2, f"Expected ~0, got {r2[valid].mean():.3f}"

    def test_known_r2_value(self):
        """Construct prediction = true_mean + model_noise.

        true_mean ~ N(0, signal_std²)
        pred = true_mean + model_error, model_error ~ N(0, err_std²)
        R² ≈ signal_std² / (signal_std² + err_std²) = 0.8
        """
        torch.manual_seed(42)
        N, n_img, R = 8, 200, 10
        signal_std = 2.0
        err_std = 1.0
        trial_noise_std = 0.5

        true_mean = torch.randn(N, n_img) * signal_std
        trial_noise = torch.randn(N, n_img, R) * trial_noise_std
        responses = true_mean.unsqueeze(2) + trial_noise

        model_error = torch.randn(N, n_img) * err_std
        predictions = true_mean + model_error

        expected_r2 = signal_std**2 / (signal_std**2 + err_std**2)  # 0.8

        r2 = noise_corrected_r2(responses, predictions)
        mean_r2 = r2.nanmean().item()
        assert abs(mean_r2 - expected_r2) < 0.1, f"Expected ~{expected_r2:.2f}, got {mean_r2:.3f}"

    def test_ragged_repeats_with_nan(self):
        """NaN padding for missing repeats is handled correctly."""
        N, n_img = 3, 20
        max_R = 5
        responses = torch.full((N, n_img, max_R), float("nan"))
        true_mean = torch.randn(N, n_img)

        for i in range(n_img):
            n_reps = 5 if i < 10 else 3
            responses[:, i, :n_reps] = true_mean[:, i].unsqueeze(1) + torch.randn(N, n_reps) * 0.3

        predictions = true_mean
        r2 = noise_corrected_r2(responses, predictions)
        assert r2.shape == (N,)
        assert not r2.isnan().all(), "Should compute valid R² despite ragged repeats"
        assert r2.nanmean().item() > 0.5, "Near-perfect predictions should yield high R²"

    def test_anti_correlated_prediction_gives_high_r2(self):
        """Prediction anti-correlated with true mean → R² ≈ 1.0.

        The ER estimator uses cov² in the numerator (squared covariance),
        so it is sign-agnostic: anti-correlated predictions explain just
        as much variance as positively correlated ones.
        """
        torch.manual_seed(3)
        N, n_img, R = 3, 100, 5
        true_mean = torch.randn(N, n_img) * 2.0
        trial_noise = torch.randn(N, n_img, R) * 0.3
        responses = true_mean.unsqueeze(2) + trial_noise
        predictions = -true_mean

        r2 = noise_corrected_r2(responses, predictions)
        assert r2.nanmean().item() > 0.9, f"Expected ~1.0 (sign-agnostic), got {r2.nanmean():.3f}"

    @pytest.mark.parametrize("tag,atol", [("f64", 1e-12), ("f32", 1e-6), ("f16", 1e-6)])
    def test_matches_reference_values(self, tag, atol):
        """Scores are pinned to stored reference values.

        ``nc_r2`` is a published encoding score, so the estimator must keep
        producing the same numbers. The fixture holds seeded inputs with ragged
        NaN-padded repeats and one zero-variance neuron (expected NaN), in
        float64 and float32.
        """
        ref = np.load(Path(__file__).parent / "data" / "noise_corrected_r2_reference.npz")
        resp = torch.from_numpy(ref[f"{tag}_resp"])
        pred = torch.from_numpy(ref[f"{tag}_pred"])
        expected = torch.from_numpy(ref[f"{tag}_expected"])

        out = noise_corrected_r2(resp, pred)

        assert out.dtype == expected.dtype
        assert torch.equal(out.isnan(), expected.isnan())
        assert out[0].isnan()  # the zero-variance neuron
        torch.testing.assert_close(out, expected, atol=atol, rtol=0, equal_nan=True)

    def test_output_shape(self):
        """Verify output is (N_neurons,)."""
        for N, n_img, R in [(1, 10, 3), (10, 50, 5), (20, 100, 2)]:
            responses = torch.randn(N, n_img, R)
            predictions = torch.randn(N, n_img)
            r2 = noise_corrected_r2(responses, predictions)
            assert r2.shape == (N,)


# ---------------------------------------------------------------------------
# Integration: temporal corr vs rate corr measure different things
# ---------------------------------------------------------------------------


class TestCorrelationPathsDiffer:
    """Verify that temporal correlation and rate correlation
    measure genuinely different things.

    A model that predicts the same temporal profile for every image
    (correct shape, wrong selectivity) should get high temporal correlation
    but low rate correlation.
    """

    def test_same_profile_all_images(self):
        """pred has correct temporal shape for every image but wrong scaling."""
        torch.manual_seed(0)
        B, T, N = 20, 50, 4

        # Shared temporal profile
        profile = torch.randn(1, T, N)
        # Per-image positive scale that doesn't match the true selectivity
        true_selectivity = torch.randn(B, 1, N).abs() + 0.1
        wrong_selectivity = torch.randn(B, 1, N).abs() + 0.1

        target = true_selectivity * profile + torch.randn(B, T, N) * 0.1
        pred = wrong_selectivity * profile

        # Temporal correlation: profile shape matches → high
        temporal_corr = batch_temporal_correlation(pred, target)
        mean_temporal = temporal_corr.nanmean().item()

        # Rate correlation: sum over T, then Pearson across images
        rate_acc = OnlinePearsonAccumulator()
        rate_acc.update("s1", pred.sum(dim=1), target.sum(dim=1), N)
        rate_corr = rate_acc.compute()["s1"].mean().item()

        assert mean_temporal > 0.8, f"Temporal corr should be high, got {mean_temporal:.3f}"
        assert abs(rate_corr) < 0.5, f"Rate corr should be low, got {rate_corr:.3f}"
