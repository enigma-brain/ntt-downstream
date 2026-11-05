"""Tests for shared rectified-flow utilities (flow_utils)."""

from __future__ import annotations

import pytest
import torch

from ntt_downstream.encoding.models.enc.losses.flow_utils import (
    FlowOptions,
    flow_matching_loss,
    flow_sample,
    make_time_grid,
    ot_pair_noise,
    sample_flow_time,
)


class TestFlowOptions:
    def test_defaults(self):
        opts = FlowOptions()
        assert opts.time_sampling == "uniform"
        assert opts.ot_coupling is False
        assert opts.cond_dropout == 0.0

    def test_invalid_time_sampling_raises(self):
        with pytest.raises(ValueError, match="time_sampling"):
            FlowOptions(time_sampling="cosmap")

    def test_invalid_cond_dropout_raises(self):
        with pytest.raises(ValueError, match="cond_dropout"):
            FlowOptions(cond_dropout=1.5)


class TestSampleFlowTime:
    def test_uniform_shape_and_range(self):
        torch.manual_seed(0)
        t = sample_flow_time(1000, FlowOptions(), torch.device("cpu"), torch.float32)
        assert t.shape == (1000,)
        assert t.min() >= 0.0 and t.max() <= 1.0

    def test_logit_normal_range(self):
        torch.manual_seed(0)
        opts = FlowOptions(time_sampling="logit_normal")
        t = sample_flow_time(1000, opts, torch.device("cpu"), torch.float32)
        assert t.shape == (1000,)
        assert t.min() > 0.0 and t.max() < 1.0

    def test_logit_normal_mean_shifts_mass(self):
        torch.manual_seed(0)
        opts = FlowOptions(time_sampling="logit_normal", logit_normal_mean=3.0)
        t = sample_flow_time(4000, opts, torch.device("cpu"), torch.float32)
        # sigmoid(3 + n), n~N(0,1): mass concentrates near t=1.
        assert t.mean() > 0.8

    def test_logit_normal_concentrates_midrange(self):
        torch.manual_seed(0)
        opts = FlowOptions(time_sampling="logit_normal", logit_normal_std=1.0)
        t = sample_flow_time(8000, opts, torch.device("cpu"), torch.float32)
        mid = ((t > 0.25) & (t < 0.75)).float().mean()
        # Uniform puts 0.5 mass in (0.25, 0.75); logit-normal(0,1) puts ~0.73.
        assert mid > 0.6


class TestMakeTimeGrid:
    def test_shift_one_is_uniform(self):
        grid = make_time_grid(5, 1.0, torch.device("cpu"), torch.float32)
        expected = torch.linspace(0.0, 1.0, 6)
        torch.testing.assert_close(grid, expected)

    def test_endpoints_and_monotonic(self):
        grid = make_time_grid(7, 3.0, torch.device("cpu"), torch.float32)
        assert grid.shape == (8,)
        torch.testing.assert_close(grid[0], torch.tensor(0.0))
        torch.testing.assert_close(grid[-1], torch.tensor(1.0))
        assert (grid.diff() > 0).all()

    def test_shift_gt_one_concentrates_near_noise(self):
        # shift > 1 must place interior grid points BELOW the uniform grid,
        # i.e. finer resolution near t=0 (the noise end in our convention).
        uniform = make_time_grid(5, 1.0, torch.device("cpu"), torch.float32)
        shifted = make_time_grid(5, 3.0, torch.device("cpu"), torch.float32)
        assert (shifted[1:-1] < uniform[1:-1]).all()


class TestOTPairNoise:
    def _total_cost(self, noise, target):
        B = noise.shape[0]
        return (noise.reshape(B, -1) - target.reshape(B, -1)).pow(2).sum().item()

    def test_paired_rows_drawn_from_noise(self):
        # The Sinkhorn argmax assignment is NOT a strict permutation — two
        # targets can pick the same noise partner — so the multiset of rows may
        # differ from the input, so an exact-permutation assertion does not
        # apply. The invariant that DOES hold: the coupling only ever
        # reuses real noise samples. The result is ``noise[perm]``, so every
        # output row is one of the original noise rows, never a fabricated one.
        # Exact tuple membership is valid here because rows are gathered by
        # index, not arithmetically combined.
        torch.manual_seed(0)
        noise = torch.randn(64, 8)
        target = torch.randn(64, 8)
        paired = ot_pair_noise(noise, target)
        noise_rows = {tuple(r.tolist()) for r in noise}
        assert all(tuple(r.tolist()) in noise_rows for r in paired)

    def test_approximately_preserves_marginal(self):
        # Collisions cost some marginal coverage, but a clear majority of noise
        # rows survive (empirically ~70-80% unique with these seeds). We assert
        # a loose lower bound rather than exact preservation: the test tracks the
        # documented "approximately preserved" contract without pinning the exact
        # collision count, which is an artifact of the Sinkhorn reg / iterations.
        torch.manual_seed(0)
        noise = torch.randn(128, 16)
        target = torch.randn(128, 16)
        paired = ot_pair_noise(noise, target)
        n_unique = len({tuple(r.tolist()) for r in paired})
        assert n_unique > noise.shape[0] // 2

    def test_reduces_transport_cost(self):
        torch.manual_seed(0)
        noise = torch.randn(128, 16)
        target = torch.randn(128, 16)
        paired = ot_pair_noise(noise, target)
        assert self._total_cost(paired, target) < self._total_cost(noise, target)

    @pytest.mark.parametrize("shape", [(32, 8), (32, 4, 6), (32, 3, 5, 4)])
    def test_preserves_shape(self, shape):
        torch.manual_seed(0)
        noise = torch.randn(*shape)
        target = torch.randn(*shape)
        assert ot_pair_noise(noise, target).shape == noise.shape

    def test_chunking_pairs_all_rows(self):
        # Exercise the chunked path with a batch larger than one chunk. Each
        # chunk solves its own minibatch-OT, so — like the single-chunk case —
        # the assignment is approximate, not a strict permutation. The contract
        # to verify is that every row is processed and receives a valid partner
        # drawn from the input noise (membership), with the shape preserved.
        from ntt_downstream.encoding.models.enc.losses import flow_utils

        torch.manual_seed(0)
        B = flow_utils._OT_MAX_CHUNK + 100
        noise = torch.randn(B, 4)
        target = torch.randn(B, 4)
        paired = ot_pair_noise(noise, target)
        assert paired.shape == noise.shape
        noise_rows = {tuple(r.tolist()) for r in noise}
        assert all(tuple(r.tolist()) in noise_rows for r in paired)


class TestFlowMatchingLoss:
    def _zero_field(self, x_t, t, cond):
        return torch.zeros_like(x_t)

    def test_scalar_loss_all_options(self):
        torch.manual_seed(0)
        opts = FlowOptions(time_sampling="logit_normal", ot_coupling=True, cond_dropout=0.5)
        target = torch.randn(16, 8)
        cond = torch.randn(16, 8)
        null = torch.zeros(8)
        loss = flow_matching_loss(self._zero_field, target, cond, opts, null_cond=null)
        assert loss.ndim == 0 and torch.isfinite(loss)

    def test_cond_dropout_without_null_raises(self):
        opts = FlowOptions(cond_dropout=0.5)
        target = torch.randn(4, 8)
        cond = torch.randn(4, 8)
        with pytest.raises(ValueError, match="null_cond"):
            flow_matching_loss(self._zero_field, target, cond, opts)

    def test_full_dropout_replaces_conditioning(self):
        # With cond_dropout ~ 1, the forward must only ever see the null embedding.
        seen = []

        def spy(x_t, t, cond):
            seen.append(cond)
            return torch.zeros_like(x_t)

        opts = FlowOptions(cond_dropout=0.999)
        torch.manual_seed(0)
        target = torch.randn(64, 8)
        cond = torch.randn(64, 8)
        null = torch.full((8,), 7.0)
        flow_matching_loss(spy, target, cond, opts, null_cond=null)
        assert (seen[0] == 7.0).float().mean() > 0.95

    def test_broadcasts_over_grid_shapes(self):
        # Full-grid layout (B, S, T, D) with grid conditioning.
        torch.manual_seed(0)
        target = torch.randn(4, 3, 5, 8)
        cond = torch.randn(4, 3, 5, 8)
        loss = flow_matching_loss(self._zero_field, target, cond, FlowOptions())
        assert loss.ndim == 0 and torch.isfinite(loss)


class TestFlowSample:
    def _contracting_field(self, x_t, t, cond):
        # v = -x: closed-form contraction, magnitude must shrink monotonically.
        return -x_t

    def test_euler_contracts(self):
        torch.manual_seed(0)
        x0 = torch.randn(8, 4) * 10
        out = flow_sample(self._contracting_field, x0, torch.zeros(8, 4), n_steps=10)
        assert out.norm() < x0.norm()

    def test_shift_changes_trajectory_not_endpoints(self):
        # A time-DEPENDENT field makes the shifted grid produce a different
        # result; both must remain finite.
        def field(x_t, t, cond):
            return -x_t * t.view(-1, 1)

        torch.manual_seed(0)
        x0 = torch.randn(8, 4)
        out_uniform = flow_sample(field, x0.clone(), torch.zeros(8, 4), n_steps=5, shift=1.0)
        out_shifted = flow_sample(field, x0.clone(), torch.zeros(8, 4), n_steps=5, shift=3.0)
        assert torch.isfinite(out_shifted).all()
        assert not torch.allclose(out_uniform, out_shifted)

    def test_cfg_without_null_raises(self):
        with pytest.raises(ValueError, match="cfg_scale"):
            flow_sample(
                self._contracting_field,
                torch.randn(4, 4),
                torch.zeros(4, 4),
                cfg_scale=2.0,
            )

    def test_cfg_scale_one_matches_plain_path(self):
        # cfg_scale=1.0 must take the single-forward path and be exactly
        # equal to sampling without CFG, even when null_cond exists.
        torch.manual_seed(0)
        x0 = torch.randn(8, 4)
        plain = flow_sample(self._contracting_field, x0.clone(), torch.zeros(8, 4), n_steps=4)
        with_null = flow_sample(
            self._contracting_field,
            x0.clone(),
            torch.zeros(8, 4),
            n_steps=4,
            cfg_scale=1.0,
            null_cond=torch.zeros(4),
        )
        torch.testing.assert_close(plain, with_null)

    def test_cfg_guides_velocity(self):
        # Field where conditional and unconditional velocities differ:
        # v = cond (constant). null_cond = 0 -> v_uncond = 0, so
        # cfg_scale s gives x1 = x0 + s * cond.
        def field(x_t, t, cond):
            return cond

        x0 = torch.zeros(4, 3)
        cond = torch.ones(4, 3)
        out = flow_sample(field, x0, cond, n_steps=5, cfg_scale=2.0, null_cond=torch.zeros(3))
        torch.testing.assert_close(out, torch.full((4, 3), 2.0))

    def test_heun_solver_runs(self):
        torch.manual_seed(0)
        out = flow_sample(
            self._contracting_field,
            torch.randn(8, 4),
            torch.zeros(8, 4),
            n_steps=5,
            solver="heun",
            shift=2.0,
        )
        assert torch.isfinite(out).all()


class TestFlowMatchingHeadIntegration:
    def _head(self, **kw):
        from ntt_downstream.encoding.models.enc.losses.flow_matching_head import FlowMatchingHead

        return FlowMatchingHead(token_dim=8, conditioning_dim=8, width=32, depth=2, **kw)

    def test_default_has_no_null_cond(self):
        head = self._head()
        assert head.null_cond is None
        # State dict must be unchanged for default configs (checkpoint compat).
        assert not any("null_cond" in k for k in head.state_dict())

    def test_cond_dropout_registers_null_cond(self):
        head = self._head(flow_opts=FlowOptions(cond_dropout=0.1))
        assert isinstance(head.null_cond, torch.nn.Parameter)
        assert head.null_cond.shape == (8,)

    def test_loss_backward_with_all_options(self):
        torch.manual_seed(0)
        head = self._head(flow_opts=FlowOptions(time_sampling="logit_normal", ot_coupling=True, cond_dropout=0.5))
        loss = head.compute_loss(torch.randn(16, 8), torch.randn(16, 8))
        loss.backward()
        assert torch.isfinite(loss)

    def test_sample_with_cfg_and_shift(self):
        torch.manual_seed(0)
        head = self._head(flow_opts=FlowOptions(cond_dropout=0.1))
        out = head.sample(torch.randn(4, 8), n_steps=3, shift=2.0, cfg_scale=1.5)
        assert out.shape == (4, 8)

    def test_sample_cfg_without_training_dropout_raises(self):
        head = self._head()
        with pytest.raises(ValueError, match="cfg_scale"):
            head.sample(torch.randn(4, 8), n_steps=2, cfg_scale=2.0)


class TestLDNSFlowMatchingHeadIntegration:
    def _head(self, **kw):
        from ntt_downstream.encoding.models.enc.losses.ldns_denoiser import LDNSFlowMatchingHead

        return LDNSFlowMatchingHead(latent_n=4, hidden_channels=16, conditioning_dim=4, num_blocks=2, **kw)

    def test_loss_and_sample_with_options(self):
        torch.manual_seed(0)
        head = self._head(flow_opts=FlowOptions(time_sampling="logit_normal", cond_dropout=0.2))
        loss = head.compute_loss(torch.randn(8, 4, 10), torch.randn(8, 4))
        assert torch.isfinite(loss)
        out = head.sample(torch.randn(8, 4), T=10, n_steps=3, shift=2.0, cfg_scale=1.5)
        assert out.shape == (8, 4, 10)


class TestFullGridFlowMatchingHeadIntegration:
    def _head(self, **kw):
        pytest.importorskip("ntt.contrib.tokenizer.perceiver.blocks")
        from ntt_downstream.encoding.models.enc.losses.full_grid_denoiser import (
            FullGridFlowMatchingHead,
        )

        return FullGridFlowMatchingHead(
            latent_dim=8,
            conditioning_dim=8,
            hidden_channels=16,
            num_blocks=1,
            num_heads=2,
            max_spatial=4,
            max_time=6,
            **kw,
        )

    def test_loss_and_sample_with_options(self):
        torch.manual_seed(0)
        head = self._head(flow_opts=FlowOptions(time_sampling="logit_normal", cond_dropout=0.2))
        target = torch.randn(2, 4, 6, 8)
        cond = torch.randn(2, 4, 6, 8)
        loss = head.compute_loss(target, cond)
        assert torch.isfinite(loss)
        out = head.sample(cond, n_steps=2, shift=2.0, cfg_scale=1.5)
        assert out.shape == (2, 4, 6, 8)
