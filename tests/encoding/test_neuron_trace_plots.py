"""Unit tests for eval-time neuron trace plotting helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for tests
import matplotlib.pyplot as plt  # noqa: E402
import torch

from ntt_downstream.encoding.models.enc.viz import (  # noqa: E402
    pick_diverse_neurons,
    pick_random_neurons,
    plot_neuron_traces,
)


class TestPickRandomNeurons:
    def test_returns_k_unique_indices(self):
        idx = pick_random_neurons(n_total=96, k=10, seed=0)
        assert len(idx) == 10
        assert len(set(idx)) == 10
        assert all(0 <= i < 96 for i in idx)

    def test_deterministic_from_seed(self):
        a = pick_random_neurons(n_total=96, k=10, seed=42)
        b = pick_random_neurons(n_total=96, k=10, seed=42)
        assert a == b

    def test_different_seeds_differ(self):
        a = pick_random_neurons(n_total=96, k=10, seed=0)
        b = pick_random_neurons(n_total=96, k=10, seed=1)
        assert a != b

    def test_k_larger_than_n_returns_all(self):
        idx = pick_random_neurons(n_total=5, k=10, seed=0)
        assert sorted(idx) == [0, 1, 2, 3, 4]


class TestPickDiverseNeurons:
    def test_returns_up_to_10_unique(self):
        r = torch.linspace(-1.0, 1.0, steps=50)
        idx = pick_diverse_neurons(r)
        assert len(idx) == 10
        assert len(set(idx)) == 10

    def test_includes_top_and_bottom(self):
        r = torch.linspace(-1.0, 1.0, steps=50)
        idx = pick_diverse_neurons(r)
        # Top 3 by r → indices 49, 48, 47 (ascending r → last 3 best)
        assert 49 in idx
        assert 48 in idx
        assert 47 in idx
        # Bottom 3 → indices 0, 1, 2
        assert 0 in idx
        assert 1 in idx
        assert 2 in idx

    def test_nan_treated_as_worst(self):
        # 15 elements so n > k=10 — forces the sort+NaN-mask branch.
        r = torch.tensor([0.5, float("nan"), 0.1, 0.9, 0.7, 0.2, 0.3, 0.8, 0.4, 0.6, 0.11, 0.12, 0.13, 0.14, 0.15])
        idx = pick_diverse_neurons(r)
        # NaN at index 1 sorts to -inf -> lands in the bottom group.
        assert 1 in idx

    def test_small_session_returns_all(self):
        r = torch.tensor([0.1, 0.5, 0.3])
        idx = pick_diverse_neurons(r)
        assert sorted(idx) == [0, 1, 2]

    def test_preserves_order_top_mid_bottom(self):
        r = torch.arange(50, dtype=torch.float32) / 50.0  # increasing
        idx = pick_diverse_neurons(r)
        # First 3 should be top (argsort desc starts at 49)
        assert idx[0] == 49
        assert idx[1] == 48
        assert idx[2] == 47


class TestPlotNeuronTraces:
    def _synthetic(self, T: int = 50, N: int = 96):
        torch.manual_seed(0)
        pred = torch.randn(T, N)
        target = torch.randn(T, N)
        return pred, target

    def test_returns_figure_with_10_axes(self):
        pred, target = self._synthetic()
        fig = plot_neuron_traces(
            pred,
            target,
            neuron_idx=list(range(10)),
            session_id="sess_A",
            global_step=100,
            kind="random",
        )
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 10
        plt.close(fig)

    def test_each_axis_has_two_lines(self):
        pred, target = self._synthetic()
        fig = plot_neuron_traces(
            pred,
            target,
            neuron_idx=list(range(10)),
            session_id="sess_A",
            global_step=100,
            kind="random",
        )
        for ax in fig.axes:
            # Each axis plots target + pred = 2 Line2D objects.
            assert len(ax.get_lines()) == 2
        plt.close(fig)

    def test_each_axis_has_psth_lines_when_provided(self):
        pred, target = self._synthetic()
        pred_psth = pred + 0.5
        target_psth = target + 0.5
        fig = plot_neuron_traces(
            pred,
            target,
            neuron_idx=list(range(10)),
            session_id="sess_A",
            global_step=100,
            kind="random",
            pred_psth=pred_psth,
            target_psth=target_psth,
        )
        for ax in fig.axes:
            assert len(ax.get_lines()) == 4
        labels = [line.get_label() for line in fig.axes[0].get_lines()]
        assert labels == ["target", "pred", "target PSTH", "pred PSTH"]
        plt.close(fig)

    def test_panel_titles_contain_neuron_idx(self):
        pred, target = self._synthetic()
        idx = [3, 17, 42, 5, 9, 11, 22, 33, 44, 55]
        fig = plot_neuron_traces(
            pred,
            target,
            neuron_idx=idx,
            session_id="sess_A",
            global_step=100,
            kind="random",
        )
        titles = [ax.get_title() for ax in fig.axes]
        for i, expected_neuron in enumerate(idx):
            assert f"n{expected_neuron}" in titles[i]
        plt.close(fig)

    def test_pearson_r_shown_when_provided(self):
        pred, target = self._synthetic()
        r = torch.arange(96, dtype=torch.float32) / 96.0
        fig = plot_neuron_traces(
            pred,
            target,
            neuron_idx=[0, 1, 2],
            session_id="sess_A",
            global_step=100,
            kind="diverse",
            pearson_r=r,
        )
        assert "r=" in fig.axes[0].get_title()
        plt.close(fig)

    def test_short_selection_hides_unused_axes(self):
        pred, target = self._synthetic(N=5)
        fig = plot_neuron_traces(
            pred,
            target,
            neuron_idx=[0, 1, 2, 3, 4],
            session_id="sess_A",
            global_step=100,
            kind="random",
        )
        # Figure should still have 10 subplot slots, but only 5 are populated
        # with data (lines). Unused axes have no lines and are invisible.
        visible_axes = [ax for ax in fig.axes if ax.get_visible() and len(ax.get_lines()) > 0]
        assert len(visible_axes) == 5
        plt.close(fig)

    def test_time_axis_hz_sets_seconds_xlabel(self):
        pred, target = self._synthetic()
        fig = plot_neuron_traces(
            pred,
            target,
            neuron_idx=list(range(10)),
            session_id="sess_A",
            global_step=0,
            kind="random",
            time_axis_hz=100.0,
        )
        # Bottom-row axes (ax_i >= 5) get the xlabel.
        assert fig.axes[5].get_xlabel() == "time (s)"
        plt.close(fig)

    def test_suptitle_includes_session_and_step(self):
        pred, target = self._synthetic()
        fig = plot_neuron_traces(
            pred,
            target,
            neuron_idx=list(range(10)),
            session_id="sess_XYZ",
            global_step=12345,
            kind="diverse",
        )
        suptitle = fig._suptitle.get_text() if fig._suptitle else ""
        assert "sess_XYZ" in suptitle
        assert "12345" in suptitle
        assert "diverse" in suptitle
        plt.close(fig)
