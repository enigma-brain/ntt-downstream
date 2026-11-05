"""Tests for dataloading utilities: SessionBatchSampler reproducibility and spike binning."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ntt_downstream.encoding.dataloading.cowley_v4 import SessionBatchSampler, numpy_collate_fn
from ntt_downstream.encoding.dataloading.spike_augmentations import apply_spike_augmentations

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_dataset(session_indices: dict[str, list[int]]) -> SimpleNamespace:
    """Return a minimal object that satisfies SessionBatchSampler's interface."""
    return SimpleNamespace(session_to_global_indices=session_indices)


def _collect_batches(sampler: SessionBatchSampler) -> list[list[int]]:
    return list(sampler)


# ---------------------------------------------------------------------------
# SessionBatchSampler reproducibility tests
# ---------------------------------------------------------------------------


class TestSessionBatchSamplerReproducibility:
    """Verify that SessionBatchSampler produces deterministic orderings."""

    @pytest.fixture()
    def dataset(self):
        return _make_mock_dataset(
            {
                "sess_A": list(range(0, 20)),
                "sess_B": list(range(20, 50)),
                "sess_C": list(range(50, 65)),
            }
        )

    def test_same_epoch_same_batches(self, dataset):
        """Calling set_epoch(0) twice must yield identical batch sequences."""
        sampler = SessionBatchSampler(dataset, batch_size=8, shuffle=True)

        sampler.set_epoch(0)
        run1 = _collect_batches(sampler)

        sampler.set_epoch(0)
        run2 = _collect_batches(sampler)

        assert run1 == run2

    def test_different_epochs_differ(self, dataset):
        """Different epoch values should (almost certainly) produce different orderings."""
        sampler = SessionBatchSampler(dataset, batch_size=8, shuffle=True)

        sampler.set_epoch(0)
        run_epoch0 = _collect_batches(sampler)

        sampler.set_epoch(1)
        run_epoch1 = _collect_batches(sampler)

        # Flatten to compare element order.
        flat0 = [idx for batch in run_epoch0 for idx in batch]
        flat1 = [idx for batch in run_epoch1 for idx in batch]
        assert flat0 != flat1, "Two different epochs produced identical orderings"

    def test_all_items_yielded(self, dataset):
        """Every global index must appear exactly once across all batches."""
        sampler = SessionBatchSampler(dataset, batch_size=8, shuffle=True, drop_last=False)
        sampler.set_epoch(42)

        all_items = sorted(idx for batch in sampler for idx in batch)
        expected = sorted(idx for indices in dataset.session_to_global_indices.values() for idx in indices)
        assert all_items == expected

    def test_all_items_yielded_no_shuffle(self, dataset):
        """Without shuffling, all items must still be yielded."""
        sampler = SessionBatchSampler(dataset, batch_size=8, shuffle=False, drop_last=False)

        all_items = sorted(idx for batch in sampler for idx in batch)
        expected = sorted(idx for indices in dataset.session_to_global_indices.values() for idx in indices)
        assert all_items == expected

    def test_drop_last(self, dataset):
        """With drop_last=True, partial batches should be dropped."""
        sampler = SessionBatchSampler(dataset, batch_size=8, shuffle=False, drop_last=True)
        for batch in sampler:
            assert len(batch) == 8

    def test_batches_within_session(self, dataset):
        """All indices in a batch must belong to the same session."""
        sampler = SessionBatchSampler(dataset, batch_size=8, shuffle=True)
        sampler.set_epoch(7)

        # Build reverse lookup: global_index -> session_id
        idx_to_session = {}
        for sid, indices in dataset.session_to_global_indices.items():
            for idx in indices:
                idx_to_session[idx] = sid

        for batch in sampler:
            sessions_in_batch = {idx_to_session[i] for i in batch}
            assert len(sessions_in_batch) == 1, f"Batch contains indices from multiple sessions: {sessions_in_batch}"


# ---------------------------------------------------------------------------
# Spike binning correctness
# ---------------------------------------------------------------------------


class TestSpikeBinning:
    """Verify that the spike-to-bin logic places spikes in correct bins.

    temporaldata.Data.slice() with reset_origin=True (the default) returns
    timestamps relative to the window start, so dividing by bin_size directly
    gives the correct bin index. These tests replicate the binning logic from
    CowleyV4MultiSessionDataset.__getitem__ on synthetic data to confirm
    correctness.
    """

    @staticmethod
    def _bin_spikes(
        spike_ts: np.ndarray,
        spike_ui: np.ndarray,
        n_units: int,
        n_time_bins: int,
        bin_size: float,
    ) -> np.ndarray:
        """Replicate the binning logic from __getitem__."""
        response = np.zeros((n_units, n_time_bins), dtype=np.float32)
        if len(spike_ts) > 0:
            bin_idx = np.minimum((spike_ts / bin_size).astype(int), n_time_bins - 1)
            np.add.at(response, (spike_ui, bin_idx), 1)
        return response

    def test_relative_timestamps_bin_correctly(self):
        """Relative timestamps (as returned by slice) should land in expected bins."""
        bin_size = 0.001  # 1 ms
        n_time_bins = 200  # 200 ms window
        n_units = 3

        # Synthetic relative timestamps (already shifted to window start = 0)
        spike_ts = np.array([0.0, 0.0005, 0.001, 0.050, 0.1995, 0.1999])
        spike_ui = np.array([0, 0, 1, 2, 1, 2])

        response = self._bin_spikes(spike_ts, spike_ui, n_units, n_time_bins, bin_size)

        # bin 0: [0, 0.001) -> ts=0.0 (unit 0), ts=0.0005 (unit 0)
        assert response[0, 0] == 2.0
        # bin 1: [0.001, 0.002) -> ts=0.001 (unit 1)
        assert response[1, 1] == 1.0
        # bin 50: [0.050, 0.051) -> ts=0.050 (unit 2)
        assert response[2, 50] == 1.0
        # bin 199: [0.199, 0.200) -> ts=0.1995 (unit 1), ts=0.1999 (unit 2)
        assert response[1, 199] == 1.0
        assert response[2, 199] == 1.0
        # Total spikes must match
        assert response.sum() == 6.0

    def test_absolute_timestamps_would_clamp(self):
        """Demonstrate that absolute (unrelatived) timestamps would all clamp
        to the last bin, which would be incorrect."""
        bin_size = 0.001
        n_time_bins = 200
        n_units = 2

        # Simulate what would happen with absolute timestamps (onset=100.0s)
        absolute_ts = np.array([100.000, 100.050, 100.100, 100.150])
        spike_ui = np.array([0, 0, 1, 1])

        response = self._bin_spikes(absolute_ts, spike_ui, n_units, n_time_bins, bin_size)

        # All spikes would be clamped to the last bin -- this is wrong
        assert response[:, -1].sum() == 4.0, "Absolute timestamps all clamp to last bin"
        assert response[:, :-1].sum() == 0.0, "No spikes in earlier bins with absolute ts"

    def test_empty_spikes(self):
        """Empty spike arrays should produce an all-zero response."""
        response = self._bin_spikes(np.array([]), np.array([]), n_units=5, n_time_bins=100, bin_size=0.001)
        assert response.shape == (5, 100)
        assert response.sum() == 0.0


class TestSpikeAugmentations:
    def test_jitter_preserves_spike_count_and_shape(self):
        torch.manual_seed(0)
        response = torch.zeros(2, 8, 3)
        response[0, 2, 1] = 1
        response[0, 4, 2] = 2
        response[1, 5, 0] = 1

        out = apply_spike_augmentations(
            response,
            {"enabled": True, "jitter": {"p": 1.0, "max_bins": 2}},
            time_dim=1,
        )

        assert out.shape == response.shape
        assert torch.equal(out.sum(dim=(1, 2)), response.sum(dim=(1, 2)))

    def test_jitter_fraction_only_moves_subset_of_spikes(self):
        torch.manual_seed(3)
        response = torch.zeros(1, 10, 1)
        response[0, 5, 0] = 10

        out = apply_spike_augmentations(
            response,
            {"enabled": True, "jitter": {"p": 1.0, "max_bins": 1, "fraction": 0.3}},
            time_dim=1,
        )

        assert out.sum() == response.sum()
        assert out[0, 5, 0] >= 7

    def test_jitter_num_spikes_caps_number_of_affected_spikes(self):
        torch.manual_seed(5)
        response = torch.zeros(1, 10, 1)
        response[0, 5, 0] = 10

        out = apply_spike_augmentations(
            response,
            {"enabled": True, "jitter": {"p": 1.0, "max_bins": 1, "num_spikes": 2}},
            time_dim=1,
        )

        assert out.sum() == response.sum()
        assert out[0, 5, 0] >= 8

    def test_dropout_can_remove_all_spikes(self):
        response = torch.ones(2, 5, 3)
        out = apply_spike_augmentations(
            response,
            {"enabled": True, "dropout": {"p": 1.0, "drop_prob": 1.0}},
            time_dim=1,
        )
        assert out.sum() == 0

    def test_dropout_max_spikes_caps_removed_spikes_per_sample(self):
        torch.manual_seed(0)
        response = torch.ones(2, 5, 3)
        out = apply_spike_augmentations(
            response,
            {"enabled": True, "dropout": {"p": 1.0, "drop_prob": 1.0, "max_spikes": 4}},
            time_dim=1,
        )
        assert torch.equal(response.sum(dim=(1, 2)) - out.sum(dim=(1, 2)), torch.tensor([4.0, 4.0]))

    def test_addition_can_add_to_all_bins(self):
        response = torch.zeros(2, 5, 3)
        out = apply_spike_augmentations(
            response,
            {"enabled": True, "addition": {"p": 1.0, "add_prob": 1.0}},
            time_dim=1,
        )
        assert torch.equal(out, torch.ones_like(response))

    def test_addition_max_spikes_caps_added_spikes_per_sample(self):
        torch.manual_seed(0)
        response = torch.zeros(2, 5, 3)
        out = apply_spike_augmentations(
            response,
            {"enabled": True, "addition": {"p": 1.0, "add_prob": 1.0, "max_spikes": 4}},
            time_dim=1,
        )
        assert torch.equal(out.sum(dim=(1, 2)), torch.tensor([4.0, 4.0]))

    def test_cowley_collate_applies_spike_augmentations_on_time_last_response(self):
        batch = [
            {
                "image": np.zeros((3, 4, 4), dtype=np.float32),
                "response": np.ones((2, 5), dtype=np.float32),
                "image_id": i,
                "session_id": "sess",
                "n_units": 2,
            }
            for i in range(2)
        ]

        _sid, out = numpy_collate_fn(
            batch,
            spike_augmentations={"enabled": True, "dropout": {"p": 1.0, "drop_prob": 1.0}},
        )
        assert out["response"].shape == (2, 2, 5)
        assert out["response"].sum() == 0
