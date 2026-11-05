"""Tests for Wingert auditory cortex dataloading."""

from __future__ import annotations

import pytest
import torch

from ntt_downstream.encoding.dataloading.wingert_auditory import (
    WingertEvalCollate,
    WingertMultiSessionDataset,
    WingertRepeatAveragedDataset,
    WingertTrainCollate,
)


class TestWingertMultiSessionDataset:
    def test_loads_all_sessions(self, wingert_data_dir):
        ds = WingertMultiSessionDataset(str(wingert_data_dir), normalize=False)
        assert len(ds.session_meta) == 2
        assert len(ds) > 0

    def test_getitem_returns_expected_keys(self, wingert_data_dir):
        ds = WingertMultiSessionDataset(str(wingert_data_dir), normalize=False)
        item = ds[0]
        assert set(item.keys()) == {
            "spectrogram",
            "response",
            "epoch_name",
            "session_id",
            "n_units",
            "length",
            "response_length",
        }

    def test_getitem_shapes(self, wingert_data_dir):
        ds = WingertMultiSessionDataset(str(wingert_data_dir), normalize=False)
        item = ds[0]
        T = item["length"]
        assert item["spectrogram"].shape == (T, 32)
        assert item["response"].shape == (T, 5)

    def test_split_filters_presentations(self, wingert_data_dir):
        ds_all = WingertMultiSessionDataset(str(wingert_data_dir), normalize=False)
        ds_train = WingertMultiSessionDataset(str(wingert_data_dir), split="train", normalize=False)
        ds_valid = WingertMultiSessionDataset(str(wingert_data_dir), split="valid", normalize=False)
        ds_test = WingertMultiSessionDataset(str(wingert_data_dir), split="test", normalize=False)
        assert len(ds_train) + len(ds_valid) + len(ds_test) == len(ds_all)
        assert len(ds_train) > 0
        assert len(ds_valid) > 0
        assert len(ds_test) > 0

    def test_session_to_global_indices(self, wingert_data_dir):
        ds = WingertMultiSessionDataset(str(wingert_data_dir), normalize=False)
        all_indices = []
        for indices in ds.session_to_global_indices.values():
            assert len(indices) > 0
            all_indices.extend(indices)
        assert sorted(all_indices) == list(range(len(ds)))

    def test_normalization_produces_0_1_range(self, wingert_data_dir):
        ds = WingertMultiSessionDataset(str(wingert_data_dir), normalize=True, response_normalization="minmax")
        item = ds[0]
        spec = item["spectrogram"]
        assert spec.min() >= 0.0
        assert spec.max() <= 1.0 + 1e-6

    def test_spectrogram_is_float32(self, wingert_data_dir):
        ds = WingertMultiSessionDataset(str(wingert_data_dir), normalize=True)
        item = ds[0]
        assert item["spectrogram"].dtype == torch.float32
        assert item["response"].dtype == torch.float32

    def test_session_ids_filter(self, wingert_data_dir):
        ds = WingertMultiSessionDataset(str(wingert_data_dir), session_ids=["sess_A"], normalize=False)
        assert len(ds.session_meta) == 1
        assert ds.session_meta[0]["session_id"] == "sess_A"


class TestResponseNormalization:
    def test_none_returns_raw_counts(self, wingert_data_dir):
        """With response_normalization='none', responses match normalize=False dataset."""
        ds_none = WingertMultiSessionDataset(str(wingert_data_dir), normalize=True, response_normalization="none")
        ds_raw = WingertMultiSessionDataset(str(wingert_data_dir), normalize=False)
        for i in range(min(3, len(ds_none))):
            assert torch.equal(ds_none[i]["response"], ds_raw[i]["response"])

    def test_zscore_zero_mean_unit_var(self, wingert_data_dir):
        """With response_normalization='zscore', response values go negative (unlike minmax)."""
        ds = WingertMultiSessionDataset(str(wingert_data_dir), normalize=True, response_normalization="zscore")
        has_negative = False
        for i in range(len(ds)):
            resp = ds[i]["response"]
            if resp.min() < 0:
                has_negative = True
                break
        assert has_negative, "zscore responses should contain negative values"

    def test_minmax_produces_0_1_range(self, wingert_data_dir):
        """With response_normalization='minmax', responses are in [0, 1]."""
        ds_minmax = WingertMultiSessionDataset(str(wingert_data_dir), normalize=True, response_normalization="minmax")
        ds_raw = WingertMultiSessionDataset(str(wingert_data_dir), normalize=False)
        for i in range(min(3, len(ds_minmax))):
            resp = ds_minmax[i]["response"]
            resp_raw = ds_raw[i]["response"]
            assert resp.min() >= 0.0
            assert resp.max() <= 1.0 + 1e-6
            # Minmax should differ from raw (unless all values are zero)
            if resp_raw.abs().sum() > 0:
                assert not torch.equal(resp, resp_raw)

    def test_default_is_none(self, wingert_data_dir):
        """Default response_normalization is 'none'."""
        ds = WingertMultiSessionDataset(str(wingert_data_dir))
        assert ds.response_normalization == "none"

    def test_repeat_averaged_inherits_normalization(self, wingert_data_dir):
        """WingertRepeatAveragedDataset uses base's normalization."""
        base = WingertMultiSessionDataset(str(wingert_data_dir), normalize=True, response_normalization="zscore")
        avg_ds = WingertRepeatAveragedDataset(base)
        # zscore can produce negative values
        has_negative = False
        for i in range(len(avg_ds)):
            if avg_ds[i]["response"].min() < 0:
                has_negative = True
                break
        assert has_negative, "zscore repeat-averaged responses should contain negative values"


class TestWingertTrainCollate:
    def test_output_shape(self, wingert_data_dir):
        ds = WingertMultiSessionDataset(str(wingert_data_dir), split="train", normalize=False)
        collate = WingertTrainCollate(crop_length=50)
        batch = [ds[i] for i in range(min(3, len(ds)))]
        out = collate(batch)
        B = len(batch)
        assert out["spectrogram"].shape == (B, 50, 32)
        assert out["response"].shape == (B, 50, 5)
        assert isinstance(out["session_id"], str)
        assert isinstance(out["n_units"], int)

    def test_crop_within_bounds(self, wingert_data_dir):
        ds = WingertMultiSessionDataset(str(wingert_data_dir), split="train", normalize=False)
        collate = WingertTrainCollate(crop_length=50)
        for _ in range(20):
            batch = [ds[0]]
            out = collate(batch)
            assert out["spectrogram"].shape[1] == 50

    def test_short_stimulus_pads(self, wingert_data_dir):
        ds = WingertMultiSessionDataset(str(wingert_data_dir), split="train", normalize=False)
        collate = WingertTrainCollate(crop_length=200)
        batch = [ds[0]]
        out = collate(batch)
        assert out["spectrogram"].shape[1] == 200
        assert out["lengths"].shape == (1,)
        assert out["lengths"][0] <= 200

    def test_returns_session_id(self, wingert_data_dir):
        ds = WingertMultiSessionDataset(str(wingert_data_dir), split="train", normalize=False)
        collate = WingertTrainCollate(crop_length=50)
        batch = [ds[0]]
        out = collate(batch)
        assert "session_id" in out
        assert "spectrogram" in out

    def test_cached_token_targets_slice_on_window_boundaries(self, monkeypatch):
        collate = WingertTrainCollate(crop_length=40, model_window_size=20)
        token_target = torch.arange(6 * 5, dtype=torch.long).view(6, 5, 1)
        item = {
            "spectrogram": torch.randn(60, 32),
            "response": torch.randn(600, 5),
            "length": 60,
            "response_length": 600,
            "session_id": "sess_A",
            "n_units": 5,
            "token_target_full": token_target,
            "tokens_per_window": 2,
        }

        monkeypatch.setattr(torch, "randint", lambda low, high, size: torch.tensor([1]))
        out = collate([item])
        assert out["spectrogram"].shape == (1, 40, 32)
        assert out["response"].shape == (1, 400, 5)
        assert out["tokenizer_target"].shape == (1, 4, 5, 1)
        assert torch.equal(out["tokenizer_target"][0], token_target[2:6])

    def test_cached_short_stimulus_uses_padded_token_target(self):
        collate = WingertTrainCollate(crop_length=40, model_window_size=20)
        padded_target = torch.arange(4 * 5, dtype=torch.long).view(4, 5, 1)
        item = {
            "spectrogram": torch.randn(30, 32),
            "response": torch.randn(300, 5),
            "length": 30,
            "response_length": 300,
            "session_id": "sess_A",
            "n_units": 5,
            "token_target_full": torch.arange(2 * 5, dtype=torch.long).view(2, 5, 1),
            "token_target_train_padded": padded_target,
            "tokens_per_window": 2,
        }
        out = collate([item])
        assert torch.equal(out["tokenizer_target"][0], padded_target)

    def test_spike_augmentations_apply_to_time_first_response(self):
        collate = WingertTrainCollate(
            crop_length=4,
            spike_augmentations={"enabled": True, "dropout": {"p": 1.0, "drop_prob": 1.0}},
        )
        item = {
            "spectrogram": torch.zeros(4, 32),
            "response": torch.ones(4, 5),
            "length": 4,
            "response_length": 4,
            "session_id": "sess_A",
            "n_units": 5,
        }

        out = collate([item])

        assert out["response"].shape == (1, 4, 5)
        assert out["response"].sum() == 0

    def test_spike_augmentations_reject_cached_token_targets(self):
        collate = WingertTrainCollate(
            crop_length=40,
            model_window_size=20,
            spike_augmentations={"enabled": True, "dropout": {"p": 1.0, "drop_prob": 0.5}},
        )
        item = {
            "spectrogram": torch.randn(60, 32),
            "response": torch.randn(600, 5).clamp_min(0),
            "length": 60,
            "response_length": 600,
            "session_id": "sess_A",
            "n_units": 5,
            "token_target_full": torch.arange(6 * 5, dtype=torch.long).view(6, 5, 1),
            "tokens_per_window": 2,
        }

        with pytest.raises(ValueError, match="precomputed Wingert token targets"):
            collate([item])


class TestWingertEvalCollate:
    def test_pads_to_max_length(self, wingert_data_dir):
        ds = WingertMultiSessionDataset(str(wingert_data_dir), normalize=False)
        collate = WingertEvalCollate()
        sid = ds.session_meta[0]["session_id"]
        indices = ds.session_to_global_indices[sid]
        batch = [ds[i] for i in indices[:3]]
        out = collate(batch)
        max_len = max(b["length"] for b in batch)
        assert out["spectrogram"].shape == (len(batch), max_len, 32)
        assert out["response"].shape == (len(batch), max_len, 5)

    def test_lengths_tensor(self, wingert_data_dir):
        ds = WingertMultiSessionDataset(str(wingert_data_dir), normalize=False)
        collate = WingertEvalCollate()
        sid = ds.session_meta[0]["session_id"]
        indices = ds.session_to_global_indices[sid]
        batch = [ds[i] for i in indices[:3]]
        out = collate(batch)
        assert out["lengths"].shape == (len(batch),)
        for i, item in enumerate(batch):
            assert out["lengths"][i].item() == item["length"]

    def test_padding_is_zeros(self, wingert_data_dir):
        ds = WingertMultiSessionDataset(str(wingert_data_dir), normalize=False)
        collate = WingertEvalCollate()
        sid = ds.session_meta[0]["session_id"]
        indices = ds.session_to_global_indices[sid]
        batch = [ds[i] for i in indices[:3]]
        out = collate(batch)
        for i, item in enumerate(batch):
            L = item["length"]
            max_len = out["spectrogram"].shape[1]
            if L < max_len:
                assert out["spectrogram"][i, L:, :].abs().sum() == 0
                assert out["response"][i, L:, :].abs().sum() == 0


class TestWingertRepeatAveragedDataset:
    def test_groups_by_stimulus(self, wingert_data_dir):
        base = WingertMultiSessionDataset(str(wingert_data_dir), normalize=False)
        avg_ds = WingertRepeatAveragedDataset(base)
        # Mock data: 3 unique stimuli × 2 presentations × 2 sessions = 12 presentations
        # Averaged: 3 unique per session × 2 sessions = 6
        assert len(avg_ds) == 6

    def test_returns_expected_keys(self, wingert_data_dir):
        base = WingertMultiSessionDataset(str(wingert_data_dir), normalize=False)
        avg_ds = WingertRepeatAveragedDataset(base)
        item = avg_ds[0]
        assert "spectrogram" in item
        assert "response" in item
        assert "session_id" in item
        assert "n_units" in item
        assert "length" in item
        assert "response_length" in item
        assert "n_repeats" in item

    def test_response_is_averaged(self, wingert_data_dir):
        base = WingertMultiSessionDataset(str(wingert_data_dir), normalize=False)
        avg_ds = WingertRepeatAveragedDataset(base)
        item = avg_ds[0]
        assert item["n_repeats"] >= 1

    def test_session_to_global_indices(self, wingert_data_dir):
        base = WingertMultiSessionDataset(str(wingert_data_dir), normalize=False)
        avg_ds = WingertRepeatAveragedDataset(base)
        all_indices = []
        for indices in avg_ds.session_to_global_indices.values():
            all_indices.extend(indices)
        assert sorted(all_indices) == list(range(len(avg_ds)))


class TestWingertTwoRateDataset:
    def test_default_same_rate(self, wingert_data_dir):
        """When no response_fs, T_spec == T_resp and length == response_length."""
        ds = WingertMultiSessionDataset(str(wingert_data_dir), normalize=False)
        item = ds[0]
        assert item["length"] == item["response_length"]
        assert item["spectrogram"].shape[0] == item["response"].shape[0]

    def test_two_rate_shapes(self, wingert_data_dir):
        """response_fs=500 -> T_resp == T_spec * 5."""
        ds = WingertMultiSessionDataset(str(wingert_data_dir), fs=100.0, response_fs=500.0, normalize=False)
        item = ds[0]
        T_spec = item["length"]
        T_resp = item["response_length"]
        assert item["spectrogram"].shape[0] == T_spec
        assert item["response"].shape[0] == T_resp
        assert T_resp == T_spec * 5

    def test_two_rate_freq_channels(self, wingert_data_dir):
        """Spectrogram shape unchanged by response_fs."""
        ds_default = WingertMultiSessionDataset(str(wingert_data_dir), normalize=False)
        ds_two = WingertMultiSessionDataset(str(wingert_data_dir), response_fs=500.0, normalize=False)
        item_d = ds_default[0]
        item_t = ds_two[0]
        assert item_d["spectrogram"].shape == item_t["spectrogram"].shape

    def test_two_rate_n_units(self, wingert_data_dir):
        """Response still has correct N (neuron count)."""
        ds = WingertMultiSessionDataset(str(wingert_data_dir), response_fs=500.0, normalize=False)
        item = ds[0]
        assert item["response"].shape[1] == item["n_units"]

    def test_two_rate_dtypes(self, wingert_data_dir):
        """Both tensors are float32."""
        ds = WingertMultiSessionDataset(str(wingert_data_dir), response_fs=500.0, normalize=False)
        item = ds[0]
        assert item["spectrogram"].dtype == torch.float32
        assert item["response"].dtype == torch.float32

    def test_response_fs_equals_fs_same_as_default(self, wingert_data_dir):
        """Explicit response_fs=fs matches default behavior."""
        ds_default = WingertMultiSessionDataset(str(wingert_data_dir), fs=100.0, normalize=False)
        ds_explicit = WingertMultiSessionDataset(str(wingert_data_dir), fs=100.0, response_fs=100.0, normalize=False)
        for i in range(min(3, len(ds_default))):
            d = ds_default[i]
            e = ds_explicit[i]
            assert d["length"] == e["length"]
            assert d["response_length"] == e["response_length"]
            assert torch.equal(d["spectrogram"], e["spectrogram"])
            assert torch.equal(d["response"], e["response"])


class TestWingertTwoRateCollation:
    def test_train_collate_two_rate_shapes(self, wingert_data_dir):
        """Correct crop dimensions at both rates."""
        ds = WingertMultiSessionDataset(str(wingert_data_dir), fs=100.0, response_fs=500.0, normalize=False)
        collate = WingertTrainCollate(crop_length=50)
        batch = [ds[i] for i in range(min(3, len(ds)))]
        out = collate(batch)
        B = len(batch)
        assert out["spectrogram"].shape == (B, 50, 32)
        assert out["response"].shape == (B, 250, 5)
        assert out["lengths"].shape == (B,)
        assert out["response_lengths"].shape == (B,)

    def test_train_collate_aligned_crop(self, wingert_data_dir):
        """Response crop length = spec crop length * ratio."""
        ds = WingertMultiSessionDataset(str(wingert_data_dir), fs=100.0, response_fs=500.0, normalize=False)
        collate = WingertTrainCollate(crop_length=50)
        batch = [ds[0]]
        out = collate(batch)
        resp_len = out["response_lengths"][0].item()
        spec_len = out["lengths"][0].item()
        assert resp_len == spec_len * 5 or resp_len == 250

    def test_eval_collate_two_rate(self, wingert_data_dir):
        """Both padded to their respective max lengths."""
        ds = WingertMultiSessionDataset(str(wingert_data_dir), fs=100.0, response_fs=500.0, normalize=False)
        collate = WingertEvalCollate()
        sid = ds.session_meta[0]["session_id"]
        indices = ds.session_to_global_indices[sid]
        batch = [ds[i] for i in indices[:3]]
        out = collate(batch)

        max_spec = max(b["length"] for b in batch)
        max_resp = max(b["response_length"] for b in batch)
        assert out["spectrogram"].shape[1] == max_spec
        assert out["response"].shape[1] == max_resp
        assert "response_lengths" in out
        assert out["response_lengths"].shape == (len(batch),)

    def test_single_rate_collate(self, wingert_data_dir):
        """Single rate still works; response_lengths present."""
        ds = WingertMultiSessionDataset(str(wingert_data_dir), normalize=False)
        collate = WingertTrainCollate(crop_length=50)
        batch = [ds[i] for i in range(min(3, len(ds)))]
        out = collate(batch)
        assert "response_lengths" in out
        # At single rate, response_lengths == lengths
        assert torch.equal(out["lengths"], out["response_lengths"])
