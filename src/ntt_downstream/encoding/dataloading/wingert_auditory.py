"""Wingert auditory cortex multi-session dataset and collation utilities.

Each item is one stimulus presentation: (spectrogram, binned_response).
Preprocessing reproduces the published NEMS pipeline:
  - Spectrograms: log((x + 0.1) / 0.1), then per-channel minmax
  - Responses: rasterize at fs Hz, per-channel minmax
  - Normalization is over the full continuous recording per session
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ntt_downstream.encoding.dataloading.spike_augmentations import (
    apply_spike_augmentations,
    spike_augmentations_enabled,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def per_channel_minmax(arr: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """Minmax normalize each row (channel) independently. Matches NEMS."""
    mn = arr.min(axis=1, keepdims=True)
    shifted = arr - mn
    mx = shifted.max(axis=1, keepdims=True)
    mx[mx == 0] = 1
    normed = shifted / mx
    normed[normed < floor] = 0
    return normed


# ---------------------------------------------------------------------------
# Multi-session dataset
# ---------------------------------------------------------------------------


class WingertMultiSessionDataset(Dataset):
    """Multi-session auditory encoding dataset.

    Per-session data is pre-computed lazily on first access and cached
    per-process (PID-keyed), making this safe for num_workers > 0 and DDP.

    Args:
        data_dir: Directory containing exported .h5 session files.
        session_ids: Subset of session IDs to include. None = all.
        split: "train", "valid", "test", or None for all.
        fs: Sampling rate for spectrogram rasterization (default 100 Hz).
        response_fs: Sampling rate for response rasterization. None = use fs.
        normalize: Apply log-compress + minmax (default True).
        response_normalization: How to normalize responses.
            "none" — raw spike counts (for Poisson loss).
            "zscore" — per-neuron z-score from full recording (for MSE loss).
            "minmax" — per-channel minmax.
    """

    def __init__(
        self,
        data_dir: str,
        session_ids: list[str] | None = None,
        split: str | None = None,
        fs: float = 100.0,
        response_fs: float | None = None,
        normalize: bool = True,
        response_normalization: str = "none",
        token_target_cache_dir: str | None = None,
    ):
        self.fs = fs
        self.response_fs = response_fs if response_fs is not None else fs
        self.normalize = normalize
        self.response_normalization = response_normalization
        self.split = split
        self.token_target_cache_dir = token_target_cache_dir

        all_files = sorted(glob.glob(f"{data_dir}/*.h5"))
        if session_ids is not None:
            all_files = [f for f in all_files if Path(f).stem in session_ids]

        self.session_meta: list[dict] = []
        self._global_to_local: list[tuple[int, int]] = []
        self.session_boundaries: list[tuple[int, int]] = []

        global_offset = 0
        for fpath in all_files:
            sid = Path(fpath).stem
            with h5py.File(fpath, "r") as fh:
                n_units = len(cast(h5py.Dataset, fh["units/id"]))
                all_starts = np.array(fh["stimuli/start"])
                all_ends = np.array(fh["stimuli/end"])
                all_names = np.array(
                    [
                        n.decode() if isinstance(n, bytes) else str(n)
                        for n in cast(h5py.Dataset, fh["stimuli/epoch_name"])
                    ]
                )
                all_spec_idx = np.array(fh["stimuli/spectrogram_index"])

                total_bins = int(fh.attrs.get("recording_total_bins", 0))
                if total_bins == 0:
                    total_bins = int(np.round(float(np.array(fh["domain/end"])[-1]) * fs))

                if split is not None:
                    domain_key = f"{split}_domain"
                    if domain_key not in fh:
                        continue
                    d_s = np.array(fh[f"{domain_key}/start"])
                    d_e = np.array(fh[f"{domain_key}/end"])
                    mask = np.zeros(len(all_starts), dtype=bool)
                    for a, b in zip(d_s, d_e, strict=False):
                        mask |= (all_starts >= a - 0.01) & (all_starts <= b + 0.01)
                    all_starts = all_starts[mask]
                    all_ends = all_ends[mask]
                    all_names = all_names[mask]
                    all_spec_idx = all_spec_idx[mask]

            n_stim = len(all_starts)
            if n_stim == 0:
                continue

            self.session_meta.append(
                {
                    "session_id": sid,
                    "filepath": fpath,
                    "n_units": n_units,
                    "total_bins": total_bins,
                    "stim_starts": all_starts,
                    "stim_ends": all_ends,
                    "stim_names": all_names,
                    "stim_spec_idx": all_spec_idx,
                }
            )
            self.session_boundaries.append((global_offset, global_offset + n_stim))
            for local_idx in range(n_stim):
                self._global_to_local.append((len(self.session_meta) - 1, local_idx))
            global_offset += n_stim

        self.session_to_global_indices: dict[str, list[int]] = {}
        for sess_idx, meta in enumerate(self.session_meta):
            start, end = self.session_boundaries[sess_idx]
            self.session_to_global_indices[meta["session_id"]] = list(range(start, end))

        self._process_cache: dict[int, dict[int, dict]] = {}
        self._token_target_process_cache: dict[int, dict[int, dict[str, Any] | None]] = {}

    def _get_token_target_cache(self, sess_idx: int) -> dict[str, Any] | None:
        """Load optional precomputed token targets for one split/session."""
        if self.token_target_cache_dir is None or self.split is None:
            return None

        pid = os.getpid()
        if pid not in self._token_target_process_cache:
            self._token_target_process_cache[pid] = {}
        cache = self._token_target_process_cache[pid]
        if sess_idx in cache:
            return cache[sess_idx]

        sid = self.session_meta[sess_idx]["session_id"]
        cache_path = Path(self.token_target_cache_dir) / str(self.split) / f"{sid}.pt"
        if not cache_path.exists():
            cache[sess_idx] = None
            return None

        loaded = torch.load(cache_path, map_location="cpu", weights_only=False)
        items = loaded.get("items")
        if not isinstance(items, list):
            raise ValueError(f"Token target cache at {cache_path} is missing an 'items' list")
        cache[sess_idx] = loaded
        return loaded

    def _get_session_data(self, sess_idx: int) -> dict:
        """Lazily pre-compute and cache per-epoch data for a session.

        Spectrograms are cached as dense tensors (small at 100 Hz).
        Responses are cached as per-epoch spike timestamps + normalization
        stats, then binned on-the-fly in ``__getitem__`` to avoid holding
        large dense arrays at high response_fs.
        """
        pid = os.getpid()
        if pid not in self._process_cache:
            self._process_cache[pid] = {}
        cache = self._process_cache[pid]

        if sess_idx not in cache:
            meta = self.session_meta[sess_idx]
            fh = h5py.File(meta["filepath"], "r")

            timestamps = np.array(fh["spikes/timestamps"])
            unit_index = np.array(fh["spikes/unit_index"])
            spec_data = np.array(fh["spectrograms/data"])
            n_freq = spec_data.shape[1]
            n_units = meta["n_units"]
            total_bins = meta["total_bins"]
            total_resp_bins = int(np.round((total_bins / self.fs) * self.response_fs))

            # All stimulus intervals (unfiltered, needed for normalization)
            all_s = np.array(fh["stimuli/start"])
            all_e = np.array(fh["stimuli/end"])
            all_si = np.array(fh["stimuli/spectrogram_index"])

            # Build full continuous spectrogram at self.fs
            stim_c = np.zeros((n_freq, total_bins), dtype=np.float32)
            for i in range(len(all_s)):
                nb = int(np.round((all_e[i] - all_s[i]) * self.fs))
                sb = int(np.round(all_s[i] * self.fs))
                eb = min(sb + nb, total_bins)
                stim_c[:, sb:eb] = spec_data[all_si[i]][:, : eb - sb]

            if self.normalize:
                stim_c = np.log((stim_c + 0.1) / 0.1)
                stim_c = per_channel_minmax(stim_c)

            # Compute response normalization stats from full continuous recording.
            # We only keep the per-unit stats — the dense array is freed.
            resp_min = None
            resp_range = None
            resp_mean = None
            resp_std = None
            if self.normalize and self.response_normalization not in ("none", "sqrt"):
                resp_c = np.zeros((n_units, total_resp_bins), dtype=np.float32)
                total_duration = total_bins / self.fs
                valid = timestamps < total_duration
                b = np.clip(np.floor(timestamps[valid] * self.response_fs).astype(int), 0, total_resp_bins - 1)
                np.add.at(resp_c, (unit_index[valid], b), 1)
                if self.response_normalization == "minmax":
                    resp_min = resp_c.min(axis=1).astype(np.float32)  # (n_units,)
                    resp_shifted = resp_c - resp_min[:, None]
                    resp_range = resp_shifted.max(axis=1).astype(np.float32)  # (n_units,)
                    resp_range[resp_range == 0] = 1.0
                    del resp_shifted
                elif self.response_normalization == "zscore":
                    resp_mean = resp_c.mean(axis=1).astype(np.float32)  # (n_units,)
                    resp_std = resp_c.std(axis=1).astype(np.float32)  # (n_units,)
                    resp_std[resp_std == 0] = 1.0
                del resp_c

            # Extract per-epoch data
            starts = meta["stim_starts"]
            ends = meta["stim_ends"]
            total_duration = total_bins / self.fs
            valid_mask = timestamps < total_duration
            valid_ts = timestamps[valid_mask]
            valid_ui = unit_index[valid_mask]

            epochs = []
            for i in range(len(starts)):
                # Spectrogram window at self.fs (dense, small)
                sb = int(np.round(starts[i] * self.fs))
                ep_len = int(np.round((ends[i] - starts[i]) * self.fs))
                eb = min(sb + ep_len, total_bins)
                len_spec = eb - sb
                x = torch.from_numpy(stim_c[:, sb:eb].astype(np.float32).T)  # (len_spec, n_freq)

                # Response: store spike timestamps + unit indices (tiny)
                len_resp = int(np.round((ends[i] - starts[i]) * self.response_fs))
                spike_mask = (valid_ts >= starts[i]) & (valid_ts < ends[i])
                ep_ts = (valid_ts[spike_mask] - starts[i]).astype(np.float32)
                ep_ui = valid_ui[spike_mask].astype(np.int32)

                epochs.append((x, ep_ts, ep_ui, len_spec, len_resp))

            fh.close()
            del stim_c, timestamps, unit_index, spec_data, valid_ts, valid_ui
            cache[sess_idx] = {
                "epochs": epochs,
                "resp_min": resp_min,
                "resp_range": resp_range,
                "resp_mean": resp_mean,
                "resp_std": resp_std,
                "n_units": n_units,
            }

        return cache[sess_idx]

    def _bin_spikes(self, ep_ts: np.ndarray, ep_ui: np.ndarray, n_units: int, len_resp: int) -> torch.Tensor:
        """Bin per-epoch spike timestamps into a dense (len_resp, n_units) tensor."""
        resp = np.zeros((len_resp, n_units), dtype=np.float32)
        if len(ep_ts) > 0:
            bins = np.clip(np.floor(ep_ts * self.response_fs).astype(np.int32), 0, len_resp - 1)
            np.add.at(resp, (bins, ep_ui), 1)
        return torch.from_numpy(resp)

    def __len__(self) -> int:
        return len(self._global_to_local)

    def __getitem__(self, idx: int) -> dict:
        sess_idx, local_idx = self._global_to_local[idx]
        meta = self.session_meta[sess_idx]
        data = self._get_session_data(sess_idx)
        spec, ep_ts, ep_ui, length, resp_length = data["epochs"][local_idx]

        resp = self._bin_spikes(ep_ts, ep_ui, data["n_units"], resp_length)

        if self.normalize:
            if self.response_normalization == "minmax" and data["resp_min"] is not None:
                resp = (resp - data["resp_min"][None, :]) / data["resp_range"][None, :]
                resp[resp < 1e-8] = 0
            elif self.response_normalization == "zscore" and data["resp_mean"] is not None:
                resp = (resp - data["resp_mean"][None, :]) / data["resp_std"][None, :]
            elif self.response_normalization == "sqrt":
                resp = torch.sqrt(resp)

        out = {
            "spectrogram": spec,
            "response": resp,
            "epoch_name": meta["stim_names"][local_idx],
            "session_id": meta["session_id"],
            "n_units": meta["n_units"],
            "length": length,
            "response_length": resp_length,
        }
        token_cache = self._get_token_target_cache(sess_idx)
        if token_cache is not None:
            cache_items = token_cache["items"]
            if local_idx >= len(cache_items):
                raise IndexError(
                    f"Token target cache for session {meta['session_id']} has {len(cache_items)} items, "
                    f"but local_idx={local_idx} was requested"
                )
            token_item = cache_items[local_idx]
            out["token_target_full"] = token_item["full_target"]
            out["tokens_per_window"] = int(token_item["tokens_per_window"])
            train_padded = token_item.get("train_padded_target")
            if train_padded is not None:
                out["token_target_train_padded"] = train_padded
            out["token_target_is_discrete"] = bool(token_item.get("is_discrete", False))
        return out


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


class WingertTrainCollate:
    """Collate with random cropping for training.

    Samples a random contiguous window of crop_length bins from each item.
    If a stimulus is shorter than crop_length, it is zero-padded.

    Args:
        crop_length: Number of time bins to crop to.
    """

    def __init__(
        self,
        crop_length: int,
        model_window_size: int | None = None,
        spike_augmentations=None,
    ):
        self.crop_length = crop_length
        self.model_window_size = model_window_size
        self.spike_augmentations = spike_augmentations

    def __call__(self, batch: list[dict]) -> dict:
        specs, resps, lengths, resp_lengths = [], [], [], []
        token_targets = []
        C = self.crop_length
        use_cached_targets = "token_target_full" in batch[0]
        if use_cached_targets and spike_augmentations_enabled(self.spike_augmentations):
            raise ValueError(
                "Spike augmentations cannot be used with precomputed Wingert token targets. "
                "Disable data.token_target_cache_dir so tokenizer targets are computed from augmented responses."
            )
        crop_windows = None
        if use_cached_targets:
            if self.model_window_size is None:
                raise ValueError("model_window_size is required when collating cached token targets")
            if C % self.model_window_size != 0:
                raise ValueError(
                    f"crop_length={C} must be divisible by model_window_size={self.model_window_size} "
                    "for cached token-target slicing."
                )
            crop_windows = C // self.model_window_size

        for item in batch:
            spec = item["spectrogram"]  # (T_spec, F)
            resp = item["response"]  # (T_resp, N)
            T_spec = item["length"]
            T_resp = item["response_length"]
            ratio = T_resp / T_spec if T_spec > 0 else 1.0
            C_resp = int(round(C * ratio))
            token_target_crop = None

            if T_spec <= C:
                pad_spec = C - T_spec
                pad_resp = C_resp - T_resp
                spec_crop = torch.nn.functional.pad(spec[:T_spec], (0, 0, 0, pad_spec))
                resp_crop = torch.nn.functional.pad(resp[:T_resp], (0, 0, 0, max(pad_resp, 0)))
                lengths.append(T_spec)
                resp_lengths.append(T_resp)
                if use_cached_targets:
                    token_target_crop = item.get("token_target_train_padded")
                    if token_target_crop is None:
                        raise ValueError(
                            "Cached token targets are enabled, but item is missing token_target_train_padded "
                            "for a short stimulus."
                        )
            else:
                if use_cached_targets:
                    # Both are set whenever use_cached_targets is true (see above).
                    assert self.model_window_size is not None and crop_windows is not None
                    tokens_per_window = int(item["tokens_per_window"])
                    full_windows = item["token_target_full"].shape[0] // tokens_per_window
                    if full_windows < crop_windows:
                        raise ValueError(
                            f"Cached token target has only {full_windows} windows, but crop requires {crop_windows}"
                        )
                    start_window = torch.randint(0, full_windows - crop_windows + 1, (1,)).item()
                    start_spec = start_window * self.model_window_size
                    token_start = start_window * tokens_per_window
                    token_end = token_start + crop_windows * tokens_per_window
                    token_target_crop = item["token_target_full"][token_start:token_end]
                else:
                    start_spec = torch.randint(0, T_spec - C, (1,)).item()
                start_resp = int(round(start_spec * ratio))
                spec_crop = spec[start_spec : start_spec + C]
                resp_crop = resp[start_resp : start_resp + C_resp]
                # Handle edge case: resp slice may be slightly short
                if resp_crop.shape[0] < C_resp:
                    pad = C_resp - resp_crop.shape[0]
                    resp_crop = torch.nn.functional.pad(resp_crop, (0, 0, 0, pad))
                lengths.append(C)
                resp_lengths.append(C_resp)

            specs.append(spec_crop)
            resps.append(resp_crop)
            if use_cached_targets:
                token_targets.append(token_target_crop)

        response = torch.stack(resps)
        if self.spike_augmentations is not None:
            response = apply_spike_augmentations(response, self.spike_augmentations, time_dim=1)

        out = {
            "spectrogram": torch.stack(specs),
            "response": response,
            "session_id": batch[0]["session_id"],
            "n_units": batch[0]["n_units"],
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "response_lengths": torch.tensor(resp_lengths, dtype=torch.long),
        }
        if use_cached_targets:
            out["tokenizer_target"] = torch.stack(token_targets)
            out["tokenizer_tokens_per_window"] = int(batch[0]["tokens_per_window"])
            assert crop_windows is not None  # set whenever use_cached_targets is true
            out["tokenizer_n_windows"] = int(crop_windows)
        return out


class WingertEvalCollate:
    """Collate with zero-padding to max length within batch for evaluation.

    Returns a lengths tensor so metrics can mask padded positions.
    """

    def __call__(self, batch: list[dict]) -> dict:
        lengths = [item["length"] for item in batch]
        resp_lengths = [item["response_length"] for item in batch]
        max_len = max(lengths)
        max_resp_len = max(resp_lengths)

        specs, resps = [], []
        for item in batch:
            spec = item["spectrogram"]
            resp = item["response"]
            T_spec = item["length"]
            T_resp = item["response_length"]

            pad_spec = max_len - T_spec
            if pad_spec > 0:
                spec = torch.nn.functional.pad(spec[:T_spec], (0, 0, 0, pad_spec))
            else:
                spec = spec[:T_spec]

            pad_resp = max_resp_len - T_resp
            if pad_resp > 0:
                resp = torch.nn.functional.pad(resp[:T_resp], (0, 0, 0, pad_resp))
            else:
                resp = resp[:T_resp]

            specs.append(spec)
            resps.append(resp)

        return {
            "spectrogram": torch.stack(specs),
            "response": torch.stack(resps),
            "session_id": batch[0]["session_id"],
            "n_units": batch[0]["n_units"],
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "response_lengths": torch.tensor(resp_lengths, dtype=torch.long),
            "epoch_names": [item["epoch_name"] for item in batch],
        }


# ---------------------------------------------------------------------------
# Repeat-averaged dataset
# ---------------------------------------------------------------------------


class WingertRepeatAveragedDataset(Dataset):
    """Wraps a WingertMultiSessionDataset to average responses across repeats.

    Groups presentations by (session_id, epoch_name) and returns the mean
    response across repeats. Used for noise-ceiling evaluation.

    Args:
        base_dataset: The underlying multi-session dataset.
    """

    def __init__(self, base_dataset: WingertMultiSessionDataset):
        self.base = base_dataset
        self._averaged_items: list[dict] = []
        self.session_to_global_indices: dict[str, list[int]] = {}

        global_idx = 0
        for sess_idx, meta in enumerate(base_dataset.session_meta):
            sid = meta["session_id"]
            self.session_to_global_indices[sid] = []

            name_to_locals: dict[str, list[int]] = {}
            for local_idx, name in enumerate(meta["stim_names"]):
                name_to_locals.setdefault(name, []).append(local_idx)

            for name, local_indices in name_to_locals.items():
                self._averaged_items.append(
                    {
                        "sess_idx": sess_idx,
                        "epoch_name": name,
                        "local_indices": local_indices,
                        "n_repeats": len(local_indices),
                    }
                )
                self.session_to_global_indices[sid].append(global_idx)
                global_idx += 1

    def __len__(self) -> int:
        return len(self._averaged_items)

    def __getitem__(self, idx: int) -> dict:
        info = self._averaged_items[idx]
        sess_idx = info["sess_idx"]
        meta = self.base.session_meta[sess_idx]
        data = self.base._get_session_data(sess_idx)
        n_units = data["n_units"]

        local_indices = info["local_indices"]
        resps = []
        for li in local_indices:
            _spec, ep_ts, ep_ui, _length, resp_length = data["epochs"][li]
            resp = self.base._bin_spikes(ep_ts, ep_ui, n_units, resp_length)
            if self.base.normalize:
                if self.base.response_normalization == "minmax" and data["resp_min"] is not None:
                    resp = (resp - data["resp_min"][None, :]) / data["resp_range"][None, :]
                    resp[resp < 1e-8] = 0
                elif self.base.response_normalization == "zscore" and data["resp_mean"] is not None:
                    resp = (resp - data["resp_mean"][None, :]) / data["resp_std"][None, :]
                elif self.base.response_normalization == "sqrt":
                    resp = torch.sqrt(resp)
            resps.append(resp)

        spec, _ep_ts, _ep_ui, length, resp_length = data["epochs"][local_indices[0]]
        mean_resp = torch.stack(resps).mean(dim=0)

        return {
            "spectrogram": spec,
            "response": mean_resp,
            "session_id": meta["session_id"],
            "n_units": meta["n_units"],
            "length": length,
            "response_length": resp_length,
            "epoch_name": info["epoch_name"],
            "n_repeats": info["n_repeats"],
        }
