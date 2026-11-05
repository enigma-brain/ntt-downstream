"""Cowley V4 multi-session dataset and session-aware batch samplers.

Provides (image, binned_spike_response) pairs per stimulus presentation across
multiple recording sessions. Sessions have different numbers of units (72-96),
so batches must come from a single session to avoid padding the neuron dimension.
"""

from __future__ import annotations

import copy
import glob
import os
import warnings
from typing import cast

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from temporaldata import Data
from torch.utils.data import Dataset

from ntt_downstream.encoding.dataloading.spike_augmentations import apply_spike_augmentations


class CowleyV4MultiSessionDataset(Dataset):
    """MapDataset over stimulus presentations across multiple Cowley V4 sessions.

    Each item returns a dict with:
        image:      (3, H, W) float32 in [0, 1] (or ImageNet-normalized if enabled)
        response:   (n_units, n_time_bins) float32 binned spike counts
        image_id:   int
        session_id: str
        n_units:    int

    Args:
        data_dir: Directory containing per-session HDF5 files.
        response_window: Duration of the response window in seconds.
        bin_size: Seconds per time bin (default 1ms).
        response_offset: Seconds from stimulus onset to start of response window
            (default 0.0). E.g., 0.050 to skip the first 50ms of neural latency.
        session_ids: Optional list of session IDs to include.
        split: If given ("train", "valid", "test"), restrict to stimulus
            presentations whose image belongs to that split (per session).
        author_split: If given ("train", "val", "test"), only include sessions
            with that author-designated split.
        image_size: Optional (H, W) tuple. If provided and different from native
            resolution, images are resized via bilinear interpolation.

    Multi-worker & DDP safe: HDF5 files are opened lazily per-process (keyed
    by PID), so each DataLoader worker and DDP rank gets its own file handles.

    Returns numpy arrays from __getitem__. Use numpy_collate_fn for the DataLoader.
    Sessions without embedded images are skipped with a warning.
    """

    # ImageNet normalization constants (used when imagenet_normalize=True)
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def __init__(
        self,
        data_dir: str,
        response_window: float = 0.200,
        bin_size: float = 0.001,
        response_offset: float = 0.0,
        session_ids: list[str] | None = None,
        split: str | None = None,
        author_split: str | None = None,
        image_size: tuple[int, int] | None = None,
        subjects: str | list[str] | None = None,
        min_snr: float | None = None,
        imagenet_normalize: bool = False,
    ):
        self.data_dir = data_dir
        self.response_window = response_window
        self.bin_size = bin_size
        self.response_offset = response_offset
        self.n_time_bins = int(round(response_window / bin_size))
        self.split = split
        self.image_size = image_size
        self.imagenet_normalize = imagenet_normalize

        # Normalize subjects filter to a list (or None)
        if isinstance(subjects, str):
            subjects = [subjects]
        self._subjects_filter = subjects

        # Discover session files
        all_files = sorted(glob.glob(f"{data_dir}/*.h5"))
        if session_ids is not None:
            all_files = [f for f in all_files if any(s in f for s in session_ids)]

        # Phase 1: quick metadata scan (open/close each file)
        self.session_meta: list[dict] = []
        self._global_to_local: list[tuple[int, int]] = []
        self.session_boundaries: list[tuple[int, int]] = []
        skipped = []
        skipped_author = []
        skipped_subject = []

        global_offset = 0
        for fpath in all_files:
            sid = fpath.split("/")[-1].replace(".h5", "")
            with h5py.File(fpath, "r") as fh:
                if "images" not in fh:
                    skipped.append(sid)
                    continue

                # Read session metadata from temporaldata
                data_tmp = Data.from_hdf5(fh)

                # Filter by author_split if requested
                sess_author_split = getattr(data_tmp, "author_split", None)
                if author_split is not None and sess_author_split != author_split:
                    skipped_author.append(sid)
                    continue

                # Extract subject_id and filter by subjects if requested
                # `subject` is a dynamic temporaldata attribute; sessions may lack it.
                sess_subject_id = getattr(getattr(data_tmp, "subject", None), "id", None)
                if self._subjects_filter is not None and sess_subject_id not in self._subjects_filter:
                    skipped_subject.append(sid)
                    continue

                # Always read full image_id array for the lookup table
                all_stim_image_ids = np.array(fh["stimuli/image_id"])
                stim_onsets = np.array(fh["stimuli/start"])
                stim_image_ids = all_stim_image_ids
                n_units = cast(h5py.Dataset, fh["units/id"]).shape[0]

                # Read native image size
                native_h = cast(h5py.Dataset, fh["images/pixels"]).shape[1]
                native_w = cast(h5py.Dataset, fh["images/pixels"]).shape[2]

                # Filter by split mask if requested
                if split is not None:
                    mask_key = f"stimuli/{split}_mask"
                    if mask_key not in fh:
                        raise ValueError(f"No {mask_key} in {fpath} — was the data exported with image-based splits?")
                    mask = np.array(fh[mask_key])
                    stim_onsets = stim_onsets[mask]
                    stim_image_ids = stim_image_ids[mask]

                # Build image lookup from full (unfiltered) image_ids
                img_id_to_idx = {int(iid): i for i, iid in enumerate(np.unique(all_stim_image_ids))}

            n_stim = len(stim_onsets)
            if n_stim == 0:
                continue

            self.session_meta.append(
                {
                    "session_id": sid,
                    "subject_id": sess_subject_id,
                    "filepath": fpath,
                    "n_units": n_units,
                    "author_split": sess_author_split,
                    "stim_onsets": stim_onsets,
                    "stim_image_ids": stim_image_ids,
                    "img_id_to_idx": img_id_to_idx,
                    "native_hw": (native_h, native_w),
                }
            )
            self.session_boundaries.append((global_offset, global_offset + n_stim))
            for local_idx in range(n_stim):
                self._global_to_local.append((len(self.session_meta) - 1, local_idx))
            global_offset += n_stim

        if skipped:
            warnings.warn(f"Skipped {len(skipped)} sessions without images: {skipped}", stacklevel=2)
        if skipped_author:
            warnings.warn(
                f"Skipped {len(skipped_author)} sessions not matching author_split={author_split!r}: {skipped_author}",
                stacklevel=2,
            )
        if skipped_subject:
            warnings.warn(
                f"Skipped {len(skipped_subject)} sessions not matching "
                f"subjects={self._subjects_filter!r}: {skipped_subject}",
                stacklevel=2,
            )

        # Phase 2: optional SNR-based unit filtering
        if min_snr is not None and self.session_meta:
            self._apply_snr_filter(min_snr)

        # Session index for batch sampler
        self.session_to_global_indices: dict[str, list[int]] = {}
        for sess_idx, meta in enumerate(self.session_meta):
            start, end = self.session_boundaries[sess_idx]
            self.session_to_global_indices[meta["session_id"]] = list(range(start, end))

        # Per-process HDF5 handles (lazily opened, keyed by PID)
        self._process_cache: dict[int, dict] = {}

        # Determine if resizing is needed
        if self.image_size is not None and len(self.session_meta) > 0:
            native_hw = self.session_meta[0]["native_hw"]
            self._needs_resize = self.image_size != native_hw
        else:
            self._needs_resize = False

        total = len(self._global_to_local)
        if self.session_meta:
            min_units = min(m["n_units"] for m in self.session_meta)
            max_units = max(m["n_units"] for m in self.session_meta)
            n_range = f"{min_units}–{max_units}"
        else:
            n_range = "0"
        split_str = f", split={split}" if split else ""
        author_str = f", author_split={author_split}" if author_split else ""
        resize_str = f", resize={self.image_size}" if self._needs_resize else ""
        print(
            f"CowleyV4MultiSessionDataset: {len(self.session_meta)} sessions, "
            f"{total} presentations, n_units: {n_range}"
            f"{split_str}{author_str}{resize_str}"
        )

    def _apply_snr_filter(self, min_snr: float):
        """Compute per-unit SNR for each session and filter out low-SNR units.

        Uses the unbiased SNR metric from Pospisil & Bair (2021), matching
        Cowley et al.'s ``compute_SNRs`` implementation. SNR is always computed
        on ALL presentations (ignoring split) so the result is stable across
        train/val/test. Values are cached in the HDF5 file for fast reuse.
        """
        for meta in self.session_meta:
            snr = self._get_or_compute_unit_snr(meta)
            kept = np.where(snr > min_snr)[0]
            n_orig = meta["n_units"]
            if len(kept) < n_orig:
                meta["kept_units"] = kept
                meta["n_units_original"] = n_orig
                meta["n_units"] = len(kept)
                print(f"  SNR filter ({meta['session_id']}): {n_orig} → {len(kept)} units (threshold={min_snr})")

    @staticmethod
    def _snr_cache_key(response_offset: float, response_window: float) -> str:
        """Key for cached SNR values within the sidecar .npz file."""
        return f"snr_off{response_offset:.4f}_win{response_window:.4f}"

    @staticmethod
    def _snr_cache_path(h5_filepath: str) -> str:
        """Path to the sidecar SNR cache file alongside the HDF5."""
        return h5_filepath.replace(".h5", ".snr_cache.npz")

    def _get_or_compute_unit_snr(self, meta: dict) -> np.ndarray:
        """Load cached SNR or compute and cache it in a sidecar .npz file.

        SNR is computed on ALL presentations in the session (ignoring any
        split filter) so the metric is stable regardless of train/val/test.
        """
        cache_key = self._snr_cache_key(self.response_offset, self.response_window)
        cache_path = self._snr_cache_path(meta["filepath"])

        # Try reading from cache
        try:
            cached = np.load(cache_path)
            if cache_key in cached:
                return cached[cache_key]
        except Exception:
            pass

        # Compute from raw spikes (using ALL presentations, not split-filtered)
        snr = self._compute_unit_snr(meta)

        # Save to sidecar cache (merge with existing entries if any)
        try:
            existing = {}
            try:
                existing = dict(np.load(cache_path))
            except Exception:
                pass
            existing[cache_key] = snr
            np.savez(cache_path, **existing)
        except Exception:
            pass  # read-only filesystem — skip caching

        return snr

    def _compute_unit_snr(self, meta: dict) -> np.ndarray:
        """Compute unbiased SNR per unit for one session (Pospisil & Bair 2021).

        Always uses ALL presentations from the HDF5 file (ignoring any split
        filter on ``meta["stim_onsets"]``) so the metric is stable.

        Returns:
            snr: (n_units,) array of SNR values.
        """
        from collections import defaultdict

        n_units = meta["n_units"]

        with h5py.File(meta["filepath"], "r") as fh:
            # Read ALL stim onsets/image_ids (not the split-filtered ones in meta)
            all_stim_onsets = np.array(fh["stimuli/start"])
            all_stim_image_ids = np.array(fh["stimuli/image_id"])
            spike_ts = np.array(fh["spikes/timestamps"])
            spike_ui = np.array(fh["spikes/unit_index"])

        # Use searchsorted for fast binning
        sort_order = np.argsort(spike_ts)
        spike_ts = spike_ts[sort_order]
        spike_ui = spike_ui[sort_order]

        t_starts = all_stim_onsets + self.response_offset
        t_ends = all_stim_onsets + self.response_offset + self.response_window
        left_idx = np.searchsorted(spike_ts, t_starts, side="left")
        right_idx = np.searchsorted(spike_ts, t_ends, side="left")

        # Group presentations by image, accumulate spike counts
        image_presentations: dict[int, list[np.ndarray]] = defaultdict(list)
        for i in range(len(all_stim_onsets)):
            counts = np.zeros(n_units, dtype=np.float64)
            li, ri = left_idx[i], right_idx[i]
            if ri > li:
                np.add.at(counts, spike_ui[li:ri], 1)
            image_presentations[int(all_stim_image_ids[i])].append(counts)

        # Build (n_units, n_images, max_repeats) array with NaN padding
        sorted_ids = sorted(image_presentations.keys())
        n_images = len(sorted_ids)
        max_repeats = max(len(v) for v in image_presentations.values())

        raw = np.full((n_units, n_images, max_repeats), np.nan, dtype=np.float64)
        for j, img_id in enumerate(sorted_ids):
            reps = np.array(image_presentations[img_id])  # (n_reps, n_units)
            raw[:, j, : reps.shape[0]] = reps.T

        # Cowley's unbiased SNR (Pospisil & Bair 2021)
        repeats_per_image = np.sum(~np.isnan(raw[0]), axis=1)
        n = np.median(repeats_per_image)
        m = n_images

        vars_resid = np.mean(np.nanvar(raw, axis=2, ddof=1), axis=1)  # (n_units,)
        vars_stim = np.var(np.nanmean(raw, axis=2), axis=1)  # (n_units,)
        vars_stim = vars_stim - (m - 1) / m * vars_resid / n  # unbias

        # Avoid division by zero for silent units
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = vars_stim / vars_resid
        snr = np.nan_to_num(snr, nan=-np.inf)

        return snr

    def _get_session_handles(self, sess_idx: int):
        """Lazily open HDF5 + create lazy Data per process."""
        pid = os.getpid()
        if pid not in self._process_cache:
            self._process_cache[pid] = {}

        cache = self._process_cache[pid]
        if sess_idx not in cache:
            meta = self.session_meta[sess_idx]
            fh = h5py.File(meta["filepath"], "r")
            data = Data.from_hdf5(fh, lazy=True)
            cache[sess_idx] = {"fh": fh, "data": data}

        return cache[sess_idx]

    def __len__(self):
        return len(self._global_to_local)

    def __getitem__(self, idx):
        sess_idx, local_idx = self._global_to_local[idx]
        meta = self.session_meta[sess_idx]
        handles = self._get_session_handles(sess_idx)

        onset = float(meta["stim_onsets"][local_idx])
        image_id = int(meta["stim_image_ids"][local_idx])
        n_units = meta["n_units"]

        # Lazy copy + slice: O(1), no disk read yet
        data = copy.copy(handles["data"])
        sliced = data.slice(onset + self.response_offset, onset + self.response_offset + self.response_window)

        # Accessing attributes triggers lazy HDF5 read of just this slice.
        # temporaldata.Data.slice() with reset_origin=True (the default)
        # returns timestamps relative to the window start, so spike_ts
        # ranges from 0 to response_window. No need to subtract window_start.
        spike_ts = np.array(sliced.spikes.timestamps)
        spike_ui = np.array(sliced.spikes.unit_index)

        n_units_bin = meta.get("n_units_original", n_units)
        response = np.zeros((n_units_bin, self.n_time_bins), dtype=np.float32)
        if len(spike_ts) > 0:
            bin_idx = np.minimum((spike_ts / self.bin_size).astype(int), self.n_time_bins - 1)
            np.add.at(response, (spike_ui, bin_idx), 1)

        # Apply SNR unit filter if active
        kept_units = meta.get("kept_units")
        if kept_units is not None:
            response = response[kept_units]

        # Image: single-row HDF5 read, CHW format
        img_lookup_idx = meta["img_id_to_idx"][image_id]
        image = np.array(handles["fh"]["images/pixels"][img_lookup_idx]).astype(np.float32) / 255.0
        image = np.ascontiguousarray(image.transpose(2, 0, 1))  # (3, H, W)

        # ImageNet normalization for pretrained backbones
        if self.imagenet_normalize:
            image = (image - self.IMAGENET_MEAN) / self.IMAGENET_STD

        # Optional resize
        if self._needs_resize:
            image_t = torch.from_numpy(image).unsqueeze(0)  # (1, 3, H, W)
            image_t = F.interpolate(image_t, size=self.image_size, mode="bilinear", align_corners=False)
            image = image_t.squeeze(0).numpy()

        return {
            "image": image,
            "response": response,
            "image_id": image_id,
            "session_id": meta["session_id"],
            "subject_id": meta["subject_id"],
            "n_units": n_units,
        }

    def close(self):
        for cache in self._process_cache.values():
            for handles in cache.values():
                handles["fh"].close()
        self._process_cache.clear()


class CowleyV4RepeatAveragedDataset(Dataset):
    """Dataset yielding (image, mean_response) per unique image per session.

    For each unique image in a session, all single-trial responses are loaded
    and averaged. This is used for repeat-averaged evaluation: the model predicts
    once per image, and we compare against the trial-averaged neural response.

    Uses the same lazy HDF5 loading as CowleyV4MultiSessionDataset.
    Returns numpy arrays; use ``numpy_collate_fn`` for the DataLoader.

    Args:
        base_dataset: A CowleyV4MultiSessionDataset instance to derive from.
            Must already be constructed (metadata loaded). The base dataset's
            HDF5 handles and session metadata are reused.
    """

    def __init__(self, base_dataset: CowleyV4MultiSessionDataset):
        self.base = base_dataset
        self.n_time_bins = base_dataset.n_time_bins
        self.bin_size = base_dataset.bin_size
        self.response_window = base_dataset.response_window
        self.response_offset = base_dataset.response_offset
        self._needs_resize = base_dataset._needs_resize
        self.image_size = base_dataset.image_size

        # Build index: (sess_idx, image_id) → list of local_indices into base
        # Then flatten to a list of (sess_idx, image_id, [local_indices])
        self._items: list[tuple[int, int, list[int]]] = []
        self.session_to_global_indices: dict[str, list[int]] = {}

        global_offset = 0
        for sess_idx, meta in enumerate(base_dataset.session_meta):
            image_to_locals: dict[int, list[int]] = {}
            for local_idx, iid in enumerate(meta["stim_image_ids"]):
                image_to_locals.setdefault(int(iid), []).append(local_idx)

            sid = meta["session_id"]
            session_indices = []
            for image_id, local_indices in image_to_locals.items():
                self._items.append((sess_idx, image_id, local_indices))
                session_indices.append(global_offset)
                global_offset += 1
            self.session_to_global_indices[sid] = session_indices

        # Copy session_meta reference for SessionBatchSampler compatibility
        self.session_meta = base_dataset.session_meta

        total = len(self._items)
        n_sessions = len(base_dataset.session_meta)
        avg_repeats = np.mean([len(item[2]) for item in self._items]) if self._items else 0
        print(
            f"CowleyV4RepeatAveragedDataset: {n_sessions} sessions, "
            f"{total} unique images, avg {avg_repeats:.1f} repeats/image"
        )

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        sess_idx, image_id, local_indices = self._items[idx]
        meta = self.base.session_meta[sess_idx]
        handles = self.base._get_session_handles(sess_idx)
        n_units = meta["n_units"]

        # Average responses across all repeats of this image
        n_units_bin = meta.get("n_units_original", n_units)
        sum_response = np.zeros((n_units_bin, self.n_time_bins), dtype=np.float64)
        for local_idx in local_indices:
            onset = float(meta["stim_onsets"][local_idx])
            data = copy.copy(handles["data"])
            sliced = data.slice(onset + self.response_offset, onset + self.response_offset + self.response_window)
            # temporaldata.Data.slice() with reset_origin=True (the default)
            # returns timestamps relative to the window start, so spike_ts
            # ranges from 0 to response_window. No need to subtract window_start.
            spike_ts = np.array(sliced.spikes.timestamps)
            spike_ui = np.array(sliced.spikes.unit_index)

            response = np.zeros((n_units_bin, self.n_time_bins), dtype=np.float64)
            if len(spike_ts) > 0:
                bin_idx = np.minimum((spike_ts / self.bin_size).astype(int), self.n_time_bins - 1)
                np.add.at(response, (spike_ui, bin_idx), 1)
            sum_response += response

        # Apply SNR unit filter if active
        kept_units = meta.get("kept_units")
        if kept_units is not None:
            sum_response = sum_response[kept_units]

        mean_response = (sum_response / len(local_indices)).astype(np.float32)

        # Image (same for all repeats — load once)
        img_lookup_idx = meta["img_id_to_idx"][image_id]
        image = np.array(handles["fh"]["images/pixels"][img_lookup_idx]).astype(np.float32) / 255.0
        image = np.ascontiguousarray(image.transpose(2, 0, 1))

        if self._needs_resize:
            image_t = torch.from_numpy(image).unsqueeze(0)
            image_t = F.interpolate(image_t, size=self.image_size, mode="bilinear", align_corners=False)
            image = image_t.squeeze(0).numpy()

        return {
            "image": image,
            "response": mean_response,
            "image_id": image_id,
            "session_id": meta["session_id"],
            "n_units": n_units,
        }

    def close(self):
        pass  # HDF5 handles owned by base_dataset


def numpy_collate_fn(batch, spike_augmentations=None):
    """Collate via np.stack then convert to torch once.

    Returns (session_id, batch_dict) tuple to match the training loop interface.
    """
    response = torch.from_numpy(np.stack([b["response"] for b in batch]))
    if spike_augmentations is not None:
        response = apply_spike_augmentations(response, spike_augmentations, time_dim=2)

    batch_dict = {
        "image": torch.from_numpy(np.stack([b["image"] for b in batch])),
        "response": response,
        "image_id": torch.tensor([b["image_id"] for b in batch]),
        "n_units": batch[0]["n_units"],
    }
    session_id = batch[0]["session_id"]  # all same within a batch
    return session_id, batch_dict


from ntt_downstream.encoding.dataloading.samplers import (  # noqa: F401
    DistributedSessionBatchSampler,
    SessionBatchSampler,
)
