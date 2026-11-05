# tests/conftest.py
"""Shared test fixtures for ntt-downstream."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest


@pytest.fixture()
def wingert_data_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with 2 small synthetic Wingert-format HDF5 files."""
    rng = np.random.RandomState(42)

    for sess_name in ["sess_A", "sess_B"]:
        fpath = tmp_path / f"{sess_name}.h5"
        n_units = 5
        fs = 100.0
        n_freq = 32
        spec_lengths = [100, 150, 120]
        n_unique = len(spec_lengths)
        max_spec_len = max(spec_lengths)

        spec_data = rng.rand(n_unique, n_freq, max_spec_len).astype(np.float32)
        for i, slen in enumerate(spec_lengths):
            spec_data[i, :, slen:] = 0.0

        spec_ids = np.array([f"STIM_seq{i:04d}" for i in range(n_unique)], dtype="S")

        stim_starts, stim_ends, stim_spec_idx, stim_names = [], [], [], []
        t = 0.5
        for _pres in range(2):
            for si in range(n_unique):
                dur = spec_lengths[si] / fs
                stim_starts.append(t)
                stim_ends.append(t + dur)
                stim_spec_idx.append(si)
                stim_names.append(f"STIM_seq{si:04d}")
                t += dur + 0.5

        stim_starts = np.array(stim_starts)
        stim_ends = np.array(stim_ends)
        stim_spec_idx = np.array(stim_spec_idx, dtype=np.int64)
        stim_names_arr = np.array(stim_names, dtype="S")
        total_duration = t + 0.5
        total_bins = int(np.round(total_duration * fs))

        spike_times, spike_units = [], []
        for u in range(n_units):
            n_spk = rng.randint(20, 50)
            ts = rng.uniform(0, total_duration, size=n_spk)
            spike_times.append(ts)
            spike_units.append(np.full(n_spk, u, dtype=np.int64))
        spike_times = np.concatenate(spike_times)
        spike_units = np.concatenate(spike_units)
        sort_idx = np.argsort(spike_times)
        spike_times = spike_times[sort_idx]
        spike_units = spike_units[sort_idx]

        # Split domains
        train_start = stim_starts[0] - 0.01
        train_end = stim_ends[3] + 0.01
        valid_start = stim_starts[4] - 0.01
        valid_end = stim_ends[4] + 0.01
        test_start = stim_starts[5] - 0.01
        test_end = stim_ends[5] + 0.01

        unit_ids = np.array([f"{sess_name}-{u:03d}" for u in range(n_units)], dtype="S")
        brain_areas = np.array(["A1"] * n_units, dtype="S")

        with h5py.File(fpath, "w") as f:
            spk = f.create_group("spikes")
            spk.create_dataset("timestamps", data=spike_times)
            spk.create_dataset("unit_index", data=spike_units)
            spk.attrs["type"] = "IrregularTimeSeries"

            units = f.create_group("units")
            units.create_dataset("id", data=unit_ids)
            units.create_dataset("brain_area", data=brain_areas)
            units.attrs["type"] = "ArrayDict"

            stim = f.create_group("stimuli")
            stim.create_dataset("start", data=stim_starts)
            stim.create_dataset("end", data=stim_ends)
            stim.create_dataset("spectrogram_index", data=stim_spec_idx)
            stim.create_dataset("epoch_name", data=stim_names_arr)
            stim.attrs["type"] = "Interval"

            specs = f.create_group("spectrograms")
            specs.create_dataset("id", data=spec_ids)
            specs.create_dataset("data", data=spec_data)
            specs.create_dataset("length", data=np.array(spec_lengths, dtype=np.int64))
            specs.attrs["type"] = "ArrayDict"

            f.attrs["spectrogram_fs"] = fs
            f.attrs["recording_total_bins"] = total_bins

            dom = f.create_group("domain")
            dom.create_dataset("start", data=np.array([0.0]))
            dom.create_dataset("end", data=np.array([total_duration]))
            dom.attrs["type"] = "Interval"

            for name, s, e in [
                ("train_domain", train_start, train_end),
                ("valid_domain", valid_start, valid_end),
                ("test_domain", test_start, test_end),
            ]:
                d = f.create_group(name)
                d.create_dataset("start", data=np.array([s]))
                d.create_dataset("end", data=np.array([e]))
                d.attrs["type"] = "Interval"

    return tmp_path
