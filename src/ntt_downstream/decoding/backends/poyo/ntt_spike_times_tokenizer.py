"""NTTSpikeTimesTokenizer — dynamic per-spike tokenizer.

Emits one constant-content token per spike, using spike timestamps as token
times and unit indices as unit_ids. Intended as an NTT-native analog of the
FakeTokenizer that operates on NTT data structures.
"""

from __future__ import annotations

import torch
from ntt.core.components.types import BinnedSpikesInput, SpikeTimesInput
from torch import Tensor, nn

from ntt_downstream.decoding.backends.poyo.tokenizer_output import TokenizerOutput


class NTTSpikeTimesTokenizer(nn.Module):
    """Dynamic per-spike tokenizer for NTT inputs.

    Accepts either SpikeTimesInput (preferred) or BinnedSpikesInput (converted
    to spike times). Outputs constant-content tokens with per-spike timestamps
    and unit IDs, padded to the batch max length.

    Args:
        token_dim: Dimension of the token embeddings.
        token_value: Constant fill value for token content (default 0.0).
    """

    def __init__(self, token_dim: int, token_value: float = 0.0) -> None:
        super().__init__()
        assert isinstance(token_dim, int) and token_dim > 0, f"token_dim must be a positive integer, got {token_dim!r}"
        self.token_dim = token_dim
        self.token_value = float(token_value)

    def forward(self, batch: BinnedSpikesInput | SpikeTimesInput) -> TokenizerOutput:
        """Produce one constant-content token per spike.

        Args:
            batch: SpikeTimesInput (preferred) or BinnedSpikesInput.

        Returns:
            TokenizerOutput with combines_units=False, token_bin_size_s=None.
        """
        if isinstance(batch, SpikeTimesInput):
            return self._from_padded_spike_times(batch.spike_timestamps, batch.spike_unit_index, batch.spike_pad_mask)
        if isinstance(batch, BinnedSpikesInput):
            ts_list, uid_list = self._binned_to_spike_times(batch)
            return self._from_lists(ts_list, uid_list)

        raise TypeError(f"Expected SpikeTimesInput or BinnedSpikesInput, got {type(batch).__name__}")

    def _from_padded_spike_times(
        self,
        spike_timestamps: Tensor,
        spike_unit_index: Tensor,
        spike_pad_mask: Tensor,
    ) -> TokenizerOutput:
        """Build TokenizerOutput from padded spike-time tensors."""
        if spike_timestamps.ndim != 2 or spike_unit_index.ndim != 2 or spike_pad_mask.ndim != 2:
            raise ValueError("SpikeTimesInput tensors must be [B, N] padded arrays.")
        if spike_timestamps.shape != spike_unit_index.shape or spike_timestamps.shape != spike_pad_mask.shape:
            raise ValueError(
                "SpikeTimesInput tensors must have matching shapes for timestamps, unit_index, and pad_mask."
            )

        B, N = spike_timestamps.shape
        device = spike_timestamps.device

        tokens = torch.full(
            (B, N, self.token_dim),
            self.token_value,
            device=device,
            dtype=torch.float32,
        )
        timestamps = spike_timestamps.to(device=device, dtype=torch.float32)
        unit_ids = spike_unit_index.to(device=device, dtype=torch.long)
        mask = spike_pad_mask.to(device=device).bool()

        return TokenizerOutput(
            tokens=tokens,
            timestamps=timestamps,
            unit_ids=unit_ids,
            mask=mask,
            combines_units=False,
            token_bin_size_s=None,
        )

    def _from_lists(
        self,
        spike_timestamps: list[Tensor],
        spike_unit_ids: list[Tensor],
    ) -> TokenizerOutput:
        """Build TokenizerOutput from per-sample spike lists.

        Pads each sample to ``max(n_i, 1)`` and delegates to the padded path.
        """
        if not isinstance(spike_timestamps, list) or not isinstance(spike_unit_ids, list):
            raise TypeError(
                f"Expected lists, got spike_timestamps: {type(spike_timestamps)}, "
                f"spike_unit_ids: {type(spike_unit_ids)}"
            )
        batch_size = len(spike_timestamps)
        if batch_size != len(spike_unit_ids):
            raise ValueError(f"Batch size mismatch: {batch_size} timestamps vs {len(spike_unit_ids)} unit_ids")
        if batch_size == 0:
            raise ValueError("Empty batch: spike_timestamps and spike_unit_ids must be non-empty lists")

        for i, (ts, uid) in enumerate(zip(spike_timestamps, spike_unit_ids, strict=True)):
            if not isinstance(ts, Tensor):
                raise TypeError(f"spike_timestamps[{i}] must be a Tensor, got {type(ts)}")
            if not isinstance(uid, Tensor):
                raise TypeError(f"spike_unit_ids[{i}] must be a Tensor, got {type(uid)}")
            if ts.ndim != 1:
                raise ValueError(f"spike_timestamps[{i}] must be 1D, got shape {ts.shape}")
            if uid.ndim != 1:
                raise ValueError(f"spike_unit_ids[{i}] must be 1D, got shape {uid.shape}")
            if ts.shape[0] != uid.shape[0]:
                raise ValueError(f"Sample {i}: timestamp count {ts.shape[0]} != unit_id count {uid.shape[0]}")

        n_max = max((ts.shape[0] for ts in spike_timestamps), default=0)
        n_max = max(n_max, 1)

        device = spike_timestamps[0].device
        padded_ts = torch.zeros(batch_size, n_max, device=device, dtype=torch.float32)
        padded_uid = torch.zeros(batch_size, n_max, device=device, dtype=torch.long)
        padded_mask = torch.zeros(batch_size, n_max, device=device, dtype=torch.long)
        for i, (ts, uid) in enumerate(zip(spike_timestamps, spike_unit_ids, strict=True)):
            n_i = ts.shape[0]
            if n_i > 0:
                padded_ts[i, :n_i] = ts.to(device=device, dtype=torch.float32)
                padded_uid[i, :n_i] = uid.to(device=device, dtype=torch.long)
                padded_mask[i, :n_i] = 1

        return self._from_padded_spike_times(padded_ts, padded_uid, padded_mask)

    @staticmethod
    def _binned_to_spike_times(
        batch: BinnedSpikesInput,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Convert BinnedSpikesInput to per-sample spike times and unit IDs.

        Emits one spike event per count at the bin center time. Uses the
        pad_and_gap_mask to ignore invalid bins.
        """
        spike_matrix = batch.spike_matrix  # [B, T, U], int
        mask = batch.pad_and_gap_mask  # [B, T, U], int/bool (1=valid)
        abs_start = batch.absolute_start  # [B]

        B, T, U = spike_matrix.shape
        sr = batch.sampling_rate
        if isinstance(sr, torch.Tensor):
            assert sr.numel() == 1 or torch.all(torch.eq(sr, sr.flatten()[0])), (
                f"sampling_rate tensor has multiple distinct values: {sr}. "
                f"All batch elements must share the same sampling rate."
            )
            sr = sr.flatten()[0].item()
        bin_size_s = 1.0 / sr

        spike_timestamps_list: list[Tensor] = []
        spike_unit_ids_list: list[Tensor] = []

        for b in range(B):
            start = abs_start[b].item()
            valid_spikes = spike_matrix[b] * mask[b]  # [T, U]
            t_idx, u_idx = valid_spikes.nonzero(as_tuple=True)
            counts = valid_spikes[t_idx, u_idx]

            t_repeated = t_idx.repeat_interleave(counts)
            u_repeated = u_idx.repeat_interleave(counts)

            timestamps = start + (t_repeated.float() + 0.5) * bin_size_s

            spike_timestamps_list.append(timestamps)
            spike_unit_ids_list.append(u_repeated.long())

        return spike_timestamps_list, spike_unit_ids_list
