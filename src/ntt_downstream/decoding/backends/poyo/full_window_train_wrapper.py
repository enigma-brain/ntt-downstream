"""Experimental POYO wrapper with a stricter frozen-tokenizer data contract.

This path is intentionally separate from ``train_wrapper.V2TrainWrapper``, which
keeps the active-unit / ``t_min`` raster contract (the default for
non-session-keyed tokenizers).
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, cast

import torch
from ntt.core.components.types import BinnedSpikesInput
from ntt.evaluation.utils.window_chunking import requires_fixed_input_window

from ntt_downstream.decoding.backends.poyo.tokenizer_output import TokenizerOutput
from ntt_downstream.decoding.backends.poyo.train_wrapper import (
    V2TrainWrapper,
    _clamp_output_timestamps,
    _session_id_field,
)

if TYPE_CHECKING:
    from ntt_downstream.decoding.backends.poyo.frozen_ntt_tokenizer import FrozenNTTTokenizer


class FullWindowUnitV2TrainWrapper(V2TrainWrapper):
    """Frozen-tokenizer wrapper that preserves the POYO window and all units.

    The base ``V2TrainWrapper._ntt_frozen_tokenize`` rebuilds a binned input from
    spike tokens only, using each sample's first/last spike as the raster bounds.
    That is a poor contract for session-keyed population tokenizers such as the
    Perceiver: silent units disappear, and POYO-global unit indices are passed to
    tokenizer code that expects local ``0..N_session-1`` unit numbers.

    This subclass instead:
      * bins the full ``[0, model.sequence_length]`` POYO window,
      * includes every session unit recovered from POYO START sentinels,
      * feeds local unit ids into the frozen tokenizer,
      * maps emitted token unit ids back to POYO-global ids for the adapter.
    """

    def _ntt_frozen_tokenize(self, batch: dict) -> TokenizerOutput:
        """Build a fixed-window, all-unit ``BinnedSpikesInput`` for NTT."""
        assert self.ntt_tokenizer is not None
        # nn.Module's __getattr__ hides the frozen tokenizer's own attributes.
        ntt_tokenizer = cast("FrozenNTTTokenizer", self.ntt_tokenizer)

        mi = batch["model_inputs"]
        input_token_type = mi["input_token_type"]  # [B, N]
        input_unit_index = mi["input_unit_index"]  # [B, N], POYO-global ids
        input_timestamps = mi["input_timestamps"]  # [B, N], window-relative seconds
        input_mask = mi["input_mask"]  # [B, N]

        B = input_token_type.shape[0]
        device = input_timestamps.device
        window_end = float(self.model.sequence_length)

        bin_size_ms = float(ntt_tokenizer.input_bin_size_ms)
        bin_size_s = bin_size_ms / 1000.0
        sampling_rate = 1000.0 / bin_size_ms
        # Fixed-window tokenizers are chunked by FrozenNTTTokenizer.forward into
        # chunk_width-bin pieces, so the raster MUST be an exact multiple of the
        # chunk width — otherwise _forward_chunked raises. Derive n_bins as
        # n_chunks * chunk_width (setup validation guarantees window_end is a
        # clean multiple of the native window in seconds). Variable-T tokenizers
        # are fed unchunked and keep the spike-derived ceil raster.
        if requires_fixed_input_window(ntt_tokenizer.tokenizer.spec):
            chunk_width = ntt_tokenizer.chunk_width_bins()
            n_chunks = max(round(window_end / (chunk_width * bin_size_s)), 1)
            n_bins = n_chunks * chunk_width
        else:
            n_bins = _num_fixed_bins(window_end, bin_size_s)

        global_units = _global_units_from_start_sentinels(
            input_unit_index=input_unit_index,
            input_token_type=input_token_type,
            input_mask=input_mask,
        )
        max_U = max(max((u.numel() for u in global_units), default=0), 1)
        zero_pad_to_U = self._zero_unit_padding_target(max_U)
        if zero_pad_to_U is not None:
            max_U = zero_pad_to_U

        spike_matrix = torch.zeros(B, n_bins, max_U, device=device, dtype=torch.int32)
        pad_and_gap_mask = torch.zeros(B, n_bins, max_U, device=device, dtype=torch.int32)
        local_unit_ids = torch.zeros(B, max_U, device=device, dtype=torch.long)
        global_unit_ids = torch.zeros(B, max_U, device=device, dtype=torch.long)
        unit_ids_pad_mask = torch.zeros(B, max_U, device=device, dtype=torch.int32)

        is_spike = (input_token_type == 0) & input_mask
        for b in range(B):
            units_b = global_units[b].to(device=device, dtype=torch.long)
            n_units = int(units_b.numel())
            if n_units == 0:
                continue

            local_unit_ids[b, :n_units] = torch.arange(n_units, device=device)
            global_unit_ids[b, :n_units] = units_b
            unit_ids_pad_mask[b, :n_units] = 1
            pad_and_gap_mask[b, :, :n_units] = 1
            if zero_pad_to_U is not None and n_units < max_U:
                # Optional fixed-neuron padding (as the WanNeural tokenizers expect): append explicit
                # zero-spike "unit 0" columns and mark them valid so the
                # tokenizer processes them instead of replacing the gap with
                # duplicated real neurons. global_unit_ids stays zero there,
                # so _restore_global_unit_ids maps those output tokens to
                # POYO's reserved NA/zero unit id.
                local_unit_ids[b, n_units:max_U] = torch.arange(n_units, max_U, device=device)
                unit_ids_pad_mask[b, n_units:max_U] = 1
                pad_and_gap_mask[b, :, n_units:max_U] = 1

            unit_to_local = {int(uid.item()): i for i, uid in enumerate(units_b)}
            spike_mask_b = is_spike[b] & (input_timestamps[b] >= 0.0) & (input_timestamps[b] <= window_end)
            if not bool(spike_mask_b.any()):
                continue

            spike_ts = input_timestamps[b][spike_mask_b]
            spike_global = input_unit_index[b][spike_mask_b].to(torch.long)
            local_cols = torch.tensor(
                [unit_to_local.get(int(uid.item()), -1) for uid in spike_global],
                device=device,
                dtype=torch.long,
            )
            known = local_cols >= 0
            if not bool(known.any()):
                continue

            spike_ts = spike_ts[known]
            local_cols = local_cols[known]
            bin_idx = (spike_ts / bin_size_s).to(torch.long).clamp(0, n_bins - 1)
            spike_matrix[b].index_put_(
                (bin_idx, local_cols),
                torch.ones_like(bin_idx, dtype=spike_matrix.dtype),
                accumulate=True,
            )

        binned_input = BinnedSpikesInput(
            spike_matrix=spike_matrix,
            pad_and_gap_mask=pad_and_gap_mask,
            sampling_rate=sampling_rate,
            unit_ids=local_unit_ids,
            unit_ids_pad_mask=unit_ids_pad_mask,
            duration_sec=torch.full((B,), window_end, device=device),
            session_id=_session_id_field(batch, B),
            subject_id=("dummy",) * B,
            species=("dummy",) * B,
            absolute_start=torch.zeros(B, device=device, dtype=torch.float64),
            absolute_end=torch.full((B,), window_end, device=device, dtype=torch.float64),
        )

        out = self.ntt_tokenizer(binned_input)
        out = self._restore_global_unit_ids(out, global_unit_ids)
        return _clamp_output_timestamps(out, window_end)

    def _restore_global_unit_ids(self, out: TokenizerOutput, global_unit_ids: torch.Tensor) -> TokenizerOutput:
        """Map tokenizer-local unit ids back to POYO-global embedding ids."""
        if out.unit_ids is None:
            return out

        slot_global_ids = getattr(self.ntt_tokenizer, "slot_global_ids", None)
        if out.combines_units and slot_global_ids is not None and int(slot_global_ids.numel()) > 0:
            # Global slot proxies already stamp POYO-global "_slot_k" ids in _flatten_4d.
            return out

        global_unit_ids = global_unit_ids.to(device=out.unit_ids.device, dtype=torch.long)
        local_ids = out.unit_ids.to(torch.long).clamp(0, global_unit_ids.shape[1] - 1)
        restored = torch.gather(global_unit_ids, dim=1, index=local_ids)
        return dataclasses.replace(out, unit_ids=restored)

    def _zero_unit_padding_target(self, current_units: int) -> int | None:
        """Return the fixed zero-pad width (WanNeural-style) when explicitly requested."""
        tokenizer_cfg = getattr(self.cfg, "tokenizer", None)
        enabled = bool(
            tokenizer_cfg is not None
            and (
                tokenizer_cfg.get("include_padded_zero_unit_tokens", False)
                or tokenizer_cfg.get("full_window_zero_unit_padding", False)
            )
        )
        if not enabled:
            return None

        if self.ntt_tokenizer is None or bool(getattr(self.ntt_tokenizer, "combines_units", False)):
            return None

        expected = getattr(self.ntt_tokenizer, "_expected_n_neurons", None)
        if expected is None:
            expected = getattr(getattr(self.ntt_tokenizer, "tokenizer", None), "spec", None)
            expected = getattr(expected, "n_neurons", None)
        if expected is None:
            return None
        return max(current_units, int(expected))


def _num_fixed_bins(window_end: float, bin_size_s: float) -> int:
    ratio = window_end / bin_size_s
    rounded = int(round(ratio))
    if math.isclose(ratio, rounded, rel_tol=0.0, abs_tol=1e-6):
        return max(rounded, 1)
    return max(int(math.ceil(ratio)), 1)


def _global_units_from_start_sentinels(
    *,
    input_unit_index: torch.Tensor,
    input_token_type: torch.Tensor,
    input_mask: torch.Tensor,
) -> list[torch.Tensor]:
    """Recover all POYO-global units in local unit order for each sample."""
    units: list[torch.Tensor] = []
    B = input_unit_index.shape[0]
    for b in range(B):
        start_mask = (input_token_type[b] == 1) & input_mask[b]
        units_b = input_unit_index[b][start_mask].to(torch.long)
        if units_b.numel() == 0:
            # Fallback for synthetic tests or non-POYO transforms: preserve first
            # occurrence order instead of sorting, because this order defines the
            # local unit id consumed by session-keyed tokenizers.
            any_unit = input_mask[b]
            units_b = _unique_preserve_order(input_unit_index[b][any_unit].to(torch.long))
        units.append(units_b)
    return units


def _unique_preserve_order(x: torch.Tensor) -> torch.Tensor:
    seen: set[int] = set()
    vals: list[torch.Tensor] = []
    for v in x:
        key = int(v.item())
        if key in seen:
            continue
        seen.add(key)
        vals.append(v)
    if not vals:
        return x.new_zeros(0, dtype=torch.long)
    return torch.stack(vals).to(torch.long)
