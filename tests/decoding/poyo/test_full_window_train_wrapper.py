from __future__ import annotations

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from torch import nn

from ntt_downstream.decoding.backends.poyo.full_window_train_wrapper import (
    FullWindowUnitV2TrainWrapper,
)
from ntt_downstream.decoding.backends.poyo.tokenizer_output import TokenizerOutput


def _minimal_cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "log_every_n_steps": 1,
            "gpus": 1,
            "num_nodes": 1,
            "batch_size": 4,
            "optim": {
                "base_lr": 1e-6,
                "weight_decay": 0.0,
                "warmup_pct": 0.1,
                "div_factor": 25,
                "final_div_factor": 100,
            },
        }
    )


class _DummyPOYO(nn.Module):
    def __init__(self, dim: int = 64, sequence_length: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.sequence_length = sequence_length
        self.unit_emb = nn.Embedding(128, dim)
        self.token_type_emb = nn.Embedding(8, dim)
        self.session_emb = nn.Embedding(8, dim)


class _DummyModalitySpec:
    def loss_fn(self, *args, **kwargs):
        return torch.tensor(0.0)


class _CapturingFrozenTokenizer(nn.Module):
    input_bin_size_ms = 10.0
    output_dim = 1
    discrete_input_mode = "z_q"

    def __init__(self, *, combines_units: bool = False, expected_n_neurons: int | None = None) -> None:
        super().__init__()
        self.combines_units = combines_units
        self._expected_n_neurons = expected_n_neurons
        self.slot_global_ids = torch.empty(0, dtype=torch.long)
        self.last_batch = None
        # The wrapper reads ntt_tokenizer.tokenizer.spec.variable_t to decide
        # whether to chunk + how to size the raster. This stub stands in for a
        # variable-T (Perceiver-style) tokenizer, so expose a matching spec —
        # requires_fixed_input_window(spec) is then False and the wrapper keeps
        # the unchanged spike-derived raster.
        self.tokenizer = SimpleNamespace(spec=SimpleNamespace(variable_t=True))

    def forward(self, batch):
        self.last_batch = batch
        B, T, U = batch.spike_matrix.shape
        if self.combines_units:
            slot_ids = torch.arange(2, device=batch.spike_matrix.device).view(1, 1, 2)
            unit_ids = slot_ids.expand(B, T, 2).reshape(B, T * 2)
            timestamps = (
                torch.arange(T, device=batch.spike_matrix.device, dtype=torch.float32)
                .add(0.5)
                .mul(0.01)
                .view(1, T, 1)
                .expand(B, T, 2)
                .reshape(B, T * 2)
            )
            return TokenizerOutput(
                tokens=torch.zeros(B, T * 2, 1, device=batch.spike_matrix.device),
                timestamps=timestamps,
                unit_ids=unit_ids,
                mask=torch.ones(B, T * 2, dtype=torch.bool, device=batch.spike_matrix.device),
                combines_units=True,
                token_bin_size_s=0.01,
            )

        unit_ids = batch.unit_ids.unsqueeze(1).expand(B, T, U).reshape(B, T * U)
        timestamps = (
            torch.arange(T, device=batch.spike_matrix.device, dtype=torch.float32)
            .add(0.5)
            .mul(0.01)
            .view(1, T, 1)
            .expand(B, T, U)
            .reshape(B, T * U)
        )
        return TokenizerOutput(
            tokens=torch.zeros(B, T * U, 1, device=batch.spike_matrix.device),
            timestamps=timestamps,
            unit_ids=unit_ids,
            mask=batch.pad_and_gap_mask.bool().reshape(B, T * U),
            combines_units=False,
            token_bin_size_s=0.01,
        )


def _wrapper(
    ntt_tokenizer: nn.Module, *, include_padded_zero_unit_tokens: bool = False
) -> FullWindowUnitV2TrainWrapper:
    cfg = _minimal_cfg()
    cfg.tokenizer = {
        "include_padded_zero_unit_tokens": include_padded_zero_unit_tokens,
    }
    return FullWindowUnitV2TrainWrapper(
        cfg=cfg,
        model=_DummyPOYO(dim=64, sequence_length=0.1),
        modality_spec=_DummyModalitySpec(),
        tokenizer_mode="ntt_frozen",
        unit_embedding_mode="add",
        sentinels=False,
        ntt_tokenizer=ntt_tokenizer,
    )


def _poyo_batch() -> dict:
    # Three all-unit START sentinels in local unit order: global ids 10,20,30.
    # Only units 20 and 30 spike; unit 10 must still be present in the NTT raster.
    token_type = torch.tensor([[1, 2, 1, 2, 1, 2, 0, 0, 0]])
    unit_index = torch.tensor([[10, 10, 20, 20, 30, 30, 20, 30, 30]])
    timestamps = torch.tensor([[0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.015, 0.025, 0.099]])
    return {
        "model_inputs": {
            "input_token_type": token_type,
            "input_unit_index": unit_index,
            "input_timestamps": timestamps,
            "input_mask": torch.ones_like(token_type, dtype=torch.bool),
        },
        "session_id": ["sess0"],
    }


def test_full_window_handoff_feeds_local_ids_and_all_units():
    tokenizer = _CapturingFrozenTokenizer(combines_units=False)
    out = _wrapper(tokenizer)._ntt_frozen_tokenize(_poyo_batch())
    binned = tokenizer.last_batch

    assert binned.spike_matrix.shape == (1, 10, 3)
    assert binned.unit_ids.tolist() == [[0, 1, 2]]
    assert binned.unit_ids_pad_mask.tolist() == [[1, 1, 1]]
    assert binned.pad_and_gap_mask[:, :, 0].all()
    assert int(binned.spike_matrix[:, :, 0].sum()) == 0  # silent unit retained
    assert int(binned.spike_matrix[:, :, 1].sum()) == 1
    assert int(binned.spike_matrix[:, :, 2].sum()) == 2
    assert float(binned.absolute_start.item()) == 0.0
    assert float(binned.absolute_end.item()) == 0.1

    assert out.unit_ids.reshape(1, 10, 3)[0, 0].tolist() == [10, 20, 30]
    assert float(out.timestamps[out.mask].min()) >= 0.0
    assert float(out.timestamps[out.mask].max()) <= 0.1


def test_full_window_handoff_restores_slot_proxy_ids_to_global_ids():
    tokenizer = _CapturingFrozenTokenizer(combines_units=True)
    out = _wrapper(tokenizer)._ntt_frozen_tokenize(_poyo_batch())

    assert tokenizer.last_batch.unit_ids.tolist() == [[0, 1, 2]]
    assert out.combines_units is True
    assert out.unit_ids.reshape(1, 10, 2)[0, 0].tolist() == [10, 20]


def test_full_window_handoff_can_append_valid_zero_unit_columns():
    tokenizer = _CapturingFrozenTokenizer(
        combines_units=False,
        expected_n_neurons=5,
    )
    out = _wrapper(
        tokenizer,
        include_padded_zero_unit_tokens=True,
    )._ntt_frozen_tokenize(_poyo_batch())
    binned = tokenizer.last_batch

    assert binned.spike_matrix.shape == (1, 10, 5)
    assert binned.unit_ids.tolist() == [[0, 1, 2, 3, 4]]
    assert binned.unit_ids_pad_mask.tolist() == [[1, 1, 1, 1, 1]]
    assert binned.pad_and_gap_mask[:, :, 3:].all()
    assert int(binned.spike_matrix[:, :, 3:].sum()) == 0

    restored = out.unit_ids.reshape(1, 10, 5)
    assert restored[0, 0].tolist() == [10, 20, 30, 0, 0]


def _random_poyo_batch(seed: int, B: int = 4, window: float = 0.1) -> tuple[dict, list[list[int]]]:
    """Padded POYO batch with START/END sentinels and spikes, plus each sample's unit order.

    Some spikes come from units without a START sentinel (must be dropped) and
    some fall just outside the window (must be dropped).
    """
    g = torch.Generator().manual_seed(seed)
    rows, units = [], []
    for _ in range(B):
        n_units = int(torch.randint(1, 9, (1,), generator=g))
        uids = torch.randperm(200, generator=g)[:n_units] + 1  # global ids; 0 = padding id
        n_spikes = int(torch.randint(0, 40, (1,), generator=g))
        pool = torch.cat([uids, torch.tensor([250, 251])])  # 250/251: no START sentinel
        spike_u = pool[torch.randint(0, len(pool), (n_spikes,), generator=g)]
        spike_t = torch.rand(n_spikes, generator=g) * (window + 0.02) - 0.01
        ttype = torch.cat([torch.tensor([1, 2]).repeat(n_units), torch.zeros(n_spikes, dtype=torch.long)])
        uidx = torch.cat([uids.repeat_interleave(2), spike_u])
        ts = torch.cat([torch.tensor([0.0, window]).repeat(n_units), spike_t])
        rows.append((ttype, uidx, ts))
        units.append(uids.tolist())
    L = max(len(r[0]) for r in rows)

    def _pad(xs, fill):
        return torch.stack([torch.cat([x, torch.full((L - len(x),), fill, dtype=x.dtype)]) for x in xs])

    batch = {
        "model_inputs": {
            "input_token_type": _pad([r[0] for r in rows], 0),
            "input_unit_index": _pad([r[1] for r in rows], 0),
            "input_timestamps": _pad([r[2] for r in rows], 0.0),
            "input_mask": torch.stack([torch.arange(L) < len(r[0]) for r in rows]),
        },
        "session_id": ["sess0"] * B,
    }
    return batch, units


def test_full_window_raster_matches_reference_binning():
    """Each in-window spike of a sentinel unit is counted once, in its local column."""
    window, bin_s = 0.1, 0.01
    for seed in range(5):
        batch, units = _random_poyo_batch(seed, window=window)
        tokenizer = _CapturingFrozenTokenizer(combines_units=False)
        _wrapper(tokenizer)._ntt_frozen_tokenize(batch)
        raster = tokenizer.last_batch.spike_matrix
        n_bins = raster.shape[1]

        mi = batch["model_inputs"]
        expected = torch.zeros_like(raster)
        for b, uids in enumerate(units):
            col = {u: i for i, u in enumerate(uids)}
            for tt, u, t, m in zip(
                mi["input_token_type"][b].tolist(),
                mi["input_unit_index"][b].tolist(),
                mi["input_timestamps"][b].tolist(),
                mi["input_mask"][b].tolist(),
                strict=True,
            ):
                if m and tt == 0 and u in col and 0.0 <= t <= window:
                    expected[b, min(int(t / bin_s), n_bins - 1), col[u]] += 1
        assert torch.equal(raster, expected), f"seed {seed}"
