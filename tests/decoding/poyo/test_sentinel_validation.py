"""Validate sentinel correctness: original POYO vs adapter with sentinels=True.

Compares sentinel tokens produced by:
1. Original POYO path: tokens where input_token_type != 0 (START=1, END=2)
2. ntt path: _ntt_tokenize() + adapt(sentinels=True)
"""

import os

import pytest
import torch
from torch.utils.data import DataLoader
from torch_brain.data import collate
from torch_brain.data.sampler import RandomFixedWindowSampler
from torch_brain.registry import MODALITY_REGISTRY
from torch_brain.transforms import Compose

from ntt_downstream.decoding.backends.poyo.adapter import adapt
from ntt_downstream.decoding.backends.poyo.datasets import SingleSessionCODataset
from ntt_downstream.decoding.backends.poyo.model import POYOWithEmbeddingBypass
from ntt_downstream.decoding.backends.poyo.ntt_spike_times_tokenizer import NTTSpikeTimesTokenizer

# Override with NTT_DOWNSTREAM_DATA to point at a brainsets `processed/`
# parent dir that contains `perich_miller_population_2018/<SESSION>.h5`. The
# test is skipped when that file is missing.
DATA_ROOT = os.environ.get("NTT_DOWNSTREAM_DATA", "./data/processed/")
SESSION = "c_20131003_center_out_reaching"


@pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA_ROOT, "perich_miller_population_2018", f"{SESSION}.h5")),
    reason=f"Test dataset not available at {DATA_ROOT}",
)
def test_sentinel_validation(capsys):
    # --- Setup ---
    from copy import deepcopy

    from temporaldata import Data

    from ntt_downstream.decoding.backends.poyo.datasets import CO_READOUT_CONFIG

    # Use simplified weights to avoid missing movement_phases in small datasets
    simple_config = deepcopy(CO_READOUT_CONFIG)
    simple_config["readout"]["weights"] = {}

    class SimplifiedDataset(SingleSessionCODataset):
        def get_recording_hook(self, data: Data):
            data.config = deepcopy(simple_config)
            # Call grandparent hook (skip CO hook that sets the full weights)
            from brainsets.datasets import PerichMillerPopulation2018

            PerichMillerPopulation2018.get_recording_hook(self, data)

    ds = SimplifiedDataset(root=DATA_ROOT, recording_id=SESSION)
    readout_id = simple_config["readout"]["readout_id"]
    readout_spec = MODALITY_REGISTRY[readout_id]

    # Model: 1.3M config, dim=64
    model = POYOWithEmbeddingBypass(
        readout_spec=readout_spec,
        sequence_length=1.0,
        latent_step=0.125,
        num_latents_per_step=16,
        dim=64,
        depth=6,
        dim_head=64,
        cross_heads=2,
        self_heads=8,
        ffn_dropout=0.2,
        lin_dropout=0.4,
        atn_dropout=0.2,
    )

    unit_ids = ds.get_unit_ids()
    session_ids = ds.recording_ids
    model.unit_emb.initialize_vocab(unit_ids)
    model.session_emb.initialize_vocab(session_ids)
    ds.transform = Compose([model.tokenize])

    # DataLoader: batch_size=1 for clarity
    sampler = RandomFixedWindowSampler(
        sampling_intervals=ds.get_sampling_intervals("train"),
        window_length=model.sequence_length,
        generator=torch.Generator().manual_seed(42),
    )
    loader = DataLoader(ds, sampler=sampler, collate_fn=collate, batch_size=1)
    batch = next(iter(loader))

    model.eval()

    mi = batch["model_inputs"]
    input_token_type = mi["input_token_type"]  # [1, N]
    input_unit_index = mi["input_unit_index"]  # [1, N]
    input_timestamps = mi["input_timestamps"]  # [1, N]
    input_mask = mi["input_mask"]  # [1, N]

    print("=" * 60)
    print("SENTINEL VALIDATION TEST")
    print("=" * 60)

    # ============================================================
    # 1. ORIGINAL POYO PATH: extract sentinels from batch
    # ============================================================
    with torch.no_grad():
        is_sentinel_orig = (input_token_type != 0) & input_mask  # [1, N]
        is_start = (input_token_type == 1) & input_mask
        is_end = (input_token_type == 2) & input_mask

        orig_count = is_sentinel_orig[0].sum().item()
        orig_start_count = is_start[0].sum().item()
        orig_end_count = is_end[0].sum().item()

        # Extract sentinel details
        orig_sentinel_ts = input_timestamps[0][is_sentinel_orig[0]]
        orig_sentinel_uid = input_unit_index[0][is_sentinel_orig[0]]
        orig_sentinel_types = input_token_type[0][is_sentinel_orig[0]]

        # Compute original sentinel embeddings: unit_emb(uid) + token_type_emb(type)
        orig_sentinel_emb = (
            model.unit_emb(orig_sentinel_uid.unsqueeze(0)) + model.token_type_emb(orig_sentinel_types.unsqueeze(0))
        ).squeeze(0)  # [n_sentinels, D]

    print("\n--- ORIGINAL POYO ---")
    print(f"Total sentinel tokens: {orig_count}")
    print(f"  START tokens (type=1): {orig_start_count}")
    print(f"  END tokens (type=2): {orig_end_count}")
    print(f"Unique units in sentinels: {orig_sentinel_uid.unique().shape[0]}")
    print(f"Sentinel timestamps range: [{orig_sentinel_ts.min():.4f}, {orig_sentinel_ts.max():.4f}]")

    # Count spike tokens
    is_spike = (input_token_type == 0) & input_mask
    n_spikes = is_spike[0].sum().item()
    n_total = input_mask[0].sum().item()
    print(f"Spike tokens: {n_spikes}, Total valid: {n_total}")

    # ============================================================
    # 2. NTT PATH: _ntt_tokenize + adapt(sentinels=True)
    # ============================================================
    # We replicate what V2TrainWrapper._ntt_tokenize does:
    # Filter to spike tokens (type==0), build SpikeTimesInput, run tokenizer
    from ntt.core.components.types import SpikeTimesInput

    with torch.no_grad():
        B = 1
        spike_ts = input_timestamps[0][is_spike[0]]
        spike_uid = input_unit_index[0][is_spike[0]]

        n_max = max(spike_ts.shape[0], 1)
        spike_timestamps_padded = torch.zeros(B, n_max)
        spike_unit_index_padded = torch.zeros(B, n_max, dtype=torch.long)
        spike_pad_mask = torch.zeros(B, n_max, dtype=torch.long)

        spike_timestamps_padded[0, : spike_ts.shape[0]] = spike_ts
        spike_unit_index_padded[0, : spike_uid.shape[0]] = spike_uid
        spike_pad_mask[0, : spike_ts.shape[0]] = 1

        unique_uids = torch.unique(spike_unit_index_padded[spike_pad_mask.bool()])
        unit_ids_padded = unique_uids.unsqueeze(0).expand(B, -1)
        unit_ids_mask = torch.ones_like(unit_ids_padded, dtype=torch.long)

        spike_input = SpikeTimesInput(
            spike_unit_index=spike_unit_index_padded,
            spike_timestamps=spike_timestamps_padded,
            spike_pad_mask=spike_pad_mask,
            unit_ids=unit_ids_padded,
            unit_ids_pad_mask=unit_ids_mask,
            duration_sec=torch.ones(B),
            session_id=("dummy",) * B,
            subject_id=("dummy",) * B,
            species=("dummy",) * B,
        )

        ntt_tokenizer = NTTSpikeTimesTokenizer(token_dim=64)
        tok_out = ntt_tokenizer(spike_input)

        # D_tok=64=target_dim with mode="add" → projection disabled (original POYO path).
        pre = adapt(
            tok_out,
            unit_emb=model.unit_emb,
            token_type_emb=model.token_type_emb,
            mode="add",
            sentinels=True,
            target_dim=64,
            input_proj=None,
        )

        # The adapter prepends sentinels. Content tokens come after.
        # Find how many sentinels were added: compare pre.embeddings length vs tok_out.tokens length
        n_content = tok_out.tokens.shape[1]
        n_total_v2 = pre.embeddings.shape[1]
        n_sentinel_v2 = n_total_v2 - n_content

        # Extract sentinel portion (first n_sentinel_v2 tokens)
        v2_sentinel_emb = pre.embeddings[0, :n_sentinel_v2]
        v2_sentinel_ts = pre.timestamps[0, :n_sentinel_v2]
        v2_sentinel_mask = pre.mask[0, :n_sentinel_v2]

        # Count valid sentinels
        v2_valid_sentinels = v2_sentinel_mask.sum().item()

    print("\n--- NTT PATH (adapter sentinels) ---")
    print(f"Total sentinel slots: {n_sentinel_v2}")
    print(f"Valid sentinels: {v2_valid_sentinels}")
    print(
        f"Sentinel timestamps: [{v2_sentinel_ts[v2_sentinel_mask].min():.4f}, "
        f"{v2_sentinel_ts[v2_sentinel_mask].max():.4f}]"
        if v2_valid_sentinels > 0
        else "No valid sentinels"
    )

    # ============================================================
    # 3. COMPARISON
    # ============================================================
    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print(f"{'=' * 60}")

    # --- Count comparison ---
    n_unique_units = orig_sentinel_uid.unique().shape[0]
    print(f"\nOriginal POYO sentinel count: {orig_count} (2 * {n_unique_units} units = {2 * n_unique_units})")
    print(f"adapter sentinel count:   {v2_valid_sentinels}")
    print(f"  -> KNOWN DIFFERENCE: Original has START+END per unit ({orig_count}),")
    print(f"     adapter has 1 per unique unit ({v2_valid_sentinels}).")
    count_ratio = orig_count / max(v2_valid_sentinels, 1)
    print(f"     Ratio: {count_ratio:.1f}x")

    # --- Timestamp comparison ---
    # Original: START tokens at window start, END tokens at window end
    start_ts = orig_sentinel_ts[orig_sentinel_types == 1]
    end_ts = orig_sentinel_ts[orig_sentinel_types == 2]
    v2_ts = v2_sentinel_ts[v2_sentinel_mask]

    print("\nTimestamp analysis:")
    print(
        f"  Original START timestamps: all = {start_ts[0].item():.4f}"
        if start_ts.numel() > 0 and torch.all(start_ts == start_ts[0])
        else f"  Original START timestamps: {start_ts}"
    )
    print(
        f"  Original END timestamps: all = {end_ts[0].item():.4f}"
        if end_ts.numel() > 0 and torch.all(end_ts == end_ts[0])
        else f"  Original END timestamps: {end_ts}"
    )
    print(
        f"  adapter sentinel timestamps: all = {v2_ts[0].item():.4f}"
        if v2_ts.numel() > 0 and torch.all(v2_ts == v2_ts[0])
        else f"  adapter timestamps: {v2_ts}"
    )

    if v2_ts.numel() > 0 and start_ts.numel() > 0:
        ts_match_start = torch.allclose(v2_ts.double(), start_ts[: v2_ts.shape[0]].double(), atol=1e-5)
        print(f"  adapter timestamps match START timestamps: {ts_match_start}")

    # --- SENTINEL_TYPE check ---
    print("\n--- SENTINEL_TYPE CHECK ---")
    print("Adapter uses SENTINEL_TYPE = 1 (START_OF_SEQUENCE, matching original POYO)")
    print("Original POYO uses START=1, END=2 for sentinels")

    # The adapter sentinel embedding is unit_emb(uid) + token_type_emb(1), the same
    # as original POYO's START sentinel; END sentinels use token_type_emb(2).
    with torch.no_grad():
        type_emb_0 = model.token_type_emb(torch.tensor([0]))  # DEFAULT
        type_emb_1 = model.token_type_emb(torch.tensor([1]))  # START
        type_emb_2 = model.token_type_emb(torch.tensor([2]))  # END

        print(f"\n  token_type_emb(0) [DEFAULT]: norm={type_emb_0.norm():.4f}")
        print(f"  token_type_emb(1) [START]:   norm={type_emb_1.norm():.4f}")
        print(f"  token_type_emb(2) [END]:     norm={type_emb_2.norm():.4f}")
        print(f"  Diff 0 vs 1: {(type_emb_0 - type_emb_1).norm():.4f}")
        print(f"  Diff 0 vs 2: {(type_emb_0 - type_emb_2).norm():.4f}")
        print(f"  Diff 1 vs 2: {(type_emb_1 - type_emb_2).norm():.4f}")

    # --- Embedding comparison ---
    print("\n--- EMBEDDING COMPARISON ---")
    # For matching units, compare adapter sentinel embedding to original START sentinel
    with torch.no_grad():
        # Get the unique units from adapter
        v2_unique_uids = tok_out.unit_ids[0][tok_out.mask[0]].unique()

        # For each v2 sentinel uid, find matching original START sentinel
        orig_start_mask = orig_sentinel_types == 1
        orig_start_uids = orig_sentinel_uid[orig_start_mask]
        orig_start_embs = orig_sentinel_emb[orig_start_mask]

        # Build lookup: uid -> embedding for original START sentinels
        matched = 0
        emb_diffs = []
        for i, uid in enumerate(v2_unique_uids):
            # Find this uid in original START sentinels
            match_mask = orig_start_uids == uid
            if match_mask.any():
                orig_emb_i = orig_start_embs[match_mask][0]
                v2_emb_i = v2_sentinel_emb[i]

                diff = (orig_emb_i - v2_emb_i).abs().max().item()
                emb_diffs.append(diff)
                matched += 1

        if emb_diffs:
            print(f"  Matched {matched}/{v2_unique_uids.shape[0]} units")
            print(f"  Max abs diff between adapter and original START sentinels: {max(emb_diffs):.6f}")
            print(f"  Mean abs diff: {sum(emb_diffs) / len(emb_diffs):.6f}")

            if max(emb_diffs) < 1e-5:
                print("  -> MATCH: adapter sentinels equal original START sentinels")
            else:
                print("  -> MISMATCH: adapter sentinels differ from original START sentinels")

        # Cross-check against unit_emb(uid) + token_type_emb(1) computed directly.
        print("\n  Direct recomputation with token_type_emb(1) (START):")
        hypo_diffs = []
        for uid in v2_unique_uids:
            match_mask = orig_start_uids == uid
            if match_mask.any():
                orig_emb_i = orig_start_embs[match_mask][0]
                # Recompute: unit_emb(uid) + token_type_emb(1)
                hypo_emb = model.unit_emb(uid.unsqueeze(0)).squeeze(0) + type_emb_1.squeeze(0)
                diff = (orig_emb_i - hypo_emb).abs().max().item()
                hypo_diffs.append(diff)
        if hypo_diffs:
            print(f"  Max abs diff: {max(hypo_diffs):.6f}")
            if max(hypo_diffs) < 1e-5:
                print("  -> MATCH: original START sentinels equal unit_emb + token_type_emb(1)")

    # ============================================================
    # 4. SUMMARY
    # ============================================================
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"""
1. COUNT: Original POYO has {orig_count} sentinels (START+END per unit = 2*{n_unique_units}).
   Adapter has {v2_valid_sentinels} sentinels (1 per unique unit): the adapter merges
   the separate START and END tokens into one.

2. TIMESTAMPS: Original START sentinels sit at window start, END at window end.
   The adapter places all sentinels at the window start (min timestamp).

3. TOKEN_TYPE EMBEDDING: The adapter uses SENTINEL_TYPE=1, the same token type
   as the original START sentinels, so its sentinel embeddings are comparable to them.
""")


if __name__ == "__main__":
    test_sentinel_validation(None)
