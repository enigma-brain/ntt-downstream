# Decoding (POYO bypass)

The decoding subpackage wraps frozen ntt tokenizers around the POYO decoder
for behavioral decoding (cursor velocity from spikes).

Entry point: `python -m ntt_downstream.decoding.train`
(`src/ntt_downstream/decoding/train.py`).

> **For running evals, prefer the stable CLI contract:** see [cli.md](cli.md)
> for the `ntt-downstream-decode` launcher (promoted flags, escape hatches).
> This doc explains the internals behind that launcher.

## Pipeline

The POYO bypass keeps everything below the input layer **identical** to
upstream POYO and only changes how the input tokens are built (the classes are
`POYOWithEmbeddingBypass` and `V2TrainWrapper`). The original
`unit_emb(unit_id) + token_type_emb(token_type)` lookup is replaced by a
**tokenizer-bypass path**: an arbitrary ntt tokenizer turns spikes into tokens,
an adapter glues unit identity and sentinels onto them, and the resulting
`[B, N, D_model]` tensor is fed into POYO via the `input_embeddings` argument.

```
   Original POYO                         POYO with embedding bypass
   ─────────────                         (ntt tokenizer)
                                         ──────────────────────────

   POYO Dataset                          POYO Dataset
   (spikes, behavior)                    (spikes, behavior)
        │                                     │
        ▼                                     ▼
   model.tokenize()                      model.tokenize()              ← SHARED
   (sentinels, indices,                  (sentinels, indices,
    latent grid, readout)                 latent grid, readout)
        │                                     │
        │                                ┌────┴─────────┐
        │                                │ extract       │  filter token_type==0
        │                                │ spikes        │  (drops POYO sentinels)
        │                                └────┬─────────┘
        │                                     │
        │                                ┌────┴─────────┐
        │                                │ NTT Tokenizer │  FROZEN by default
        │                                │ (fake/spike-  │
        │                                │  times/Poisson│
        │                                │  /conv/LDNS)  │
        │                                └────┬─────────┘
        │                                     │ TokenizerOutput
        │                                ┌────┴─────────┐
        │                                │ Adapter        │  unit_emb mode
        │                                │ + sentinels    │  ∈ {none, add, concat}
        │                                └────┬─────────┘
        │                                     │
        ▼                                     ▼
   ┌─────────────────┐              ┌──────────────────────┐
   │ POYO.forward     │              │ POYO.forward         │
   │ unit_emb +       │              │ input_embeddings=    │  ← BYPASS
   │ token_type_emb   │              │ (pre-computed)       │
   │ → [B, N, D]      │              │ → [B, N, D]          │
   └────────┬────────┘              └──────────┬───────────┘
            │                                  │
            ▼                                  ▼
   ┌────────────────────────────────────────────────────────┐
   │  Encoder cross-attn → latent self-attn → decoder       │  ← SHARED
   │  → readout                                              │    (identical)
   └────────────────────────┬───────────────────────────────┘
                            │
                            ▼
                    Behavior prediction
```

## Fake-tokenizer equivalence: why "fake" matters

The fake tokenizer (`tokenizer.mode=fake`) sets
`tokens = token_type_emb(token_type)` and `unit_ids = input_unit_index`.
Combined with `unit_embedding_mode=add` it produces

```
input_embeddings = token_type_emb(token_type) + unit_emb(unit_id)
```

which is byte-equivalent to POYO's original `inputs` line. This is the
fake-tokenizer equivalence smoke test: any change to the bypass plumbing should
leave fake-tokenizer R² unchanged.

## Tokenizer kinds

Pick with `tokenizer=...`:

| Hydra group | `mode` | Source | Used for |
|---|---|---|---|
| `tokenizer=fake` | `fake` | `V2TrainWrapper._fake_tokenize` builds a `TokenizerOutput` from POYO's already-tokenized batch (`token_type_emb` as content) | Fake-tokenizer equivalence, a bypass-path smoke test |
| `tokenizer=spike_times` | `ntt` | `NTTSpikeTimesTokenizer` — one zero-content token per spike | NTT-native pass-through baseline; pairs with `mode=add` to recover original POYO |
| `tokenizer=poisson` | `ntt_frozen` | `FrozenNTTTokenizer.from_poisson` — config-built rate-bin tokenizer (no checkpoint) | Cheap binned-rate baseline |
| `tokenizer=frozen` | `ntt_frozen` | `FrozenNTTTokenizer.from_checkpoint` — wraps any ntt `BaseTokenizer` ckpt (conv-* VAE, LDNS, Perceiver, residual-VQ, …) | Production runs against a learned tokenizer |
| `tokenizer=frozen_full_window` | `ntt_frozen` | Same loader as `frozen`, but feeds a full `[0, sequence_length]` raster with every session unit. Use for population/session-keyed tokenizers (e.g. Perceiver). | |

`ntt_checkpoint=???` is required (Hydra refuses to start without it) when
`tokenizer=frozen` or `tokenizer=frozen_full_window`. It takes a local `.ckpt`
path or an `mlflow-artifacts:` (or `mlflow:`) URI; `hf://` checkpoints are materialised to a
local `.ckpt` by ntt's launcher before this package sees them.

### Which unit handoff a frozen tokenizer needs

`tokenizer.full_window_units` picks how POYO's spike tokens reach a frozen
tokenizer:

- **Active-unit handoff** (`false`): in each window only the units that spike
  are passed, packed into positions `0..k-1` in unit-id order. A neuron's
  position therefore changes from window to window, and silent units are
  missing.
- **All-unit handoff** (`true`): every unit of the session, in a fixed order,
  on the full `[0, sequence_length]` raster (`FullWindowUnitV2TrainWrapper`).

| Tokenizer | Reads neurons | Handoff | What `auto` picks |
|---|---|---|---|
| Per-neuron (`compress_n: false`; e.g. Conv/VAE, cs-*, PCA) | one at a time | either | active-unit |
| Population over an unordered set (`variable_n: true` without per-session unit embeddings; e.g. a transformer-vq-* tokenizer with population context) | no positional encoding over neurons | either | active-unit |
| Fixed population (`compress_n: true`, `variable_n: false`; e.g. LDNS, wavelet-*) | **by position** | **all-unit: set `tokenizer.full_window_units=true`** | active-unit, the wrong one |
| Session-keyed (per-session unit embeddings; e.g. Perceiver) | by session unit id | all-unit | all-unit |

`auto` checks only whether a tokenizer is session-keyed. A fixed-population
tokenizer under `auto` runs without any error or warning, but it reads
different neurons at the same position in different windows, so its decoding
R² understates the tokenizer. Pass `tokenizer.full_window_units=true` (or
`tokenizer=frozen_full_window`) for these tokenizers.

### The contract: `TokenizerOutput`

Every tokenizer (real or fake) returns a `TokenizerOutput` dataclass:

```python
@dataclass
class TokenizerOutput:
    tokens: Tensor              # float [B, N, D_tok] OR int64 [B, N, Q] (codebook indices)
    timestamps: Tensor          # [B, N]
    unit_ids: Tensor | None     # [B, N], None for population tokenizers
    mask: Tensor                # [B, N], True for valid tokens
    combines_units: bool        # True ⇒ population tokenizer (no per-unit dim)
    token_bin_size_s: float | None   # token time resolution; None ⇒ variable-length
    vocab_sizes: tuple[int, ...] | None  # (K_0, …, K_{Q-1}) when tokens are int; else None
```

`combines_units=True` is incompatible with `unit_embedding_mode in {add, concat}`
(no per-token unit ID to look up). The adapter raises loudly if you try.

### Per-unit vs. population tokenizers

- **Per-unit** (`compress_n=False`, `has_unit_dim=True`): emit `[B, T', U, D]`
  latents; the wrapper flattens `T'×U → N` and stamps per-token `unit_ids`.
  Variable `U` across sessions is fine.
- **Population** (`compress_n=True`): emit `[B, T', 1, D]` population latents;
  the wrapper squeezes `U=1` and sets `combines_units=True`,
  `unit_ids=None`. These only support `unit_embedding_mode=none`. If
  `n_neurons` is fixed and the input has fewer neurons,
  `_pad_neurons_if_needed` duplicates existing neurons to fill the gap.

## Adapter (`tokenizer.unit_embedding_mode`)

The adapter builds the `[B, N, D_model]` POYO input from `tok_out`:

| Mode | Formula | Requires |
|---|---|---|
| `none` | `x = tok_out.tokens` (or `input_proj(x)` if `D_tok != D_model`) | nothing — works for both per-unit and population tokenizers |
| `add` | `x = (input_proj(x) if needed else x) + unit_emb(unit_id)` | `unit_ids != None` (per-unit tokenizer) |
| `concat` | `x = input_proj(cat(x, unit_emb(unit_id)))` | `unit_ids != None` and `input_proj` is required (no identity path) |

`input_proj` is materialised only when needed:
- `mode="concat"`: a `Linear(D_tok + D_model, D_model)` MLP (always required).
- `mode in {none, add}` and `D_tok != D_model`: a `Linear(D_tok, D_model)` MLP.
- `mode in {none, add}` and `D_tok == D_model`: `input_proj=None` (identity),
  preserving the exact `unit_emb + token_type_emb` formula for the spike-times
  path. The MLP is 1 hidden layer with GELU; hidden width = `D_model`.

**Sentinels** (`tokenizer.sentinels=true`) prepend one
`unit_emb(uid) + token_type_emb(SENTINEL_TYPE=1)` token per unique spiking
unit per sample, placed at `window_start=0.0`. Every `configs/decoding/tokenizer/`
group defaults to `true`; the tokenizer benchmark and the
`ntt-downstream-decode` entrypoint config pin `false`. `adapter.py` is the
source of truth. The sentinel set is **slightly different**
from upstream POYO, which emits START + END sentinels for *every* session
unit, including silent ones — the adapter only sees tokenizer output, not the
full session unit list.

## Frozen ntt checkpoints

`FrozenNTTTokenizer` wraps any ntt `BaseTokenizer` instance. By default it
calls `tokenizer.requires_grad_(False)` and `tokenizer.eval()`. Setting
`tokenizer.freeze=false` leaves params trainable, and
`V2TrainWrapper.configure_optimizers` adds a third optimizer group for the
trainable tokenizer params with peak LR `max_lr * cfg.optim.tokenizer_lr_factor`.
A warning banner is logged at construction time so this cannot happen by
accident.

Frozen tokenizers do per-sample binning of POYO-tokenized spikes back into
`BinnedSpikesInput` shape on GPU, then call the wrapped tokenizer's `encode()`.
The base `V2TrainWrapper._ntt_frozen_tokenize` is fully vectorized (single
batched `scatter_add`, two host syncs); the `FullWindowUnitV2TrainWrapper`
override still uses a per-sample loop because its global→local unit-id
remapping is sample-specific and doesn't fold cleanly into a batched scatter.

### Pydantic spec compatibility

Checkpoints can pickle `TokenizerSpec` instances whose fields differ from the
installed `ntt` schema. `_loosen_legacy_spec_validation` flips every loaded
`TokenizerSpec` subclass from `extra='forbid'` to `extra='ignore'` before
`torch.load` runs, so unpickling succeeds at the cost of silently dropping
unknown fields. A warning lists the patched classes; whether the dropped fields
were architecture-affecting is the caller's responsibility to verify.

## Configs

Hydra config root: `configs/decoding/` (repo root, resolved via `.project-root`).

```
configs/decoding/
├── train.yaml             # entry point (imports defaults)
├── defaults.yaml          # all knobs, fully commented (the readable reference)
├── data/
│   ├── ntt_reaching_strict.yaml      # multi-session + tier eval
│   └── calibration_multi_base.yaml
├── model/
│   ├── poyo_1.3M.yaml             # paper config, default
│   ├── poyo_1.3M_no_crossattn.yaml
│   └── poyo_11.8M.yaml
└── tokenizer/
    ├── fake.yaml          # fake-tokenizer equivalence
    ├── spike_times.yaml   # NTTSpikeTimesTokenizer
    ├── poisson.yaml       # Poisson rate-bin baseline
    ├── frozen.yaml        # frozen ntt checkpoint
    └── frozen_full_window.yaml  # full-window/all-unit handoff
```

`train.yaml` imports `defaults`, picks `model=poyo_1.3M`, and `tokenizer=fake`.
The minimum override needed for a real run is at least the tokenizer and
either a `session` (single-session) or `+data=ntt_reaching_strict`
(multi-session). `defaults.yaml` is the readable reference — every knob has an
inline comment.

## Knobs

### Single-session vs. multi-session

The same script handles both. The branching key is `cfg.dataset`:

- `dataset: null` (default) → `SingleSessionCODataset(recording_id=cfg.session)`.
- `dataset: <selection-list>` → `build_multi_brainset_co_dataset(selection=cfg.dataset, ...)`.

`+data=ntt_reaching_strict` sets `dataset` to
`${ntt_split:ntt_reaching_strict_train}` and turns on `tiers.enabled=true`
with the right tier 1/2/3 selections. The `ntt_reaching_strict*` selection
lists come from ntt (`NTT_SPLITS_DIR` or a source install; see
[environment.md](environment.md#ntt-selections-directory)). The session lists
in `configs/decoding/` are defaults; ntt supplies its own selections.

### Training schedule

| Key | Default | Notes |
|---|---|---|
| `epochs` | 1000 | Used in epoch mode (when `max_steps` is null). |
| `max_steps` | null | Step-based override. When set, `max_epochs=-1` and the trainer is driven by total optimizer steps. Used for calibration sweeps where data-budget changes per run but step count must stay constant. |
| `eval_epochs` | 4 | Validation cadence in epoch mode. |
| `eval_every_n_steps` | 200 | Validation cadence in step mode. |
| `checkpoint_every_n_steps` | 200 | Checkpoint cadence in step mode. |
| `seed` | 42 | Run seed; threaded into per-rank sampler RNGs and worker seeds. |

`epochs`, `max_steps`, and `*_every_n_steps` are all **1-GPU-equivalent counts
at effective batch `cfg.batch_size * world_size`**. The script divides by
`world_size` so the OneCycleLR ramp and total-step budget are constant across
W-GPU configurations.

In step mode (`max_steps != null`):
- `max_epochs=-1` (no cap)
- `val_check_interval = eval_every_n_steps` and
  `check_val_every_n_epoch=None` (Lightning rejects `val_check_interval >
  steps_per_epoch` otherwise)
- `ModelCheckpoint` switches to `every_n_train_steps` and
  `save_on_train_epoch_end=False`

### Optimizer

`SparseLamb` with `OneCycleLR` warmup. `peak_lr = base_lr * effective_batch_size`
where effective batch is `batch_size * gpus * num_nodes`.

| Key | Default | Notes |
|---|---|---|
| `optim.base_lr` | 3.125e-5 | Multiplicative — see formula. `1.953125e-6 * 512 = 1.0e-3` for a multi-session conv-* run. |
| `optim.weight_decay` | 1.0e-4 | |
| `optim.warmup_pct` | 0.05 | OneCycleLR `pct_start`. |
| `optim.div_factor` | 25 | Initial LR = `peak / 25`. |
| `optim.final_div_factor` | 40 | Final LR = `peak / (25 * 40) = peak / 1000`. |
| `optim.tokenizer_lr_factor` | 1.0 | Multiplier on the tokenizer-param-group peak LR; only used when `tokenizer.freeze=false`. |

### Data

| Key | Default | Notes |
|---|---|---|
| `session` | `c_20150319_center_out_reaching` | Single-session recording ID. Ignored when `dataset` is set. |
| `data_root` | `./data/processed/` | Brainsets processed root (parent of `<brainset>/<session>.h5`). |
| `dataset` | null | Multi-session selection list (resolver output). |
| `restrict_brainsets` | `[]` | Brainset allowlist for multi-session runs. Typo-guarded. |
| `co_only` | true | Filter Perich `_random_target_reaching` sessions out (CO-only). |
| `temporal_split.*` | disabled | Time-aware trial split (see [below](#temporal-split)). |
| `train_decimation.*` | disabled | Decimate train trials to a fixed count (see [below](#train-decimation)). |
| `tiers.*` | disabled | Tier-1/2/3 eval pools (only consumed when `dataset` is set). |

### Calibration / fine-tune

| Key | Default | Notes |
|---|---|---|
| `init_ckpt` | null | Path to a Lightning `.ckpt` to initialise the wrapper from. `InfiniteVocabEmbedding` pre-hooks materialise from the ckpt's vocab; new units/sessions are then `extend_vocab()`-ed in. |
| `freeze_mode` | `none` | `none` ‖ `unit_identification`. The latter freezes everything except `unit_emb` and `session_emb` (the unit-identification fine-tune from the POYO paper). |
| `ft_train_input_proj` | false | When `freeze_mode=unit_identification`, also unfreeze `input_proj`. Useful for `combines_units=True` tokenizers (e.g. Perceiver) where the body relies on `input_proj`-projected slot content. |
| `eval_only` | false | Skip `trainer.fit()`; only run `validate()` + `test()`. Used by the `ft_trials=0` leg of the calibration sweep. |

### Logging

| Key | Default | Notes |
|---|---|---|
| `log_every_n_steps` | 10 | Lightning logger flush + grad-norm compute cadence. |
| `wandb.enabled` | true | |
| `wandb.{project,name,entity,group,tags,notes}` | various | Forwarded to `wandb.init()`. |
| `mlflow.enabled` | false | When true, reads `MLFLOW_TRACKING_URI` etc. from env; true with no URI stops the run with an error. Sweeps set this per cell (see [Running sweeps without MLflow](#running-sweeps-without-mlflow)). |

### Trainer / hardware

| Key | Default | Notes |
|---|---|---|
| `gpus` | 1 | |
| `num_nodes` | 1 | |
| `strategy` | `auto` | Lightning strategy. `auto` → single-device when `gpus=1`, DDP when `gpus>1`. POYO needs `find_unused_parameters=True` under DDP; `_wo_crossattention` relies on this too — don't "fix" it. |
| `precision` | `"32"` | Set to `bf16-mixed` if you turn on `force_flash_attn`. |
| `force_flash_attn` | false | Monkey-patches torch_brain to route every attention call through FlashAttention's varlen kernel — sidesteps PyTorch SDPA's "no `attn_mask` in flash kernel" restriction. Requires `flash_attn` and bf16/fp16 inputs. |
| `num_workers` | 8 | |
| `unit_dropout` | false | When true, prepends `UnitDropout(min=30, mode=100, max=300, peak=4)` to the train transform pipeline (POYO paper defaults). |

### Frozen-tokenizer-specific knobs

Surfaced under `tokenizer.*` for `tokenizer={frozen, frozen_full_window}`.
The two Hydra YAMLs (`configs/decoding/tokenizer/frozen.yaml`,
`configs/decoding/tokenizer/frozen_full_window.yaml`) are the source of truth
for defaults; the table below is a reference summary:

| Key | Default | Notes |
|---|---|---|
| `tokenizer.ntt_checkpoint` | required | Local `.ckpt` path or `mlflow-artifacts:` (or `mlflow:`) URI. |
| `tokenizer.freeze` | true | Set to false to fine-tune the ntt tokenizer end-to-end (a warning banner is logged). |
| `tokenizer.discrete_input_mode` | `z_q` | `"z_q"` (continuous embedding) or `"embedding"` (learnable codebook lookup) for discrete-codebook tokenizers. |
| `tokenizer.discrete_indices_source` | `auto` | `"auto"` ‖ `"encode"` ‖ `"bottleneck"` ‖ `"forward"` — where to read codebook indices from in the wrapped tokenizer. |
| `tokenizer.warm_start_unit_emb_proxy` | false | At FT init, copy base session's first-M slot-proxy rows into the new session's first-M proxy rows. Only meaningful for `combines_units=True` tokenizers used with `init_ckpt`. |
| `tokenizer.use_global_slot_proxies` | false | Register `M` universal `_slot_k` tokens in `unit_emb` instead of using per-session real-unit_ids as slot proxies. Architecturally honest for abstract-latent tokenizers (slot k means the same thing across sessions). |
| `tokenizer.full_window_units` | `auto` (`true` in `frozen_full_window`) | Feed every session unit, with local unit ids, on the `[0, sequence_length]` raster (POYO→NTT handoff). `auto` turns it on for session-keyed tokenizers (per-session unit embeddings, e.g. Perceiver) and off otherwise; `false` on a session-keyed tokenizer raises, since the active-unit handoff's POYO-global ids would be silently clamped. **Set `true` for fixed-population tokenizers** (`compress_n: true`, `variable_n: false`, e.g. LDNS, wavelet-*): `auto` leaves them on the active-unit handoff, which moves neurons between positions from window to window. See [Which unit handoff a frozen tokenizer needs](#which-unit-handoff-a-frozen-tokenizer-needs). |
| `tokenizer.include_padded_zero_unit_tokens` | false (full_window only) | Include zero-spike padding tokens for absent units. |

### Hydra resolvers

`${ntt_split:<name>}` — registered by `register_ntt_split_resolver()` at script
import. Reads `<ntt>/config/data/selections/<name>.yaml`, drops upstream
`_target_` keys, stringifies all session/subject IDs (defends against YAML
auto-typing all-digit IDs as ints), and rewrites into the `torch_brain.data.Dataset`
selection-list shape. See [environment.md](environment.md#ntt-selections-directory)
for resolution order.

## Data & splits

### Datasets

| Path | Class | When |
|---|---|---|
| Single-session | `SingleSessionCODataset(recording_id=cfg.session)` | `cfg.dataset is None` (default) |
| Multi-session | `build_multi_brainset_co_dataset(selection=cfg.dataset, ...)` returning a `torch_brain.data.NestedSpikingDataset` | `cfg.dataset is not None` |

Despite their names ("CO" = center-out), both classes accept Perich
`_center_out_reaching` and `_random_target_reaching` sessions — the right
per-recording readout config is dispatched by suffix in `get_recording_hook`.

### Multi-brainset support

`build_multi_brainset_co_dataset` consumes the resolved output of a
`${ntt_split:...}` selection list. For each brainset entry it instantiates the
right class:

| Brainset | Class | Readout |
|---|---|---|
| `perich_miller_population_2018` | `PerichMillerPopulation2018` | `CO_READOUT_CONFIG` or `RT_READOUT_CONFIG` (by session-id suffix) |
| `odoherty_sabes_nonhuman_2017` | `OdohertySabesNonhuman2017` | `ODOHERTY_READOUT_CONFIG` |
| `churchland_shenoy_neural_2012` | `ChurchlandShenoyNeural2012` | `CHURCHLAND_READOUT_CONFIG` |

Each child dataset is wrapped in a `_with_readout` subclass whose
`get_recording_hook` stamps `data.config = deepcopy(readout_selector(session_id))`.
Recording IDs in the nested dataset follow `{brainset}/{session}` — this is
what `train_ds.recording_ids` returns and what shows up in Lightning metrics
keys for per-session R².

### Tier evaluation pools

Multi-session runs evaluate against a list of held-out pools, each wired as a
separate `DataLoader` and each producing per-session R²s through the shared
`SafeDecodingStitchEvaluator`. The `TierAggregator` callback then computes
per-tier averages and per-brainset breakdowns:

```
val/tier1_val_r2                          # mean over tier 1 sessions
val/tier1_val_r2_perich_miller_co         # subset by brainset / task type
val/tier2_val_r2_odoherty
test/tier3_test_r2
```

`ntt_reaching_strict` ships five tier YAMLs:

| Tier | Selection | Meaning |
|---|---|---|
| Tier 1 (xtime val) | `ntt_reaching_strict_tier1_xtime_val` | Same training sessions, held-out time slice — `split=valid` |
| Tier 2 (xsession val/test) | `..._tier2_xsession_val/test` | Same subjects, different sessions. `ignore_splits=true` upstream → `split=null` (full session domain). |
| Tier 3 (xsubject val/test) | `..._tier3_xsubject_val/test` | Different subjects entirely. `ignore_splits=true` → `split=null`. |

Vocab initialization walks **every** dataset (train + all tier loaders) and
`initialize_vocab(union_unit_ids)` / `initialize_vocab(union_session_ids)`
once before `trainer.fit`. Without this, POYO's `tokenize` would key-error on
tier sessions whose units were never registered in `unit_emb`.

### Temporal split

`temporal_split.enabled=true` re-derives `train_domain` / `valid_domain` /
`test_domain` at dataset-construction time from the loaded `Data.trials`,
ignoring the HDF5-baked random splits. **Trials are kept whole, never sliced.**

`compute_temporal_domains` lays trials out as
`[ train (n_train) | gap (discarded) | valid (n_valid) | test (n_test) ]`
with train pinned to the start and valid+test pinned to the end, so shrinking
train across a sweep keeps the same held-out trials.

| Mode | Behavior |
|---|---|
| `temporal` (default) | Order array sorted ascending by trial start time. Eval is the chronological tail of the recording — leakage-safe. Test typically scores lower than valid because eval lives in the temporally-distant future. |
| `random` | `np.random.default_rng(seed).permutation(n)`. Eval trials scattered uniformly through the recording; with a fixed seed the assignment is reproducible across train-ratio sweeps. |

`train_domain` is the full session domain minus `(gap ∪ valid ∪ test)` dilated
by `dilation` seconds (default 1.0), matching the upstream brainsets pipeline
buffer. The override is gated by a `brainsets` allowlist — Odoherty (already
temporal-block-based) is left untouched by default.

### Train decimation

`train_decimation.trials=N` replaces train sampling intervals with the time
spans of `N` seeded-random valid trials drawn from the baseline train domain.
Composes on top of whichever split source produced that domain (HDF5 baked or
`temporal_split` override). Valid/test pass through unchanged.

This is the canonical "use exactly N train trials" knob, used by the
calibration sweeps to vary the train-data budget while keeping the held-out
eval layout fixed.

- Selection is by trial *index* (after `is_valid=True` filtering), not by
  time, so the same `cfg.seed` picks the same set regardless of split layout.
- A trial is eligible iff a single baseline interval covers both its start
  and its end.
- `min_duration` is forwarded from the trainer (= `model.sequence_length`),
  so trials shorter than the sampler's window are excluded **before** the
  seeded random pick.
- `brainsets` allowlist mirrors `temporal_split.brainsets`. `null` means
  "decimate every brainset"; set to a list to restrict.

### Per-rank seeding

`RandomFixedWindowSampler` does **not** shard under DDP — each rank draws its
own random windows from the full pool. Per-rank seeding (`cfg.seed + rank`)
keeps ranks decorrelated; effective batch is `world_size * cfg.batch_size`.

`DistributedStitchingFixedWindowSampler` for eval is constructed with
`num_replicas=1, rank=0` so every rank evaluates the **full pool**. Sharding
eval would deadlock `SafeDecodingStitchEvaluator`'s per-rank filter at the
parent's `metric_fn.compute()` all-reduce. Duplicate eval work is cheap
relative to training.

## Outputs and logging

The training script wires three loggers (CSV always; wandb and mlflow opt-in).
A run writes, under `csv_logs/version_*/`:

- `metrics.csv` — Lightning CSVLogger output
- `run_info.json` — token stats, trainable-param count (logged at `on_fit_start`)
- `downstream_decoding_results.json` — the run's results record (schema v2;
  see [results.md](results.md)): `val_best` / `val_last` / `test`, uploaded to
  MLflow / W&B. A benchmark also merges each tokenizer's cells into a cohort
  record, which is what ntt's aggregation reads.
- `run_result.json` — a flat local record, written by `_save_results` via
  `results_storage.py`. It is sweep bookkeeping (completion checks,
  plain-eval mirror, calibration CSV) and is **not uploaded**.
- Best/last checkpoint (from `ModelCheckpoint`)
- `wandb/` — when `wandb.enabled=true`

The record is the run's output, so a run that cannot write it fails, and
`run_result.json` is then not written either (a resumed `--missing-only`
calibrate sweep would otherwise treat the cell as done). The config is fully
resolved at startup, since the record stores the resolved config, so a broken
interpolation stops the run before training instead of after it.

### `run_result.json`

Flat keys from `results_storage.build_run_result`: `timestamp, session,
tokenizer_type, tokenizer_checkpoint, unit_embedding_mode, sentinels,
unit_dropout, model_depth, model_dim, model_params, token_dims_per_sec,
epochs, max_steps, test_r2, val_best_r2, log_dir`, plus the calibration-sweep
fields `val_last_r2`, `init_ckpt`, `freeze_mode`, `eval_only`,
`metric_schema_version` and (with train decimation) `requested_trials` /
`kept_trials`.

`ntt_downstream.decoding.calibration_csv` slices a calibration sweep into
per-FT-mode summary CSVs.

### `metric_schema_version`

`run_info.json` stamps `metric_schema_version: 2`, meaning
`token_dims_per_sec` is derived from the tokenizer mask (`mask_sum` already
includes the `T'` factor of fixed-rate per-unit tokenizers).

## Calibration sweeps

`scripts/decoding/calibration_general_singlesession.sbatch` and
`calibration_general_multisession.sbatch` (thin wrappers over
`orchestration calibrate [--multi-session]`; see
[decoding_orchestration.md](decoding_orchestration.md)) implement the
POYO-paper "transfer to a new session" protocol:

1. **Base train** on `TRAIN_SESSION` with the HDF5-baked random ~70% train
   domain — no temporal-split override, no decimation.
2. For each session in `OTHER_SESSIONS`:
   - **Plain eval** (`n_trials=0`, `eval_only=true`): load the base ckpt,
     extend the unit/session vocab, run `validate()` + `test()`.
   - **Fine-tune sweep** at each `FT_TRIAL_COUNTS = {4, 8, 16, 32, 64, 86}` in two
     modes:
     - `unit_identification` — freeze everything but `unit_emb` /
       `session_emb` (the POYO paper's unit-identification fine-tune).
     - `full` — update all weights.
     Each FT run sets `train_decimation.trials=$n_trials`. Valid/test stay on
     the HDF5-baked random splits so every FT run is comparable to the
     `n_trials=0` baseline.

Counts exceeding `floor(0.7 * n_valid_trials)` are skipped per session. The
orchestrator aggregates `val_last_r2` across all FT runs into two CSVs (one
per FT mode) under the per-base-run directory.

Driven by `configs/decoding/single_session/` and `configs/decoding/multi_session/`
(non-Hydra YAMLs loaded by `ntt_downstream.decoding.config_loader` and consumed
by the orchestration engine).

## Running sweeps without MLflow

A single `train.py` run uses no MLflow unless you set `mlflow.enabled=true`.
The `benchmark` and `calibrate` sweeps take an `mlflow` knob (`--mlflow`, the
`mlflow:` key in `configs/cli/*.yaml`, or `mlflow.enabled` in the calibration
YAML) with three values:

- `auto` (the default): MLflow is on when `MLFLOW_TRACKING_URI` is set and off
  otherwise, so a sweep runs without a tracking server with no extra flags.
- `true`: MLflow is required. With no `MLFLOW_TRACKING_URI` the sweep stops
  before launching any cell.
- `false`: no MLflow, even when a URI is set.

With MLflow off, each cell still writes `downstream_decoding_results.json` and
`run_result.json`, and the sweep still writes its cohort record. You lose the
MLflow parent/child runs and their aggregated metrics. `MLFLOW_ENABLED` in
`scripts/decoding/scratch_scaling_benchmark.sh` takes the same three values,
with the same default and the same early stop for `true` without a URI.

## Variant: no encoder cross-attention

`POYOWithEmbeddingBypass_wo_crossattention` (selected via
`model=poyo_1.3M_no_crossattn`) skips the encoder cross-attention and the
latent grid entirely — input tokens are fed directly into `proc_layers`
self-attention, and the decoder cross-attention queries the processed input
tokens. A `LayerNorm` (`input_pre_norm`) is added before `proc_layers` because
the residual stream from a frozen ntt tokenizer can have a very different
scale from POYO's `unit_emb` initialization (`std=0.02`).

Because `enc_atn`, `enc_ffn`, and `latent_emb` are still constructed by the
parent `__init__` but unused at forward time, multi-GPU training needs
`find_unused_parameters=True`.
