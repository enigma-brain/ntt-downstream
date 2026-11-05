# Repo organization

```
ntt-downstream/
├── README.md                     # overview, install, quick start
├── LICENSE                       # MIT
├── LICENSES/                     # licence texts for adapted third-party code
├── CITATION.cff                  # citation metadata
├── pyproject.toml                # pinned deps; uv source overrides for ntt + torch_brain; console scripts
├── uv.lock
├── .env.example                  # optional MLflow credentials and SLURM submission knobs
├── .pre-commit-config.yaml
├── .project-root                 # marker file for rootutils / Hydra config discovery
├── .github/                      # CI (env build, lint, tests) and dependabot
├── docs/                         # this directory
├── configs/
│   ├── cli/                      # curated entrypoint configs for the four launcher verbs
│   ├── decoding/                 # Hydra tree for decoding + non-Hydra calibration YAMLs
│   └── encoding/                 # Hydra tree for the encoding entry point
├── scripts/
│   ├── decoding/                 # sbatch wrappers + submit.sh over the orchestration engine
│   ├── encoding/                 # tokenizer_roundtrip.py
│   └── fixtures/                 # regenerates tests/fixtures/results_v2/
├── src/ntt_downstream/
│   ├── cli/                      # the ntt-downstream-{decode,encode,calibrate,benchmark} launcher
│   ├── common/results/           # results-record schema, writer, provenance, Lightning callback
│   ├── decoding/                 # POYO bypass library + orchestration engine
│   └── encoding/                 # encoding library
├── tests/                        # cli/, common/, decoding/, encoding/, fixtures/
├── data/                         # placeholder for processed data (contents are gitignored)
└── tokenizer_ckpts/              # placeholder for tokenizer checkpoints (contents are gitignored)
```

`outputs/`, `experiments_*/` and `results/` are gitignored directories that
runs create: per-run outputs, calibration and benchmark sweep trees, and
aggregated cross-run results. Frozen ntt checkpoints for the decoding benchmark
are read from `results/tokenizer_checkpoints/` by default (`--ckpt-root`, or
`CKPT_ROOT` in the sbatch scripts), laid out as
`<root>/<name>/checkpoints/model.ckpt`. `tokenizer_ckpts/` is the location the
checkpoint-backed tests look in by default (`NTT_TOKENIZER_CKPTS` overrides it).

Every knob is a config key under `configs/`. If you need a new one, add a key
there rather than a CLI argument or a hard-coded constant.

## Decoding: `src/ntt_downstream/decoding/`

Single Hydra entry point: `train.py`. The shape of a run is determined
entirely by Hydra overrides; the script handles single- vs multi-session
branching internally.

| File | Role |
|---|---|
| `train.py` | Hydra entry: `python -m ntt_downstream.decoding.train`. |
| `orchestration.py` | The `calibrate` / `benchmark` sweep engine (see [decoding_orchestration.md](decoding_orchestration.md)). |
| `config_loader.py` | Loads the calibration YAMLs (`extends:` chain) through the `TRAINING_KEYS` / `DATA_KEYS` whitelists. |
| `calibration_csv.py` | Builds the per-fine-tune-mode calibration CSVs. |
| `cohort_results.py` | Merges benchmark cells into one cohort record per tokenizer. |
| `mlflow_utils.py`, `utils/mlflow_parent.py` | MLflow parent-run helpers and the `create` / `aggregate` / `log-artifacts` / `export-metrics` subprocess CLI. |
| `backends/poyo/model.py` | `POYOWithEmbeddingBypass`, a minimal POYO subclass overriding `forward()` to accept `input_embeddings`, plus a `_wo_crossattention` variant. |
| `backends/poyo/poyo.py` | Vendored POYO core, kept so the bypass class can subclass it cleanly. |
| `backends/poyo/tokenizer_output.py` | The `TokenizerOutput` dataclass, the contract every tokenizer satisfies. |
| `backends/poyo/adapter.py` | `adapt()` converts `TokenizerOutput` to `[B, N, D_model]`, applies the unit-embedding mode, prepends sentinels. |
| `backends/poyo/train_wrapper.py` | `V2TrainWrapper` (Lightning): owns the tokenizer, optimizer setup, run-info banner. Three tokenize modes: `fake`, `ntt`, `ntt_frozen`. |
| `backends/poyo/full_window_train_wrapper.py` | Variant for full-window (session-keyed) tokenizers. |
| `backends/poyo/frozen_ntt_tokenizer.py` | Wraps a frozen ntt checkpoint (local path or `mlflow-artifacts:` URI). |
| `backends/poyo/ntt_spike_times_tokenizer.py` | Spike-times tokenizer that emits `TokenizerOutput`. |
| `backends/poyo/flash_attn_varlen.py` | FlashAttention varlen wrapper used under DDP. |
| `backends/poyo/datasets.py` | Per-session Perich-Miller dataset (CO + RT). |
| `backends/poyo/multi_brainset_dataset.py` | Multi-brainset center-out reaching dataset. |
| `backends/poyo/multi_session_setup.py` | Multi-session and tiered-eval setup helpers. |
| `backends/poyo/ntt_splits.py` | `${ntt_split:...}` Hydra resolver. |
| `backends/poyo/temporal_split.py` | Deterministic temporal-split sampler. |
| `backends/poyo/train_decimation.py` | Train-trial decimation with a seeded eligibility filter. |
| `backends/poyo/stitch_evaluator.py` | Per-session R² stitch evaluator (suppresses verbose val tables). |
| `backends/poyo/results_parser.py` | Parses Lightning metrics into the shared results-record metric blocks. |
| `backends/poyo/results_storage.py` | Writer for the flat local `run_result.json` (sweep bookkeeping; the results record is written via `common/results/`, see [results.md](results.md)). |

## Encoding: `src/ntt_downstream/encoding/`

| File | Role |
|---|---|
| `train.py` | Hydra entry: `python -m ntt_downstream.encoding.train`. |
| `models/enc/hiera/base_model.py` | `ImageEncodingHiera`: backbone + cross-attention readout + per-loss prediction head. |
| `models/enc/hiera/mar_model.py` | `ImageEncodingHieraMAR`, the MAR (masked autoregressive) variant. |
| `models/enc/hiera/model_factory.py` | Builds the right model from `cfg`; wires diffusion / energy / cross-entropy heads. |
| `models/enc/hiera/modded_hiera.py` | Hiera backbone with custom mask-unit handling. |
| `models/enc/hiera/evals.py` | `OnlinePearsonAccumulator`, `TemporalCorrAccumulator`, rebinning helpers. |
| `models/enc/hiera/noise_corrected_r2.py` | Noise-corrected R² after Pospisil & Bair (2021), implemented in torch. |
| `models/enc/audio/base_model.py` | `AudioEncodingModel`: CausalConv backbone + cross-attention readout for audio. |
| `models/enc/audio/model_factory.py` | Audio model builder. |
| `models/enc/conv/` | Cowley-paper-matched ConvCoreReadout baseline (stacked 2D conv core + Gaussian readout). |
| `models/enc/losses/diffloss.py`, `losses/diffusion/` | DiffLoss head (DDPM diffusion in latent space) and its diffusion utilities. |
| `models/enc/losses/energy_loss.py` | EAR energy-score loss head. |
| `models/enc/losses/flow_matching_head.py`, `flow_utils.py` | Pointwise flow-matching head and shared rectified-flow utilities. |
| `models/enc/losses/full_grid_denoiser.py` | `FullGridFlowMatchingHead`: joint denoiser over the full latent grid. |
| `models/enc/losses/ldns_denoiser.py` | LDNS-style temporal denoiser. |
| `models/enc/viz.py` | Neuron-trace plotting helpers for eval logging. |
| `tokenizer_wrappers/encoding_wrapper.py` | `TokenizerWrapper`: bridges ntt tokenizers and encoding-pipeline tensors; supports `compress_t` (per-neuron) and `compress_n` (population) tokenizers. |
| `training/encoding_module.py` | `EncodingLitModule` (image, Cowley V4). |
| `training/audio_encoding_module.py` | `AudioEncodingLitModule` (audio, Wingert). |
| `training/data_module.py`, `training/wingert_data_module.py` | `CowleyV4DataModule` and `WingertAuditoryDataModule`. |
| `training/final_eval.py`, `training/prediction.py`, `training/ce_marginal.py` | End-of-training evaluation, the loss-mode to rate-prediction dispatch, and the cross-entropy marginal-rate approximation. |
| `training/tokenizer_startup_roundtrip.py` | Pre-fit tokenizer encode-decode ceiling diagnostic. |
| `training/results_parser.py` | Parses Lightning metrics into the shared results-record metric blocks. |
| `dataloading/cowley_v4.py`, `dataloading/wingert_auditory.py` | Dataset adapters. |
| `dataloading/samplers.py` | `DistributedSessionBatchSampler`: partitions sessions across DDP ranks. |
| `dataloading/spike_augmentations.py` | Spike-train augmentations applied at the data-module level. |
| `analysis/radar.py` | Aggregated radar plots across tokenizers; its docstring documents the `manifest.csv` columns (`tokenizer`, `eval_report_path`, `<source>_eval_report_path`) it reads. |

## Launcher and results: `cli/`, `common/results/`

`cli/` implements the four `ntt-downstream-*` console scripts (see
[cli.md](cli.md)). `common/results/` defines the v2 results record and writes it
for both tasks (see [results.md](results.md)).

## Configs

Decoding Hydra configs live under `configs/decoding/` (the `train.py`
`@hydra.main` tree: `train.yaml`, `defaults.yaml`, `model/`, `tokenizer/`,
`data/`). Encoding configs live under `configs/encoding/`. The
`configs/decoding/` directory also holds the non-Hydra calibration YAMLs
(`single_session/`, `multi_session/`) consumed by the orchestration engine
through `ntt_downstream.decoding.config_loader`. `configs/cli/` holds the
curated entrypoint configs of the launcher verbs.

## Scripts

`scripts/decoding/`:

| Script | What it runs |
|---|---|
| `calibration_general_singlesession.sbatch` | Single-session calibration sweep (base train + per-session fine-tune sweep); thin SLURM wrapper over `python -m ntt_downstream.decoding.orchestration calibrate`. |
| `calibration_general_multisession.sbatch` | Multi-session variant (`orchestration calibrate --multi-session`): joint base train over the whole train pool. |
| `tokenizer_benchmark.sbatch` | R² / wall-clock benchmark across tokenizers (plus the spike-times baseline via `SPIKE_TIMES_ONLY=1`); thin wrapper over `orchestration benchmark`. |
| `scratch_scaling_benchmark.sh` | From-scratch trial-budget scaling benchmark. |
| `submit.sh` | Submits an sbatch script with `--partition` / `--gres` injected from `.env`. |

`scripts/encoding/tokenizer_roundtrip.py` encodes and decodes through a
tokenizer to inspect reconstruction quality. `scripts/fixtures/make_results_v2_fixtures.py`
regenerates the results fixtures.

The sbatch scripts are generic examples: each carries a `gpu` partition
fallback and an "adjust the `#SBATCH` resource lines to your cluster" header.
Edit them, or submit through `submit.sh` (see [environment.md](environment.md)).

## Tests

`tests/decoding/poyo/` covers adapter modes, sentinel placement, the
embedding-bypass forward path, FlashAttention varlen parity, frozen-ntt
loading, train-decimation eligibility, and `metric_schema_version=2`
accounting. `tests/decoding/` also covers the orchestration engine, cohort
records and MLflow helpers. `tests/encoding/` covers data-module shapes,
tokenizer-wrapper round-trips, loss-head numerical parity, and DDP sampler
partitioning. `tests/cli/` and `tests/common/` cover the launcher and the
results record.

Skip markers: `-m "not slow"`. Some checkpoint-backed tests skip when
`NTT_TOKENIZER_CKPTS` (default `./tokenizer_ckpts/`) or `NTT_DOWNSTREAM_DATA`
(default `./data/processed/`) lack the referenced fixture.

## Third-party code

The repo's own code is MIT licensed (see [LICENSE](../LICENSE)). These files
adapt third-party code and keep the upstream licence, as stated in each file's
header. The upstream licence texts and copyright notices are in
[`LICENSES/`](../LICENSES/).

| File(s) | Upstream | Licence |
|---|---|---|
| `decoding/backends/poyo/{poyo,model,train_wrapper,multi_session_setup,flash_attn_varlen}.py` | [torch_brain](https://github.com/neuro-galaxy/torch_brain), Copyright 2025 NeuroGalaxy | Apache-2.0 |
| `encoding/models/enc/hiera/modded_hiera.py`, `hiera/utils.py` | [Hiera](https://github.com/facebookresearch/hiera), Meta Platforms | Apache-2.0 |
| `encoding/models/enc/hiera/mar_model.py`, `losses/diffloss.py` | [MAR](https://github.com/LTH14/mar), Copyright (c) 2024 Tianhong Li; `TimestepEmbedder`, `FinalLayer` and `modulate` originate in [DiT](https://github.com/facebookresearch/DiT) (CC BY-NC 4.0), used as redistributed by MAR under MIT | MIT |
| `encoding/models/enc/losses/diffusion/` (`gaussian_diffusion.py`, `respace.py`, `diffusion_utils.py`) | [OpenAI diffusion](https://github.com/openai/improved-diffusion), Copyright (c) 2021 OpenAI, via MAR; `create_diffusion` originates in [DiT](https://github.com/facebookresearch/DiT) (CC BY-NC 4.0), used as redistributed by MAR under MIT | MIT |
| `encoding/models/enc/losses/energy_loss.py` | [EAR](https://github.com/shaochenze/EAR) (Shao et al., 2025), MIT via MAR; `modulate` and the residual blocks originate in [DiT](https://github.com/facebookresearch/DiT) (CC BY-NC 4.0), used as redistributed by MAR under MIT | MIT |
| `encoding/models/enc/conv/{core,layers,model,readout}.py` | [open-retina](https://github.com/open-retina/open-retina), Copyright (c) 2025 open-retina | MIT |
| `encoding/models/enc/losses/ldns_denoiser.py` | [LDNS](https://github.com/mackelab/LDNS), Copyright (c) 2024 mackelab; its `TimestepEmbedder` originates in [DiT](https://github.com/facebookresearch/DiT) (CC BY-NC 4.0), used as redistributed by MAR under MIT | MIT |
| `encoding/models/enc/hiera/noise_corrected_r2.py` | Estimator from Pospisil & Bair (2021), implemented from the paper; the Cowley group's code was the inspiration | n/a (no upstream code) |

Pretrained Hiera weights, downloaded only by the optional `hiera_pretrained_*`
backbones, are licensed CC BY-NC 4.0 by Meta. The tokenizer weights that
`ntt` publishes on Hugging Face are CC-BY-NC-4.0, and each dataset keeps the
licence of its upstream source (see the `ntt` documentation).
