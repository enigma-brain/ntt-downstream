# Environment

Cache locations, data paths, and runtime identity vars used by both training
pipelines. Most knobs are config-driven; the env vars below only matter when
the default doesn't fit your setup.

## MLflow / wandb

| Var | Purpose |
|---|---|
| `MLFLOW_TRACKING_URI` | Encoding `train.py` reads this directly (`trainer.mlflow_tracking_uri` falls back to it). Decoding reads it when `mlflow.enabled=true` or when resolving an `mlflow-artifacts:` checkpoint; the `benchmark` / `calibrate` sweeps enable MLflow only when it is set ([decoding.md](decoding.md#running-sweeps-without-mlflow)). |
| `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` | Auth for the tracking server. |
| `WANDB_*` | Standard wandb config. The decoding logger is on by default (`cfg.wandb.enabled=true`). Encoding does not use wandb. |

With logging on (`trainer.logging=true`, the default), the encoding entry point
refuses to start without `MLFLOW_TRACKING_URI` set unless you pass
`trainer.mlflow_tracking_uri=...` on the CLI. With `trainer.logging=false` it
needs no MLflow configuration at all (see
[encoding.md](encoding.md#running-without-mlflow)). MLflow is optional for
decoding sweeps and encoding runs. The decoding SLURM scripts source a `.env`
file at the repo root (see `.env.example`); for other entry points, export the
variables in your shell.

## NTT selections directory

The decoding `${ntt_split:...}` Hydra resolver looks for selection YAMLs in:

1. `$NTT_SPLITS_DIR` (explicit override)
2. PEP 610 `direct_url.json` next to the installed `ntt` dist-info
3. Walk up from `ntt.__file__` to find a sibling `config/data/selections/`

Set `NTT_SPLITS_DIR` if `ntt` is pip-installed from a built wheel (cases 2
and 3 don't apply).

## Running against a working-tree `ntt`

`uv sync` installs the pinned `ntt` release. To run downstream evals against an
*in-progress* `ntt` checkout instead (co-development), swap just that package in
the venv without re-locking:

```bash
uv pip install -e /abs/path/to/ntt --no-deps --python .venv/bin/python
uv run --no-sync python -m ntt_downstream.encoding.train --config-name ...
uv sync   # revert to the pinned ntt
```

See [getting-started.md](getting-started.md#standalone-vs-embedded-in-ntt)
for the standalone-vs-embedded install modes and why the flags matter.

## Cache locations

If `$HOME` is on a small or slow filesystem, redirect heavyweight caches to a
larger work dir before invoking `uv sync` or any training run:

```bash
export UV_CACHE_DIR=/path/to/cache/uv
export TORCH_HOME=/path/to/cache/torch
export HF_HOME=/path/to/cache/huggingface
export WANDB_CACHE_DIR=/path/to/cache/wandb
```

`UV_CACHE_DIR` controls the uv pip cache (frequently the largest); `TORCH_HOME`
and `HF_HOME` cover model/weight downloads triggered by torch hub and
HuggingFace respectively. None of these are required — defaults under
`~/.cache/` work everywhere.

## Data roots

**Decoding.** Brainsets processed data is laid out as
`<data_root>/<brainset>/<session>.h5`. Pass it to `train.py` via
`data_root=...` on the CLI; the sbatch scripts under `scripts/decoding/`
accept the same value through the `DATA_ROOT` env var. Default:
`./data/processed/`.

**Encoding.** Both datasets are the preprocessed `cowley_v4_compact_2024`
and `wingert_auditory_cortex_2026` brainsets of the public NTT dataset (see
the [ntt documentation](https://github.com/enigma-brain/ntt)). Cowley V4 lives at `data.dataset_path` (default
`./data/cowley_v4_compact_2024`); Wingert lives at default
`./data/wingert_auditory_cortex_2026`. Override on the CLI:
`data.dataset_path=/abs/path`.

## Tokenizer checkpoints

Decoding takes a local `.ckpt` path or an `mlflow-artifacts:` URI as
`tokenizer.ntt_checkpoint`; `FrozenNTTTokenizer._resolve_ckpt_path` downloads
URIs via `mlflow.artifacts.download_artifacts`. When `ntt` launches a run, it
also accepts `hf://` checkpoints and hands this package a local `.ckpt`. The
decoding benchmark discovers tokenizers under a checkpoint root laid out as
`<root>/<name>/checkpoints/model.ckpt`, which defaults to
`results/tokenizer_checkpoints/`. Change it with `--ckpt-root` or the
`CKPT_ROOT` env var of the sbatch scripts.

Encoding takes a local path or an MLflow URI under `tokenizer.checkpoint_path`.
The Perceiver recipes also require `tokenizer.embeddings_path` pointing at the
NuCLR pre-computed unit embeddings (NuCLR, Azabou et al., is released with
`ntt`).

## SLURM identity

Decoding's `multi_session_setup.dist_rank_world()` reads `RANK` / `WORLD_SIZE`
first, then falls back to `SLURM_PROCID` / `SLURM_NTASKS`. Lightning's
launcher sets the first pair; bare SLURM `srun` sets the SLURM ones.
Single-process runs default to `(0, 1)`. This is read **at sampler-construction
time** — Lightning hasn't initialised the process group yet at that point, so
we can't use `torch.distributed.is_initialized()`.

Encoding's DDP support is via Lightning's `ddp_find_unused_parameters_true`
strategy by default when `trainer.devices > 1`. The encoding `train.py` also
threads `NTT_HYDRA_RUN_DIR` through Lightning's subprocess launcher so all
ranks resolve the same Hydra output dir (otherwise each child resolves
`${now:...}` independently and produces empty per-rank dirs).

## SLURM partition / gres (cross-cluster portability)

The decoding sbatch scripts under `scripts/decoding/` carry a neutral
`#SBATCH --partition=gpu` fallback. SLURM parses the `#SBATCH` block *before*
the shell runs, so — unlike every runtime knob — partition/gres can't ride the
`.env` → `--flag` forwarding the scripts use. To stay portable without editing
files, submit through the wrapper, which injects them as `sbatch` CLI flags
(CLI flags beat in-file `#SBATCH`):

```bash
PARTITION=gpu-big GRES=gpu:8 \
    scripts/decoding/submit.sh scripts/decoding/tokenizer_benchmark.sbatch
```

`scripts/decoding/submit.sh` sources the repo `.env`, so the usual way to
configure a site once is to set `PARTITION` / `GRES` (and optionally
`SBATCH_EXTRA="--time=… --mem=…"`) in `.env` — see `.env.example`. The
scripts' own env knobs pass through unchanged (`sbatch --export=ALL`). On a
cluster whose default GPU partition is named `gpu` you can skip the wrapper and
`sbatch` the scripts directly.
