# Getting started

## Install

The repo is `uv`-managed (Python 3.13). Two source-only deps come from
`tool.uv.sources`:

- `ntt` ([enigma-brain/ntt](https://github.com/enigma-brain/ntt)), pinned to a
  release tag. It provides the tokenizer specs, checkpoints and selection
  YAMLs.
- `torch_brain`, tracked from `neuro-galaxy/torch_brain@main`.

```bash
uv sync
```

`flash-attn==2.8.3` comes from a prebuilt wheel that matches
`torch==2.8.0+cu126`, so the two pins move together. If a fresh `uv sync`
leaves a venv with missing CUDA `.so` files (typically `libcudnn.so.9`), rerun
with `uv sync --reinstall` once.

### Standalone vs. embedded in `ntt`

There are two ways to get an `ntt` into this venv, and they are not mixed.

**Standalone** (a project that depends only on `ntt-downstream`): standalone
installs resolve the `ntt` release pinned in `pyproject.toml`. Use `uv run` /
`uv sync` as usual.

**Embedded** (this repo as the `downstream/` submodule of an `ntt` checkout,
or any checkout that launches it through `ntt`'s downstream jobs): the eval
runs against the *host's* `ntt`, the code that wrote the checkpoints, instead
of the pinned release, which can be older than the checkpoint format. The host
owns the env and overlays itself onto it:

```bash
# Everything from the lock except ntt, so the pinned ntt is never fetched.
uv sync --project <downstream> --frozen --no-install-package ntt --python 3.13
# The host checkout's ntt, editable, without re-resolving its deps.
VIRTUAL_ENV=<downstream>/.venv uv pip install --no-deps -e <ntt-checkout>
# Check: ntt resolves to the host, and its deps are satisfied by the lock.
<downstream>/.venv/bin/python -c 'import ntt; print(ntt.__file__)'
VIRTUAL_ENV=<downstream>/.venv uv pip check
```

Inside `ntt`, the downstream launcher (`ntt.jobs.downstream`) does this for you
and invokes the console scripts with `uv run --no-sync`. If you drive an
embedded env by hand, keep to `--no-sync` too: a plain `uv run`, `uv sync` or
`uv run --frozen` syncs the env back to the lock, which swaps the overlay for
the pinned release.

If `uv pip check` reports a missing or conflicting dependency, the host `ntt`
needs dependencies this lock does not provide. Update the `ntt` pin in
`pyproject.toml` and relock.

**Co-developing `ntt` from a standalone checkout** is the same overlay: after
`uv sync`, run the `uv pip install --no-deps -e <ntt-checkout>` step above,
drive runs with `uv run --no-sync ...`, and `uv sync` to revert to the pinned
release. `--no-deps` swaps only the `ntt` package, leaving the torch/flash-attn
pins untouched. `uv run --with-editable <ntt-checkout>` does **not** work: it
re-resolves the host's full dependency tree without this repo's `einops<0.7`
override and fails on the `vector-quantize-pytorch` conflict.

## Data

The examples use sessions from the public NTT dataset, which is built with
[brainsets](https://github.com/neuro-galaxy/brainsets). See the
[ntt documentation](https://github.com/enigma-brain/ntt) for how to obtain the
processed sessions, and pass the processed-data root as `data_root=...`
(decoding) or `data.dataset_path=...` (encoding). The encoding recipes read
the preprocessed `cowley_v4_compact_2024` and `wingert_auditory_cortex_2026`
brainsets from the same dataset: set `data.dataset_path=<processed>/cowley_v4_compact_2024`
(or `.../wingert_auditory_cortex_2026`), or pass `--data-root <processed>/<brainset>`
to `ntt-downstream-encode`. Each dataset keeps the licence of its upstream
source.

## Two entry points

| Task | Entry point | Default config |
|---|---|---|
| Decoding | `python -m ntt_downstream.decoding.train` | `configs/decoding/train.yaml` |
| Encoding | `python -m ntt_downstream.encoding.train` | `configs/encoding/cowley_v4_base.yaml` |

Both are Hydra-driven. Override knobs as `key=value` on the CLI. The
`ntt-downstream-{decode,encode,calibrate,benchmark}` console scripts wrap them
with a stable flag surface; see [cli.md](cli.md).

## Environment variables

See [environment.md](environment.md) for the full reference. The minimum:

| Var | When you need it |
|---|---|
| `MLFLOW_TRACKING_URI` (+ user/password) | Encoding runs log to MLflow by default (`trainer.logging=true`); pass `trainer.logging=false` to run without it. Required only when a decoding run loads an `mlflow-artifacts:` checkpoint or sets `mlflow.enabled=true`; decoding sweeps use MLflow only when it is set. |
| `NTT_SPLITS_DIR` | Decoding multi-session runs that consume `${ntt_split:...}` resolvers when `ntt` isn't installed from a source path. |

The encoding `train.py` reads `MLFLOW_TRACKING_URI` from the environment. Export
it yourself; the SLURM scripts under `scripts/decoding/` also source a `.env`
file at the repo root (see `.env.example`). No MLflow server? Add
`trainer.logging=false` to any encoding command; see
[encoding.md](encoding.md#running-without-mlflow).

## Decoding — first run (single session, fake tokenizer)

The "fake" tokenizer reproduces POYO's original computation through the bypass
path; it's the equivalence smoke test.

```bash
uv run python -m ntt_downstream.decoding.train \
    tokenizer=fake \
    session=c_20150319_center_out_reaching \
    data_root=/path/to/processed \
    epochs=10 \
    wandb.enabled=false
```

Outputs land under `outputs/poyo/` (override with `log_dir=...`). The run
writes `downstream_decoding_results.json` (see [results.md](results.md)),
`run_result.json` and `metrics.csv` under
`csv_logs/version_*/` (see [decoding.md § Outputs and logging](decoding.md#outputs-and-logging)).

## Decoding — first run (frozen ntt tokenizer)

```bash
uv run python -m ntt_downstream.decoding.train \
    tokenizer=frozen \
    tokenizer.ntt_checkpoint=/path/to/tokenizer.ckpt \
    tokenizer.unit_embedding_mode=add \
    session=c_20150319_center_out_reaching \
    data_root=/path/to/processed
```

`tokenizer.ntt_checkpoint` takes a local `.ckpt` path or an `mlflow-artifacts:` (or `mlflow:`)
URI. Inside `ntt`, the launcher also accepts `hf://` checkpoints and passes
this repo a local `.ckpt`.

## Decoding — multi-session with tiered eval

```bash
uv run python -m ntt_downstream.decoding.train \
    +data=ntt_reaching_strict \
    tokenizer=frozen \
    tokenizer.ntt_checkpoint=/path/to/tokenizer.ckpt \
    tokenizer.unit_embedding_mode=add
```

`+data=ntt_reaching_strict` sets `dataset` to a multi-session selection list
and turns on tier-1/2/3 evaluation. See [decoding.md](decoding.md#data--splits)
for tier semantics and the `${ntt_split:...}` resolver.

## Encoding — first run (Cowley V4 image baseline)

```bash
uv run python -m ntt_downstream.encoding.train
```

This launches the default config (`cowley_v4_base.yaml`): small Hiera trained
from scratch at 112×112, Poisson NLL loss, no tokenizer. Override the config
name with `--config-name`:

```bash
# Diffusion / flow / energy / MAR variants:
uv run python -m ntt_downstream.encoding.train --config-name cowley_v4_diffusion
uv run python -m ntt_downstream.encoding.train --config-name cowley_v4_full_grid_flow
uv run python -m ntt_downstream.encoding.train --config-name cowley_v4_perceiver_mse
```

The Perceiver (`cowley_v4_perceiver_*`), `cowley_v4_wanneural_vq` and
`cowley_v4_mar_energy` recipes load a trained tokenizer and require
`tokenizer.checkpoint_path=...`; the Perceiver recipes also require
`tokenizer.embeddings_path=...`. The other recipes build their tokenizer from
a spec and need no checkpoint.

Smoke-test a checkpoint-free recipe without MLflow (checkpoints and the
results record are still written to the Hydra output dir):

```bash
uv run python -m ntt_downstream.encoding.train \
    --config-name cowley_v4_full_grid_flow \
    trainer.max_steps=20 \
    trainer.limit_train_batches=2 \
    trainer.limit_val_batches=1 \
    trainer.logging=false
```

## Encoding — first run (Wingert auditory)

```bash
uv run python -m ntt_downstream.encoding.train --config-name wingert_base
```

Variants: `wingert_mse`, `wingert_diffusion`, `wingert_flow`, `wingert_energy`,
`wingert_cross_entropy`. The cross-entropy config requires a discrete-token
tokenizer checkpoint — see [encoding.md](encoding.md#wingert-auditory-cortex-experiments).

## Tests

```bash
uv run pytest tests/             # full suite
uv run pytest tests/ -m "not slow"
uv run pytest tests/decoding/    # decoding only
uv run pytest tests/encoding/    # encoding only
```

Tests that need a tokenizer checkpoint or processed data read
`NTT_TOKENIZER_CKPTS` (default `./tokenizer_ckpts/`) and `NTT_DOWNSTREAM_DATA`
(default `./data/processed/`), and skip when those are missing.
