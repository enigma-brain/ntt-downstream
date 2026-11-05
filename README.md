# ntt-downstream

Downstream evaluation of [`ntt`](https://github.com/enigma-brain/ntt) neural
tokenizers on two tasks:

- **Decoding**: predict cursor velocity from spikes via the POYO decoder
  (frozen ntt tokenizer, adapter, POYO bypass path).
- **Encoding**: predict neural responses from a stimulus (Cowley V4 image
  responses; Wingert auditory cortex), optionally with a frozen tokenizer in
  the loss.

Each subpackage has its own Hydra entry point and config tree. `ntt` drives
these evals through its `downstream_*` jobs (`ntt job=downstream_*`); this
repo also runs standalone.

## Install

```bash
uv sync
```

`uv` resolves `ntt` (the release pinned in `pyproject.toml`) and `torch_brain`
from source; `flash-attn==2.8.3` is locked to `torch==2.8.0+cu126`. When this
repo is used as the `downstream` submodule of `ntt`, the `ntt` launcher
overlays the host checkout's `ntt` onto the venv instead. See
[docs/getting-started.md](docs/getting-started.md#standalone-vs-embedded-in-ntt)
for both install modes, the dev-against-working-tree-`ntt` recipe, cache
redirects and `MLFLOW_TRACKING_URI` setup. MLflow is optional: encoding runs
without a tracking server when you pass `trainer.logging=false` (see
[docs/encoding.md](docs/encoding.md#running-without-mlflow)).

## Data

The examples use sessions of the public NTT dataset, built with
[brainsets](https://github.com/neuro-galaxy/brainsets). See the
[ntt documentation](https://github.com/enigma-brain/ntt) for how to obtain the
processed sessions. The encoding recipes use the `cowley_v4_compact_2024`
and `wingert_auditory_cortex_2026` brainsets of that dataset; point
`data.dataset_path` at `<processed>/cowley_v4_compact_2024` (or the Wingert
directory).

## Quick start

```bash
# Decoding: single-session smoke run. The fake tokenizer reproduces the
# original POYO computation through the bypass path.
uv run python -m ntt_downstream.decoding.train \
    tokenizer=fake \
    session=c_20150319_center_out_reaching \
    data_root=/path/to/processed \
    wandb.enabled=false

# Encoding: Cowley V4 image baseline (Hiera + Poisson NLL on V4 rates).
uv run python -m ntt_downstream.encoding.train trainer.logging=false
```

The same runs are available through the stable launcher, which `ntt` uses:

```bash
uv run ntt-downstream-decode --tokenizer-kind fake --data-root /path/to/processed --dry-run
```

The four console scripts are `ntt-downstream-decode`, `ntt-downstream-encode`,
`ntt-downstream-calibrate` and `ntt-downstream-benchmark`; see
[docs/cli.md](docs/cli.md).

## Documentation

| File | Topic |
|---|---|
| [docs/README.md](docs/README.md) | Index + 30-second mental model |
| [docs/getting-started.md](docs/getting-started.md) | Install, env vars, first runs |
| [docs/cli.md](docs/cli.md) | The eval CLI contract and how ntt integrates |
| [docs/decoding.md](docs/decoding.md) | POYO bypass pipeline, tokenizer kinds, all knobs |
| [docs/decoding_orchestration.md](docs/decoding_orchestration.md) | The sweep engine behind `calibrate` / `benchmark` |
| [docs/encoding.md](docs/encoding.md) | Cowley V4 + Wingert configs, loss modes, all knobs |
| [docs/results.md](docs/results.md) | The results record every run writes (schema v2), cohort records |
| [docs/environment.md](docs/environment.md) | Cache paths, MLflow / wandb / SLURM env vars |
| [docs/repo-organization.md](docs/repo-organization.md) | What lives in each directory; third-party code |

## Tests

```bash
uv run pytest tests/             # full suite
uv run pytest tests/ -m "not slow"
```

## License

The code is released under the [MIT License](LICENSE). Some modules adapt
third-party code and keep their original licence, as noted in each file's
header: torch_brain (Apache-2.0; `decoding/backends/poyo/{poyo,model,train_wrapper,multi_session_setup,flash_attn_varlen}.py`),
Hiera (Apache-2.0), MAR and the OpenAI diffusion code (MIT, with DiT-derived
helpers), EAR (MIT), open-retina (MIT) and LDNS (MIT). The full table is in
[docs/repo-organization.md](docs/repo-organization.md#third-party-code), and
the upstream licence texts with their copyright notices are in
[`LICENSES/`](LICENSES/).
`TimestepEmbedder`, `FinalLayer`, `modulate` and `create_diffusion` (in
`diffloss.py`, `energy_loss.py`, `ldns_denoiser.py` and `diffusion/__init__.py`)
originate in [DiT](https://github.com/facebookresearch/DiT) (CC BY-NC 4.0) and
are used here as redistributed by MAR under MIT.
Pretrained Hiera weights, used only by the optional `hiera_pretrained_*`
backbones, are licensed CC BY-NC 4.0. The tokenizer weights that `ntt`
publishes on Hugging Face are CC-BY-NC-4.0, and each dataset keeps the licence
of its upstream source (see the `ntt` documentation).
