# The eval CLI contract

`ntt-downstream` exposes a small, **versioned** command-line front door for
running evals. It is a thin Hydra launcher: it does not own modeling logic, it
maps a handful of stable flags onto the right Hydra keys and hands off to the
existing train apps and the decoding orchestration engine.

This is the **stable invocation contract** — the surface `ntt` (and any other
caller) should depend on. The deep-dive docs explain the internals behind it:

- [decoding.md](decoding.md) — POYO bypass pipeline, tokenizer kinds, all knobs.
- [encoding.md](encoding.md) — Cowley V4 + Wingert recipes, loss modes, all knobs.
- [decoding_orchestration.md](decoding_orchestration.md) — the sweep engine that
  `calibrate` / `benchmark` drive.

## Overview

The CLI has **four verbs**, each available as a console script (recommended)
and equivalently runnable as a module:

| verb | console script (stable contract) | drives |
|---|---|---|
| `decode` | `ntt-downstream-decode` | single decoding run (`ntt_downstream.decoding.train`) |
| `encode` | `ntt-downstream-encode` | single encoding run (`ntt_downstream.encoding.train`) |
| `calibrate` | `ntt-downstream-calibrate` | decoding **sweep** (orchestration engine) |
| `benchmark` | `ntt-downstream-benchmark` | decoding tokenizer-cohort **sweep** (orchestration engine) |

Use the **console scripts** as the stable entry. The equivalent module form is
`python -m ntt_downstream.cli.app <verb> …`, useful for local runs from a
source checkout.

All four verbs share the same flag surface (see [Promoted flags](#promoted-flags))
and the same escape hatches (`--config-name` / `--config-dir`, raw `key=value`
passthrough, `--dry-run`, `--print-resolved-config`).

## The four verbs

### `decode` / `encode` — single runs

Each verb defaults to its curated entrypoint config `configs/cli/<verb>.yaml`,
maps the promoted flags to that verb's Hydra keys, **appends raw passthrough
last**, and subprocesses the train app. The entrypoint config composes the
per-task tree and pins the contract defaults (e.g. `decode` pins
`model: poyo_1.3M_no_crossattn` and `tokenizer.sentinels=false`).

```console
$ ntt-downstream-decode --tokenizer-kind frozen --tokenizer-ckpt mlflow-artifacts:/m/x \
    --data-root /data --output-root /out --dry-run
.venv/bin/python3 -m ntt_downstream.decoding.train \
  --config-dir <repo>/configs/cli --config-name decode \
  tokenizer=frozen tokenizer.ntt_checkpoint=mlflow-artifacts:/m/x data_root=/data log_dir=/out
```

```console
$ ntt-downstream-encode --tokenizer-ckpt /m/x.ckpt --data-root /data --seed 7 --dry-run
.venv/bin/python3 -m ntt_downstream.encoding.train \
  --config-dir <repo>/configs/cli --config-name encode \
  ++tokenizer.checkpoint_path=/m/x.ckpt ++tokenizer.enabled=true data.dataset_path=/data seed=7
```

(The `--config-dir` and the interpreter path (`.venv/bin/python3`) are shown
abbreviated; the launcher emits absolute paths for both.)

### `calibrate` / `benchmark` — sweeps

Each loads `configs/cli/<verb>.yaml` (the source of truth for the grid + engine
knobs), overlays the promoted flags, and drives the **existing** decoding
orchestration engine. The sweep config carries
the cell grid and engine parameters. With `--dry-run` the engine prints its plan
instead of launching cells:

```console
$ ntt-downstream-calibrate --tokenizer-ckpt mlflow-artifacts:/m/tokA --tokenizer-name tokA --dry-run
WARNING ntt_downstream.decoding.orchestration: preflight: …/c_20131217_random_target_reaching.h5 missing — assuming no trial clamp (dry-run)
… (one preflight line per session in the sweep grid)
INFO ntt_downstream.decoding.orchestration: calibrate kind=frozen tokenizer=tokA model=poyo_1.3M_no_crossattn sweep=tokA__<ts> parent=(none)
INFO ntt_downstream.decoding.orchestration: queue: 4 (train_session, seed) jobs on workers ['0:0', '1:0', …]
INFO ntt_downstream.decoding.orchestration: [dry-run] cell (GPU 0) -> experiments_final/single_tokA_…/ft_trials_004/run.log
    <repo>/.venv/bin/python3 -m ntt_downstream.decoding.train model=poyo_1.3M_no_crossattn session=t_20130822_random_target_reaching … tokenizer=frozen tokenizer.ntt_checkpoint=mlflow-artifacts:/m/tokA … seed=43 train_decimation.trials=4 …
… (one [dry-run] cell + train.py command per cell in the grid; no cells are launched)
```

(`…` / `<repo>` abbreviate long paths and override lists; the launcher emits the
full, unabbreviated commands.)

## Promoted flags

The promoted flags are the **stable, versioned host-boundary surface**. The flag
*names* are stable across releases; the internal Hydra keys they map to are not,
and differ per verb. Any flag left unset is skipped (no override emitted).

| promoted flag | decode key | encode key |
|---|---|---|
| `--tokenizer-kind` | `tokenizer=<group>` (config-group select) | n/a — encode keys off loss/ckpt, not a tokenizer group |
| `--tokenizer-ckpt` | `tokenizer.ntt_checkpoint=<v>` | `tokenizer.checkpoint_path=<v>` + `tokenizer.enabled=true` |
| `--tokenizer-name` | sweep run-naming / MLflow tags only — **no-op for single-run** | same (tags) |
| `--data-root` | `data_root=<v>` | `data.dataset_path=<v>` |
| `--output-root` | `log_dir=<v>` | `hydra.run.dir=<v>` |
| `--seed` | `seed=<v>` | `seed=<v>` (top-level `cfg.seed`) |

`--tokenizer-name` is **a no-op for single-run** `decode` / `encode`: it maps to
no Hydra key there and a stray `--tokenizer-name` is silently ignored. It only
matters for the sweep verbs (run naming / MLflow tags).

Tokenizer checkpoint values **pass through unchanged**. A checkpoint is a local
`.ckpt` path or an `mlflow-artifacts:/` URI; the launcher does not pre-check
existence, the train app resolves it. `hf://` checkpoints are materialised to a
local `.ckpt` by ntt's launcher before it calls these verbs.

### Promoted overlay for sweeps

The sweep verbs map the promoted flags onto the orchestration engine's fields:

- **`calibrate`**: tokenizer kind / ckpt / name → its sweep tokenizer fields;
  `--output-root` → `calib_root`; `--seed` → a one-element seeds list; raw
  passthrough → the engine's `extra` (appended to every per-cell `train.py`
  call). Ckpt URIs pass through verbatim.
- **`benchmark`**: maps `--data-root` and `--seed`, and `--tokenizer-name` → a
  one-element `tokenizers` cohort filter. `--tokenizer-ckpt` and
  `--tokenizer-kind` are **intentionally ignored** for `benchmark`: it discovers
  a cohort of N frozen tokenizers from a ckpt root, and a single ckpt/kind cannot
  parameterize that cohort.

## Config selection and escape hatches

- **Default entrypoint** — `configs/cli/<verb>.yaml`, the curated contract config.
  It composes the per-task config tree and pins the defaults as data, not code.
- **`--config-name X` / `--config-dir D`** — point the launcher at a *different*
  config name or search-path dir. This is a first-class escape hatch: `ntt` may
  point at its **own external Hydra config / dir** here. Note the semantics
  differ by verb family: for **single runs** `--config-dir` is forwarded to
  Hydra, which *adds* the dir to the search path (and the default
  `--config-name` only exists in `configs/cli/`, so pass both flags together);
  for **sweeps** the launcher reads `<config-dir>/<config-name>.yaml` directly,
  *replacing* the default location.
- **Raw `key=value` passthrough** — any trailing args go straight to Hydra,
  appended **after** the promoted-flag overrides, so a power user can override
  even a promoted-derived value (Hydra last-wins):

```console
$ ntt-downstream-decode --tokenizer-kind fake --dry-run batch_size=8
.venv/bin/python3 -m ntt_downstream.decoding.train \
  --config-dir <repo>/configs/cli --config-name decode \
  tokenizer=fake batch_size=8
```

## `--dry-run` and `--print-resolved-config`

Two GPU-free preview modes, **mutually exclusive**:

- **`--dry-run`** — prints the assembled launch command and exits. No subprocess,
  no training. For the sweep verbs it runs the engine's own dry-run, which prints
  the sweep plan and launches no cells. This is the intended **contract test**.
- **`--print-resolved-config`** — prints the resolved config and exits. No
  training. For `decode` / `encode` it composes the full Hydra config (via
  `--cfg job`); for `calibrate` / `benchmark` it prints the overlaid sweep
  config (sweep YAML + promoted flags + passthrough) without touching the
  engine. Useful to inspect exactly what the entrypoint config + promoted
  flags + passthrough resolve to:

```console
$ ntt-downstream-decode --print-resolved-config | head
session: c_20150319_center_out_reaching
data_root: ./data/processed/
log_dir: ./outputs/poyo/
epochs: 1000
batch_size: 128
eval_epochs: 4
seed: 42
max_steps: null
eval_every_n_steps: 200
checkpoint_every_n_steps: 200
```

## Versioning and sweep-config validation

The contract is versioned by the package release: the promoted flag *names*,
the four verbs, and the preview flags stay stable between releases, and `ntt`
pins its `downstream` submodule to the release tag. There is no separate
version marker inside the configs; they travel with the pinned checkout.

External **sweep** configs are validated structurally instead: every top-level
key must be a real engine parameter (argparse dest), so a typo'd or stale key
fails loudly rather than being silently ignored:

```console
$ ntt-downstream-calibrate --config-dir /some/dir --config-name calibrate --dry-run
error: unknown calibrate sweep config key 'jobs_per_gpus'
# exit code: 2
```

## For ntt integrators

`ntt` (and any other caller) drives this package through the contract verbs:

```
uv run --project <downstream> --no-sync ntt-downstream-<verb> \
  <promoted flags> [--config-name X | --config-dir D] [key=val ...]
```

The caller's side of the contract:

- **Invoke the verbs, not the internals.** The launcher and train apps own the
  mapping from promoted flags to Hydra keys, so a caller does not shell into
  `python -m ntt_downstream.decoding.train`,
  `python -m ntt_downstream.encoding.train` or
  `python -m ntt_downstream.decoding.orchestration` directly.
- **Do not mirror the launcher's internals.** Default-architecture knowledge
  (the entrypoint config pins the architecture), the set of tokenizer-free
  losses, and the `<root>/<name>/checkpoints/model.ckpt` layout that
  `benchmark` discovers belong to this package.
- **Pass checkpoint values through verbatim.** The launcher and train app
  resolve local paths and `mlflow-artifacts:/` URIs, so a caller does not
  pre-check them. `hf://` checkpoints are the one exception: the caller
  materialises them to a local `.ckpt` first.
- **Own the environment and overlay its own `ntt`.** Run the verbs with
  `uv run --project <downstream> --no-sync ...`, never `--frozen`, which
  reverts the overlay. See
  [getting-started.md § Standalone vs. embedded](getting-started.md#standalone-vs-embedded-in-ntt).
- **Use `--dry-run` as a GPU-free contract test.** It composes the exact launch
  command (or sweep plan) without touching a GPU.
