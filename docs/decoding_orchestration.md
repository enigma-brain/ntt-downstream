# Decoding orchestration engine

`src/ntt_downstream/decoding/orchestration.py` is the engine behind the
decoding `calibrate` and `benchmark` sweeps. It owns the job queue, the GPU
worker pool, the per-cell `train.py` override lists, idempotent resume, the
MLflow parent-run lifecycle and the CSV / cohort aggregation. The SLURM
wrappers in `scripts/decoding/`, the `ntt-downstream-calibrate` /
`ntt-downstream-benchmark` launchers and `ntt job=downstream_*` are all thin
clients of it, so editing a `#SBATCH` line cannot change the pipeline.

> **To run sweeps, prefer the stable CLI contract** in [cli.md](cli.md). This
> page describes what the engine does behind it.

## Subcommands

```
python -m ntt_downstream.decoding.orchestration calibrate  [flags]
python -m ntt_downstream.decoding.orchestration benchmark  [flags]
```

- **`calibrate`**: trains a base model on one training session (or one pool of
  sessions with `--multi-session`) per seed, then fine-tunes or evaluates it on
  every calibration session at several trial budgets (`ft_trials`) and
  fine-tune modes (`unit_identification`, `full`). `ft_trials` is the only
  calibration sweep axis; a budget of `0` is a plain evaluation (`eval_only`).
- **`benchmark`**: trains one single-session run per (tokenizer, session) cell
  for a cohort of frozen tokenizers, optionally plus the spike-times baseline.

Flags:

```
calibrate  --training-config --data-config --tokenizer-kind {frozen,spike_times}
           --ckpt --tokenizer-name --train-sessions --other-sessions
           --ft-trial-counts --ft-modes --seeds --model-cfg --no-crossattn
           --unit-embedding-mode --sentinels --batch-size --num-workers
           --base-max-steps --ft-max-steps --num-latents-per-step
           --eval-every-n-steps --ckpt-every-n-steps --ft-eval-every
           --ft-ckpt-every --precision --force-flash-attn
           --temporal-split-enabled --brainset --calib-root --gpus
           --jobs-per-gpu --skip-base-train --base-ckpt
           --reuse-base-if-exists --missing-only --mlflow {auto,true,false}
           --mlflow-experiment-name --mlflow-log-model --extra "<raw hydra>"
           --repo-root --data-root --dry-run
           [--multi-session --hydra-data-config]

benchmark  (--ckpt-root | --tokenizers) --tokenizers-limit --sessions
           --data-config --model-cfg --unit-embedding-mode --sentinels
           --epochs --batch-size --seed --num-workers --comparison-tag
           --include-spike-times --spike-times-only --gpus --mlflow
           --mlflow-experiment-name --mlflow-log-model --extra --repo-root
           --data-root --dry-run
```

Unset `calibrate` flags fall back to the training and data YAMLs
(`configs/decoding/{single,multi}_session/{tokenizer_training,data}.yaml`),
which `config_loader.py` reads through a whitelist of dotted keys
(`TRAINING_KEYS`, `DATA_KEYS`). A missing whitelisted key is a hard error, so a
typo in a YAML fails at load. `benchmark` reads only its data YAML and applies
literal defaults inline (400 epochs, batch size 128, seed 42, 8 workers,
`model=poyo_1.3M_no_crossattn`). The checkpoint root defaults to
`<repo>/results/tokenizer_checkpoints`, laid out as
`<ckpt_root>/<name>/checkpoints/model.ckpt`; every subdirectory holding such a
file is a cohort member unless `--tokenizers` names them.

`--dry-run` resolves everything and prints the job queue plus each cell's exact
`train.py` override list without executing anything.

## Module layout

All modules live under `src/ntt_downstream/decoding/`.

| Module | Role |
|---|---|
| `orchestration.py` | The engine and its `argparse` CLI. |
| `config_loader.py` | YAML loading with the `extends:` chain, deep merge, and the `TRAINING_KEYS` / `DATA_KEYS` whitelists. |
| `calibration_csv.py` | Builds `calibration_<ft_mode>.csv` from the per-cell `run_result.json` files. |
| `cohort_results.py` | Merges a tokenizer's benchmark cells into one cohort record (see [results.md](results.md#cohort-records-decoding-benchmark)). |
| `mlflow_utils.py` | Pure MLflow helpers (run URIs, tag parsing, metric aggregation). |
| `utils/mlflow_parent.py` | Stateful MLflow helper with the subcommands `create`, `aggregate`, `log-artifacts` and `export-metrics`. |

The pure modules are imported in-process. The engine's building blocks are
module-level functions rather than classes:

- **Config resolution**: `_resolve_calibrate` turns the parsed flags plus the
  YAML defaults into a typed `CalibrateConfig`. A flag wins when it is not
  `None`; otherwise the YAML value is used. It fails early on inconsistent
  input (a frozen tokenizer without `--ckpt`, a frozen tokenizer with
  cross-attention, a multi-session pool smaller than two sessions, an empty GPU
  list).
- **GPU pool**: `--gpus` or `nvidia-smi` discovery. `--jobs-per-gpu` N creates
  N worker tags `gpu:slot` per device.
- **Job queue and workers**: `_run_pool` starts one thread per worker tag. The
  threads pull jobs from a shared in-process `queue.Queue`, so no file locking
  is involved and the queue works on any filesystem. Each job runs `train.py`
  as a subprocess with `CUDA_VISIBLE_DEVICES` set to the worker's GPU,
  `PYTHONHASHSEED` set to the cell's seed and `CUBLAS_WORKSPACE_CONFIG` set
  from the training YAML (an explicit environment value wins).
- **Train invocation**: `_base_argv`, `_ft_argv` and the benchmark cell build
  deterministic Hydra override lists (below). Cells run from the repo root so
  relative data and log paths resolve.
- **MLflow lifecycle**: `_mlflow_create_parent`, `_mlflow_aggregate` and
  `_mlflow_log_artifacts` call `utils/mlflow_parent.py` as subprocesses.
- **Results collection**: `calibration_csv.build_csv` per fine-tune mode, and
  `cohort_results` for benchmarks.

## The subprocess boundary

Two things are subprocesses by design:

- **`python -m ntt_downstream.decoding.train <overrides>` per cell.** `train.py`
  is `@hydra.main`-decorated (one composed run per process), has import-time
  side effects (the `ntt_split` resolver registration, matmul precision), and
  each cell needs its own process for `CUDA_VISIBLE_DEVICES` and
  Lightning / CUDA state. The engine never trains in-process.
- **`mlflow_parent create|aggregate|log-artifacts`.** The subprocess isolates
  the MLflow run context, so a flaky tracking server cannot take down the
  orchestrator loop. These calls are best-effort and degrade to no-ops when
  MLflow is disabled.

## Per-phase `train.py` overrides

**Calibrate, base train.** `model=<model_cfg> session=<train_session>` (or
`+data=<hydra_data_config> _train_sessions_list=[...]` in multi-session mode)
`data_root= log_dir=<outputs/<base_run>>`, the tokenizer overrides
(`tokenizer=<kind>`, `tokenizer.ntt_checkpoint=` for frozen,
`tokenizer.unit_embedding_mode=`, `tokenizer.sentinels=`),
`model.num_latents_per_step=` when the model uses cross-attention,
`temporal_split.enabled= max_steps=<base_max_steps> eval_every_n_steps=
checkpoint_every_n_steps= batch_size= seed= num_workers= precision=
force_flash_attn=`, `wandb.{name,group,tags}`,
`mlflow.{enabled,experiment_name,run_name,log_model,parent_run_id}` and
`+mlflow.tags.*` (`sweep_id`, `sweep_timestamp`, `tokenizer_name`,
`tokenizer_kind`, `stage=base`, `train_session` or `pool_size` / `pool_hash`,
`seed`). The base train sets no `init_ckpt`.

**Calibrate, fine-tune cell.** The same set for `session=<other_session>` with
`stage=ft`, plus `init_ckpt=<base ckpt> freeze_mode={none|unit_identification}
eval_only={true|false} train_decimation.trials=<N> train_decimation.seed=<seed>`
and the tags `ft_mode`, `ft_trials` and `calib_session`. `ft_mode=full` maps to
`freeze_mode=none`. A budget of `0` sets `eval_only=true`, `max_steps=1`,
`eval_every_n_steps=1` and `checkpoint_every_n_steps=1`. A budget larger than
`floor(0.7 * n_valid_trials)` of the target session is skipped; the valid-trial
count is read from the session HDF5 in a preflight step (a missing HDF5 is an
error, except under `--dry-run`).

**Benchmark cell.** `model= session= data_root= log_dir=<run dir>`, the
tokenizer overrides (`tokenizer={frozen|spike_times}`), `epochs= batch_size=
seed= num_workers=`, the fixed `precision=bf16-mixed force_flash_attn=true`,
and the same `wandb.*` / `mlflow.*` / `+mlflow.tags.*` families (plus
`session`). There is no `init_ckpt`, `freeze_mode` or `eval_only`.

## Queues, resume and idempotence

- **Queue contents.** Single-session `calibrate`: `train_sessions x seeds`.
  Multi-session `calibrate`: `seeds` only (one base per seed over the whole
  pool). `benchmark`: `(tokenizer, session)`. One calibrate job runs its base
  train and then all of its fine-tune cells on one GPU.
- **Output layout.** Calibrate base runs go to `outputs/<base_run>/`; fine-tune
  cells to `<calib_root>/<base_run>/<ft_mode>/<other_session>/ft_trials_NNN/`.
  Benchmark cells go to `experiments_benchmark/<comparison_tag>/<tag>__<tokenizer>__<session>/`.
- **Base reuse.** `--skip-base-train` fine-tunes from a supplied `--base-ckpt`
  (valid for a single train session; the checkpoint must be complete).
  `--reuse-base-if-exists` (default on) reuses the newest `last.ckpt` of the
  base run whose `global_step` is at least `base_max_steps`, so a partial base
  is never reused.
- **Cell resume.** `--missing-only` (default on) skips a fine-tune cell whose
  newest `run_result.json` is complete, meaning it contains `val_last_r2` and is
  not truncated. `run_result.json` is local bookkeeping; see
  [results.md](results.md#local-files).
- **Plain-eval mirroring.** The `ft_trials=0` evaluation does not depend on the
  fine-tune mode, so it is computed once per calibration session and copied into
  the other mode's subtree, which keeps the CSV layout symmetric.
- **CSV.** After the pool finishes, `calibration_<ft_mode>.csv` is written per
  fine-tune mode under each base run's experiment root.
- **Cohort records.** After a benchmark pool finishes, one cohort record per
  tokenizer is written next to the cells and uploaded to the tokenizer's MLflow
  parent. A cell counts only if its record was written after the sweep started.
- **MLflow parent.** Created once per calibrate sweep and once per tokenizer in
  a benchmark, tagged with a deterministic UTC timestamp and sweep id. At the
  end the engine aggregates the children's metrics onto it and archives each
  cell's results record, `metrics.csv` and the calibration CSVs under it.

## SLURM wrappers

A wrapper carries only the `#SBATCH` header, the environment setup and one
`exec`. For example:

```bash
#!/usr/bin/env bash
# EXAMPLE: adjust the #SBATCH resource lines to your cluster.
#SBATCH --job-name=calibration_singlesession
#SBATCH --partition=gpu --gres=gpu:4 --cpus-per-task=64 --mem=256G --time=24:00:00
#SBATCH --output=outputs/slurm/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; mkdir -p outputs/slurm
[[ -f .env ]] && { set -a; source .env; set +a; }
exec uv run python -m ntt_downstream.decoding.orchestration calibrate \
  --training-config "${TRAINING_CONFIG:-configs/decoding/single_session/tokenizer_training.yaml}" \
  --data-config     "${DATA_CONFIG:-configs/decoding/single_session/data.yaml}" \
  --ckpt "${CKPT:-}" --gpus "${GPUS:-}" --calib-root "${CALIB_ROOT:-experiments_final}" "$@"
```

The in-file partition fallback is `gpu`. On clusters with a different partition
or GRES, submit through `scripts/decoding/submit.sh`, which injects them as
`sbatch` command-line flags from `.env` (see [environment.md](environment.md)).
The scripts assume `uv sync` has already been run.

## Pulling results and computing tables

Every sweep dispatches many cells; each is a first-class run that can be found
and aggregated by facet through two layers.

- **MLflow.** Each cell is a child of the sweep's MLflow parent
  (`mlflow.parentRunId`) and is tagged with `seed`, `stage` (`base` / `ft`),
  `ft_mode`, `ft_trials`, `calib_session` (multi-session adds `pool_size` and
  `pool_hash`), `tokenizer_name`, `tokenizer_kind` and `sweep_id`. The parent
  carries the mean and median of every child metric. Pull a tidy per-cell table
  (tags and metrics) for a parent with:

  ```bash
  python -m ntt_downstream.decoding.utils.mlflow_parent export-metrics \
      --run-id <parent_run_id> --out metrics.csv
  ```

  The CSV has one row per cell (`run_id`, `run_name`, `status`, `tag.*`,
  `metric.*`). Load it in pandas and `groupby("tag.seed")` or `pivot(...)` to
  compute any table. MLflow can also be queried directly:
  `client.search_runs(filter_string="tags.\`mlflow.parentRunId\` = '<parent>'")`.

- **W&B.** Every cell of a sweep shares one `wandb.group`
  (`<tokenizer>[_multi]__<timestamp>` for calibrate) and carries the same facets
  as `wandb.tags` (`seed-...`, `ftmode-...`, `fttrials-...`, `calibsess-...`,
  `sess-...`). Pull them with `wandb.Api().runs(ENTITY/PROJECT,
  filters={"group": "<group>"})` and read each run's `.config`, `.summary` and
  `.tags`.
