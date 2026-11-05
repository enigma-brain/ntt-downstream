# Results records (schema v2)

Every downstream run writes **one** machine-readable record,
`downstream_{encoding,decoding}_results.json`. It's built by
`src/ntt_downstream/common/results/` and uploaded to MLflow / W&B. Consumers,
in particular ntt's aggregation, read only these records. Per-run CSVs and
`run_result.json` are local reports and bookkeeping, not contracts.

A run that cannot write its record fails (non-zero exit) instead of finishing
without it. Both tasks resolve the full config at startup, because the record
stores it, so a broken interpolation fails before training. Uploading the record
to MLflow / W&B stays best-effort: a failed upload is logged and skipped.

## Where it lands

| Producer | Location | `kind` |
|---|---|---|
| Encoding run | the Hydra run dir | `run` |
| Decoding run | `<log_dir>/csv_logs/version_N/`, next to `run_result.json` | `run` |
| Decoding benchmark, one per tokenizer | `experiments_benchmark/<tag>/<tag>__<tokenizer>__cohort/`, also uploaded to the tokenizer's MLflow parent run at the **artifact root** | `cohort` |

The decoding run record stays in the Lightning version dir because decoding sets
no `hydra.run.dir`: its Hydra dir is the shared default `outputs/<date>/<time>`.
The version dir is also the tree the benchmark sweep walks and uploads.

## Layout

```jsonc
{
  "schema_version": 2,
  "task": "encoding" | "decoding",
  "kind": "run" | "cohort",
  "metadata": { ... },
  "metrics": { "<block>": { "aggregated": {...}, "per_session": {...}, ...groups } }
}
```

**Blocks mean the same thing in both tasks:**

| Block | Meaning | Encoding source | Decoding source |
|---|---|---|---|
| `test` | **Authoritative**: the val-selected checkpoint evaluated on the held-out test split | final-eval test table (`metadata.final_eval.ckpt_kind` says `best` or the `last` fallback) | `trainer.test(ckpt="best")`; `"last"` when the run has no validation, the loaded checkpoint for `eval_only` |
| `val_best` | periodic val at the selected checkpoint (diagnostic) | metric snapshot | metric snapshot |
| `val_last` | periodic val at the end of training (diagnostic) | metric snapshot | metric snapshot |
| `val_final` | final-eval val table (encoding only) | final-eval val table | — |

**A missing `test` means "no test score"**, for example when encoding final eval
failed or was disabled. Never substitute a val block. Wingert (audio) has no
final eval, so its records have no `test`.

**Metric names are native to each task.** Encoding uses `nc_r2`,
`rate_pearson`, `temp_pearson` and their `*_psth` variants (final-eval blocks),
or the periodic-val names (`rate_corr`, …). Decoding uses `r2`. The consumer
chooses which metric to report.

**`aggregated.<metric>`:**
- `value`, `direction`, `value_range`, and optionally `median`, `std`, `n`;
- `aggregated_over`, which says what `std` is:
  - `"sessions:mean_of_means"` (encoding final eval): `std` is a **standard
    error** over sessions;
  - `"sessions:mean"` (decoding): `std` is the across-session sample std.

**`metadata`:**
- the run name, MLflow run id, git sha, timestamp, host, resolved config,
  checkpoint info and sessions;
- `tokenizer_checkpoint`: the evaluated tokenizer (`null` for tokenizer-free
  runs);
- `ntt_version` / `ntt_path` / `ntt_commit`: which ntt the run imported.
  `ntt_path` shows whether a run used ntt's parent checkout or the pinned tag
  (see [getting-started.md § Standalone vs. embedded](getting-started.md#standalone-vs-embedded-in-ntt)).
  `ntt_commit` shows which code: from the install's `direct_url.json` for a
  git install, or `git rev-parse HEAD` of an editable checkout, whose version
  string doesn't change between commits;
- encoding only: `final_eval: {"status": "ok"|"failed"|"disabled", "error"?: str}`.

## Cohort records (decoding benchmark)

The benchmark trains one single-session run per (tokenizer, session) cell. After
the pool finishes, it merges each tokenizer's cells into one `kind: "cohort"`
record (`src/ntt_downstream/decoding/cohort_results.py`):

- **`metrics.{test,val_best,val_last}`:**
  - `per_session[<session>].r2` = that cell's value; `null` if non-finite;
  - `aggregated.r2` = mean / median / std / n over the finite values.
  - Every block is always present; `aggregated` is empty when no cell has a value.
- **`metadata.cohort`:**
  - `expected`, `found` and `missing` session lists, and `complete`;
  - `stale` (cells whose newest record predates the sweep) and `not_before`;
  - `children` (per session: run name / id, git sha, ntt version, record path);
  - `mixed_provenance`: true if the found cells disagree on git sha, ntt version
    or ntt commit.

A cell counts as *found* only if the newest `version_N` of its run dir holds a
parseable record with a `test` block **written after the sweep started**
(`metadata.cohort.not_before`). A crashed rerun therefore makes a cell
**missing**; it never falls back to an older attempt. That holds even when the
rerun dies before Lightning creates a new version dir; such cells are also
listed under `stale`. The record is built from
local dirs, so offline / `--mlflow false` sweeps get it too. Building it never
fails the sweep.

## Local files

- **`run_result.json` (decoding):** a local record read by the sweep machinery
  (completion checks and `--missing-only` resume, the calibrate plain-eval
  mirror, the calibration CSV, `scripts/decoding/scratch_scaling_benchmark.sh`).
  It is **not uploaded** and is not a contract.
- **`final_eval.csv` / `final_eval_test.csv` (encoding):** human-readable
  reports, not uploaded.

## Fixtures

`tests/fixtures/results_v2/` holds records produced by the real writers, for
consumers' tests. Regenerate them after a writer change with
`uv run python scripts/fixtures/make_results_v2_fixtures.py`. A test fails while
they are stale.
