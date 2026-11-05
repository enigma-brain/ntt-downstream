#!/usr/bin/env bash
#SBATCH --job-name=scratch_scaling_benchmark
#SBATCH --partition=gpu   # neutral fallback; override per-site via scripts/decoding/submit.sh (PARTITION/GRES in .env)
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err
#
# EXAMPLE: adjust the #SBATCH resource lines to your cluster.
#
# POYO from-scratch trial-budget *scaling* benchmark.
#
# A self-contained bash orchestrator (GPU-queue runner included), in the same
# style as `tokenizer_benchmark.sbatch`.
#
# What "scratch scaling" means here:
#   * scratch  — POYO is trained *from scratch* per (session, seed): no
#                `init_ckpt`, `freeze_mode=none`. The NTT tokenizer is still
#                frozen (loaded from a checkpoint); only the POYO decoder is
#                randomly initialised and trained.
#   * scaling  — each session is run at several train-trial budgets, expressed
#                as fractions of that session's eligible train trials (e.g.
#                half / all). `train_decimation.trials` is the knob; the
#                baked HDF5 valid/test split is left intact, so scores stay
#                comparable across budgets.
#
# A "cell" is one POYO run = (condition x session x scale x seed). Cells are
# scheduled across all visible GPUs, one concurrent run per GPU; when a run
# finishes the worker pulls the next cell off a shared queue. Re-running the
# orchestrator skips cells that completed successfully (marked by a `.complete`
# sentinel written only on rc==0 + a real run_result.json); killed/incomplete
# cells are retried (resume).
#
# Outputs land under
#   experiments_benchmark/<COMPARISON_TAG>/<condition>/<scale>/<session>/seed<seed>/
# and a wide summary is written to
#   experiments_benchmark/<COMPARISON_TAG>/scratch_scaling.csv
# with columns: condition,kind,session,scale,seed,requested_trials,
#               kept_trials,val_best_r2,test_r2,run_dir
#
# Grouping (mirrors tokenizer_benchmark.sbatch): one wandb.group and one MLflow
# parent run *per condition* (suffixed with a sweep timestamp so re-runs do
# not silently merge). The cohort-wide COMPARISON_TAG survives as a tag
# (wandb `cmp-<tag>`, MLflow `sweep_id`). MLflow parents get mean/median of
# every child metric logged at the end.
#
# Examples:
#   # one tokenizer, with zero-unit padding tokens:
#   DATA_ROOT=/path/to/data/processed \
#   TOKENIZERS="my_tokenizer" \
#   ZERO_UNIT_TOKENS=true \
#     sbatch scripts/decoding/scratch_scaling_benchmark.sh
#
#   # every tokenizer in results/tokenizer_checkpoints, 2 seeds, half+all:
#   DATA_ROOT=/path/to/data/processed \
#     bash scripts/decoding/scratch_scaling_benchmark.sh
#
#   # spike-times from-scratch baseline only:
#   DATA_ROOT=... TOKENIZERS="" INCLUDE_SPIKE_TIMES=true \
#     bash scripts/decoding/scratch_scaling_benchmark.sh
#
#   DRY_RUN=1 DATA_ROOT=... bash scripts/decoding/scratch_scaling_benchmark.sh
#
#   # non-default cluster? submit via the wrapper (injects --partition/--gres from .env):
#   PARTITION=gpu-big DATA_ROOT=... \
#       scripts/decoding/submit.sh scripts/decoding/scratch_scaling_benchmark.sh
#
# Env knobs:
#   COMPARISON_TAG    experiments_benchmark/<tag>/ subdir + cmp-<tag>
#                     wandb/MLflow tag. (default: scratch_scaling_<UTC date>)
#   DATA_CONFIG       YAML providing `data_root`, `brainset`, `other_sessions`
#                     (default: configs/decoding/single_session/data.yaml)
#   SESSIONS          space-separated session ids; overrides the YAML list
#   DATA_ROOT         dataset root; overrides $DATA_CONFIG.data_root. The YAML
#                     default (./data/processed) is a placeholder; pass the
#                     root of your processed data.
#   BRAINSET          brainset dir under DATA_ROOT (default: from YAML, else
#                     perich_miller_population_2018)
#   CKPT_ROOT         dir holding <name>/<CKPT_FILE> tokenizer checkpoints
#                     (default: results/tokenizer_checkpoints)
#   CKPT_FILE         checkpoint path relative to <CKPT_ROOT>/<name>/
#                     (default: checkpoints/model.ckpt)
#   TOKENIZERS        explicit space-separated tokenizer names; overrides
#                     auto-discovery from $CKPT_ROOT. Empty string disables
#                     frozen conditions (use with INCLUDE_SPIKE_TIMES=true).
#   INCLUDE_SPIKE_TIMES  true|false. true -> also run a spike_times-from-scratch
#                     condition. (default: false)
#   SCALES            space-separated <name>:<fraction> pairs; the train-trial
#                     budget for a session is max(1, int(n_eligible*fraction)).
#                     (default: "scratch_half:0.5 scratch_all:1.0")
#   SEEDS             space-separated run seeds (default: "42 43")
#   MIN_TRIAL_DURATION  trials shorter than this (s) are not counted as
#                     eligible (default: 1.0)
#   ZERO_UNIT_TOKENS  tokenizer.include_padded_zero_unit_tokens for frozen
#                     conditions (default: false)
#   GPUS              space-separated GPU indices the pool may use
#                     (default: every index reported by nvidia-smi)
#   MODEL_CFG         (default: poyo_1.3M_no_crossattn)
#   UE_MODE           tokenizer.unit_embedding_mode (default: add)
#   SENTINELS         tokenizer.sentinels for frozen conditions (default:
#                     false). spike_times keeps its config default (true).
#   MAX_STEPS         step-based training budget per cell (default: 3400).
#                     Keep well above ~40 — OneCycleLR warmup is
#                     warmup_pct*MAX_STEPS-1 and must stay positive.
#   EVAL_EVERY_N_STEPS / CHECKPOINT_EVERY_N_STEPS  (default: 200 / 200)
#   BATCH_SIZE        (default: 128)
#   NUM_WORKERS       data-loader workers per run (default: 8)
#   WANDB_ENABLED     true|false (default: true)
#   MLFLOW_ENABLED    auto|true|false (default: auto = on iff MLFLOW_TRACKING_URI
#                     is set; true without a URI exits before any cell).
#                     MLFLOW_TRACKING_URI / _USERNAME / _PASSWORD must
#                     already be in env / .env.
#   MLFLOW_EXPERIMENT_NAME  (default: ntt-eval-poyo)
#   MLFLOW_LOG_MODEL  true|false (default: false)
#   EXTRA             extra args appended verbatim to every train.py call
#   DRY_RUN           1 -> print plan and exit without launching workers

set -euo pipefail

# Under sbatch, BASH_SOURCE[0] points into /var/spool/slurmd/... (a copy of
# the script), so dirname-based resolution would land outside the repo.
# Prefer $SLURM_SUBMIT_DIR when SLURM gives it to us.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$REPO_ROOT"

# Auto-load repo .env (MLflow credentials etc.) so every uv-spawned worker
# inherits them. `set -a` exports anything the file assigns.
if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
fi

mkdir -p outputs/slurm

# --- Data config: read data_root / brainset / sessions from YAML ---
DATA_CONFIG="${DATA_CONFIG:-configs/decoding/single_session/data.yaml}"
[[ -f "$DATA_CONFIG" ]] || { echo "ERROR: DATA_CONFIG not found: $DATA_CONFIG" >&2; exit 1; }

# One Python invocation emits shell-eval-safe assignments for the three
# fields we need; keeps the bash side free of YAML parsing.
eval "$(uv run python - "$DATA_CONFIG" <<'PY'
import shlex, sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
print(f"_YAML_DATA_ROOT={shlex.quote(str(cfg.get('data_root', '')))}")
print(f"_YAML_BRAINSET={shlex.quote(str(cfg.get('brainset', 'perich_miller_population_2018')))}")
sess = cfg.get("other_sessions", []) or []
items = " ".join(shlex.quote(str(s)) for s in sess)
print(f"_YAML_SESSIONS=({items})")
PY
)"

DATA_ROOT="${DATA_ROOT:-$_YAML_DATA_ROOT}"
BRAINSET="${BRAINSET:-$_YAML_BRAINSET}"
if [[ -z "$DATA_ROOT" ]]; then
    echo "ERROR: DATA_ROOT unresolved; pass DATA_ROOT=... explicitly" >&2
    exit 1
fi
if [[ -n "${SESSIONS:-}" ]]; then
    read -ra SESSIONS_ARR <<< "$SESSIONS"
else
    SESSIONS_ARR=("${_YAML_SESSIONS[@]}")
fi
if (( ${#SESSIONS_ARR[@]} == 0 )); then
    echo "ERROR: no sessions resolved (DATA_CONFIG=$DATA_CONFIG)" >&2
    exit 1
fi

# --- Configuration ---
COMPARISON_TAG="${COMPARISON_TAG:-scratch_scaling_$(date -u +%Y%m%d)}"
CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/results/tokenizer_checkpoints}"
CKPT_FILE="${CKPT_FILE:-checkpoints/model.ckpt}"
INCLUDE_SPIKE_TIMES="${INCLUDE_SPIKE_TIMES:-false}"
MIN_TRIAL_DURATION="${MIN_TRIAL_DURATION:-1.0}"
ZERO_UNIT_TOKENS="${ZERO_UNIT_TOKENS:-false}"
DRY_RUN="${DRY_RUN:-0}"

MODEL_CFG="${MODEL_CFG:-poyo_1.3M_no_crossattn}"
UE_MODE="${UE_MODE:-add}"
SENTINELS="${SENTINELS:-false}"
MAX_STEPS="${MAX_STEPS:-3400}"
EVAL_EVERY_N_STEPS="${EVAL_EVERY_N_STEPS:-200}"
CHECKPOINT_EVERY_N_STEPS="${CHECKPOINT_EVERY_N_STEPS:-200}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"

WANDB_ENABLED="${WANDB_ENABLED:-true}"
MLFLOW_ENABLED="${MLFLOW_ENABLED:-auto}"
# auto: log to MLflow only when a tracking server is configured, so the sweep
# runs without one (train.py rejects mlflow.enabled=true with no URI).
_mlflow_uri="${MLFLOW_TRACKING_URI:-}"
if [[ "${MLFLOW_ENABLED,,}" == auto ]]; then
    if [[ -n "${_mlflow_uri// /}" ]]; then MLFLOW_ENABLED=true; else MLFLOW_ENABLED=false; fi
fi
# Explicit true with no server: fail now rather than have every cell die in
# train.py after the sweep has started.
case "$MLFLOW_ENABLED" in
    true|True|TRUE|1)
        if [[ -z "${_mlflow_uri// /}" ]]; then
            echo "MLFLOW_ENABLED=$MLFLOW_ENABLED but MLFLOW_TRACKING_URI is unset. Set it, or use MLFLOW_ENABLED=auto (the default) / false to run without MLflow." >&2
            exit 1
        fi
        ;;
esac
MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-ntt-eval-poyo}"
MLFLOW_LOG_MODEL="${MLFLOW_LOG_MODEL:-false}"

SEEDS_RAW="${SEEDS:-42 43}"
read -ra SEEDS_ARR <<< "$SEEDS_RAW"

# Scales as <name>:<fraction> pairs. Parsed into two parallel arrays.
SCALES_RAW="${SCALES:-scratch_half:0.5 scratch_all:1.0}"
SCALE_NAMES=()
SCALE_FRACS=()
for _spec in $SCALES_RAW; do
    if [[ "$_spec" != *:* ]]; then
        echo "ERROR: SCALES entry '$_spec' is not <name>:<fraction>" >&2
        exit 1
    fi
    SCALE_NAMES+=("${_spec%%:*}")
    SCALE_FRACS+=("${_spec##*:}")
done
(( ${#SCALE_NAMES[@]} > 0 )) || { echo "ERROR: empty SCALES" >&2; exit 1; }

# Default GPU pool: every index reported by nvidia-smi. Override with $GPUS.
if [[ -z "${GPUS:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader \
                | tr '\n' ' ' | sed 's/ *$//')"
    fi
    GPUS="${GPUS:-0}"
fi
read -ra GPU_ARR <<< "$GPUS"

# Determinism: cuBLAS needs CUBLAS_WORKSPACE_CONFIG once
# torch.use_deterministic_algorithms(True) is on. PYTHONHASHSEED is set
# per-cell (it depends on the cell's seed) on the train.py invocation.
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

EXP_ROOT="${REPO_ROOT}/experiments_benchmark/${COMPARISON_TAG}"
LOG_ROOT="${EXP_ROOT}/_orchestrator_logs"
mkdir -p "$EXP_ROOT" "$LOG_ROOT"

# Pin a UTC timestamp at orchestrator start; every wandb group and MLflow
# parent created in this invocation gets it as a suffix.
SWEEP_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

# W&B rejects group/name over 128 chars and tags over 64. Truncate with a
# deterministic checksum suffix when needed — on-disk run dirs stay unchanged.
short_wandb_id() {
    local raw="$1" limit="${2:-120}"
    if (( ${#raw} <= limit )); then echo "$raw"; return 0; fi
    local hash
    hash=$(printf '%s' "$raw" | cksum | awk '{print $1}')
    echo "${raw:0:$(( limit - ${#hash} - 1 ))}_${hash}"
}

# --- Condition discovery ---
# A "condition" is a tokenizer. Frozen conditions come from $CKPT_ROOT (or an
# explicit $TOKENIZERS list); the optional spike_times condition has no ckpt.
# Each entry: NAME|KIND|CKPT_OR_EMPTY  with KIND in {frozen, spike_times}.
CONDITIONS_LIST=()

case "$INCLUDE_SPIKE_TIMES" in
    true|True|TRUE|1)  CONDITIONS_LIST+=("spike_times|spike_times|") ;;
    false|False|FALSE|0) ;;
    *) echo "ERROR: INCLUDE_SPIKE_TIMES must be true|false, got: $INCLUDE_SPIKE_TIMES" >&2; exit 1 ;;
esac

# `TOKENIZERS` unset -> auto-discover; set-but-empty -> no frozen conditions.
if [[ -n "${TOKENIZERS+x}" ]]; then
    if [[ -n "${TOKENIZERS// }" ]]; then
        read -ra _RAW <<< "$TOKENIZERS"
        for name in "${_RAW[@]}"; do
            ckpt="${CKPT_ROOT}/${name}/${CKPT_FILE}"
            if [[ ! -f "$ckpt" ]]; then
                echo "ERROR: tokenizer ckpt missing for '$name': $ckpt" >&2
                exit 1
            fi
            CONDITIONS_LIST+=("${name}|frozen|${ckpt}")
        done
    fi
else
    if [[ ! -d "$CKPT_ROOT" ]]; then
        echo "ERROR: CKPT_ROOT does not exist: $CKPT_ROOT" >&2
        exit 1
    fi
    while IFS= read -r name; do
        ckpt="${CKPT_ROOT}/${name}/${CKPT_FILE}"
        if [[ -f "$ckpt" ]]; then
            CONDITIONS_LIST+=("${name}|frozen|${ckpt}")
        fi
    done < <(ls -1 "$CKPT_ROOT" | sort)
fi

if (( ${#CONDITIONS_LIST[@]} == 0 )); then
    echo "ERROR: no conditions resolved (no tokenizers discovered and" \
         "INCLUDE_SPIKE_TIMES=false)" >&2
    exit 1
fi

# --- Per-session trial budgets ---
# For each session, count eligible train trials from its HDF5 (valid trials
# whose span falls inside a baseline train-domain interval and clears the
# minimum-duration floor), then derive the budget for each scale fraction.
# The rule is the one `train_decimation` applies (a trial is eligible iff a
# single baseline interval covers it and it clears the minimum duration), so
# budgets match what can actually be selected at run time.
declare -A NALL BUDGET
_sessions_csv="$(IFS=,; echo "${SESSIONS_ARR[*]}")"
_scales_csv=""
for i in "${!SCALE_NAMES[@]}"; do
    _scales_csv+="${SCALE_NAMES[$i]}:${SCALE_FRACS[$i]}"
    (( i < ${#SCALE_NAMES[@]} - 1 )) && _scales_csv+=","
done

while IFS='|' read -r _tag _a _b _c; do
    case "$_tag" in
        ERR)    echo "ERROR: trial-budget probe failed for $_a: $_b" >&2; exit 1 ;;
        NALL)   NALL["$_a"]="$_b" ;;
        BUDGET) BUDGET["${_a}|${_b}"]="$_c" ;;
    esac
done < <(uv run python - "$DATA_ROOT" "$BRAINSET" "$MIN_TRIAL_DURATION" \
                          "$_sessions_csv" "$_scales_csv" <<'PY'
import sys
from pathlib import Path

import h5py
import numpy as np

data_root, brainset, min_dur = sys.argv[1], sys.argv[2], float(sys.argv[3])
sessions = [s for s in sys.argv[4].split(",") if s]
scales = []
for spec in sys.argv[5].split(","):
    name, frac = spec.rsplit(":", 1)
    scales.append((name, float(frac)))

for session in sessions:
    path = Path(data_root) / brainset / f"{session}.h5"
    if not path.exists():
        print(f"ERR|{session}|missing HDF5 {path}")
        continue
    try:
        with h5py.File(path, "r") as f:
            is_valid = np.asarray(f["trials/is_valid"]).astype(bool)
            starts = np.asarray(f["trials/start"], dtype=np.float64)[is_valid]
            ends = np.asarray(f["trials/end"], dtype=np.float64)[is_valid]
            bs = np.asarray(f["train_domain/start"], dtype=np.float64)
            be = np.asarray(f["train_domain/end"], dtype=np.float64)
    except Exception as exc:  # noqa: BLE001 - surfaced to the bash caller
        print(f"ERR|{session}|{exc}")
        continue
    if starts.shape[0] == 0 or bs.shape[0] == 0:
        n_all = 0
    else:
        # For each valid trial, find the train-domain interval that could
        # contain it (searchsorted on sorted interval starts) and keep the
        # trial only if it is fully inside that interval and long enough.
        order = np.argsort(bs, kind="stable")
        bs, be = bs[order], be[order]
        idx = np.clip(np.searchsorted(bs, starts, side="right") - 1,
                      0, bs.shape[0] - 1)
        eligible = ((bs[idx] <= starts) & (be[idx] >= ends)
                    & ((ends - starts) >= min_dur))
        n_all = int(eligible.sum())
    print(f"NALL|{session}|{n_all}")
    for name, frac in scales:
        budget = max(1, int(n_all * frac)) if n_all > 0 else 0
        print(f"BUDGET|{session}|{name}|{budget}")
PY
)

# --- Per-condition grouping handles (wandb.group + MLflow parent) ---
declare -A COND_PARENT_ID COND_PARENT_URI COND_WANDB_GROUP
COND_ORDER=()

create_parent_for_condition() {
    local cond_name="$1" cond_kind="$2"
    local group_name="${cond_name}__${SWEEP_TIMESTAMP}"
    COND_WANDB_GROUP[$cond_name]="$(short_wandb_id "$group_name" 120)"
    COND_ORDER+=("$cond_name")

    # MLFLOW_ENABLED=false or DRY_RUN=1 -> create with --disabled so the
    # helper prints empty RUN_ID/RUN_URI and avoids MLflow side effects.
    local disabled_flag=""
    case "${MLFLOW_ENABLED}:${DRY_RUN}" in
        true:0|True:0|TRUE:0|1:0) disabled_flag="" ;;
        *)                        disabled_flag="--disabled" ;;
    esac
    local create_out
    create_out=$(
        uv run python -m ntt_downstream.decoding.utils.mlflow_parent create \
            --experiment-name "$MLFLOW_EXPERIMENT_NAME" \
            --run-name "$group_name" \
            --tag "sweep_id=${COMPARISON_TAG}" \
            --tag "sweep_timestamp=${SWEEP_TIMESTAMP}" \
            --tag "sweep_kind=scratch_scaling" \
            --tag "tokenizer_name=${cond_name}" \
            --tag "tokenizer_kind=${cond_kind}" \
            --tag "model_cfg=${MODEL_CFG}" \
            $disabled_flag
    ) || {
        echo "WARN: _mlflow_parent.create failed for ${cond_name}; proceeding without a parent" >&2
        COND_PARENT_ID[$cond_name]=""
        COND_PARENT_URI[$cond_name]=""
        return 0
    }
    local parsed_run_id="" parsed_run_uri="" key value
    while IFS='=' read -r key value; do
        case "$key" in
            RUN_ID)  parsed_run_id="$value" ;;
            RUN_URI) parsed_run_uri="$value" ;;
        esac
    done <<< "$create_out"
    COND_PARENT_ID[$cond_name]="$parsed_run_id"
    COND_PARENT_URI[$cond_name]="$parsed_run_uri"
}

for entry in "${CONDITIONS_LIST[@]}"; do
    IFS='|' read -r _cn _ck _cc <<< "$entry"
    create_parent_for_condition "$_cn" "$_ck"
done

# --- Build job queue + cells manifest ---
# Job line: CONDITION|KIND|CKPT|SESSION|SCALE|N_TRIALS|SEED|PARENT_ID|WANDB_GROUP
# The cells manifest (TSV) is consumed by the summary builder at the end.
JOBS=()
CELLS_TSV="${LOG_ROOT}/cells.tsv"
: > "$CELLS_TSV"
SKIPPED_EMPTY=0

for entry in "${CONDITIONS_LIST[@]}"; do
    IFS='|' read -r cond kind ckpt <<< "$entry"
    parent_id="${COND_PARENT_ID[$cond]}"
    wandb_group="${COND_WANDB_GROUP[$cond]}"
    for session in "${SESSIONS_ARR[@]}"; do
        for i in "${!SCALE_NAMES[@]}"; do
            scale="${SCALE_NAMES[$i]}"
            n_trials="${BUDGET[${session}|${scale}]:-0}"
            # A session with no eligible trials yields a 0 budget — skip it
            # rather than launch a train_decimation run that cannot draw any.
            if (( n_trials < 1 )); then
                echo "WARN: skipping ${cond}/${scale}/${session} — 0 eligible trials" >&2
                SKIPPED_EMPTY=$((SKIPPED_EMPTY + 1))
                continue
            fi
            for seed in "${SEEDS_ARR[@]}"; do
                run_dir="${EXP_ROOT}/${cond}/${scale}/${session}/seed${seed}"
                JOBS+=("${cond}|${kind}|${ckpt}|${session}|${scale}|${n_trials}|${seed}|${parent_id}|${wandb_group}")
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "$cond" "$kind" "$scale" "$session" "$seed" "$n_trials" "$run_dir" \
                    >> "$CELLS_TSV"
            done
        done
    done
done

# --- Plan ---
echo "=== POYO scratch-scaling benchmark ==="
echo "=== COMPARISON_TAG=${COMPARISON_TAG}"
echo "=== EXP_ROOT=${EXP_ROOT}"
echo "=== DATA_CONFIG=${DATA_CONFIG}"
echo "=== DATA_ROOT=${DATA_ROOT}  BRAINSET=${BRAINSET}"
echo "=== sessions (${#SESSIONS_ARR[@]}): ${SESSIONS_ARR[*]}"
echo "=== scales: ${SCALES_RAW}"
echo "=== seeds: ${SEEDS_ARR[*]}"
echo "=== GPUs=${GPU_ARR[*]}  workers=${#GPU_ARR[@]}"
echo "=== model=${MODEL_CFG}  ue_mode=${UE_MODE}  sentinels=${SENTINELS}  zero_unit_tokens=${ZERO_UNIT_TOKENS}"
echo "=== max_steps=${MAX_STEPS}  batch=${BATCH_SIZE}  num_workers=${NUM_WORKERS}"
echo "=== conditions (${#CONDITIONS_LIST[@]}):"
i=1
for ent in "${CONDITIONS_LIST[@]}"; do
    IFS='|' read -r _cn _ck _cc <<< "$ent"
    printf '   %3d. %-44s kind=%s\n' "$i" "$_cn" "$_ck"
    i=$((i + 1))
done
echo "=== per-session eligible trials:"
for session in "${SESSIONS_ARR[@]}"; do
    printf '   %-44s n_eligible=%s\n' "$session" "${NALL[$session]:-?}"
done
echo "=== mlflow=${MLFLOW_ENABLED}  experiment=${MLFLOW_EXPERIMENT_NAME}  wandb=${WANDB_ENABLED}"
echo "=== sweep_timestamp=${SWEEP_TIMESTAMP}"
echo "=== per-condition grouping handles:"
for cond in "${COND_ORDER[@]}"; do
    printf '   %-44s wandb_group=%s\n' "$cond" "${COND_WANDB_GROUP[$cond]}"
    if [[ -n "${COND_PARENT_URI[$cond]}" ]]; then
        printf '   %-44s mlflow_parent=%s\n' "" "${COND_PARENT_URI[$cond]}"
    fi
done
(( SKIPPED_EMPTY > 0 )) && echo "=== skipped ${SKIPPED_EMPTY} empty-session cell group(s)"
echo "=== total cells queued: ${#JOBS[@]}"

if [[ "$DRY_RUN" != "0" ]]; then
    echo "=== DRY_RUN=1, exiting before launching workers ==="
    for j in "${JOBS[@]}"; do echo "    $j"; done
    exit 0
fi

if (( ${#JOBS[@]} == 0 )); then
    echo "ERROR: empty job queue" >&2
    exit 1
fi

# --- Shared queue with flock-protected pop ---
QUEUE="$(mktemp -t poyo_scratch_queue.XXXXXX)"
LOCK="${QUEUE}.lock"
: > "$LOCK"
printf '%s\n' "${JOBS[@]}" > "$QUEUE"
trap 'rm -f "$QUEUE" "$LOCK" "$QUEUE.tmp"' EXIT

# --- Worker: one per GPU, processes cells sequentially on its device ---
run_worker() {
    local gpu="$1"
    local worker_failed=0
    while :; do
        local entry=""
        # Atomically pop the head of the queue.
        {
            flock -x 9
            entry="$(head -n1 "$QUEUE" || true)"
            if [[ -n "$entry" ]]; then
                tail -n +2 "$QUEUE" > "$QUEUE.tmp"
                mv "$QUEUE.tmp" "$QUEUE"
            fi
        } 9>"$LOCK"

        [[ -z "$entry" ]] && break

        local cond kind ckpt session scale n_trials seed parent_id wandb_group
        IFS='|' read -r cond kind ckpt session scale n_trials seed parent_id wandb_group <<< "$entry"

        local run_dir="${EXP_ROOT}/${cond}/${scale}/${session}/seed${seed}"
        mkdir -p "$run_dir"
        local stdout_log="${run_dir}/run.log"

        # Resume: skip only cells that finished successfully, marked by the
        # .complete sentinel written below on rc==0. Mere existence of a
        # run_result.json is NOT enough — a killed run can leave a partial/stale
        # one, which would otherwise be skipped forever and reported as NaN.
        if [[ -f "${run_dir}/.complete" ]]; then
            echo "[$(date +%H:%M:%S)] [GPU ${gpu}] skip (done) ${cond}/${scale}/${session}/seed${seed}"
            continue
        fi

        # Re-running an incomplete cell: wipe any prior csv_logs so this attempt
        # starts clean. Otherwise a stale version_*/run_result.json from a killed
        # attempt could make the `.complete` check below (and the summary) key off
        # an old result even when THIS run wrote none (e.g. rc==0 with no metric).
        # `:?` guards against an empty run_dir ever expanding to `rm -rf /csv_logs`.
        rm -rf "${run_dir:?}/csv_logs"

        local run_name="${COMPARISON_TAG}__${cond}__${scale}__${session}__seed${seed}"
        local wandb_run_name
        wandb_run_name=$(short_wandb_id "$run_name" 120)

        echo "[$(date +%H:%M:%S)] [GPU ${gpu}] start ${kind}/${cond} ${scale}/${session} seed=${seed} trials=${n_trials} -> ${stdout_log}"

        # Per-kind tokenizer args. `frozen_full_window` feeds the frozen NTT
        # tokenizer the full POYO window and all session units.
        local -a tokenizer_args
        if [[ "$kind" == "frozen" ]]; then
            tokenizer_args=(
                "tokenizer=frozen_full_window"
                "tokenizer.ntt_checkpoint=${ckpt}"
                "tokenizer.unit_embedding_mode=${UE_MODE}"
                "tokenizer.sentinels=${SENTINELS}"
                "tokenizer.include_padded_zero_unit_tokens=${ZERO_UNIT_TOKENS}"
            )
        else
            # spike_times keeps its own config default (sentinels=true) — the
            # frozen SENTINELS knob deliberately does not apply here.
            tokenizer_args=(
                "tokenizer=spike_times"
                "tokenizer.unit_embedding_mode=${UE_MODE}"
            )
        fi

        # wandb rejects any single tag over 64 chars; condition / session
        # names can exceed that, so route every tag through short_wandb_id.
        local _wandb_tag_list=(
            "scratch"
            "cmp-${COMPARISON_TAG}"
            "sweep-${SWEEP_TIMESTAMP}"
            "tok-${cond}"
            "kind-${kind}"
            "sess-${session}"
            "scale-${scale}"
            "arch-${MODEL_CFG}"
            "seed-${seed}"
        )
        local _wandb_tags_trunc=()
        for _t in "${_wandb_tag_list[@]}"; do
            _wandb_tags_trunc+=("$(short_wandb_id "$_t" 64)")
        done
        local wandb_tags
        wandb_tags=$(IFS=, ; echo "${_wandb_tags_trunc[*]}")

        # parent_run_id is empty when the helper short-circuited (MLflow off
        # or unreachable) — pass null so train.py keeps a flat run layout.
        local mlflow_parent_arg
        if [[ -n "$parent_id" ]]; then
            mlflow_parent_arg="mlflow.parent_run_id=${parent_id}"
        else
            mlflow_parent_arg="mlflow.parent_run_id=null"
        fi

        set +e
        PYTHONHASHSEED="$seed" \
        CUDA_VISIBLE_DEVICES="$gpu" \
        uv run python -m ntt_downstream.decoding.train \
            model="$MODEL_CFG" \
            session="$session" \
            data_root="$DATA_ROOT" \
            log_dir="$run_dir" \
            "${tokenizer_args[@]}" \
            max_steps="$MAX_STEPS" \
            eval_every_n_steps="$EVAL_EVERY_N_STEPS" \
            checkpoint_every_n_steps="$CHECKPOINT_EVERY_N_STEPS" \
            batch_size="$BATCH_SIZE" \
            seed="$seed" \
            num_workers="$NUM_WORKERS" \
            precision=bf16-mixed \
            force_flash_attn=true \
            temporal_split.enabled=false \
            train_decimation.trials="$n_trials" \
            freeze_mode=none \
            eval_only=false \
            "wandb.enabled=${WANDB_ENABLED}" \
            wandb.name="$wandb_run_name" \
            wandb.group="$wandb_group" \
            "wandb.tags=[${wandb_tags}]" \
            "mlflow.enabled=${MLFLOW_ENABLED}" \
            mlflow.experiment_name="$MLFLOW_EXPERIMENT_NAME" \
            mlflow.run_name="$wandb_run_name" \
            "mlflow.log_model=${MLFLOW_LOG_MODEL}" \
            "$mlflow_parent_arg" \
            "+mlflow.tags.sweep_id=${COMPARISON_TAG}" \
            "+mlflow.tags.sweep_timestamp=${SWEEP_TIMESTAMP}" \
            "+mlflow.tags.sweep_kind=scratch_scaling" \
            "+mlflow.tags.tokenizer_name=${cond}" \
            "+mlflow.tags.tokenizer_kind=${kind}" \
            "+mlflow.tags.session=${session}" \
            "+mlflow.tags.scale=${scale}" \
            "+mlflow.tags.requested_trials=${n_trials}" \
            ${EXTRA:-} \
            >"$stdout_log" 2>&1
        local rc=$?
        set -e

        if (( rc == 0 )) && compgen -G "${run_dir}/csv_logs/version_*/run_result.json" >/dev/null; then
            # Mark done only when the run BOTH exited cleanly AND produced a
            # result. train.py can exit 0 without writing run_result.json (no
            # test/val metric -> "skipping results save"); such a cell must NOT
            # be marked complete, or resume would skip it forever as NaN.
            : > "${run_dir}/.complete"
            echo "[$(date +%H:%M:%S)] [GPU ${gpu}] OK   ${cond} ${scale}/${session} seed=${seed}"
        elif (( rc == 0 )); then
            # Clean exit but no result file — leave unmarked so a later resume retries.
            echo "[$(date +%H:%M:%S)] [GPU ${gpu}] OK(rc=0) but NO run_result.json — not marking complete; will retry on resume: ${cond} ${scale}/${session} seed=${seed}" >&2
        else
            worker_failed=1
            echo "[$(date +%H:%M:%S)] [GPU ${gpu}] FAIL ${cond} ${scale}/${session} seed=${seed} rc=${rc}" >&2
        fi
    done
    # Return non-zero if ANY cell failed so the parent's `wait` sets the overall
    # exit code (we still don't abort early — every cell gets attempted).
    echo "[$(date +%H:%M:%S)] [GPU ${gpu}] worker done (failed=${worker_failed})"
    return "$worker_failed"
}

echo "=== launching ${#GPU_ARR[@]} workers ==="
pids=()
for g in "${GPU_ARR[@]}"; do
    run_worker "$g" 2> >(tee -a "${LOG_ROOT}/gpu${g}.err.log" >&2) \
                  > >(tee -a "${LOG_ROOT}/gpu${g}.out.log") &
    pids+=($!)
done

# Wait on every worker; don't abort on a single failure.
set +e
fail=0
for p in "${pids[@]}"; do
    if ! wait "$p"; then
        fail=1
    fi
done
set -e

echo ""

# --- Build the scratch_scaling.csv summary ---
# Walk the cells manifest, read each cell's latest run_result.json, and emit
# one wide row per cell. Cells without a result yield NaN metric cells.
SUMMARY_CSV="${EXP_ROOT}/scratch_scaling.csv"
uv run python - "$CELLS_TSV" "$SUMMARY_CSV" <<'PY'
import csv
import json
import sys
from pathlib import Path

cells_tsv, out_csv = Path(sys.argv[1]), Path(sys.argv[2])


def latest_result(run_dir: Path) -> Path | None:
    """Newest csv_logs/version_*/run_result.json under ``run_dir``."""
    results = sorted(
        (run_dir / "csv_logs").glob("version_*/run_result.json"),
        key=lambda p: int(p.parts[-2].split("_", 1)[1]),
    )
    return results[-1] if results else None


rows: list[dict[str, str]] = []
for line in cells_tsv.read_text().splitlines():
    if not line.strip():
        continue
    cond, kind, scale, session, seed, requested, run_dir = line.split("\t")
    row = {
        "condition": cond,
        "kind": kind,
        "session": session,
        "scale": scale,
        "seed": seed,
        "requested_trials": requested,
        "kept_trials": "nan",
        "val_best_r2": "nan",
        "test_r2": "nan",
        "run_dir": run_dir,
    }
    # Only trust cells the worker marked complete (.complete sentinel == rc==0
    # AND a result was written). This keeps the summary consistent with the
    # resume guard: a cell without .complete is "not done" and stays NaN rather
    # than reporting a stale/partial run_result.json from a killed attempt.
    rp = latest_result(Path(run_dir)) if (Path(run_dir) / ".complete").is_file() else None
    if rp is not None:
        try:
            data = json.loads(rp.read_text())
            if not isinstance(data, dict):
                raise ValueError(f"run_result.json is not a JSON object: {type(data).__name__}")
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            # A run killed mid-write can leave a truncated/garbage run_result.json.
            # Degrade this one cell to NaN (row defaults above) rather than
            # crashing the whole summary, which would lose scratch_scaling.csv.
            print(f"WARNING: skipping corrupt {rp}: {exc}", file=sys.stderr)
        else:
            for key in ("kept_trials", "val_best_r2", "test_r2"):
                val = data.get(key)
                if val is not None:
                    row[key] = str(val)
    rows.append(row)

if rows:
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    n_done = sum(1 for r in rows if r["test_r2"] != "nan")
    print(f"wrote {out_csv} ({len(rows)} cells, {n_done} with results)")
else:
    print(f"no cells to summarise; {out_csv} not written")
PY

# --- Aggregate child metrics onto each condition's MLflow parent run ---
for cond in "${COND_ORDER[@]}"; do
    parent_id="${COND_PARENT_ID[$cond]}"
    [[ -z "$parent_id" ]] && continue
    echo "[$(date +%H:%M:%S)] aggregating parent for ${cond}: ${COND_PARENT_URI[$cond]}"
    uv run python -m ntt_downstream.decoding.utils.mlflow_parent aggregate \
        --run-id "$parent_id" || \
        echo "WARN: aggregation failed for ${cond} parent=${parent_id}" >&2
done

if (( fail == 0 )); then
    echo "=== all workers completed; results in ${EXP_ROOT}/ ==="
    echo "=== summary: ${SUMMARY_CSV}"
else
    echo "=== one or more workers reported failures (see ${LOG_ROOT}/) ==="
    exit 1
fi
