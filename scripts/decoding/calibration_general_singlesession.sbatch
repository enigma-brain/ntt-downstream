#!/usr/bin/env bash
#SBATCH --job-name=calibration_singlesession
#SBATCH --partition=gpu   # neutral fallback; override per-site via scripts/decoding/submit.sh (PARTITION/GRES in .env)
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err
#
# EXAMPLE: adjust the #SBATCH resource lines to your cluster.
#
# Thin SLURM wrapper over the stable orchestration CLI:
#   python -m ntt_downstream.decoding.orchestration calibrate
#
# ALL benchmarking/calibration logic lives in the package; this file carries
# only cluster (#SBATCH) directives + env -> flag forwarding, so editing a
# #SBATCH knob can never change the pipeline. See docs/decoding_orchestration.md.
# Every env knob below is optional — unset ones fall back to the YAML defaults
# the CLI resolves from configs/decoding/single_session/{training,data}.yaml.
#
# Examples:
#   sbatch scripts/decoding/calibration_general_singlesession.sbatch    # uses YAML defaults
#   # non-default cluster? submit via the wrapper (injects --partition/--gres from .env):
#   PARTITION=gpu-big scripts/decoding/submit.sh scripts/decoding/calibration_general_singlesession.sbatch
#   CKPT=/abs/<exp>/checkpoints/model.ckpt GPUS="0 1 2 3" \
#       bash scripts/decoding/calibration_general_singlesession.sbatch
#   DRY_RUN=1 CKPT=/abs/m.ckpt bash scripts/decoding/calibration_general_singlesession.sbatch

set -euo pipefail

# Under sbatch, BASH_SOURCE points into the spool copy; prefer SLURM_SUBMIT_DIR.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p outputs/slurm
[[ -f "${REPO_ROOT}/.env" ]] && { set -a; source "${REPO_ROOT}/.env"; set +a; }

# Determinism is owned by the orchestration engine: it pins PYTHONHASHSEED to
# each cell's seed and injects CUBLAS_WORKSPACE_CONFIG (from the training YAML's
# runtime.cublas_workspace_config) into every train.py subprocess. Export
# CUBLAS_WORKSPACE_CONFIG here only to override the YAML value at run time.

args=()
[[ -n "${TRAINING_CONFIG:-}"        ]] && args+=(--training-config "$TRAINING_CONFIG")
[[ -n "${DATA_CONFIG:-}"            ]] && args+=(--data-config "$DATA_CONFIG")
[[ -n "${TOKENIZER_KIND:-}"         ]] && args+=(--tokenizer-kind "$TOKENIZER_KIND")
[[ -n "${CKPT:-}"                   ]] && args+=(--ckpt "$CKPT")
[[ -n "${TOKENIZER_NAME:-}"         ]] && args+=(--tokenizer-name "$TOKENIZER_NAME")
[[ -n "${TRAIN_SESSIONS:-}"         ]] && args+=(--train-sessions "$TRAIN_SESSIONS")
[[ -n "${OTHER_SESSIONS:-}"         ]] && args+=(--other-sessions "$OTHER_SESSIONS")
[[ -n "${FT_TRIAL_COUNTS:-}"        ]] && args+=(--ft-trial-counts "$FT_TRIAL_COUNTS")
[[ -n "${FT_MODES:-}"               ]] && args+=(--ft-modes "$FT_MODES")
[[ -n "${SEEDS:-}"                  ]] && args+=(--seeds "$SEEDS")
[[ -n "${MODEL_SIZE:-}"             ]] && args+=(--model-cfg "$MODEL_SIZE")
[[ -n "${NO_CROSSATTN:-}"           ]] && args+=(--no-crossattn "$NO_CROSSATTN")
[[ -n "${UE_MODE:-}"                ]] && args+=(--unit-embedding-mode "$UE_MODE")
[[ -n "${TOKENIZER_SENTINELS:-}"    ]] && args+=(--sentinels "$TOKENIZER_SENTINELS")
[[ -n "${BATCH_SIZE:-}"             ]] && args+=(--batch-size "$BATCH_SIZE")
[[ -n "${NUM_WORKERS:-}"            ]] && args+=(--num-workers "$NUM_WORKERS")
[[ -n "${BASE_MAX_STEPS:-}"         ]] && args+=(--base-max-steps "$BASE_MAX_STEPS")
[[ -n "${FT_MAX_STEPS:-}"           ]] && args+=(--ft-max-steps "$FT_MAX_STEPS")
[[ -n "${NUM_LATENTS_PER_STEP:-}"   ]] && args+=(--num-latents-per-step "$NUM_LATENTS_PER_STEP")
[[ -n "${EVAL_EVERY_N_STEPS:-}"     ]] && args+=(--eval-every-n-steps "$EVAL_EVERY_N_STEPS")
[[ -n "${CKPT_EVERY_N_STEPS:-}"     ]] && args+=(--ckpt-every-n-steps "$CKPT_EVERY_N_STEPS")
[[ -n "${FT_EVAL_EVERY:-}"          ]] && args+=(--ft-eval-every "$FT_EVAL_EVERY")
[[ -n "${FT_CKPT_EVERY:-}"          ]] && args+=(--ft-ckpt-every "$FT_CKPT_EVERY")
[[ -n "${PRECISION:-}"              ]] && args+=(--precision "$PRECISION")
[[ -n "${FORCE_FLASH_ATTN:-}"       ]] && args+=(--force-flash-attn "$FORCE_FLASH_ATTN")
[[ -n "${TEMPORAL_SPLIT_ENABLED:-}" ]] && args+=(--temporal-split-enabled "$TEMPORAL_SPLIT_ENABLED")
[[ -n "${BRAINSET:-}"               ]] && args+=(--brainset "$BRAINSET")
[[ -n "${MLFLOW_LOG_MODEL:-}"       ]] && args+=(--mlflow-log-model "$MLFLOW_LOG_MODEL")
[[ -n "${CALIB_ROOT:-}"             ]] && args+=(--calib-root "$CALIB_ROOT")
[[ -n "${DATA_ROOT:-}"              ]] && args+=(--data-root "$DATA_ROOT")
[[ -n "${GPUS:-}"                   ]] && args+=(--gpus "$GPUS")
[[ -n "${JOBS_PER_GPU:-}"           ]] && args+=(--jobs-per-gpu "$JOBS_PER_GPU")
[[ -n "${MLFLOW_ENABLED:-}"         ]] && args+=(--mlflow "$MLFLOW_ENABLED")
[[ -n "${MLFLOW_EXPERIMENT_NAME:-}" ]] && args+=(--mlflow-experiment-name "$MLFLOW_EXPERIMENT_NAME")
[[ -n "${REUSE_BASE_IF_EXISTS:-}"   ]] && args+=(--reuse-base-if-exists "$REUSE_BASE_IF_EXISTS")
[[ -n "${MISSING_ONLY:-}"           ]] && args+=(--missing-only "$MISSING_ONLY")
[[ -n "${SKIP_BASE_TRAIN:-}"        ]] && args+=(--skip-base-train "$SKIP_BASE_TRAIN")
[[ -n "${BASE_CKPT:-}"              ]] && args+=(--base-ckpt "$BASE_CKPT")
[[ -n "${EXTRA:-}"                  ]] && args+=(--extra "$EXTRA")
[[ "${DRY_RUN:-0}" != "0"           ]] && args+=(--dry-run)

exec uv run python -m ntt_downstream.decoding.orchestration calibrate "${args[@]}" "$@"
