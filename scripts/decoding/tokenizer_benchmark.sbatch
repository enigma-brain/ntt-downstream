#!/usr/bin/env bash
#SBATCH --job-name=tokenizer_benchmark
#SBATCH --partition=gpu   # neutral fallback; override per-site via scripts/decoding/submit.sh (PARTITION/GRES in .env)
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err
#
# EXAMPLE: adjust the #SBATCH resource lines to your cluster.
#
# Thin SLURM wrapper over the stable orchestration CLI:
#   python -m ntt_downstream.decoding.orchestration benchmark
#
# Covers two recipes:
#   * tokenizer benchmark — frozen tokenizers (+ optional spike_times)
#   * spike_times baseline — set SPIKE_TIMES_ONLY=1 (skips frozen discovery, so
#     no CKPT_ROOT is needed, and defaults MODEL_CFG to the crossattn poyo_1.3M)
#
# ALL benchmarking logic lives in the package; this file carries only cluster
# (#SBATCH) directives + env -> flag forwarding. See docs/decoding_orchestration.md.
# Every env knob is optional — unset ones fall back to the CLI defaults / the
# data YAML.
#
# Examples:
#   sbatch scripts/decoding/tokenizer_benchmark.sbatch
#   # non-default cluster? submit via the wrapper (injects --partition/--gres from .env):
#   PARTITION=gpu-big scripts/decoding/submit.sh scripts/decoding/tokenizer_benchmark.sbatch
#   GPUS="0 1 2 3" CKPT_ROOT=results/tokenizer_checkpoints \
#       bash scripts/decoding/tokenizer_benchmark.sbatch
#   DRY_RUN=1 TOKENIZERS="tokenizer_a tokenizer_b" bash scripts/decoding/tokenizer_benchmark.sbatch
#   # spike_times baseline only:
#   SPIKE_TIMES_ONLY=1 SESSIONS="t_..._random_target_reaching c_..." \
#       sbatch scripts/decoding/tokenizer_benchmark.sbatch

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p outputs/slurm
[[ -f "${REPO_ROOT}/.env" ]] && { set -a; source "${REPO_ROOT}/.env"; set +a; }

# Determinism is owned by the orchestration engine: it pins PYTHONHASHSEED to
# the run's seed and injects CUBLAS_WORKSPACE_CONFIG into every train.py
# subprocess. Export CUBLAS_WORKSPACE_CONFIG here only to override at run time.

args=()
[[ -n "${DATA_CONFIG:-}"            ]] && args+=(--data-config "$DATA_CONFIG")
[[ -n "${CKPT_ROOT:-}"              ]] && args+=(--ckpt-root "$CKPT_ROOT")
[[ -n "${TOKENIZERS:-}"             ]] && args+=(--tokenizers "$TOKENIZERS")
[[ -n "${TOKENIZERS_LIMIT:-}"       ]] && args+=(--tokenizers-limit "$TOKENIZERS_LIMIT")
[[ -n "${INCLUDE_SPIKE_TIMES:-}"    ]] && args+=(--include-spike-times "$INCLUDE_SPIKE_TIMES")
[[ -n "${SPIKE_TIMES_ONLY:-}"       ]] && args+=(--spike-times-only "$SPIKE_TIMES_ONLY")
[[ -n "${SESSIONS:-}"               ]] && args+=(--sessions "$SESSIONS")
[[ -n "${DATA_ROOT:-}"              ]] && args+=(--data-root "$DATA_ROOT")
[[ -n "${MODEL_CFG:-}"              ]] && args+=(--model-cfg "$MODEL_CFG")
[[ -n "${UE_MODE:-}"                ]] && args+=(--unit-embedding-mode "$UE_MODE")
[[ -n "${SENTINELS:-}"              ]] && args+=(--sentinels "$SENTINELS")
[[ -n "${EPOCHS:-}"                 ]] && args+=(--epochs "$EPOCHS")
[[ -n "${BATCH_SIZE:-}"             ]] && args+=(--batch-size "$BATCH_SIZE")
[[ -n "${SEED:-}"                   ]] && args+=(--seed "$SEED")
[[ -n "${NUM_WORKERS:-}"            ]] && args+=(--num-workers "$NUM_WORKERS")
[[ -n "${COMPARISON_TAG:-}"         ]] && args+=(--comparison-tag "$COMPARISON_TAG")
[[ -n "${GPUS:-}"                   ]] && args+=(--gpus "$GPUS")
[[ -n "${MLFLOW_ENABLED:-}"         ]] && args+=(--mlflow "$MLFLOW_ENABLED")
[[ -n "${MLFLOW_EXPERIMENT_NAME:-}" ]] && args+=(--mlflow-experiment-name "$MLFLOW_EXPERIMENT_NAME")
[[ -n "${MLFLOW_LOG_MODEL:-}"       ]] && args+=(--mlflow-log-model "$MLFLOW_LOG_MODEL")
[[ -n "${EXTRA:-}"                  ]] && args+=(--extra "$EXTRA")
[[ "${DRY_RUN:-0}" != "0"           ]] && args+=(--dry-run)

exec uv run python -m ntt_downstream.decoding.orchestration benchmark "${args[@]}" "$@"
