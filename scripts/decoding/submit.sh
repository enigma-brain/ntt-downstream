#!/usr/bin/env bash
#
# Thin submission wrapper for the decoding sbatch scripts.
#
# Why this exists: SLURM parses the `#SBATCH` directives at the top of a job
# script *before* the shell ever runs, so they cannot read environment
# variables — `#SBATCH --partition=$PARTITION` would literally request a
# partition named "$PARTITION". The cluster-specific bits (GPU partition,
# generic-resource request) therefore can't be folded into the same
# `.env` -> `--flag` forwarding the scripts use for every runtime knob.
#
# This wrapper closes that gap the only way SLURM allows: it sources the repo
# `.env`, then submits the target script with `--partition` / `--gres` passed
# as `sbatch` *command-line* flags. CLI flags beat the in-file `#SBATCH`
# directives, so the values here win over the neutral `gpu` fallback baked
# into each script — making the scripts portable across clusters without
# editing them. The in-file fallback works on clusters with a partition named
# `gpu`; the wrapper is only needed when your partition/gres differ.
#
# Configure via `.env` (or inline env):
#   PARTITION      GPU partition name        (e.g. gpu-big; default: script's #SBATCH)
#   GRES           generic resource request  (e.g. gpu:8;   default: script's #SBATCH)
#   SBATCH_EXTRA   extra sbatch flags, verbatim (e.g. "--time=4:00:00 --mem=64G")
#
# Usage:
#   scripts/decoding/submit.sh <script.sbatch> [script env-knobs / args...]
#
# Examples:
#   PARTITION=gpu-big GRES=gpu:8 \
#       scripts/decoding/submit.sh scripts/decoding/tokenizer_benchmark.sbatch
#   # the scripts' own env knobs pass through unchanged (sbatch --export=ALL):
#   PARTITION=gpu TOKENIZERS="my_tokenizer" \
#       scripts/decoding/submit.sh scripts/decoding/tokenizer_benchmark.sbatch
#
# For a dry run, bypass this wrapper and run the script directly — the
# `#SBATCH` lines are inert outside `sbatch`:
#   DRY_RUN=1 bash scripts/decoding/tokenizer_benchmark.sbatch

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Inline `PARTITION=… submit.sh` must win over a value in `.env`, so snapshot
# the wrapper knobs before sourcing and reassert any that were set inline.
# (The .env values still serve as the per-site default; the sbatch script
# re-sources `.env` itself at runtime, so the job keeps the rest of it.)
_inline_partition="${PARTITION:-}"
_inline_gres="${GRES:-}"
_inline_sbatch_extra="${SBATCH_EXTRA:-}"
if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
fi
[[ -n "$_inline_partition" ]] && PARTITION="$_inline_partition"
[[ -n "$_inline_gres" ]] && GRES="$_inline_gres"
[[ -n "$_inline_sbatch_extra" ]] && SBATCH_EXTRA="$_inline_sbatch_extra"

if (( $# < 1 )); then
    echo "usage: $0 <script.sbatch> [script env-knobs / args...]" >&2
    exit 2
fi
script="$1"; shift
[[ -f "$script" ]] || { echo "ERROR: sbatch script not found: $script" >&2; exit 1; }

sbatch_args=()
if [[ -n "${PARTITION:-}" ]]; then sbatch_args+=(--partition="$PARTITION"); fi
if [[ -n "${GRES:-}" ]]; then sbatch_args+=(--gres="$GRES"); fi
# SBATCH_EXTRA is split on whitespace on purpose so it can carry several flags.
if [[ -n "${SBATCH_EXTRA:-}" ]]; then
    read -ra _sbatch_extra <<< "$SBATCH_EXTRA"
    sbatch_args+=("${_sbatch_extra[@]}")
fi

exec sbatch "${sbatch_args[@]}" "$script" "$@"
