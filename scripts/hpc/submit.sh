#!/bin/bash
# Run ONE (split, fold, seed) on a single GPU via papermill (spec §1.1, §11.5, §15).
# Works both as an sbatch script and when invoked directly (env vars set the job).
#
#   SPLIT_FILE=splits_loso_foldA.csv SEED=0 OUT=runs/foldA_s0 bash scripts/hpc/submit.sh
#   sbatch --export=ALL,SPLIT_FILE=...,SEED=0,OUT=... scripts/hpc/submit.sh
#
#SBATCH --job-name=openidh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail

# Stage B is offline; weights are staged locally (spec §1.4).
export OPENIDH_PATHS="${OPENIDH_PATHS:-configs/paths.hpc.yaml}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

: "${SPLIT_FILE:?set SPLIT_FILE (e.g. splits_loso_foldA.csv)}"
SEED="${SEED:-0}"
FOLD_ID="${FOLD_ID:-}"
TRAIN_YAML="${TRAIN_YAML:-train.yaml}"
OUT="${OUT:-runs/${SLURM_JOB_ID:-manual}}"
PATHS_FILE="${OPENIDH_PATHS}"
mkdir -p "${OUT}"

echo "[submit] split=${SPLIT_FILE} fold_id=${FOLD_ID:-<none>} seed=${SEED} out=${OUT} gpu=${CUDA_VISIBLE_DEVICES:-?}"

PM_ARGS=(-p split_file "${SPLIT_FILE}" -p seed "${SEED}" -p output_dir "${OUT}"
         -p train_yaml "${TRAIN_YAML}" -p device cuda -p paths_file "${PATHS_FILE}")
[ -n "${FOLD_ID}" ] && PM_ARGS+=(-p fold_id "${FOLD_ID}")

uv run papermill notebooks/03_train.ipynb "${OUT}/03_train_executed.ipynb" "${PM_ARGS[@]}"
