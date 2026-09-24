#!/bin/bash
# Experiment: score every run as if T2/FLAIR (and T1) had never been acquired.
# Inference only — the named sequences are zeroed before the forward pass, so the
# unified trunk sees the absence too, and their heads leave the fusion. This is
# the missing-sequence path the model was trained with (modality dropout p=0.2),
# NOT the head-level arithmetic of scripts/head_ablation.py.
#
#   qsub qsub/modality_drop.sh                                   # 13 runs x 2 drop-sets
#   qsub -v RUNS="vendor_s0 vendor_s1 vendor_s2" qsub/modality_drop.sh
#   qsub -v DROPS="T2+FLAIR" qsub/modality_drop.sh               # just the first question
#
# Writes runs/<run>/predictions_drop-<tag>.csv (same schema as predictions.csv),
# so the comparison afterwards needs no GPU. Copy those CSVs back — they are
# small (~50-150KB each) and gitignored is NOT an issue: they are plain text.
# Submit from the repo root: `#$ -cwd` makes the job run there.
# One GPU; 13 runs x 2 passes x (inner-val + test) is roughly 15 min, I/O bound.
#$ -cwd
#$ -jc gtn-container_g1
#$ -ac d=nvcr-pytorch-2401
#$ -N openidh_drop
#$ -j y

# ── container env (same as qsub/dump_features.sh) ─────────────────────────────
/usr/local/bin/nvidia_entrypoint.sh || true
. /fefs/opt/dgx/env_set/nvcr-pytorch-2401.sh

export MY_PROXY_URL="http://10.1.10.1:8080/"
export HTTP_PROXY=$MY_PROXY_URL
export HTTPS_PROXY=$MY_PROXY_URL
export FTP_PROXY=$MY_PROXY_URL
export http_proxy=$MY_PROXY_URL
export https_proxy=$MY_PROXY_URL
export ftp_proxy=$MY_PROXY_URL
export LDFLAGS=-L/usr/local/nvidia/lib64

unset PYTHONPATH PYTHONUSERBASE PREFIX
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
export UV_LINK_MODE=copy
export UV_CACHE_DIR="${UV_CACHE_DIR:-${HOME}/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${HOME}/.local/share/uv/python}"

set -euo pipefail

cd "${REPO_DIR:-${HOME}/OpenIDH}"

export OPENIDH_PATHS="${OPENIDH_PATHS:-configs/paths.hpc.yaml}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

DEVICE="${DEVICE:-cuda}"

echo "[drop] host=$(hostname) job=${JOB_ID:-?} device=${DEVICE} runs=${RUNS:-<default 13>} drops=${DROPS:-<all>}"
nvidia-smi -L || true

ARGS=(--device "${DEVICE}" --json results/modality_drop.json)
# RUNS / DROPS are space-separated lists; unquoted on purpose so they split.
[ -n "${RUNS:-}" ] && ARGS+=(--runs ${RUNS})
[ -n "${DROPS:-}" ] && ARGS+=(--drops ${DROPS})

uv run --frozen python scripts/modality_drop_eval.py "${ARGS[@]}"

echo "[drop] done — runs/*/predictions_drop-*.csv + results/modality_drop.json"
echo "[drop] to analyse elsewhere: rsync -av --include='*/' --include='predictions_drop-*' --exclude='*' runs/ <dest>/runs/"
