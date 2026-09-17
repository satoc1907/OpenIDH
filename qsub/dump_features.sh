#!/bin/bash
# Experiment A (shift decomposition): dump CLS features + head weights — inference only.
# Reloads each runs/<name>/model_best.pt, scores inner-val and test once, and writes
# runs/<name>/{features_val,features_test,head_weights}.npz. No weights are updated;
# nothing is retrained. notebooks/05_shift_decomposition.ipynb then runs on the npz
# files alone (no model, no GPU), so copy runs/*/features_*.npz + head_weights.npz +
# features_meta.json back to the analysis machine.
#
#   qsub qsub/dump_features.sh                                   # 12 shift runs + rand0_s0
#   qsub -v RUNS="vendor_s0 vendor_s1 vendor_s2" qsub/dump_features.sh   # just those
#
# Submit from the repo root: `#$ -cwd` makes the job run there.
# One GPU is enough — 13 runs x (inner-val + test) is well under 30 min, I/O bound.
#$ -cwd
#$ -jc gtn-container_g1
#$ -ac d=nvcr-pytorch-2401
#$ -N openidh_feat
#$ -j y

# ── container env (same as qsub/calibrate.sh) ─────────────────────────────────
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

# uv brings its own Python (pyproject needs >=3.12, the container ships 3.10), so
# the container's 3.10 site-packages must not leak into that venv.
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

echo "[feat] host=$(hostname) job=${JOB_ID:-?} device=${DEVICE} runs=${RUNS:-<default 13>}"
nvidia-smi -L || true

ARGS=(--device "${DEVICE}")
# RUNS is a space-separated list; unquoted on purpose so it splits into arguments.
[ -n "${RUNS:-}" ] && ARGS+=(--runs ${RUNS})

uv run --frozen python scripts/dump_features.py "${ARGS[@]}"

echo "[feat] done — runs/*/features_{val,test}.npz + head_weights.npz + features_meta.json"
echo "[feat] to analyse elsewhere: rsync -av --include='*/' --include='features_*' --include='head_weights.npz' --exclude='*' runs/ <dest>/runs/"
