#!/bin/bash
# Post-hoc temperature scaling over finished runs (spec §10) — inference only.
# Reloads each runs/<name>/model_best.pt, fits T on inner-val, applies it to test,
# writes runs/<name>/calibration.json, and prints the before/after markdown table
# at the end of the job log. No weights are updated; nothing is retrained.
#
#   qsub qsub/calibrate.sh                              # every run with a checkpoint
#   qsub -v RUNS="foldB_s0 foldB_s1 foldB_s2" qsub/calibrate.sh   # just those
#
# Submit from the repo root: `#$ -cwd` makes the job run there.
# One GPU is enough — 27 runs x (inner-val + test) is roughly 30 min, I/O bound.
#$ -cwd
#$ -jc gtn-container_g1
#$ -ac d=nvcr-pytorch-2401
#$ -N openidh_calib
#$ -j y

# ── container env (same as qsub/train_one.sh) ─────────────────────────────────
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

echo "[calib] host=$(hostname) job=${JOB_ID:-?} device=${DEVICE} runs=${RUNS:-<all>}"
nvidia-smi -L || true

ARGS=(--device "${DEVICE}")
# RUNS is a space-separated list; unquoted on purpose so it splits into arguments.
[ -n "${RUNS:-}" ] && ARGS+=(--runs ${RUNS})

uv run --frozen python scripts/calibrate_runs.py "${ARGS[@]}"

echo "[calib] done — per-run results in runs/*/calibration.json"
