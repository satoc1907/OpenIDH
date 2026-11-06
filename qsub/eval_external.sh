#!/bin/bash
# External validation (EGD, Japan): score the 6 LOSO checkpoints — inference only.
# Step 1 builds the evaluation lists from the cohort tables in
# /home/satoc/data/OpenIDH/metadata/ (external_metadata_dir) and stops if the case
# counts differ from the data-management reports. Step 2 reloads each
# runs/<name>/model_best.pt and writes, outside the repo (the per-case file has labels):
#   /home/satoc/data/OpenIDH/external_eval/{predictions.csv,summary.json,summary.md}
#
#   qsub qsub/eval_external.sh                                      # foldA_s0-2, foldB_s0-2
#   qsub -v RUNS=foldA_s0,COHORTS=EGD qsub/eval_external.sh         # a quick check first
#
# Submit from the repo root: `#$ -cwd` makes the job run there.
# One GPU is enough — 6 runs x ~1,090 cases, I/O bound (each case gunzips 5 volumes).
#$ -cwd
#$ -jc gtn-container_g1
#$ -ac d=nvcr-pytorch-2401
#$ -N openidh_ext
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

echo "[ext] host=$(hostname) job=${JOB_ID:-?} device=${DEVICE} runs=${RUNS:-<6 LOSO>} cohorts=${COHORTS:-EGD Japan}"
nvidia-smi -L || true

SPLIT_ARGS=()
ARGS=(--device "${DEVICE}")
# RUNS / COHORTS are space-separated lists; unquoted on purpose so they split.
[ -n "${RUNS:-}" ] && ARGS+=(--runs ${RUNS})
if [ -n "${COHORTS:-}" ]; then
  SPLIT_ARGS+=(--cohorts ${COHORTS})
  ARGS+=(--cohorts ${COHORTS})
fi

uv run --frozen python scripts/build_external_splits.py ${SPLIT_ARGS[@]+"${SPLIT_ARGS[@]}"}
uv run --frozen python scripts/eval_external.py "${ARGS[@]}"

echo "[ext] done — summary.md is aggregate only; predictions.csv has labels, keep it off git"
