#!/bin/bash
# Run ONE (split, fold, seed) on one GPU under raiden's UGE scheduler (qsub).
# Container env is taken verbatim from `qsub/gpu_0 original.sh`; only the
# Python part differs (see the uv note below).
#
#   # single job
#   qsub -v SPLIT_FILE=splits_loso_foldA.csv,SEED=0,OUT=runs/foldA_s0 qsub/train_one.sh
#   # all 9 fold-units x seeds
#   bash qsub/submit_all.sh
#
# Submit from the repo root: `#$ -cwd` makes the job run there.
#$ -cwd
#$ -jc gtn-container_g1
#$ -ac d=nvcr-pytorch-2401
#$ -N openidh
#$ -j y

# ── container env (as in the original template) ───────────────────────────────
/usr/local/bin/nvidia_entrypoint.sh || true
. /fefs/opt/dgx/env_set/nvcr-pytorch-2401.sh

# Compute nodes reach the outside world only through this proxy.
export MY_PROXY_URL="http://10.1.10.1:8080/"
export HTTP_PROXY=$MY_PROXY_URL
export HTTPS_PROXY=$MY_PROXY_URL
export FTP_PROXY=$MY_PROXY_URL
export http_proxy=$MY_PROXY_URL
export https_proxy=$MY_PROXY_URL
export ftp_proxy=$MY_PROXY_URL
export LDFLAGS=-L/usr/local/nvidia/lib64

# ── Python: uv, NOT the container interpreter ─────────────────────────────────
# pyproject requires Python >=3.12 but nvcr-pytorch-2401 ships 3.10, so uv builds
# its own interpreter + venv from uv.lock. The template's PYTHONPATH/PYTHONUSERBASE
# point at 3.10 site-packages; they must NOT leak into that venv, hence the unset.
unset PYTHONPATH PYTHONUSERBASE PREFIX
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
export UV_LINK_MODE=copy                                    # fefs has no cross-fs hardlinks
export UV_CACHE_DIR="${UV_CACHE_DIR:-${HOME}/.cache/uv}"    # ~5GB with torch cu128
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${HOME}/.local/share/uv/python}"

set -euo pipefail

cd "${REPO_DIR:-${HOME}/OpenIDH}"

# Stage B reads HPC paths and never fetches weights at runtime (spec §1.4):
# stage weights/ beforehand with `bash qsub/prepare_env.sh` on the login node.
export OPENIDH_PATHS="${OPENIDH_PATHS:-configs/paths.hpc.yaml}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Without this, Python block-buffers stdout when it is a file, so per-epoch
# progress only lands in the log at the very end — indistinguishable from a hang.
export PYTHONUNBUFFERED=1
# Pin the main process's thread count: left unpinned it varies with machine load
# (dataloader workers compete for cores), which reorders float reductions and makes
# reruns differ by ~1e-5. Fixed here so a rerun of the same seed reproduces exactly.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

: "${SPLIT_FILE:?set SPLIT_FILE (e.g. splits_loso_foldA.csv)}"
SEED="${SEED:-0}"
FOLD_ID="${FOLD_ID:-}"
TRAIN_YAML="${TRAIN_YAML:-train.yaml}"
OUT="${OUT:-runs/${JOB_ID:-manual}}"
ENTRY="${ENTRY:-papermill}"   # papermill = executed notebook artifact; py = plain script
mkdir -p "${OUT}"

echo "[qsub] host=$(hostname) job=${JOB_ID:-?} split=${SPLIT_FILE} fold_id=${FOLD_ID:-<none>} seed=${SEED} out=${OUT}"
nvidia-smi -L || true
uv --version

if [ "${ENTRY}" = "py" ]; then
  ARGS=(--split-file "${SPLIT_FILE}" --seed "${SEED}" --output-dir "${OUT}"
        --train-yaml "${TRAIN_YAML}" --device cuda --paths-file "${OPENIDH_PATHS}")
  [ -n "${FOLD_ID}" ] && ARGS+=(--fold-id "${FOLD_ID}")
  uv run --frozen python scripts/run_fold.py "${ARGS[@]}"
else
  PM_ARGS=(-p split_file "${SPLIT_FILE}" -p seed "${SEED}" -p output_dir "${OUT}"
           -p train_yaml "${TRAIN_YAML}" -p device cuda -p paths_file "${OPENIDH_PATHS}")
  [ -n "${FOLD_ID}" ] && PM_ARGS+=(-p fold_id "${FOLD_ID}")
  # --log-output: without it papermill keeps cell stdout inside the notebook and
  # writes it only at the end, so a 20h job shows zero progress in the qsub log.
  uv run --frozen papermill --log-output --no-progress-bar \
    notebooks/03_train.ipynb "${OUT}/03_train_executed.ipynb" "${PM_ARGS[@]}"
fi

echo "[qsub] done -> ${OUT}"
