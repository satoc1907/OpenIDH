#!/bin/bash
# Shared environment for every rikyu script (sourced by prepare_env.sh and the
# *.sbatch files). Mirrors the env block of qsub/train_one.sh minus raiden's
# proxy and container juggling: rikyu's login/compute nodes are Ubuntu 24.04
# aarch64 with the NVIDIA driver on the host, so the uv venv runs directly.
#
# Storage rules (home is 50GB, fixed):
#   uv cache + uv-managed CPython -> group area (/data1, write-once, ~7GB)
#   venv                          -> inside the repo on home (SSD Lustre; the
#                                    import hot path, ~6.5GB)
#   set UV_PROJECT_ENVIRONMENT to move the venv to /data1 if home gets tight.

export RIKYU_GROUP="${RIKYU_GROUP:-/data1/rkp00078/satoc}"
export REPO_DIR="${REPO_DIR:-${HOME}/OpenIDH}"

# --- Python: uv only, nothing from the host or ~/.local may leak in -----------
unset PYTHONPATH PYTHONUSERBASE PREFIX
export PYTHONNOUSERSITE=1
export PATH="${HOME}/.local/bin:${PATH}"
export UV_LINK_MODE=copy                          # Lustre: no cross-fs hardlinks
export UV_CACHE_DIR="${UV_CACHE_DIR:-${RIKYU_GROUP}/uv/cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${RIKYU_GROUP}/uv/python}"
export UV_PYTHON_PREFERENCE=only-managed          # never pick up /usr/bin/python3

# --- run-time: offline, unbuffered, pinned threads (see qsub/train_one.sh) ----
export OPENIDH_PATHS="${OPENIDH_PATHS:-configs/paths.rikyu.yaml}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

# Optional: run the same venv inside the Apptainer image instead of on the host
# (USE_CONTAINER=1). Only needed if the host userland ever misbehaves.
export SIF="${SIF:-${RIKYU_GROUP}/sif/pytorch-2511.sif}"
run_py() {
  if [ "${USE_CONTAINER:-0}" = "1" ]; then
    apptainer exec --nv -B /data1/rkp00078 "${SIF}" uv run --frozen "$@"
  else
    uv run --frozen "$@"
  fi
}

# --- data staging to the node-local NVMe (/tmp, wiped at job end) -------------
# stage_data <subject-list-file|all>
#   Copies the needed subject directories from the HDD Lustre data tree into
#   $STAGE_DIR/v1.0.0/ and writes $STAGE_PATHS, a derived paths yaml whose
#   data_dir points there. Export OPENIDH_PATHS="$STAGE_PATHS" afterwards.
#   The list file holds one "<cohort>/<subject_dir>" per line (relative to
#   v1.0.0/); "all" stages the three training cohorts (~46GB).
stage_data() {
  local list="${1:?stage_data <list-file|all>}"
  local src; src="$(run_py python -c 'from openidh_model.utils.paths import load_paths; print(load_paths().data_dir)')"
  export STAGE_DIR="${STAGE_DIR:-/tmp/${SLURM_JOB_ID:-manual}/openidh}"
  mkdir -p "${STAGE_DIR}/v1.0.0"
  local t0; t0=$(date +%s)
  if [ "${list}" = "all" ]; then
    rsync -a --exclude '_backup_*' --exclude 'egd' "${src}/" "${STAGE_DIR}/v1.0.0/"
  elif command -v rsync >/dev/null 2>&1; then
    rsync -a --files-from="${list}" -r "${src}/" "${STAGE_DIR}/v1.0.0/"
  else  # no rsync on the node: plain copy, one subject dir per line
    while read -r rel; do
      [ -n "${rel}" ] || continue
      mkdir -p "${STAGE_DIR}/v1.0.0/$(dirname "${rel}")"
      cp -r "${src}/${rel}" "${STAGE_DIR}/v1.0.0/${rel}"
    done < "${list}"
  fi
  echo "[stage] $(du -sh "${STAGE_DIR}/v1.0.0" | cut -f1) staged to ${STAGE_DIR} in $(( $(date +%s) - t0 ))s"
  export STAGE_PATHS="${STAGE_DIR}/paths.yaml"
  # same as configs/paths.rikyu.yaml with data_dir swapped; root stays explicit
  sed -e "s|^data_dir:.*|data_dir: ${STAGE_DIR}/v1.0.0|" \
      -e "s|^root:.*|root: ${REPO_DIR}|" "${REPO_DIR}/configs/paths.rikyu.yaml" > "${STAGE_PATHS}"
  echo "[stage] OPENIDH_PATHS -> ${STAGE_PATHS}"
}

# --- what every job should print first (the "unknowns" list) ------------------
node_report() {
  echo "[node] host=$(hostname) job=${SLURM_JOB_ID:-?} gpus=${SLURM_GPUS:-?} cpus=${SLURM_CPUS_ON_NODE:-?}"
  nvidia-smi -L || true
  echo "[node] /dev/shm: $(df -h /dev/shm | tail -1)"
  echo "[node] ulimit -n $(ulimit -n)  ulimit -u $(ulimit -u)  /tmp free: $(df -h /tmp | tail -1 | awk '{print $4}')"
  echo "[node] uv $(uv --version 2>/dev/null || echo MISSING)"
}
