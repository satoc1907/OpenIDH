#!/bin/bash
# Step 2 on rikyu: the 2-epoch smoke training (train.smoke.yaml) on one GPU,
# with the data staged to node-local /tmp first.
#   sbatch slurm/smoke.sbatch                              # from the repo root
#   sbatch --export=ALL,OUT=runs/smoke_rikyu_s1,SEED=1 slurm/smoke.sbatch
# Known answer: runs/smoke_foldA_s0/result.json (CPU run: best_val_nll 0.679,
# n_train 20 / n_val 10 / n_test 20). A GPU run lands within ~1e-2 of it (TF32);
# the subject counts and the ordering of metrics must match exactly.
# ~2 minutes of GPU -> ~10 yen.
#SBATCH --job-name=openidh_smoke
#SBATCH --account=rkp00078
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/%x.%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source slurm/env.sh
node_report

SPLIT_FILE="${SPLIT_FILE:-splits_loso_foldA.csv}"
TRAIN_YAML="${TRAIN_YAML:-train.smoke.yaml}"
SEED="${SEED:-0}"
OUT="${OUT:-runs/smoke_rikyu_s${SEED}}"
mkdir -p "${OUT}"

# 1. the 50 subjects the smoke config will pick, staged to /tmp (~1.5GB)
LIST="${OUT}/staged_subjects.txt"
run_py python scripts/smoke_subjects.py --split-file "${SPLIT_FILE}" --train-yaml "${TRAIN_YAML}" > "${LIST}"
echo "[smoke] $(wc -l < "${LIST}") subject dirs to stage"
stage_data "${LIST}"
export OPENIDH_PATHS="${STAGE_PATHS}"

# 2. train (same entry point as raiden's qsub/train_one.sh, ENTRY=py)
# train.smoke.yaml says device: cpu / num_workers: 0 — override the device only.
echo "[smoke] split=${SPLIT_FILE} yaml=${TRAIN_YAML} seed=${SEED} out=${OUT}"
run_py python scripts/run_fold.py --split-file "${SPLIT_FILE}" --seed "${SEED}" \
    --output-dir "${OUT}" --train-yaml "${TRAIN_YAML}" --device cuda --paths-file "${OPENIDH_PATHS}"

# 3. compare with the known answer
echo "[smoke] result vs runs/smoke_foldA_s0 (CPU reference):"
run_py python - "${OUT}" <<'PY'
import json, sys
new = json.load(open(f"{sys.argv[1]}/result.json")); ref = json.load(open("runs/smoke_foldA_s0/result.json"))
for k in ("n_train", "n_val", "n_test", "best_epoch"):
    print(f"  {k:12s} new {new[k]:>8}  ref {ref[k]:>8}  {'ok' if new[k] == ref[k] else 'DIFF'}")
print(f"  {'best_val_nll':12s} new {new['best_val_nll']:.4f}  ref {ref['best_val_nll']:.4f}")
for k in ("nll", "auc", "median_S"):
    print(f"  test {k:8s} new {new['test_metrics'][k]:.4f}  ref {ref['test_metrics'][k]:.4f}")
PY
echo "[smoke] done -> ${OUT}"
