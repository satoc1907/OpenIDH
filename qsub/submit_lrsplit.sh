#!/bin/bash
# Experiment: separate learning rates for the trunks and the heads.
#
# The tabular head contributes evidence 1.26 against 6-12 for the image heads, so
# age is effectively unused — while age alone beats the whole model out of
# distribution (AUC 0.889 vs 0.860). The heads are 8k parameters trained at a rate
# picked for 44M of pretrained trunk, so the hypothesis is undertraining.
#
# Runs LOSO-B x 3 seeds first; widen only if the tabular evidence actually moves.
#
#   bash qsub/submit_lrsplit.sh                     # lr_head 1e-3
#   LR=3 bash qsub/submit_lrsplit.sh                # lr_head 3e-3
#   SPLITS="foldA" bash qsub/submit_lrsplit.sh      # LOSO-A once B looks good
#   DRY_RUN=1 bash qsub/submit_lrsplit.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

LR="${LR:-1}"                                   # 1 -> train.lrsplit.yaml, 3 -> lrsplit3
TRAIN_YAML="train.lrsplit$([ "${LR}" = "3" ] && echo 3 || echo "").yaml"
SEEDS="${SEEDS:-0 1 2}"
SPLITS="${SPLITS:-foldB}"
JC="${JC:-gtn-container_g1}"
REPO_DIR="${REPO_DIR:-$(pwd)}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"

[ -f "configs/${TRAIN_YAML}" ] || { echo "configs/${TRAIN_YAML} not found"; exit 1; }
echo "[lrsplit] config=${TRAIN_YAML} splits='${SPLITS}' seeds='${SEEDS}'"

n=0
for tag in ${SPLITS}; do
  case "${tag}" in
    foldA) split="splits_loso_foldA.csv" ;;
    foldB) split="splits_loso_foldB.csv" ;;
    *) echo "unknown split tag '${tag}' (expected foldA or foldB)"; exit 1 ;;
  esac
  for seed in ${SEEDS}; do
    name="openidh_lr${LR}_${tag}_s${seed}"
    out="runs/lrsplit${LR}_${tag}_s${seed}"
    vars="SPLIT_FILE=${split},FOLD_ID=,SEED=${seed},OUT=${out},TRAIN_YAML=${TRAIN_YAML},REPO_DIR=${REPO_DIR}"
    if [ -n "${DRY_RUN:-}" ]; then
      echo "qsub -N ${name} -jc ${JC} -o ${LOG_DIR}/${name}.log -j y -v '${vars}' qsub/train_one.sh"
    else
      qsub -N "${name}" -jc "${JC}" -o "${LOG_DIR}/${name}.log" -j y -v "${vars}" qsub/train_one.sh
    fi
    n=$(( n + 1 ))
  done
done
echo "submitted ${n} job(s)${DRY_RUN:+ (DRY_RUN — nothing was actually submitted)}"
echo "compare with: uv run python scripts/compare_lrsplit.py --variant lrsplit${LR}"
