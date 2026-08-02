#!/bin/bash
# Submit all 9 fold-units x SEEDS as independent 1-GPU qsub jobs (spec §15 Stage B step 14).
# Run this on the login node, from the repo root:
#
#   bash qsub/submit_all.sh                 # 9 units x 3 seeds = 27 jobs
#   SEEDS="0" bash qsub/submit_all.sh       # one seed only
#   DRY_RUN=1 bash qsub/submit_all.sh       # print the qsub lines, submit nothing
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

SEEDS="${SEEDS:-0 1 2}"
TRAIN_YAML="${TRAIN_YAML:-train.yaml}"
JC="${JC:-gtn-container_g1}"          # job class: 1 GPU per job (spec §1.1)
REPO_DIR="${REPO_DIR:-$(pwd)}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"

# "split_file|fold_id|tag"  (empty fold_id = single-fold file)
UNITS=(
  "splits_loso_foldA.csv||foldA"
  "splits_loso_foldB.csv||foldB"
  "splits_vendor_philips.csv||vendor"
  "splits_field.csv||field"
  "splits_random_5fold.csv|0|rand0"
  "splits_random_5fold.csv|1|rand1"
  "splits_random_5fold.csv|2|rand2"
  "splits_random_5fold.csv|3|rand3"
  "splits_random_5fold.csv|4|rand4"
)

n=0
for seed in ${SEEDS}; do
  for u in "${UNITS[@]}"; do
    IFS='|' read -r split fold tag <<< "$u"
    name="openidh_${tag}_s${seed}"
    out="runs/${tag}_s${seed}"
    vars="SPLIT_FILE=${split},FOLD_ID=${fold},SEED=${seed},OUT=${out},TRAIN_YAML=${TRAIN_YAML},REPO_DIR=${REPO_DIR}"
    cmd=(qsub -N "${name}" -jc "${JC}" -o "${LOG_DIR}/${name}.log" -j y -v "${vars}" qsub/train_one.sh)
    if [ -n "${DRY_RUN:-}" ]; then
      echo "qsub -N ${name} -jc ${JC} -o ${LOG_DIR}/${name}.log -j y -v '${vars}' qsub/train_one.sh"
    else
      "${cmd[@]}"
    fi
    n=$(( n + 1 ))
  done
done
echo "submitted ${n} job(s)${DRY_RUN:+ (DRY_RUN — nothing was actually submitted)}; logs -> ${LOG_DIR}/"
echo "watch with: qstat"
