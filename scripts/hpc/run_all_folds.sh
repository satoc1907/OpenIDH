#!/bin/bash
# Launch all 9 fold-units x SEEDS (spec §15 Stage B step 14).
# 9 units = 2 LOSO + vendor + field + 5 random folds. Embarrassingly parallel:
# 1 GPU per job (spec §1.1). If SLURM is present each unit is an sbatch job;
# otherwise they run locally, 4 at a time (one per GPU, round-robin).
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

SEEDS="${SEEDS:-0 1 2}"
TRAIN_YAML="${TRAIN_YAML:-train.yaml}"
N_GPUS="${N_GPUS:-4}"

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

HAVE_SLURM=0; command -v sbatch >/dev/null 2>&1 && HAVE_SLURM=1
i=0
for seed in ${SEEDS}; do
  for u in "${UNITS[@]}"; do
    IFS='|' read -r split fold tag <<< "$u"
    out="runs/${tag}_s${seed}"
    if [ "${HAVE_SLURM}" = "1" ]; then
      sbatch --export=ALL,SPLIT_FILE="${split}",FOLD_ID="${fold}",SEED="${seed}",OUT="${out}",TRAIN_YAML="${TRAIN_YAML}" \
             scripts/hpc/submit.sh
    else
      gpu=$(( i % N_GPUS ))
      echo "[local] gpu=${gpu} ${split} fold=${fold:-<none>} seed=${seed} -> ${out}"
      CUDA_VISIBLE_DEVICES="${gpu}" SPLIT_FILE="${split}" FOLD_ID="${fold}" SEED="${seed}" \
        OUT="${out}" TRAIN_YAML="${TRAIN_YAML}" bash scripts/hpc/submit.sh &
      # throttle to N_GPUS concurrent local jobs
      if (( (i + 1) % N_GPUS == 0 )); then wait; fi
    fi
    i=$(( i + 1 ))
  done
done
[ "${HAVE_SLURM}" = "1" ] || wait
echo "launched ${i} job(s)  (SLURM=${HAVE_SLURM})"
