#!/bin/bash

set -euo pipefail
LOGDIR="logs"
mkdir -p "$LOGDIR"

TEMPLATE="jobs/template.sbatch"
DATA_PATH="/mnt/data/data/processed/vhh_2K.tsv"
WANDB_PROJECT="esm2"

specs=(
  "facebook/esm2_t36_3B_UR50D 5 0 4 16"
  "facebook/esm2_t36_3B_UR50D 5 0 16 64"
  "facebook/esm2_t36_3B_UR50D 5 0 64 256"
)

for spec in "${specs[@]}"; do
  read -r model epochs zs bs bs_ds <<< "$spec"
  size=$(echo "$model" | sed -E 's/.*_([0-9]+(M|B)).*/\1/')
  wandb_name="${size}-zs${zs}-e${epochs}-bs${bs}-d2K"
  CMD="--model ${model} --batch_size ${bs} --batch_size_ds ${bs_ds} \
    --zero_stage ${zs} --epochs ${epochs} --wandb --wandb_project ${WANDB_PROJECT} \
    --wandb_run_name ${wandb_name} --data ${DATA_PATH}"
  CMD=$(echo "$CMD" | xargs)
  echo "Submitting: $CMD"
  sbatch --export=ALL,CMD="$CMD" "$TEMPLATE"
done
