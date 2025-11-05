#!/bin/bash

set -euo pipefail

TEMPLATE="jobs/template.sbatch"
LOGDIR="logs"
mkdir -p "$LOGDIR"

DATA_PATH="/mnt/data/data/processed/vhh_200.tsv"

BATCH_FLAGS="--batch_size 4 --batch_size_ds 16"
WANDB_PROJECT="esm2-v0"

specs=(
  "facebook/esm2_t6_8M_UR50D 5 0"
  "facebook/esm2_t12_35M_UR50D 5 0"
  "facebook/esm2_t30_150M_UR50D 5 0"
  "facebook/esm2_t33_650M_UR50D 5 0"
  "facebook/esm2_t36_3B_UR50D 5 0"
  "facebook/esm2_t48_15B_UR50D 5 1 --fp16"
  "facebook/esm2_t48_15B_UR50D 5 2 --fp16"
  "facebook/esm2_t48_15B_UR50D 5 3 --fp16"
)

extract_size() {
  local model="$1"
  local size
  size=$(echo "$model" | sed -E 's/.*_([0-9]+(M|B)).*/\1/')
  if [[ -z "$size" || "$size" == "$model" ]]; then
    size=$(echo "$model" | sed -E 's/.*t[0-9]+_([0-9]+).*/\1M/')
  fi
  echo "$size"
}

jobs=()
for spec in "${specs[@]}"; do
  read -r model epochs zs extra <<<"$spec"
  extra=${extra:-}
  size=$(extract_size "$model")
  wandb_name="${size}-zs${zs}-e${epochs}-bs4-d200"
  if [[ "$extra" == "--fp16" ]]; then
    wandb_name="${wandb_name}-fp16"
  fi
  CMD="--model ${model} ${BATCH_FLAGS} --zero_stage ${zs} --epochs ${epochs} \
--wandb --wandb_project ${WANDB_PROJECT} --wandb_run_name ${wandb_name} \
${extra} --data ${DATA_PATH}"
  CMD=$(echo "$CMD" | xargs)
  jobs+=("$CMD")
  echo "Submitting: $CMD"
  sbatch --export=ALL,CMD="$CMD" "$TEMPLATE"
done