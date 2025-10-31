#!/usr/bin/env bash
set -euo pipefail

TEMPLATE="esm2_job_template.sbatch"
LOGDIR="logs"
mkdir -p "$LOGDIR"

# jobs: MODEL EPOCHS ZERO_STAGE [EXTRA ARGS...]
jobs=(
  "facebook/esm2_t6_8M_UR50D 4 0"
  "facebook/esm2_t12_35M_UR50D 4 0"
  "facebook/esm2_t30_150M_UR50D 4 0"
  "facebook/esm2_t33_650M_UR50D 4 0"
  "facebook/esm2_t36_3B_UR50D 4 0"
)

for entry in "${jobs[@]}"; do
  read -r -a words <<< "$entry"
  if [ "${#words[@]}" -lt 3 ]; then
    echo "Invalid job: $entry"
    continue
  fi

  MODEL="${words[0]}"
  EPOCHS="${words[1]}"
  ZERO_STAGE="${words[2]}"

  if [ "${#words[@]}" -gt 3 ]; then
    EXTRA_ARGS="$(printf '%s ' "${words[@]:3}")"
    EXTRA_ARGS="${EXTRA_ARGS%" "}"
  else
    EXTRA_ARGS=""
  fi

  SAFE_MODEL_NAME=$(echo "$MODEL" | tr '/:' '__')
  JOB_NAME="${SAFE_MODEL_NAME}-e${EPOCHS}-zs${ZERO_STAGE}"
  OUT="$LOGDIR/${JOB_NAME}-%j.out"
  ERR="$LOGDIR/${JOB_NAME}-%j.err"

  echo "Submitting $JOB_NAME -> model=$MODEL epochs=$EPOCHS zero=$ZERO_STAGE extra='$EXTRA_ARGS'"

  sbatch --job-name="$JOB_NAME" --output="$OUT" --error="$ERR" \
    --export=ALL,MODEL="$MODEL",EPOCHS="$EPOCHS",ZERO_STAGE="$ZERO_STAGE",EXTRA_ARGS="$EXTRA_ARGS" \
    "$TEMPLATE"
done
