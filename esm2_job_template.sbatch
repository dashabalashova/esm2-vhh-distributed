#!/bin/bash
#SBATCH --job-name=esm2-deepspeed
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=140G
#SBATCH --time=24:00:00

set -euo pipefail

export GPUS_PER_NODE=${GPUS_PER_NODE:-1}
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=${MASTER_PORT:-12345}

: "${MODEL:?MODEL must be set}"
: "${EPOCHS:?EPOCHS must be set}"
: "${ZERO_STAGE:?ZERO_STAGE must be set}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "Starting job for model=$MODEL epochs=$EPOCHS zero_stage=$ZERO_STAGE $(TZ='Europe/Amsterdam' date +'%Y-%m-%d %H:%M:%S %Z')"
echo "SLURM_JOBID=$SLURM_JOBID on nodes: $SLURM_JOB_NODELIST"
echo "EXTRA_ARGS='$EXTRA_ARGS'"

if [ -n "${EXTRA_ARGS:-}" ]; then
  srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
  --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
  --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
  esm2_deepspeed.py --model "$MODEL" \
  --zero_stage "$ZERO_STAGE" --epochs "$EPOCHS" --wandb "$EXTRA_ARGS"'
else
  srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
  --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
  --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
  esm2_deepspeed.py --model "$MODEL" \
  --zero_stage "$ZERO_STAGE" --epochs "$EPOCHS" --wandb'
fi
