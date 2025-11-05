#!/bin/bash

#SBATCH --job-name=esm2
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

export GPUS_PER_NODE=${GPUS_PER_NODE:-1}
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=${MASTER_PORT:-12345}
CMD="${CMD:-}"

echo "Starting SLURM_JOBID=$SLURM_JOBID on nodes: $SLURM_JOB_NODELIST – $(TZ='Europe/Amsterdam' date +'%Y-%m-%d %H:%M:%S %Z')"
echo "CMD='$CMD'"

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
  --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
  --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
  src/esm2_deepspeed.py ${CMD}'