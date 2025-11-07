# ESM-2 + VHH fine-tuning (DeepSpeed ZeRO)
Multi-node fine-tuning of ESM-2 on VHH (nanobody) data using DeepSpeed ZeRO. Designed for the PoC footprint (4 x H200 GPUs, 2 TB network SSD, 2 TB shared filesystem). This repo contains data-prep scripts, example datasets and DeepSpeed trainig code tuned for the PoC hardware, and launcher scripts to run distributed training and evaluation.

## Quickstart — multi-GPU Slurm training and results monitoring

### Infrastructure
Create Shared Filesystems: `https://console.nebius.com/<Project_ID>/filesystems` for `filestore_jail` (1TB) and `filestore_jail_submounts` (1TB). [Launch Soperator cluster](https://github.com/nebius/nebius-solutions-library/tree/main/soperator): Terraform variables are located at `infra/terraform.tfvars`. The provisioned cluster includes 4 x NVIDIA H200 GPUs for high-throughput multi‑GPU distributed model training. Storage layout: Shared filesystem (1 TB) mounted on controller, worker, and login nodes and Persistent volume (1 TB) mounted at `/mnt/data` for datasets and training outputs.

### Prepare environment

from `/root` on Slurm login node
```
git clone https://github.com/dashabalashova/esm2-vhh-distributed.git
cd esm2-vhh-distributed
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install uv
uv pip install -r requirements.txt
wandb login
```

### Copy data
from `/root` to `/mnt/data`
```
mkdir -p /mnt/data/data/processed/
cp data/processed/vhh_200.tsv /mnt/data/data/processed/vhh_200.tsv
cp data/processed/vhh_2K.tsv /mnt/data/data/processed/vhh_2K.tsv
cp data/processed/vhh_20K.tsv /mnt/data/data/processed/vhh_20K.tsv
```

### Run training
all ESM-2 models (data=2K)
```
chmod +x jobs/job_01.sh
jobs/job_01.sh
```
ESM-2 3B model – various batch sizes (data=2K)
```
chmod +x jobs/job_02.sh
jobs/job_02.sh
```
ESM-2 3B model (data=20K)
```
chmod +x jobs/job_03.sh
jobs/job_03.sh
```
ESM-2 3B model (data=20K) on SSD NRD
```
chmod +x scripts/copy_data.sh
chmod +x jobs/job_04.sh
scripts/copy_data.sh
jobs/job_04.sh
```

### Monitoring & logs

W&B: enable with `--wandb --wandb_project <project> --wandb_run_name <name>`. The script logs losses, ROC AUC, epoch time GPU/CPU performance. Ensure wandb login was done.

Logs: `/root/esm2-vhh-distributed/logs/esm2-<JOB_ID>.out`

## Key recommendations

ZeRO stage:

Stage 1–2: good tradeoff for 4xH200 (memory saved, moderate overhead). Stage 3 gives maximum memory reduction but add complexity / IO overhead – try only if you need to squeeze larger batch/model. `--zero_stage` flag controls this.

Mixed precision:

Use `--fp16` to enable fp16 in DeepSpeed when model & dataset permit – saves memory and speeds up training on H200. Try `--fp16` with caution for numeric stability; monitor training loss.

Batching:

Two fields in the script: `--batch_size` (per-device effective microbatch) and `--batch_size_ds` (DeepSpeed train_batch_size). See `jobs/job_02.sh` for example BS combos.

Checkpoints & output:

Script saves best model & tokenizer to `args.output_dir` / `args.wandb_run_name` on rank 0; final save to `args.output_dir` at the end. Default `--output_dir=/mnt/data/outputs`. Ensure `/mnt/data` has sufficient space.