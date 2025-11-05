# ESM-2 + VHH fine-tuning (DeepSpeed ZeRO)

Multi-node fine-tuning of ESM-2 on VHH (nanobody) data using DeepSpeed ZeRO. Designed for the PoC footprint (4 x H200 GPUs, 2 TB network SSD, 2 TB shared filesystem). This repo contains data-prep scripts, example datasets and DeepSpeed trainig code tuned for the PoC hardware, and launcher scripts to run distributed training and evaluation.

Run from `/root` with the shared `/mnt/data` mounted:
```
git clone https://github.com/dashabalashova/esm2-vhh-distributed.git
mkdir -p /mnt/data/data/processed/

cp data/processed/vhh_200.tsv /mnt/data/data/processed/vhh_200.tsv
cp data/processed/vhh_2K.tsv /mnt/data/data/processed/vhh_2K.tsv
cp data/processed/vhh_20K.tsv /mnt/data/data/processed/vhh_20K.tsv

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install uv
uv pip install -r requirements.txt
wandb login

chmod +x jobs/job_01.sh
chmod +x jobs/job_02.sh
chmod +x jobs/job_03.sh

jobs/job_01.sh
jobs/job_02.sh
jobs/job_03.sh
```