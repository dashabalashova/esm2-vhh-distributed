# ESM-2 + VHH fine-tuning (DeepSpeed ZeRO)

Multi-node fine-tuning of ESM-2 on VHH (nanobody) data using DeepSpeed ZeRO. Designed for the PoC footprint (4 x H200 GPUs, 2 TB network SSD, 2 TB shared filesystem). This repo contains data-prep scripts, example datasets and DeepSpeed trainig code tuned for the PoC hardware, and launcher scripts to run distributed training and evaluation.