#!/usr/bin/env python3
# ESM-2 + VHH + DeepSpeed ZeRO
import argparse
import os
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import deepspeed
from deepspeed.accelerator import get_accelerator

try:
    import wandb

    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False


class SeqTSVDataset(Dataset):
    def __init__(self, path, seq_col="sequence", tgt_col="target"):
        """Load a TSV file and store sequences and integer targets."""
        df = pd.read_csv(path, sep="\t", header=0)
        self.seq = df[seq_col].astype(str).tolist()
        self.tgt = df[tgt_col].astype(int).tolist()

    def __len__(self):
        """Return number of examples."""
        return len(self.seq)

    def __getitem__(self, i):
        """Return a single example as a dict."""
        return {"sequence": self.seq[i], "label": self.tgt[i]}


def collate_fn(tokenizer, batch, max_length):
    """Tokenize a list of examples and return tensors including labels."""
    seqs = [b["sequence"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    enc = tokenizer(
        seqs,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc["labels"] = labels
    return enc


def set_seed(seed):
    """Set random seeds for reproducibility across numpy, random, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gather_across_ranks(obj):
    """Gather a Python object from all distributed ranks (returns list)."""
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, obj)
    return gathered


def concat_gathered_lists(gathered: List[Tuple[np.ndarray, np.ndarray]]):
    """Concatenate lists of (scores, labels) from gathered ranks into arrays."""
    scores_list = []
    labels_list = []
    for item in gathered:
        if item is None:
            continue
        sc, lb = item
        if sc is None or lb is None:
            continue
        scores_list.append(np.asarray(sc))
        labels_list.append(np.asarray(lb))
    if len(scores_list) == 0:
        return np.array([]), np.array([])
    return np.concatenate(scores_list, axis=0), np.concatenate(labels_list, axis=0)


def train(args):
    """Main training loop using DeepSpeed: data loading, training, evaluation, and saving."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    deepspeed.init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    get_accelerator().set_device(local_rank)

    if torch.distributed.get_rank() == 0:
        print(f"Experiment args: {args}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ds = SeqTSVDataset(args.data)
    if args.val_split > 0.0:
        train_idx, val_idx = train_test_split(
            list(range(len(ds))),
            test_size=args.val_split,
            random_state=args.seed,
            stratify=ds.tgt,
        )
        train_ds = Subset(ds, train_idx)
        val_ds = Subset(ds, val_idx)
    else:
        train_ds = ds
        val_ds = None

    ds_config = {
        "train_batch_size": args.batch_size_ds,
        "gradient_clipping": args.gradient_clipping,
        "fp16": {"enabled": bool(args.fp16)},
        "zero_optimization": {"stage": int(args.zero_stage)},
    }

    engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        config_params=ds_config,
    )

    device = engine.device if hasattr(engine, "device") else torch.device(
        f"cuda:{local_rank}"
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    pin_memory = True

    train_sampler = DistributedSampler(train_ds) if world_size > 1 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=lambda b: collate_fn(tokenizer, b, args.max_length),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size or args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(tokenizer, b, args.max_length),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    use_wandb = (
        args.wandb
        and _HAS_WANDB
        and int(os.environ.get("RANK", "0")) == 0
    )

    if args.wandb and not _HAS_WANDB and int(os.environ.get("RANK", "0")) == 0:
        print("wandb requested but import failed — continuing without wandb.")
        use_wandb = False

    if use_wandb:
        wandb_run_name = ''.join(["esm2-",args.model.split("_")[2],"-e",
                                 str(args.epochs),"-zs" + str(args.zero_stage)])
        wandb.init(project=args.wandb_project, name=wandb_run_name, config=vars(args))

    best_val_auc = None
    engine.train()
    step = 0
    for epoch in range(args.epochs):
        running_loss = 0.0
        epoch_loss_sum = 0.0
        epoch_steps = 0
        local_train_scores = []
        local_train_labels = []
        epoch_start_time = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            out = engine(**batch)
            loss = out.loss
            loss_val = loss.item()
            running_loss += loss_val
            epoch_loss_sum += loss_val
            epoch_steps += 1
            logits = out.logits
            probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            labels_np = batch["labels"].detach().cpu().numpy()
            local_train_scores.append(probs)
            local_train_labels.append(labels_np)

            engine.backward(loss)
            engine.step()

            if (
                int(os.environ.get("RANK", "0")) == 0
                and (step + 1) % args.log_interval == 0
            ):
                avg_loss = running_loss / (args.log_interval if args.log_interval > 0 else 1)
                print(
                    f"[step {step}] epoch {epoch} step {i} loss={loss_val:.4f} "
                    f"avg_loss={avg_loss:.4f}"
                )
                if use_wandb:
                    wandb.log({"train/log_avg_loss": avg_loss}, step=step)
                running_loss = 0.0
            step += 1
        epoch_train_end_time = time.time()
        train_epoch_time = epoch_train_end_time - epoch_start_time

        train_epoch_avg_loss = (
            epoch_loss_sum / epoch_steps if epoch_steps > 0 else float("nan")
        )
        train_scores_np = np.array([])
        train_labels_np = np.array([])
        local_scores_concat = np.concatenate(local_train_scores, axis=0)
        local_labels_concat = np.concatenate(local_train_labels, axis=0)
        gathered = gather_across_ranks((local_scores_concat, local_labels_concat))
        train_scores_np, train_labels_np = concat_gathered_lists(gathered)
        train_epoch_auc = float(roc_auc_score(train_labels_np, train_scores_np))
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                f"Epoch {epoch} train avg loss: {train_epoch_avg_loss:.4f}, "
                f"train ROC AUC: {train_epoch_auc}, train time: {train_epoch_time:.3f}s"
            )
        if use_wandb:
            wandb.log(
                {
                    "train/avg_loss": train_epoch_avg_loss,
                    "train/roc_auc": train_epoch_auc,
                    "train/epoch_time_sec": train_epoch_time,
                },
                step=step,
            )

        engine.eval()
        all_labels = []
        all_scores = []
        all_losses = []
        val_start_time = time.time()
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
                }
                outputs = engine(**batch)
                logits = outputs.logits
                if logits is not None:
                    probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                    labels = batch["labels"].detach().cpu().numpy()
                    all_scores.append(probs)
                    all_labels.append(labels)
                loss_val_tensor = outputs.loss
                lval = (
                    loss_val_tensor.item()
                    if hasattr(loss_val_tensor, "item")
                    else float(loss_val_tensor)
                )
                all_losses.append(np.array([lval]) * labels.shape[0])

            val_end_time = time.time()
            local_scores = np.concatenate(all_scores, axis=0)
            local_labels = np.concatenate(all_labels, axis=0)
            val_epoch_time = val_end_time - val_start_time
            gathered = gather_across_ranks((local_scores, local_labels))
            val_scores_np, val_labels_np = concat_gathered_lists(gathered)
            val_auc = float(roc_auc_score(val_labels_np, val_scores_np))
            local_losses = np.concatenate(all_losses, axis=0)
            gathered_losses = gather_across_ranks(local_losses)
            total_losses = np.concatenate([g for g in gathered_losses], axis=0)
            val_avg_loss = float(np.mean(total_losses))

            if int(os.environ.get("RANK", "0")) == 0:
                print(
                    f"Epoch {epoch} validation ROC AUC: {val_auc}  avg loss: {val_avg_loss}  "
                    f"val time: {val_epoch_time:.3f}s"
                )
            if use_wandb:
                wandb.log(
                    {
                        "val/roc_auc": val_auc,
                        "val/avg_loss": val_avg_loss,
                        "val/epoch_time_sec": val_epoch_time,
                    },
                    step=step,
                )

            if int(os.environ.get("RANK", "0")) == 0:
                if (
                    best_val_auc is None
                    or (val_auc is not None and not np.isnan(val_auc) and val_auc > best_val_auc)
                ):
                    best_val_auc = val_auc
                    out = Path(args.output_dir+'/'+wandb_run_name)
                    out.mkdir(parents=True, exist_ok=True)
                    model_to_save = engine.module if hasattr(engine, "module") else engine                    
                    model_to_save.save_pretrained(out)
                    tokenizer.save_pretrained(out)
                    print(f"Saved best model/tokenizer to {out} (val_auc={val_auc})")

            engine.train()

    if int(os.environ.get("RANK", "0")) == 0:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        model_to_save = engine.module if hasattr(engine, "module") else engine
        model_to_save.save_pretrained(out)
        tokenizer.save_pretrained(out)
        print(f"Saved final model/tokenizer to {out}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--data", type=str, default="data/llama_2K.tsv")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--batch_size_ds", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument(
        "--zero_stage", type=int, default=0, choices=[0, 1, 2, 3]
    )
    p.add_argument(
        "--fp16", action="store_true", help="enable fp16 in deepspeed config"
    )
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--gradient_clipping", type=float, default=1.0)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--wandb", action="store_true", help="enable wandb logging (if installed)"
    )
    p.add_argument("--wandb_project", type=str, default="esm2-deepspeed")
    p.add_argument("--wandb_run_name", type=str, default=None)

    args = p.parse_args()

    set_seed(args.seed)
    train(args)

    try:
        dist.barrier()
    except Exception:
        pass
    try:
        dist.destroy_process_group()
    except Exception:
        pass
