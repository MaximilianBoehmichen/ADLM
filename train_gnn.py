"""GNN classifier training entrypoint.

Usage:
    python train_gnn.py --dataset pneumoniamnist --epochs 50
    python train_gnn.py --dataset chestmnist --epochs 50 --batch-size 64
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

# Make `src/` importable without an `src.` prefix (matches existing modules).
ROOT = Path(__file__).parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import wandb
from dataset.gaussian2D import Gaussian2DDataset
from dataset.transforms import encode_rotation, to_undirected_transform
from model.gcn_classifier import GCNClassifier
from training.metrics import EpochMetrics, run_medmnist_evaluator
from training.task_info import (
    build_loss,
    compute_or_load_pos_weight,
    get_task_info,
    predict_scores,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="Train GCN classifier on Gaussian graphs.")
    p.add_argument("--dataset", default="pneumoniamnist",
                   choices=["pneumoniamnist", "chestmnist"])
    p.add_argument("--data-root", default="data",
                   help="Parent dir containing {dataset}/{split}/*.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--in-memory", action="store_true",
                   help="Force in-memory dataset (default: auto by dataset size)")
    p.add_argument("--num-workers", type=int, default=None,
                   help="DataLoader workers; default 0 on macOS, 4 elsewhere.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default="adlm-gnn-classifier")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-mode", default="online",
                   choices=["online", "offline", "disabled"])
    return p.parse_args()


def make_loader(ds, batch_size, shuffle, num_workers):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers)


def train_one_epoch(model, loader, loss_fn, optim, device, metrics: EpochMetrics,
                    task: str):
    model.train()
    for batch in loader:
        batch = batch.to(device)
        optim.zero_grad()
        logits = model(batch)
        targets = batch.y
        if task == "multi-label, binary-class":
            targets = targets.float().reshape(logits.shape)
            loss = loss_fn(logits, targets)
        else:
            targets = targets.long().reshape(-1)
            loss = loss_fn(logits, targets)
        loss.backward()
        optim.step()
        metrics.update(logits, targets, loss=loss.item())


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, metrics: EpochMetrics, task: str):
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        targets = batch.y
        if task == "multi-label, binary-class":
            targets = targets.float().reshape(logits.shape)
            loss = loss_fn(logits, targets)
        else:
            targets = targets.long().reshape(-1)
            loss = loss_fn(logits, targets)
        metrics.update(logits, targets, loss=loss.item())


@torch.no_grad()
def collect_scores(model, loader, device, task: str) -> torch.Tensor:
    model.eval()
    chunks = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        chunks.append(predict_scores(logits, task).cpu())
    return torch.cat(chunks, dim=0)


def main():
    args = parse_args()
    set_seed(args.seed)

    # Resolve auto defaults
    if args.num_workers is None:
        args.num_workers = 0 if sys.platform == "darwin" else 4
    in_memory = args.in_memory or (args.dataset == "pneumoniamnist")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project=args.wandb_project, entity=args.wandb_entity,
               mode=args.wandb_mode, config={**vars(args), "device": str(device),
                                              "in_memory_resolved": in_memory})

    task_info = get_task_info(args.dataset)
    task = task_info["task"]
    num_classes = task_info["num_classes"]

    transforms = Compose([encode_rotation, to_undirected_transform])
    data_root = Path(args.data_root) / args.dataset
    train_ds = Gaussian2DDataset(root=data_root, split="train",
                                  transforms=transforms, in_memory=in_memory)
    val_ds = Gaussian2DDataset(root=data_root, split="val",
                                transforms=transforms, in_memory=in_memory)
    test_ds = Gaussian2DDataset(root=data_root, split="test",
                                 transforms=transforms, in_memory=in_memory)

    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers)

    model = GCNClassifier(in_dim=7, hidden_dim=args.hidden, num_classes=num_classes,
                          num_layers=args.layers, dropout=args.dropout,
                          task=task).to(device)

    pos_weight = None
    if task == "multi-label, binary-class":
        cache_path = Path(args.data_root) / args.dataset / "pos_weight.pt"
        pos_weight = compute_or_load_pos_weight(train_ds, num_labels=num_classes,
                                                 cache_path=cache_path)

    loss_fn = build_loss(task, pos_weight, device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    for epoch in range(args.epochs):
        tm = EpochMetrics(task, num_classes, device)
        vm = EpochMetrics(task, num_classes, device)
        train_one_epoch(model, train_loader, loss_fn, optim, device, tm, task)
        evaluate(model, val_loader, loss_fn, device, vm, task)
        sched.step()

        train_out = tm.compute()
        val_out = vm.compute()
        log = {f"train/{k}": v for k, v in train_out.items()}
        log.update({f"val/{k}": v for k, v in val_out.items()})
        log["epoch"] = epoch
        log["lr"] = sched.get_last_lr()[0]
        wandb.log(log)
        print(f"epoch {epoch:3d} | "
              f"train loss {train_out['loss']:.4f} auc {train_out['auroc']:.4f} | "
              f"val loss {val_out['loss']:.4f} auc {val_out['auroc']:.4f}")

    # Final headline test metric via official MedMNIST Evaluator. Evaluator
    # asserts y_score.shape[0] == full official split size; partial preprocessing
    # (smoke runs over a subset) trips it, so fall back to torchmetrics in that case.
    scores = collect_scores(model, test_loader, device, task)
    try:
        test_eval = run_medmnist_evaluator(scores, args.dataset, "test")
        wandb.log({"test/auc": test_eval["AUC"], "test/acc": test_eval["ACC"]})
        print(f"TEST | AUC {test_eval['AUC']:.4f} | ACC {test_eval['ACC']:.4f}")
    except AssertionError:
        tm = EpochMetrics(task, num_classes, device)
        evaluate(model, test_loader, loss_fn, device, tm, task)
        test_out = tm.compute()
        wandb.log({"test/auroc": test_out["auroc"], "test/accuracy": test_out["accuracy"]})
        print(f"TEST (partial split, torchmetrics fallback) | "
              f"AUROC {test_out['auroc']:.4f} | ACC {test_out['accuracy']:.4f}")

    wandb.finish()


if __name__ == "__main__":
    main()
