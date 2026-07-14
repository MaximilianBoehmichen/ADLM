"""GNN classifier training entrypoint.

Usage:
    python train_gnn.py --dataset pneumoniamnist --epochs 50
    python train_gnn.py --dataset chestmnist --epochs 50 --batch-size 64
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import summary
from torch_geometric.transforms import Compose
from tqdm import tqdm

ROOT = Path(__file__).parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import wandb
from dataset.gaussian2D import Gaussian2DDataset
from dataset.transforms import (
    BuildKNNGraph, FeatureNormalization,
    encode_rotation, extract_layout,
)
from model.gnn_another import ResNetLikePYGGNN
from model.gnn_other import ResNetLikeGNN
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
    p.add_argument("--dataset", default="chestmnist")
    p.add_argument("--data-root", default="data",
                   help="Parent dir containing {dataset}/{split}/*.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="AdamW weight decay (L2 regularization)")
    p.add_argument("--model", default="pyg", choices=["pyg", "simple"],
                   help="Model architecture: 'pyg' (SumConv basis-decomposed) or 'simple' (KNNConv)")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--num-bases", type=int, default=8)
    p.add_argument("--neighbor-distance", default="l2", choices=["l2", "mahalanobis"])
    p.add_argument("--flow-direction", default="propagate_to", choices=["propagate_to", "propagate_from"])
    p.add_argument("--rff-features", type=int, default=0)
    p.add_argument("--rff-sigma", type=float, default=1.0)
    p.add_argument("--rotate-frame", action="store_true")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap dataset size (e.g. 32 to overfit a single batch)")
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
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=True if sys.platform != "darwin" else False,
    )


def train_one_epoch(model, loader, loss_fn, optim, device, metrics: EpochMetrics,
                    task: str) -> dict:
    model.train()
    n_batches = 0
    t_data, t_step = 0.0, 0.0
    t0 = time.perf_counter()
    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        t_data += time.perf_counter() - t0
        t0_step = time.perf_counter()
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
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        t_step += time.perf_counter() - t0_step
        n_batches += 1
        t0 = time.perf_counter()
    return {"data_ms": t_data / n_batches * 1000,
            "step_ms": t_step / n_batches * 1000}


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

    if args.num_workers is None:
        args.num_workers = 0 if sys.platform == "darwin" else 4
    in_memory = args.in_memory or (args.dataset == "pneumoniamnist")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project=args.wandb_project, entity=args.wandb_entity,
               mode=args.wandb_mode, config={**vars(args), "device": str(device),
                                             "in_memory_resolved": in_memory})

    is_3d = "3d" in args.dataset.lower()
    spatial_dim = 3 if is_3d else 2
    rotation_dim = 4 if is_3d else 1
    layout_dim = 2 * spatial_dim + rotation_dim
    encoded_rot = 2 if not is_3d else rotation_dim
    expected_in_dim = 2 * spatial_dim + encoded_rot + 1


    task_info = get_task_info(args.dataset)
    task = task_info["task"]
    num_classes = task_info["num_classes"]
    class_names = task_info["class_names"]

    knn_graph = BuildKNNGraph(metric=args.neighbor_distance, direction=args.flow_direction)

    pre = [extract_layout, knn_graph] + ([] if is_3d else [encode_rotation])
    transforms = Compose(pre)
    stats_transforms = Compose([encode_rotation]) if not is_3d else None

    data_root = Path(args.data_root) / args.dataset

    t0 = time.perf_counter()
    stats_ds = Gaussian2DDataset(root=data_root, split="train", transforms=stats_transforms, in_memory=False)
    train_ds = Gaussian2DDataset(root=data_root, split="train", transforms=transforms, in_memory=in_memory)
    val_ds = Gaussian2DDataset(root=data_root, split="val", transforms=transforms, in_memory=in_memory)
    test_ds = Gaussian2DDataset(root=data_root, split="test", transforms=transforms, in_memory=in_memory)
    t_load = time.perf_counter() - t0
    print(f"Data loaded in {t_load:.1f}s "
          f"(train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}, "
          f"in_memory={in_memory})")
    wandb.log({"time/data_load_s": t_load})

    if args.max_samples is not None:
        for ds in (train_ds, val_ds, test_ds):
            ds.files = ds.files[:args.max_samples]
            if ds.in_memory:
                ds.data = ds.data[:args.max_samples]

    stats_path = ROOT / "cache" / args.dataset / f"{args.dataset}_feature_stats.pt"
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    cache_valid = False
    if stats_path.exists() and args.max_samples is None:
        saved = torch.load(stats_path, weights_only=True)
        mean, std = saved["mean"], saved["std"]
        if mean.shape[0] == expected_in_dim:
            cache_valid = True
            print(f"Loaded cached feature stats from {stats_path}")
        else:
            print(f"Stale cache (dim {mean.shape[0]} != {expected_in_dim}), recomputing...")
    if not cache_valid:
        print("Computing feature statistics from training set...")
        mean, std = FeatureNormalization.compute_stats(stats_ds)
        if args.max_samples is None:
            torch.save({"mean": mean, "std": std}, stats_path)
    feat_norm = FeatureNormalization(mean, std)
    for ds in (train_ds, val_ds, test_ds):
        old = ds.transforms
        ds.transforms = Compose([old, feat_norm]) if old else feat_norm

    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers)

    k = 9 if spatial_dim == 2 else 27
    if args.model == "pyg":
        model = ResNetLikePYGGNN(
            in_channels=expected_in_dim,
            num_classes=num_classes,
            k=k,
            num_bases=args.num_bases,
            hidden_dim=args.hidden,
            num_layers=args.layers,
            d=spatial_dim,
            mahalanobis=True if args.neighbor_distance == "mahalanobis" else False,
            rff_features=args.rff_features,
            rff_sigma=args.rff_sigma,
            rotate=args.rotate_frame,
        ).to(device)
    else:
        model = ResNetLikeGNN(
            in_channels=expected_in_dim,
            num_classes=num_classes,
            k=k,
        ).to(device)

    N, K = 836, 15
    src = torch.arange(N, device=device).repeat_interleave(K)
    dst = torch.randint(0, N, (N * K,), device=device)
    sample = Batch.from_data_list([
        Data(
            x=torch.randn(N, expected_in_dim, device=device),
            pos=torch.rand(N, spatial_dim, device=device),
            layout=torch.randn(N, layout_dim, device=device),
            edge_index=torch.stack([src, dst]),
            y=torch.tensor([0], device=device),
        )
    ])

    print(summary(model, sample))

    pos_weight = None
    if task == "multi-label, binary-class":
        cache_path = ROOT / "cache" / args.dataset / "pos_weight.pt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pos_weight = compute_or_load_pos_weight(train_ds, num_labels=num_classes,
                                                cache_path=cache_path)

    loss_fn = build_loss(task, pos_weight, device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    best_model: torch.nn.Module = model
    best_auc: float = 0.0

    for epoch in range(args.epochs):
        tm = EpochMetrics(task, num_classes, device)
        vm = EpochMetrics(task, num_classes, device)

        t0 = time.perf_counter()
        iter_times = train_one_epoch(model, train_loader, loss_fn, optim, device,
                                     tm, task)
        t_train = time.perf_counter() - t0

        t0 = time.perf_counter()
        evaluate(model, val_loader, loss_fn, device, vm, task)
        t_val = time.perf_counter() - t0

        train_out = tm.compute()
        val_out = vm.compute()
        train_out.pop("auroc_per_class", None)
        val_per_class = val_out.pop("auroc_per_class", None)
        log = {f"train/{k}": v for k, v in train_out.items()}
        log.update({f"val/{k}": v for k, v in val_out.items()})
        if val_per_class is not None:
            for name, auc in zip(class_names, val_per_class):
                log[f"val_auc/{name}"] = auc
        sched.step()
        log["epoch"] = epoch
        log["lr"] = sched.get_last_lr()[0]
        log["time/train_s"] = t_train
        log["time/val_s"] = t_val
        log["time/epoch_s"] = t_train + t_val
        log["time/iter_data_ms"] = iter_times["data_ms"]
        log["time/iter_step_ms"] = iter_times["step_ms"]
        if device.type == "cuda":
            log["gpu/mem_peak_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
            torch.cuda.reset_peak_memory_stats(device)
        wandb.log(log)
        print(f"epoch {epoch:3d} | "
              f"train loss {train_out['loss']:.4f} auc {train_out['auroc']:.4f} | "
              f"val loss {val_out['loss']:.4f} auc {val_out['auroc']:.4f} | "
              f"{t_train + t_val:.1f}s")

        # Only virtual early stopping
        if val_out['auroc'] > best_auc:
            best_auc = val_out['auroc']
            best_model = copy.deepcopy(model)

    model = best_model

    # Final test metrics. Overall AUC/ACC via the official MedMNIST Evaluator;
    # per-class AUC via torchmetrics (the Evaluator returns only the macro mean).
    # The Evaluator asserts y_score.shape[0] == full official split size; partial
    # preprocessing (smoke runs over a subset) trips it, so fall back in that case.
    test_metrics = EpochMetrics(task, num_classes, device)
    evaluate(model, test_loader, loss_fn, device, test_metrics, task)
    test_out = test_metrics.compute()
    test_per_class = test_out.pop("auroc_per_class", None)

    test_log = {}
    if test_per_class is not None:
        for name, auc in zip(class_names, test_per_class):
            test_log[f"test_auc/{name}"] = auc

    scores = collect_scores(model, test_loader, device, task)
    try:
        test_eval = run_medmnist_evaluator(scores, args.dataset, "test")
        test_log.update({"test/auc": test_eval["AUC"], "test/acc": test_eval["ACC"]})
        print(f"TEST | AUC {test_eval['AUC']:.4f} | ACC {test_eval['ACC']:.4f}")
    except Exception:
        test_log.update({"test/auroc": test_out["auroc"], "test/accuracy": test_out["accuracy"]})
        print(f"TEST (partial split, torchmetrics fallback) | "
              f"AUROC {test_out['auroc']:.4f} | ACC {test_out['accuracy']:.4f}")
    wandb.log(test_log)

    wandb.finish()


if __name__ == "__main__":
    main()
