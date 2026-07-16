"""Linear baseline diagnostic: can simple pooled Gaussian stats predict labels?

Extracts per-image feature vectors from Gaussian graphs (no message passing,
just global mean/std/min/max of each node feature column), then fits a
logistic regression and reports train vs val AUC.

If this doesn't generalize -> representation is the bottleneck, not the GNN.
If this does generalize -> the GNN architecture is the problem.

Usage:
    python scripts/linear_baseline.py --data-root data --dataset chestmnist
    python scripts/linear_baseline.py --data-root /vol/miltank/users/hdo/data --dataset chestmnist
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from dataset.gaussian2D import Gaussian2DDataset
from dataset.transforms import encode_rotation


def extract_features(dataset) -> tuple[np.ndarray, np.ndarray]:
    """Pool each graph to a fixed-size vector: [mean, std, min, max] per feature column."""
    X, Y = [], []
    for i in range(len(dataset)):
        d = dataset[i]
        x = d.x  # [N, F]
        vec = torch.cat([
            x.mean(dim=0),
            x.std(dim=0),
            x.min(dim=0).values,
            x.max(dim=0).values,
        ]).numpy()
        X.append(vec)
        Y.append(d.y.numpy().ravel())
    return np.stack(X), np.stack(Y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data")
    p.add_argument("--dataset", default="chestmnist",
                   choices=["chestmnist", "pneumoniamnist"])
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    data_root = Path(args.data_root) / args.dataset
    in_memory = args.dataset == "pneumoniamnist"

    train_ds = Gaussian2DDataset(root=data_root, split="train",
                                  transforms=encode_rotation, in_memory=in_memory)
    val_ds = Gaussian2DDataset(root=data_root, split="val",
                                transforms=encode_rotation, in_memory=in_memory)

    if args.max_samples:
        train_ds.files = train_ds.files[:args.max_samples]
        val_ds.files = val_ds.files[:args.max_samples]
        if train_ds.in_memory:
            train_ds.data = train_ds.data[:args.max_samples]
            val_ds.data = val_ds.data[:args.max_samples]

    print(f"Extracting features from {len(train_ds)} train / {len(val_ds)} val graphs...")
    X_train, Y_train = extract_features(train_ds)
    X_val, Y_val = extract_features(val_ds)
    print(f"Feature vector shape: {X_train.shape[1]} dims")

    # Standardize using train stats
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0).clip(min=1e-6)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    is_multilabel = Y_train.ndim > 1 and Y_train.shape[1] > 1

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.multiclass import OneVsRestClassifier

    if is_multilabel:
        clf = OneVsRestClassifier(
            LogisticRegression(max_iter=1000, C=1.0), n_jobs=-1
        )
        clf.fit(X_train, Y_train)
        train_score = roc_auc_score(Y_train, clf.predict_proba(X_train), average="macro")
        val_score = roc_auc_score(Y_val, clf.predict_proba(X_val), average="macro")
    else:
        Y_train_flat = Y_train.ravel()
        Y_val_flat = Y_val.ravel()
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_train, Y_train_flat)
        train_score = roc_auc_score(Y_train_flat, clf.predict_proba(X_train)[:, 1])
        val_score = roc_auc_score(Y_val_flat, clf.predict_proba(X_val)[:, 1])

    print(f"\n--- Logistic Regression on pooled Gaussian features ---")
    print(f"Train AUC: {train_score:.4f}")
    print(f"Val   AUC: {val_score:.4f}")
    print(f"Gap:       {train_score - val_score:.4f}")

    if val_score < 0.55:
        print("\n[!] Val AUC near chance -> representation likely lacks discriminative signal.")
        print("    Consider: more Gaussians, more optimization epochs, or different features.")
    elif train_score - val_score > 0.1:
        print("\n[!] Large train-val gap even for linear model -> distribution shift or label noise.")
    else:
        print("\n[OK] Linear model generalizes -> signal exists, GNN architecture is the bottleneck.")


if __name__ == "__main__":
    main()