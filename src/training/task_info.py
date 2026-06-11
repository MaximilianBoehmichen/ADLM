"""Per-dataset task wiring: maps a MedMNIST dataset flag to its task type,
number of classes, loss function, and prediction helper."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from medmnist import INFO


def get_task_info(dataset_flag: str) -> dict:
    """Return {'task', 'num_classes', 'loss_kind'} for a MedMNIST dataset."""
    info = INFO[dataset_flag]
    task = info["task"]
    num_classes = len(info["label"])
    loss_kind = "bce" if task == "multi-label, binary-class" else "ce"
    return {"task": task, "num_classes": num_classes, "loss_kind": loss_kind}


def build_loss(task: str, pos_weight: torch.Tensor | None,
               device: torch.device) -> nn.Module:
    """Build the appropriate loss module for the given task."""
    if task == "multi-label, binary-class":
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss()


def predict_scores(logits: torch.Tensor, task: str) -> torch.Tensor:
    """Convert raw logits to the score tensor expected by medmnist.Evaluator.

    Evaluator expects:
        binary-class:                shape (N, 2) probabilities (softmax)
        multi-class:                 shape (N, C) probabilities (softmax)
        multi-label, binary-class:   shape (N, C) sigmoid outputs
    """
    if task == "multi-label, binary-class":
        return torch.sigmoid(logits)
    return torch.softmax(logits, dim=-1)


def compute_pos_weight(train_dataset, num_labels: int) -> torch.Tensor:
    """Compute per-label pos_weight = #neg / max(#pos, 1) by scanning a dataset once."""
    pos = torch.zeros(num_labels, dtype=torch.float64)
    n = 0
    for i in range(len(train_dataset)):
        y = train_dataset[i].y.reshape(-1).to(torch.float64)
        assert y.numel() == num_labels, (
            f"expected {num_labels} labels, got {y.numel()} at index {i}"
        )
        pos += y
        n += 1
    neg = n - pos
    pos_safe = pos.clamp(min=1.0)
    return (neg / pos_safe).to(torch.float32)


def compute_or_load_pos_weight(train_dataset, num_labels: int,
                               cache_path: Path) -> torch.Tensor:
    """Load cached pos_weight or compute and cache it."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return torch.load(cache_path, weights_only=True)
    pw = compute_pos_weight(train_dataset, num_labels=num_labels)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pw, cache_path)
    return pw
