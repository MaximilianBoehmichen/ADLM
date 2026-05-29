"""Per-epoch metrics aggregator and a thin wrapper around medmnist.Evaluator."""
from __future__ import annotations

import numpy as np
import torch
from medmnist import Evaluator
from torchmetrics.classification import (
    BinaryAUROC, BinaryAccuracy,
    MulticlassAUROC, MulticlassAccuracy,
    MultilabelAUROC, MultilabelAccuracy,
)

from training.task_info import predict_scores


class EpochMetrics:
    """Accumulate logits/targets/loss across an epoch; compute AUROC + ACC + loss."""

    def __init__(self, task: str, num_classes: int, device: torch.device):
        self.task = task
        self.device = device
        if task == "multi-label, binary-class":
            self.auroc = MultilabelAUROC(num_labels=num_classes, average="macro").to(device)
            self.acc = MultilabelAccuracy(num_labels=num_classes, average="macro").to(device)
        elif task == "binary-class":
            self.auroc = BinaryAUROC().to(device)
            self.acc = BinaryAccuracy().to(device)
        else:  # multi-class
            self.auroc = MulticlassAUROC(num_classes=num_classes, average="macro").to(device)
            self.acc = MulticlassAccuracy(num_classes=num_classes, average="micro").to(device)
        self._loss_sum = 0.0
        self._loss_count = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: float):
        scores = predict_scores(logits.detach(), self.task)
        if self.task == "binary-class":
            # Pass probability-of-positive to BinaryAUROC/Accuracy.
            pos_prob = scores[:, 1]
            self.auroc.update(pos_prob, targets.to(self.device).long())
            self.acc.update(pos_prob, targets.to(self.device).long())
        elif self.task == "multi-label, binary-class":
            self.auroc.update(scores, targets.to(self.device).int())
            self.acc.update(scores, targets.to(self.device).int())
        else:
            self.auroc.update(scores, targets.to(self.device).long())
            self.acc.update(scores, targets.to(self.device).long())
        self._loss_sum += float(loss)
        self._loss_count += 1

    def compute(self) -> dict:
        return {
            "loss": self._loss_sum / max(self._loss_count, 1),
            "auroc": float(self.auroc.compute()),
            "accuracy": float(self.acc.compute()),
        }


def run_medmnist_evaluator(all_scores: torch.Tensor, dataset_flag: str,
                            split: str, size: int = 224) -> dict:
    """Run the official MedMNIST Evaluator on accumulated scores.

    Args:
        all_scores: (N, C) probabilities/sigmoid outputs in the order yielded
            by a non-shuffled DataLoader over the split.
        dataset_flag: e.g. "pneumoniamnist", "chestmnist".
        split: "train" | "val" | "test".
        size: image resolution (28 / 64 / 128 / 224); this project uses 224.

    Returns:
        {"AUC": float, "ACC": float}
    """
    evaluator = Evaluator(dataset_flag, split, size=size)
    auc, acc = evaluator.evaluate(all_scores.detach().cpu().numpy().astype(np.float32))
    return {"AUC": float(auc), "ACC": float(acc)}
