"""Evaluation: score a model on a split with the MedMNIST evaluator."""

from dataclasses import dataclass

import numpy as np
import torch
from medmnist import Evaluator
from medmnist.evaluator import getACC, getAUC
from torch import nn
from torch.utils.data import DataLoader

from baseline.cli import Config
from baseline.data import DatasetInfo
from baseline.models import BaseModel


@dataclass(slots=True)
class EvalResult:
    """Metrics and raw predictions produced by :func:`evaluate`."""

    auc: float
    acc: float
    loss: float
    per_class_auc: list[float]
    per_class_acc: list[float]
    y_true: np.ndarray
    y_score: np.ndarray


@torch.no_grad()
def evaluate(
    model: BaseModel,
    loader: DataLoader,
    device: torch.device,
    config: Config,
    dataset_info: DatasetInfo,
    split: str,
    criterion: nn.Module,
) -> EvalResult:
    """Run the model over a split and score it with the MedMNIST evaluator.

    Overall AUC/ACC come from :class:`medmnist.Evaluator`; per-class values
    use MedMNIST's ``getAUC``/``getACC`` on each label as a binary problem.

    Args:
        model: The model to evaluate.
        loader: A loader for ``split`` (must be unshuffled).
        device: Device to run inference on.
        config: Training configuration (for image size and data root).
        dataset_info: Static dataset metadata.
        split: MedMNIST split name (``"val"`` or ``"test"``).

    Returns:
        Overall and per-class metrics plus the raw labels/scores.
    """
    model.eval()
    scores: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    loss_sum = 0.0
    n_batches = 0

    for images, labels in loader:
        labels = labels.to(device)
        labels = labels.float() if dataset_info.is_multilabel else labels.reshape(-1).long()

        logits = model(images.to(device))
        loss_sum += criterion(logits, labels).item()
        n_batches += 1

        if dataset_info.is_multilabel:
            scores.append(torch.sigmoid(logits).cpu().numpy())
        else:
            scores.append(torch.softmax(logits, dim=1).cpu().numpy())

        targets.append(labels.cpu().numpy())

    y_score = np.concatenate(scores)
    y_true = np.concatenate(targets)

    evaluator = Evaluator(
        dataset_info.name,
        split,
        size=config.image_size,
        root=str(config.output_dir / "medmnist_cache"),
    )
    overall_auc, overall_acc = evaluator.evaluate(y_score)

    if dataset_info.is_multilabel:
        per_class_auc = [
            getAUC(y_true[:, i], y_score[:, i], "binary-class")
            for i in range(dataset_info.num_classes)
        ]
        per_class_acc = [
            getACC(y_true[:, i], y_score[:, i], "binary-class")
            for i in range(dataset_info.num_classes)
        ]
    else:
        y_true_flat = y_true.reshape(-1)
        per_class_auc = [
            getAUC((y_true_flat == i).astype(int), y_score[:, i], "binary-class")
            for i in range(dataset_info.num_classes)
        ]
        per_class_acc = [
            getACC((y_true_flat == i).astype(int), y_score[:, i], "binary-class")
            for i in range(dataset_info.num_classes)
        ]

    return EvalResult(
        auc=float(overall_auc),
        acc=float(overall_acc),
        loss=loss_sum / n_batches,
        per_class_auc=per_class_auc,
        per_class_acc=per_class_acc,
        y_true=y_true,
        y_score=y_score,
    )
