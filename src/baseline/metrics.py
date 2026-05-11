"""Evaluation metrics for MedMNIST baselines.

The training loop currently targets the multi-label ChestMNIST setting,
so accuracy is computed at a 0.5 sigmoid threshold per class and AUC is
the class-averaged ROC AUC. ``overall`` accuracy reports the mean
per-class accuracy (i.e. element-wise correctness averaged over labels
and samples), and ``overall`` AUC reports the macro-averaged class AUC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass(slots=True)
class EvalMetrics:
    """Per-split metrics produced by :func:`compute_metrics`.

    Attributes:
        loss: Mean loss over the split.
        accuracy: Overall accuracy (mean over classes and samples).
        auc: Macro-averaged ROC AUC.
        per_class_accuracy: Accuracy for every class.
        per_class_auc: ROC AUC for every class. ``NaN`` entries indicate
            classes that had a single label value in the split and for
            which AUC is undefined.
    """

    loss: float
    accuracy: float
    auc: float
    per_class_accuracy: list[float]
    per_class_auc: list[float]


def compute_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    loss: float,
    *,
    is_multilabel: bool,
    threshold: float = 0.5,
) -> EvalMetrics:
    """Compute split-level metrics from accumulated logits and labels.

    Args:
        logits: ``(N, C)`` array of raw model outputs.
        targets: ``(N, C)`` float array (multi-label) or ``(N,)`` int
            array (multi-class).
        loss: Mean loss already computed by the caller.
        is_multilabel: Whether the dataset is multi-label.
        threshold: Sigmoid threshold used to binarise multi-label
            predictions for accuracy reporting.

    Raises:
        NotImplementedError: For multi-class data; only the multi-label
            path has been wired up so far.
    """
    if not is_multilabel:
        raise NotImplementedError(
            "Only multi-label metrics are implemented. Extend compute_metrics() "
            "to handle the multi-class case before enabling other datasets.",
        )
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(np.int32)
    targets_int = targets.astype(np.int32)

    per_class_acc = (preds == targets_int).mean(axis=0).tolist()
    accuracy = float(np.mean(per_class_acc))

    per_class_auc: list[float] = []
    for c in range(targets.shape[1]):
        if len(np.unique(targets_int[:, c])) < 2:
            per_class_auc.append(float("nan"))
            continue
        per_class_auc.append(float(roc_auc_score(targets_int[:, c], probs[:, c])))
    valid = [a for a in per_class_auc if not np.isnan(a)]
    auc = float(np.mean(valid)) if valid else float("nan")

    return EvalMetrics(
        loss=loss,
        accuracy=accuracy,
        auc=auc,
        per_class_accuracy=[float(a) for a in per_class_acc],
        per_class_auc=per_class_auc,
    )
