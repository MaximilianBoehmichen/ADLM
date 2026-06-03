"""Plotting utilities for evaluation results."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402


def save_roc_plot(
        y_true: np.ndarray,
        y_score: np.ndarray,
        class_names: list[str],
        per_class_auc: list[float],
        macro_auc: float,
        path: Path,
) -> None:
    """Save a square per-class ROC plot to ``path``.

    Args:
        y_true: Ground-truth labels, shape ``(N, num_classes)``.
        y_score: Predicted scores, shape ``(N, num_classes)``.
        class_names: Human-readable label names for the legend.
        per_class_auc: AUC per label, shown in the legend.
        macro_auc: Overall macro AUC, shown in the title.
        path: Destination image file.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        ax.plot(fpr, tpr, lw=1.2, label=f"{name} (AUC={per_class_auc[i]:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")  # square: both axes span 0..1
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC per class (macro AUC={macro_auc:.3f})")
    ax.legend(fontsize="x-small", loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
