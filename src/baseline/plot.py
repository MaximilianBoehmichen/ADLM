"""Plotting utilities for evaluation results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import auc, roc_curve  # noqa: E402


def _new_roc_axes(title: str) -> tuple[plt.Figure, plt.Axes]:
    """Create a square ROC figure with limits, labels, and title.

    The chance diagonal and legend are added later by
    :func:`_finalize_roc_plot`, after the caller has plotted its curves,
    so the legend lists the curves before "chance".

    Args:
        title: Axis title.

    Returns:
        The figure and its axes.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")  # square: both axes span 0..1
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    return fig, ax


def _finalize_roc_plot(
    fig: plt.Figure,
    ax: plt.Axes,
    path: Path,
    legend_fontsize: str,
) -> None:
    """Add the chance line and legend, then save and close the figure.

    Args:
        fig: Figure to save.
        ax: Axes the curves were plotted on.
        path: Destination image file.
        legend_fontsize: Matplotlib font-size keyword for the legend.
    """
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.legend(fontsize=legend_fontsize, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


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
    fig, ax = _new_roc_axes(f"ROC per class (macro AUC={macro_auc:.3f})")
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        ax.plot(fpr, tpr, lw=1.2, label=f"{name} (AUC={per_class_auc[i]:.3f})")

    _finalize_roc_plot(fig, ax, path, legend_fontsize="x-small")


def save_overall_roc_plot(
    y_true: np.ndarray,
    y_score: np.ndarray,
    macro_auc: float,
    path: Path,
) -> None:
    """Save a single micro-averaged ROC curve to ``path``.

    Flattens all (sample, class) pairs into one binary problem so the
    whole model is summarized by one curve, unlike the per-class plot.

    Args:
        y_true: Ground-truth labels, shape ``(N, num_classes)``.
        y_score: Predicted scores, shape ``(N, num_classes)``.
        macro_auc: Overall macro AUC (MedMNIST), shown in the title.
        path: Destination image file.
    """
    fpr, tpr, _ = roc_curve(y_true.ravel(), y_score.ravel())
    micro_auc = auc(fpr, tpr)

    fig, ax = _new_roc_axes(f"Overall ROC (macro AUC={macro_auc:.3f})")
    ax.plot(fpr, tpr, lw=1.8, label=f"micro-average (AUC={micro_auc:.3f})")
    _finalize_roc_plot(fig, ax, path, legend_fontsize="small")
