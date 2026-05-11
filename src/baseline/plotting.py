"""Matplotlib plots for the JSON history dumped by :mod:`baseline.train`.

Each call to :func:`plot_history` produces three PNG files in
``out_dir``: ``loss.png``, ``accuracy.png`` and ``auc.png``. All curves
share the fractional-epoch x-axis used during training.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _unzip(series: list[tuple[float, float]]) -> tuple[list[float], list[float]]:
    """Split a ``[(x, y), ...]`` list into parallel ``x``/``y`` lists.

    Args:
        series: List of ``(x, y)`` points.

    Returns:
        A tuple ``(xs, ys)`` of parallel lists.
    """
    xs = [point[0] for point in series]
    ys = [point[1] for point in series]
    return xs, ys


def plot_history(history_path: Path, out_dir: Path) -> None:
    """Render loss/accuracy/AUC curves from a ``history.json`` file.

    Args:
        history_path: Path to the JSON file produced by
            :class:`baseline.train.History.dump`.
        out_dir: Directory in which the PNGs are written; created on
            demand.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with history_path.open("r", encoding="utf-8") as f:
        history = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 5))
    if history["train_loss"]:
        xs, ys = _unzip(history["train_loss"])
        ax.plot(xs, ys, label="train (per opt step)", alpha=0.6)
    if history["val_loss"]:
        xs, ys = _unzip(history["val_loss"])
        ax.plot(xs, ys, label="val", marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if history["val_accuracy"]:
        xs, ys = _unzip(history["val_accuracy"])
        ax.plot(xs, ys, label="val", marker="o")
    if history.get("test"):
        ax.axhline(
            history["test"]["accuracy"],
            color="C2",
            linestyle="--",
            label=f"test (final, epoch {history['test']['epoch']})",
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if history["val_auc"]:
        xs, ys = _unzip(history["val_auc"])
        ax.plot(xs, ys, label="val", marker="o")
    if history.get("test"):
        ax.axhline(
            history["test"]["auc"],
            color="C2",
            linestyle="--",
            label=f"test (final, epoch {history['test']['epoch']})",
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("AUC")
    ax.set_title("AUC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "auc.png", dpi=150)
    plt.close(fig)
