"""Load a trained baseline checkpoint and run inference.

Typical usage::

    from baseline.inference import load_model, predict
    model, ckpt = load_model(Path("data/models/resnet18/<run>/best_*.pt"), "resnet18")
    probs, targets, metrics = predict(model, dataloader, is_multilabel=True)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from baseline.metrics import EvalMetrics, compute_metrics
from baseline.models import build_model
from baseline.models.base import (
    BaseModel,
    CheckpointMetrics,
    NormalizationStats,
)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


def load_model(
    path: Path,
    model_name: str,
    map_location: str | torch.device = "cpu",
) -> tuple[BaseModel, CheckpointMetrics]:
    """Re-create a model and load weights from ``path``.

    Args:
        path: Path to the ``.pt`` file written by
            :meth:`BaseModel.save`.
        model_name: One of the keys understood by
            :func:`baseline.models.build_model`. Required because the
            registry decides which subclass to instantiate.
        map_location: Forwarded to ``torch.load``.

    Returns:
        The reconstructed model in ``eval`` mode plus its stored
        :class:`CheckpointMetrics`.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model = build_model(
        name=model_name,
        num_classes=payload["num_classes"],
        pretrained=False,
    )
    model.load_state_dict(payload["state_dict"])
    norm = payload["normalization"]
    mean = tuple(float(v) for v in norm["mean"])
    std = tuple(float(v) for v in norm["std"])
    model.normalization = NormalizationStats(mean=mean, std=std)  # type: ignore[arg-type]
    metrics = CheckpointMetrics(**payload["metrics"])
    model.eval()
    return model, metrics


@torch.no_grad()
def predict(
    model: BaseModel,
    loader: DataLoader,
    *,
    is_multilabel: bool,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray, EvalMetrics]:
    """Run inference over a dataloader and report metrics.

    Args:
        model: A model already loaded via :func:`load_model`.
        loader: Dataloader producing ``(image, label)`` tuples that match
            the preprocessing the model was trained with.
        is_multilabel: Whether to compute multi-label metrics.
        device: Device on which to run inference.

    Returns:
        A triple ``(probs, targets, metrics)`` where ``probs`` are the
        post-sigmoid (multi-label) or post-softmax (multi-class)
        probabilities and ``metrics`` are the aggregate values.
    """
    device = torch.device(device)
    model.to(device).eval()
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu().numpy())
        all_targets.append(labels.cpu().numpy())
    logits_np = np.concatenate(all_logits, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    metrics = compute_metrics(
        logits_np,
        targets_np,
        loss=float("nan"),
        is_multilabel=is_multilabel,
    )
    if is_multilabel:
        probs = 1.0 / (1.0 + np.exp(-logits_np))
    else:
        exp = np.exp(logits_np - logits_np.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
    return probs, targets_np, metrics
