"""Base class for baseline architectures.

Every architecture exposed in the registry is wrapped in a thin subclass of
:class:`BaseModel` that owns its backbone, classification head, the
normalisation statistics that match its pretrained weights and a couple of
helpers (freeze/unfreeze, parameter groups, save/load with metrics) that the
training loop relies on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(slots=True)
class NormalizationStats:
    """Channel-wise mean/std used to normalise model inputs."""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]


# ImageNet statistics shared by every torchvision/timm model used here.
IMAGENET_NORMALIZATION = NormalizationStats(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)


@dataclass(slots=True)
class CheckpointMetrics:
    """Metric values recorded inside a checkpoint file.

    Attributes:
        epoch: Epoch index (1-based) at which the checkpoint was saved.
        val_auc: Overall AUC on the per-epoch evaluation split.
        val_accuracy: Overall accuracy on the per-epoch evaluation split.
        training_time_seconds: Wall-clock seconds elapsed between the
            start of training and the moment this checkpoint was saved.
        device: String identifier of the device that produced the
            weights (e.g. ``"cuda:0"`` or ``"cpu"``).
        test_auc: Final AUC on the held-out split (``None`` until the run
            has completed).
        test_accuracy: Final accuracy on the held-out split.
        extra: Free-form payload (per-class metrics, hparams, ...).
    """

    epoch: int
    val_auc: float
    val_accuracy: float
    training_time_seconds: float = 0.0
    device: str = "cpu"
    test_auc: float | None = None
    test_accuracy: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class BaseModel(nn.Module):
    """Common interface for every baseline architecture.

    Subclasses must populate :attr:`backbone` (everything except the head)
    and :attr:`head` (the final classification layer) so that the freeze
    helpers can operate on the right parameters.

    Attributes:
        normalization: Mean/std the model expects on its inputs.
        num_classes: Output dimensionality of the classification head.
    """

    backbone: nn.Module
    head: nn.Module

    def __init__(
        self,
        num_classes: int,
        normalization: NormalizationStats = IMAGENET_NORMALIZATION,
    ) -> None:
        """Initialise the empty wrapper; subclasses populate the modules.

        Args:
            num_classes: Output dimensionality of the classification head.
            normalization: Channel-wise mean/std the model expects on
                its inputs. Defaults to ImageNet statistics, which match
                every torchvision/timm pretrained backbone used here.
        """
        super().__init__()
        self.num_classes = num_classes
        self.normalization = normalization

    def freeze_backbone(self) -> None:
        """Freeze every parameter outside of the classification head."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Re-enable gradients on every backbone parameter."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return the parameters that currently require gradients."""
        return [p for p in self.parameters() if p.requires_grad]

    def save(self, path: Path, metrics: CheckpointMetrics) -> None:
        """Persist weights, metrics and metadata to ``path``.

        The file format is a plain ``torch.save`` of a dictionary so that
        :meth:`load` (and downstream inference scripts) can recover both
        the state dict and the validation metrics that motivated saving.

        Args:
            path: Destination ``.pt`` file. Parent directories are
                created on demand.
            metrics: Metric values recorded alongside the weights.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "metrics": metrics.__dict__,
                "num_classes": self.num_classes,
                "normalization": self.normalization.__dict__,
                "class_name": type(self).__name__,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: Path,
        map_location: str | torch.device = "cpu",
    ) -> tuple[BaseModel, CheckpointMetrics]:
        """Load weights from ``path`` together with the stored metrics.

        Args:
            path: Checkpoint produced by :meth:`save`.
            map_location: Forwarded to :func:`torch.load`.

        Returns:
            The reconstructed model plus the
            :class:`CheckpointMetrics` recorded at save time.
        """
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(num_classes=payload["num_classes"])
        model.load_state_dict(payload["state_dict"])
        norm = payload["normalization"]
        mean = tuple(float(v) for v in norm["mean"])
        std = tuple(float(v) for v in norm["std"])
        model.normalization = NormalizationStats(mean=mean, std=std)  # type: ignore[arg-type]
        metrics = CheckpointMetrics(**payload["metrics"])
        return model, metrics
