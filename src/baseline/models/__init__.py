"""Model registry.

Exposes a single :func:`build_model` factory that maps the CLI model name
to the concrete :class:`baseline.models.base.BaseModel` subclass.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

from baseline.models.base import BaseModel, CheckpointMetrics, NormalizationStats
from baseline.models.deit_tiny import DeiTTiny
from baseline.models.resnet8 import ResNet8
from baseline.models.torchvision_models import (
    DenseNet121,
    EfficientNetB0,
    MobileNetV3,
    ResNet18,
)

_REGISTRY: dict[str, Callable[..., BaseModel]] = {
    "resnet8": ResNet8,
    "resnet18": ResNet18,
    "densenet121": DenseNet121,
    "mobilenetv3": MobileNetV3,
    "efficientnetb0": EfficientNetB0,
    "deit_tiny": DeiTTiny,
}


def build_model(name: str, num_classes: int, pretrained: bool) -> BaseModel:
    """Instantiate the requested architecture.

    Args:
        name: One of the keys in :data:`_REGISTRY` (matches ``--model``).
        num_classes: Output dimensionality required by the head.
        pretrained: Whether to load ImageNet weights. Ignored (with a
            warning) for architectures without pretrained checkpoints.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model {name!r}; choose one of {sorted(_REGISTRY)}.")
    if name == "resnet8" and pretrained:
        warnings.warn(
            "ResNet8 has no pretrained weights; training from scratch instead.",
            stacklevel=2,
        )
        pretrained = False
    return _REGISTRY[name](num_classes=num_classes, pretrained=pretrained)
