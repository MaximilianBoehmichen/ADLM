"""Model registry.

Exposes a single :func:`build_model` factory that maps the CLI model name
to the concrete :class:`baseline.models.base.BaseModel` subclass.
"""

from __future__ import annotations

from collections.abc import Callable

from baseline.models.base import BaseModel, CheckpointMetrics, NormalizationStats
from baseline.models.resnet8 import ResNet8

MODELS: dict[str, Callable[..., BaseModel]] = {
    "resnet8": ResNet8,
}


def build_model(name: str, num_classes: int, normalization: NormalizationStats) -> BaseModel:
    """Instantiate the requested architecture.

    Args:
        name: One of the keys in :data:`_REGISTRY` (matches ``--model``).
        num_classes: Output dimensionality required by the head.
        normalization: The NormalizationStats.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    if name not in MODELS:
        raise KeyError(f"Unknown model {name!r}; choose one of {sorted(MODELS)}.")

    return MODELS[name](
        num_classes=num_classes,
        normalization=normalization,
        in_channels=1,  # hardcoded for now
    )
