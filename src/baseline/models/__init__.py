"""Model registry.

Exposes a single :func:`build_model` factory that maps the CLI model name
to the concrete :class:`baseline.models.base.BaseModel` subclass.
"""

from __future__ import annotations

from collections.abc import Callable

from baseline.models.base import BaseModel, NormalizationStats
from baseline.models.resnet8 import ResNet8
from baseline.models.resnet8_3d import ResNet8_3D


def _lazy_inr(class_name: str) -> Callable[..., BaseModel]:
    """Defer importing the inr2vec models until a model is actually built.

    ``inr2vec.inr_step2.model`` imports :class:`BaseModel` from this package, so
    importing it eagerly here would create a circular import. The factory runs
    only inside :func:`build_model`, by which point this package is fully loaded.
    """

    def factory(**kwargs: object) -> BaseModel:
        from inr2vec.inr_step2 import model as inr_models

        return getattr(inr_models, class_name)(**kwargs)

    return factory


MODELS: dict[str, Callable[..., BaseModel]] = {
    "resnet8": ResNet8,
    "inr2vec_paper": _lazy_inr("Inr2vecPaper"),
    "inr2vec_input": _lazy_inr("Inr2vecInput"),
    "inr2vec_full": _lazy_inr("Inr2vecFull"),
    "resnet8_3d": ResNet8_3D,
}


def build_model(
    name: str, num_classes: int, normalization: NormalizationStats
) -> BaseModel:
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


