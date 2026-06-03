from __future__ import annotations
import warnings
from collections.abc import Callable
from baseline.models.base import BaseModel

# get the original ResNet8 for comparison
from baseline.models.resnet8 import ResNet8
# get my image-net-styled ResNet8 for comparison
from baseline.models.image_net_styled_resnet8 import ImageNetResNet8

_REGISTRY: dict[str, Callable[..., BaseModel]] = {
    "resnet8": ResNet8,                  # corresponding to the original ResNet8
    "resnet8_imagenet": ImageNetResNet8, # corresponding to my ImageNet-styled ResNet8
}

def build_model(name: str, num_classes: int, pretrained: bool) -> BaseModel:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model {name!r}; choose one of {sorted(_REGISTRY)}.")
    if name in ["resnet8", "resnet8_imagenet"] and pretrained:
        warnings.warn(f"{name} has no pretrained weights; training from scratch instead.")
        pretrained = False
    return _REGISTRY[name](num_classes=num_classes, pretrained=pretrained)