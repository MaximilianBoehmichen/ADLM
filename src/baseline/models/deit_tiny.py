"""DeiT-Tiny wrapper built on top of ``timm``.

DeiT-Tiny is the ViT architecture we expose for this baseline; it has a
parameter count similar to ResNet-18 (~5.7M) and ships with strong
ImageNet weights via :mod:`timm`.
"""

from __future__ import annotations

import timm
import torch
from torch import nn

from baseline.models.base import IMAGENET_NORMALIZATION, BaseModel


class DeiTTiny(BaseModel):
    """DeiT-Tiny / patch-16 / 224 wrapper."""

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        """Instantiate ``deit_tiny_patch16_224`` from timm.

        Args:
            num_classes: Output dimensionality of the new head.
            pretrained: Whether to load the timm ImageNet checkpoint
                into the backbone.
        """
        super().__init__(num_classes=num_classes, normalization=IMAGENET_NORMALIZATION)
        net = timm.create_model(
            "deit_tiny_patch16_224",
            pretrained=pretrained,
            num_classes=0,
        )
        in_features = int(net.num_features)
        self.backbone = net
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the backbone followed by the classification head.

        Args:
            x: Input batch shaped ``(B, 3, H, W)``.

        Returns:
            Raw logits of shape ``(B, num_classes)``.
        """
        return self.head(self.backbone(x))
