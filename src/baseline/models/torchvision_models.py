"""Wrappers around torchvision architectures.

Each class instantiates the torchvision backbone, splits it into a
``backbone`` module (everything except the final classifier) and a
``head`` module (the classifier itself) so that the freeze helpers from
:class:`baseline.models.base.BaseModel` work uniformly.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from baseline.models.base import IMAGENET_NORMALIZATION, BaseModel


class ResNet18(BaseModel):
    """ResNet-18 fine-tunable on MedMNIST inputs."""

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        """Instantiate ResNet-18 with or without ImageNet weights.

        Args:
            num_classes: Output dimensionality of the new head.
            pretrained: Whether to load the torchvision ImageNet
                checkpoint into the backbone.
        """
        super().__init__(num_classes=num_classes, normalization=IMAGENET_NORMALIZATION)
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
        in_features = int(net.fc.in_features)
        net.fc = nn.Identity()
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


class DenseNet121(BaseModel):
    """DenseNet-121 fine-tunable on MedMNIST inputs."""

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        """Instantiate DenseNet-121 with or without ImageNet weights.

        Args:
            num_classes: Output dimensionality of the new head.
            pretrained: Whether to load the torchvision ImageNet
                checkpoint into the backbone.
        """
        super().__init__(num_classes=num_classes, normalization=IMAGENET_NORMALIZATION)
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.densenet121(weights=weights)
        in_features = int(net.classifier.in_features)
        net.classifier = nn.Identity()
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


class MobileNetV3(BaseModel):
    """MobileNet V3-Large fine-tunable on MedMNIST inputs."""

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        """Instantiate MobileNet V3-Large with or without ImageNet weights.

        Args:
            num_classes: Output dimensionality of the new head.
            pretrained: Whether to load the torchvision ImageNet
                checkpoint into the backbone.
        """
        super().__init__(num_classes=num_classes, normalization=IMAGENET_NORMALIZATION)
        weights = (
            models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        )
        net = models.mobilenet_v3_large(weights=weights)
        in_features = int(net.classifier[-1].in_features)
        net.classifier[-1] = nn.Identity()
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


class EfficientNetB0(BaseModel):
    """EfficientNet-B0 fine-tunable on MedMNIST inputs."""

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        """Instantiate EfficientNet-B0 with or without ImageNet weights.

        Args:
            num_classes: Output dimensionality of the new head.
            pretrained: Whether to load the torchvision ImageNet
                checkpoint into the backbone.
        """
        super().__init__(num_classes=num_classes, normalization=IMAGENET_NORMALIZATION)
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        in_features = int(net.classifier[-1].in_features)
        net.classifier[-1] = nn.Identity()
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
