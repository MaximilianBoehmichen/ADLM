"""Custom ResNet-8 implementation.

Torchvision does not ship a ResNet-8, so a small CIFAR-style residual
network is implemented here. The network has three stages of one
``BasicBlock`` each and is therefore 8 weight layers deep (initial conv +
6 conv layers in residual blocks + classification linear).

The model is randomly initialised and does not support pretraining; the
``pretrained`` flag in the factory is consequently ignored (warning is
emitted by the registry).
"""

from __future__ import annotations

import torch
from torch import nn

from baseline.models.base import IMAGENET_NORMALIZATION, BaseModel


class _BasicBlock(nn.Module):
    """A two-conv residual block with an optional downsampling shortcut."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        """Build the block.

        Args:
            in_channels: Number of channels in the input feature map.
            out_channels: Number of channels produced by both convs.
            stride: Stride of the first conv; a value greater than one
                triggers a 1x1 strided projection on the shortcut.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual block.

        Args:
            x: Input feature map of shape ``(B, in_channels, H, W)``.

        Returns:
            Output feature map of shape ``(B, out_channels, H', W')``.
        """
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return torch.relu(out)


class _ResNet8Backbone(nn.Module):
    """The convolutional trunk of ResNet-8 (everything before the head)."""

    def __init__(self) -> None:
        """Set up stem and three residual stages."""
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.stage1 = _BasicBlock(16, 16, stride=1)
        self.stage2 = _BasicBlock(16, 32, stride=2)
        self.stage3 = _BasicBlock(32, 64, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_features = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the backbone feature vector.

        Args:
            x: Input batch shaped ``(B, 3, H, W)``.

        Returns:
            Feature tensor of shape ``(B, 64)``.
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


class ResNet8(BaseModel):
    """Small CIFAR-style ResNet with 8 weight layers."""

    def __init__(self, num_classes: int, pretrained: bool = False) -> None:
        """Build a randomly-initialised ResNet-8.

        Args:
            num_classes: Output dimensionality.
            pretrained: Ignored; ResNet-8 has no pretrained weights.

        Notes:
            The ``normalization`` attribute starts as a placeholder
            (ImageNet stats) and is overwritten by
            :func:`baseline.data.build_dataloaders` with the
            channel-wise mean/std computed on the configured MedMNIST
            training split.
        """
        super().__init__(num_classes=num_classes, normalization=IMAGENET_NORMALIZATION)
        backbone = _ResNet8Backbone()
        self.backbone = backbone
        self.head = nn.Linear(backbone.out_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the backbone followed by the classification head.

        Args:
            x: Input batch shaped ``(B, 3, H, W)``.

        Returns:
            Raw logits of shape ``(B, num_classes)``.
        """
        return self.head(self.backbone(x))
