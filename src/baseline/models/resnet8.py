import torch
from torch import nn
import torch.nn.functional as F

from baseline.models.base import BaseModel, NormalizationStats


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
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.stride = stride
        self.pad = (out_channels - in_channels) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual block.

        Args:
            x: Input feature map of shape ``(B, in_channels, H, W)``.

        Returns:
            Output feature map of shape ``(B, out_channels, H', W')``.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        identity = x[:, :, :: self.stride, :: self.stride]
        if self.pad:
            identity = F.pad(identity, (0, 0, 0, 0, self.pad, self.pad))

        return F.relu(out + identity)


class _ResNet8Backbone(nn.Module):
    """The convolutional trunk of ResNet-8 (everything before the head)."""

    CHANNELS = [16, 32, 64]

    def __init__(self, in_channels: int) -> None:
        """Set up stem and three residual stages."""
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )  # uses original stem, even though we work on 224x224
        self.stage1 = _BasicBlock(self.CHANNELS[0], self.CHANNELS[0], stride=1)
        self.stage2 = _BasicBlock(self.CHANNELS[0], self.CHANNELS[1], stride=2)
        self.stage3 = _BasicBlock(self.CHANNELS[1], self.CHANNELS[2], stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the backbone feature vector.

        Args:
            x: Input batch shaped ``(B, in_cannels, H, W)``.

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

    def __init__(
        self, num_classes: int, normalization: NormalizationStats, in_channels: int
    ) -> None:
        """Build a randomly-initialized ResNet-8.

        Args:
            num_classes: Output dimensionality.
            normalization: Normalization statistics.
        """
        super().__init__(num_classes=num_classes, normalization=normalization)
        backbone = _ResNet8Backbone(in_channels)
        self.backbone = backbone
        self.head = nn.Linear(backbone.CHANNELS[2], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the backbone followed by the classification head.

        Args:
            x: Input batch shaped ``(B, in_cannels, H, W)``.

        Returns:
            Raw logits of shape ``(B, num_classes)``.
        """
        return self.head(self.backbone(x))
