import torch
from torch import nn, Tensor
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


class ResNetLikeGNN(nn.Module):
    """Another try at implementing a ResNet like GNN."""

    CHANNELS = [16, 32, 64]

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        k: int = 9,
    ) -> None:
        super().__init__()

        self.stem = ResNetStem(in_channels, self.CHANNELS[0], k)
        self.stages = nn.Sequential(
            ResNetBasicBlock(self.CHANNELS[0], self.CHANNELS[0], k),
            ResNetBasicBlock(self.CHANNELS[0], self.CHANNELS[1], k),
            ResNetBasicBlock(self.CHANNELS[1], self.CHANNELS[2], k)
        )

        self.head = nn.Linear(self.CHANNELS[-1], num_classes)

    def forward(self, data: Data) -> Tensor:
        data = self.stages(self.stem(data))
        x = global_mean_pool(data.x, data.batch)

        return self.head(x)


class NodeLayerNorm(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, data: Data) -> Data:
        assert data.x is not None
        data.x = self.norm(data.x)

        return data


class KNNConv(nn.Module):
    """KNN Convolution with aggregation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int,
    ) -> None:
        super().__init__()

        self.k = k
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, data: Data) -> Data:
        x, edge_index = data.x, data.edge_index
        assert x is not None
        assert edge_index is not None

        N = x.size(0)
        neighbour_idx = edge_index[1].view(N, -1)[:, :self.k - 1]
        concatenated = torch.cat([x.unsqueeze(1), x[neighbour_idx]], dim=1)
        data.x = self.linear(concatenated.sum(dim=1))

        return data


class ResNetStem(nn.Module):
    """ResNet like stem."""

    def __init__(
        self,
        in_channels,
        out_channels: int,
        k: int = 9,
    ) -> None:
        super().__init__()

        self.conv = KNNConv(in_channels, out_channels, k)
        self.norm = NodeLayerNorm(out_channels)

    def forward(self, data: Data) -> Data:
        data = self.norm(self.conv(data))
        data.x = F.relu(data.x)

        return data


class ResNetBasicBlock(nn.Module):
    """Basic block for ResNet like GNN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k=9,
    ) -> None:
        super().__init__()

        self.conv1 = KNNConv(in_channels, out_channels, k)
        self.norm1 = NodeLayerNorm(out_channels)
        self.conv2 = KNNConv(out_channels, out_channels, k)
        self.norm2 = NodeLayerNorm(out_channels)

        self.pad = (out_channels - in_channels) // 2

    def forward(self, data: Data) -> Data:
        x = data.x
        assert x is not None

        data.x = F.relu(self.norm1(self.conv1(data)).x)
        data = self.norm2(self.conv2(data))

        if x.size(-1) != data.x.size(-1):
            x = F.pad(x, (self.pad, self.pad))

        new_x = F.relu(data.x + x)
        data.x = new_x

        return data
