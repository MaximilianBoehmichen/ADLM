import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_max_pool, global_mean_pool
from torch_geometric.utils import add_self_loops
import torch.nn.functional as F


def prune_knn_edges(edge_index: Tensor, num_nodes: int, original_k: int = 15, keep_k: int = 8) -> Tensor:
    """Original neighbors to specified number of neighbors."""
    src, dst = edge_index

    src_pruned = src.view(num_nodes, original_k)[:, :keep_k].flatten()
    dst_pruned = dst.view(num_nodes, original_k)[:, :keep_k].flatten()

    return torch.stack([src_pruned, dst_pruned], dim=0)


class SumConv(MessagePassing):
    """MessagePassing equivalent to KNNConv. Name no longer accurate :)."""
    NUM_ADDED_FEATURES = 2
    NUM_WEIGHTING_FEATURES = 2 + 1

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr=["sum", "max"])

        self.message_mlp = nn.Linear(in_channels + self.NUM_ADDED_FEATURES, out_channels, bias=False)
        self.update_mlp = nn.Linear(out_channels * 2, out_channels, bias=False)

        self.weighting = nn.Linear(self.NUM_WEIGHTING_FEATURES, out_channels, bias=True)

    def forward(self, x: Tensor, pos: Tensor, edge_index: Tensor) -> Tensor:
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))  # Why did we drop them in the preprocessing?
        out = self.propagate(edge_index, x=x, pos=pos)

        return out

    def message(self, x_j: Tensor, pos_i: Tensor, pos_j: Tensor) -> Tensor:
        rel_pos = pos_j - pos_i
        combined = torch.cat([x_j, rel_pos], dim=-1)

        dist = rel_pos.norm(dim=-1, keepdim=True)
        weighting_features = torch.cat([rel_pos, dist], dim=-1)
        weighting = self.weighting(weighting_features)

        return self.message_mlp(combined) * weighting

    def update(self, aggr_out: Tensor) -> Tensor:
        return self.update_mlp(aggr_out)


class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = SumConv(in_channels, out_channels)
        self.norm1 = nn.LayerNorm(out_channels)

        self.conv2 = SumConv(out_channels, out_channels)
        self.norm2 = nn.LayerNorm(out_channels)

        self.shortcut = nn.Sequential()

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=False),
                nn.LayerNorm(out_channels)
            )

    def forward(self, x: Tensor, pos: Tensor, edge_index: Tensor) -> Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x, pos, edge_index)
        out = self.norm1(out)
        out = F.relu(out)

        out = self.conv2(out, pos, edge_index)
        out = self.norm2(out)

        out = out + identity

        return F.relu(out)


class ResNetLikePYGGNN(nn.Module):
    CHANNELS = [16, 32, 64]

    def __init__(self, in_channels: int, num_classes: int, k: int = 9) -> None:
        super().__init__()
        self.k = k

        self.stem_conv = SumConv(in_channels, self.CHANNELS[0])
        self.stem_norm = nn.LayerNorm(self.CHANNELS[0])

        self.stages = nn.ModuleList([
            ResNetBasicBlock(self.CHANNELS[0], self.CHANNELS[0]),
            ResNetBasicBlock(self.CHANNELS[0], self.CHANNELS[1]),
            ResNetBasicBlock(self.CHANNELS[1], self.CHANNELS[2]),
        ])

        self.head = nn.Linear(self.CHANNELS[-1] * 2, num_classes)

    def forward(self, data: Data) -> Tensor:
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch

        assert x is not None
        assert edge_index is not None
        num_nodes = x.size(0)

        edge_index = prune_knn_edges(
            edge_index,
            num_nodes=num_nodes,
            original_k=data.num_edges // num_nodes,
            keep_k=self.k - 1  # to make space for removed self loop of the preprocessing
        )

        x = self.stem_conv(x, pos, edge_index)
        x = self.stem_norm(x)
        x = F.relu(x)

        for block in self.stages:
            x = block(x, pos, edge_index)

        x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
        return self.head(x)
