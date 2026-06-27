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
    NUM_WEIGHTING_FEATURES: int
    num_bases = 2
    hidden_dim = 8
    out_channels: int

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_bases: int = 2,
        hidden_dim: int = 8,
        num_layers: int = 8,
        d: int = 2
    ) -> None:
        super().__init__(aggr=["sum"])
        self.out_channels = out_channels
        self.num_bases = num_bases
        self.hidden_dim = hidden_dim
        self.NUM_WEIGHTING_FEATURES = 9 if d == 2 else 0

        self.weighting = nn.Sequential(
            nn.Linear(self.NUM_WEIGHTING_FEATURES, self.hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
        )
        for _ in range(num_layers - 2):
            self.weighting.extend([
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.1),
            ])

        self.weighting.extend([
            nn.Linear(self.hidden_dim, self.num_bases),
            nn.LeakyReLU(0.1),
        ])

        for layer in self.weighting:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=0.1, nonlinearity='leaky_relu')

                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        self.bases = nn.Linear(in_channels, num_bases * out_channels, bias=False)

    def forward(self, x: Tensor, layout: Tensor, edge_index: Tensor) -> Tensor:
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))  # Why did we drop them in the preprocessing?
        h = self.bases(x)

        return self.propagate(edge_index, h=h, layout=layout)

    def message(self, h_j: Tensor, layout_i: Tensor, layout_j: Tensor) -> Tensor:
        pos_j, pos_i = layout_i[:, :2], layout_j[:, :2]
        scaling_i, scaling_j = layout_i[:, 2:4], layout_j[:, 2:4]
        theta_i, theta_j = layout_i[:, 4:5], layout_j[:, 4:5]

        rel_pos = pos_j - pos_i
        dist = rel_pos.norm(dim=-1, keepdim=True)
        delta_theta = theta_j - theta_i
        rot_cos = torch.cos(2 * delta_theta)
        rot_sin = torch.sin(2 * delta_theta)

        weighting_features = torch.cat([
                rel_pos,
                dist,
                scaling_i,
                scaling_j,
                rot_cos,
                rot_sin,
            ],
            dim=-1,
        )
        coeff = self.weighting(weighting_features)

        h_j = h_j.view(-1, self.num_bases, self.out_channels)
        return torch.einsum("eb,ebo->eo", coeff, h_j)

    def update(self, aggr_out: Tensor) -> Tensor:
        return aggr_out


class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_bases: int = 8, hidden_dim: int = 8, num_layers: int = 3):
        super().__init__()

        self.conv1 = SumConv(in_channels, out_channels, num_bases, hidden_dim=hidden_dim, num_layers=num_layers)
        self.norm1 = nn.LayerNorm(out_channels)

        self.conv2 = SumConv(out_channels, out_channels, num_bases, hidden_dim=hidden_dim, num_layers=num_layers)
        self.norm2 = nn.LayerNorm(out_channels)

        self.shortcut = nn.Sequential()

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=False),
                nn.LayerNorm(out_channels)
            )

    def forward(self, x: Tensor, layout: Tensor, edge_index: Tensor) -> Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x, layout, edge_index)
        out = self.norm1(out)
        out = F.relu(out)

        out = self.conv2(out, layout, edge_index)
        out = self.norm2(out)

        out = out + identity

        return F.relu(out)


class ResNetLikePYGGNN(nn.Module):
    CHANNELS = [16, 32, 64]

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        k: int = 9,
        num_bases: int = 8,
        hidden_dim: int = 8,
        num_layers: int = 8,
    ) -> None:
        super().__init__()
        self.k = k

        self.stem_conv = SumConv(in_channels, self.CHANNELS[0], num_bases=num_bases, hidden_dim=hidden_dim, num_layers=num_layers)
        self.stem_norm = nn.BatchNorm1d(self.CHANNELS[0])

        self.stages = nn.ModuleList([
            ResNetBasicBlock(self.CHANNELS[0], self.CHANNELS[0], num_bases=num_bases, hidden_dim=hidden_dim, num_layers=num_layers),
            ResNetBasicBlock(self.CHANNELS[0], self.CHANNELS[1], num_bases=num_bases, hidden_dim=hidden_dim, num_layers=num_layers),
            ResNetBasicBlock(self.CHANNELS[1], self.CHANNELS[2], num_bases=num_bases, hidden_dim=hidden_dim, num_layers=num_layers),
        ])

        self.head = nn.Linear(self.CHANNELS[-1] * 2, num_classes)

    def forward(self, data: Data) -> Tensor:
        x, layout, edge_index, batch = data.x, data.layout, data.edge_index, data.batch

        assert x is not None
        assert edge_index is not None
        assert layout is not None, "data.layout is missing — run the ExtractLayout transform in the pipeline."
        num_nodes = x.size(0)

        edge_index = prune_knn_edges(
            edge_index,
            num_nodes=num_nodes,
            original_k=data.num_edges // num_nodes,
            keep_k=self.k - 1  # to make space for removed self loop of the preprocessing
        )

        x = self.stem_conv(x, layout, edge_index)
        x = self.stem_norm(x)
        x = F.relu(x)

        for block in self.stages:
            x = block(x, layout, edge_index)

        x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
        return self.head(x)
