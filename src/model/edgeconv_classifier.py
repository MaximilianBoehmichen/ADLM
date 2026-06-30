"""EdgeConv-style GNN classifier (DGCNN-inspired).

Uses the existing edge_index from preprocessed graphs (no dynamic re-building).
Each message = MLP(x_i || x_j - x_i), which makes the convolution
edge-feature-aware and translation-invariant.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_max_pool, global_mean_pool


class EdgeConvBlock(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr='max')
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels, bias=False),
            nn.LayerNorm(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels, bias=False),
            nn.LayerNorm(out_channels),
        )
        self.skip = (
            nn.Linear(in_channels, out_channels, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return F.relu(self.skip(x) + self.propagate(edge_index, x=x))

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([x_i, x_j - x_i], dim=-1))


class EdgeConvClassifier(nn.Module):
    def __init__(self, in_dim: int = 5, hidden_dim: int = 64,
                 num_classes: int = 2, num_layers: int = 3,
                 task: str = "multi-class", **kwargs):
        super().__init__()
        self.task = task
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList(
            [EdgeConvBlock(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, edge_index)
        x = torch.cat([global_mean_pool(x, batch),
                        global_max_pool(x, batch)], dim=1)
        return self.head(x)