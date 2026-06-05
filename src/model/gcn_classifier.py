"""ResNet-inspired GNN graph classifier with edge features."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool


class ResGINEBlock(nn.Module):
    def __init__(self, dim: int, edge_dim: int):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
        )
        self.conv = GINEConv(mlp, edge_dim=edge_dim)
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x, edge_index, edge_attr)) + x)


class ResGCNClassifier(nn.Module):
    def __init__(self, in_dim: int = 5, hidden_dim: int = 64,
                 num_classes: int = 2, num_layers: int = 3,
                 task: str = "multi-class", edge_dim: int = 2):
        super().__init__()
        self.task = task
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.edge_proj = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.blocks = nn.ModuleList(
            [ResGINEBlock(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = self.edge_proj(data.edge_attr)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, edge_index, edge_attr)
        x = torch.cat([global_mean_pool(x, batch),
                        global_max_pool(x, batch)], dim=1)
        return self.head(x)
