"""ResNet-inspired GCN graph classifier."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class ResGCNBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = GCNConv(dim, dim, bias=False)
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x, edge_index)) + x)


class ResGCNClassifier(nn.Module):
    def __init__(self, in_dim: int = 7, hidden_dim: int = 64,
                 num_classes: int = 2, num_layers: int = 3,
                 task: str = "multi-class"):
        super().__init__()
        self.task = task
        # Input projection: node features → hidden space (like ResNet's initial conv)
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        # Residual blocks (GCNConv → BN → skip → ReLU)
        self.blocks = nn.ModuleList(
            [ResGCNBlock(hidden_dim) for _ in range(num_layers)]
        )
        # Classifier head (global avg pool + FC, like ResNet)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.head(x)