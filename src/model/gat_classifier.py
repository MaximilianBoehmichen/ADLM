"""GAT classifier with multi-head attention and global attention pooling."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.aggr import AttentionalAggregation


class GATClassifier(nn.Module):
    def __init__(self, in_dim: int = 5, hidden_dim: int = 64,
                 num_classes: int = 2, num_layers: int = 3,
                 task: str = "multi-class", num_heads: int = 4,
                 dropout_p: float = 0.1, **kwargs):
        super().__init__()
        self.task = task
        self.dropout_p = dropout_p

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        head_dim = hidden_dim // num_heads

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(hidden_dim, head_dim, heads=num_heads,
                          dropout=dropout_p, concat=True)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.pool = AttentionalAggregation(gate_nn)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x, edge_index) + x))
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.pool(x, batch)
        return self.head(x)