"""3-layer GCN graph classifier with input BatchNorm, mean+max pooling, MLP head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool


class GCNClassifier(nn.Module):
    def __init__(self, in_dim: int = 7, hidden_dim: int = 64,
                 num_classes: int = 2, num_layers: int = 3,
                 dropout: float = 0.3, task: str = "multi-class"):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        self.task = task
        self.dropout = dropout

        self.input_bn = nn.BatchNorm1d(in_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        last = in_dim
        for _ in range(num_layers):
            self.convs.append(GCNConv(last, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            last = hidden_dim

        # Pooling produces (B, 2*hidden_dim); MLP head -> num_classes logits.
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.input_bn(x)
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        pooled = torch.cat(
            [global_mean_pool(x, batch), global_max_pool(x, batch)],
            dim=-1,
        )
        return self.head(pooled)
