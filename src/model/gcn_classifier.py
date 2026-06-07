"""ResNet-style GNN classifier with relative positional message passing."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, ReLU, Sequential
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool


class RelPosResBlock(MessagePassing):
    """ResNet-style message passing block with relative positional encoding.

    Computes rel_pos = pos_j - pos_i inside message(), making the block
    translation-invariant without pre-computing edge attributes.
    """

    def __init__(self, channels: int, pos_dim: int = 2):
        super().__init__(aggr='mean')
        self.message_mlp = Sequential(
            Linear(channels + pos_dim, channels),
            ReLU(),
            Linear(channels, channels),
            LayerNorm(channels),
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        aggr_out = self.propagate(edge_index, x=x, pos=pos)
        return F.relu(x + aggr_out)

    def message(self, x_j: torch.Tensor, pos_i: torch.Tensor,
                pos_j: torch.Tensor) -> torch.Tensor:
        rel_pos = pos_j - pos_i
        return self.message_mlp(torch.cat([x_j, rel_pos], dim=-1))


class ResGCNClassifier(nn.Module):
    def __init__(self, in_dim: int = 5, hidden_dim: int = 64,
                 num_classes: int = 2, num_layers: int = 3,
                 task: str = "multi-class", **kwargs):
        super().__init__()
        self.task = task
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList(
            [RelPosResBlock(hidden_dim, pos_dim=2) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, data) -> torch.Tensor:
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, pos, edge_index)
        x = torch.cat([global_mean_pool(x, batch),
                        global_max_pool(x, batch)], dim=1)
        return self.head(x)