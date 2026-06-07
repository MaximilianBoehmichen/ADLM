"""ResNet-style GNN classifier with relative positional message passing."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, ReLU, Sequential
from torch_geometric.nn import MessagePassing, global_max_pool, global_mean_pool


class FourierRelPos(nn.Module):
    """Encodes 2D relative positions as multi-frequency sin/cos features.

    Makes the message MLP able to distinguish fine-grained spatial patterns
    without needing large capacity. With num_freq=4, out_dim=16.
    """

    def __init__(self, num_freq: int = 4):
        super().__init__()
        freqs = torch.pi * (2.0 ** torch.arange(num_freq))
        self.register_buffer('freqs', freqs)

    @property
    def out_dim(self) -> int:
        return 4 * self.freqs.shape[0]

    def forward(self, rel_pos: torch.Tensor) -> torch.Tensor:
        x = rel_pos.unsqueeze(-1) * self.freqs   # [E, 2, F]
        return torch.cat([x.sin(), x.cos()], dim=-1).flatten(1)  # [E, 4F]


class RelPosResBlock(MessagePassing):
    """ResNet-style message passing with Fourier-encoded relative positions.

    Translation-invariant: rel_pos = pos_j - pos_i computed in message().
    Uses sum aggregation (more expressive than mean, as shown by GIN paper).
    """

    def __init__(self, channels: int, num_freq: int = 4):
        super().__init__(aggr='sum')
        self.fourier = FourierRelPos(num_freq)
        self.message_mlp = Sequential(
            Linear(channels + self.fourier.out_dim, channels),
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
        return self.message_mlp(torch.cat([x_j, self.fourier(rel_pos)], dim=-1))


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
            [RelPosResBlock(hidden_dim) for _ in range(num_layers)]
        )
        # Virtual node: after each block, pool globally → MLP → add back to nodes
        self.vn_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
            ) for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, data) -> torch.Tensor:
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch
        x = self.input_proj(x)
        for block, vn_mlp in zip(self.blocks, self.vn_mlps):
            x = block(x, pos, edge_index)
            vn = global_mean_pool(x, batch)
            x = x + vn_mlp(vn)[batch]
        x = torch.cat([global_mean_pool(x, batch),
                        global_max_pool(x, batch)], dim=1)
        return self.head(x)

    @property
    def num_layers(self) -> int:
        return len(self.blocks)