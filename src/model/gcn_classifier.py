"""GNN classifier model zoo: four levels of complexity for systematic evaluation.

Level 0 (mlp):    No graph structure. Pool node features -> classify.
Level 1 (gcn):    Standard GCNConv message passing.
Level 2 (relpos): Custom MessagePassing with relative positions.
Level 3 (full):   RelPos + virtual node + Fourier positional encoding.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, ReLU, Sequential
from torch_geometric.nn import GCNConv, MessagePassing, global_mean_pool, global_max_pool


# ---------------------------------------------------------------------------
# Level -1: Linear baseline (replicates sklearn logistic regression)
# ---------------------------------------------------------------------------

class LinearBaseline(nn.Module):
    """Pool raw features with mean/max/std/min, then linear head. No learned
    projection before pooling — matches the sklearn logistic regression baseline."""

    def __init__(self, in_dim: int = 7, hidden_dim: int = 64,
                 num_classes: int = 2, num_layers: int = 0,
                 dropout_p: float = 0.3, task: str = "multi-class", **kwargs):
        super().__init__()
        self.task = task
        self.head = Linear(in_dim * 4, num_classes)

    def forward(self, data) -> torch.Tensor:
        x, batch = data.x, data.batch
        mean = global_mean_pool(x, batch)
        max_ = global_max_pool(x, batch)
        min_ = -global_max_pool(-x, batch)
        mean_sq = global_mean_pool(x ** 2, batch)
        std = (mean_sq - mean ** 2).clamp(min=1e-6).sqrt()
        x = torch.cat([mean, std, min_, max_], dim=1)
        return self.head(x)

    @property
    def num_layers(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# Level 0: MLP baseline (no graph structure)
# ---------------------------------------------------------------------------

class MLPPooling(nn.Module):
    def __init__(self, in_dim: int = 7, hidden_dim: int = 64,
                 num_classes: int = 2, num_layers: int = 0,
                 dropout_p: float = 0.3, task: str = "multi-class", **kwargs):
        super().__init__()
        self.task = task
        self.proj = Sequential(
            Linear(in_dim, hidden_dim),
            LayerNorm(hidden_dim),
            ReLU(),
        )
        self.drop = nn.Dropout(dropout_p)
        self.head = Linear(hidden_dim * 2, num_classes)

    def forward(self, data) -> torch.Tensor:
        x = self.proj(data.x)
        x = self.drop(x)
        x = torch.cat([global_mean_pool(x, data.batch),
                        global_max_pool(x, data.batch)], dim=1)
        return self.head(x)

    @property
    def num_layers(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# Level 1: Simple GCNConv
# ---------------------------------------------------------------------------

class SimpleGCN(nn.Module):
    def __init__(self, in_dim: int = 7, hidden_dim: int = 64,
                 num_classes: int = 2, num_layers: int = 2,
                 dropout_p: float = 0.3, task: str = "multi-class", **kwargs):
        super().__init__()
        self.task = task
        self._num_layers = num_layers
        self.dropout_p = dropout_p
        self.proj = Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList(
            [LayerNorm(hidden_dim) for _ in range(num_layers)]
        )
        self.head = Linear(hidden_dim * 2, num_classes)

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.proj(x))
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = torch.cat([global_mean_pool(x, batch),
                        global_max_pool(x, batch)], dim=1)
        return self.head(x)

    @property
    def num_layers(self) -> int:
        return self._num_layers


# ---------------------------------------------------------------------------
# Level 2: Relative position message passing
# ---------------------------------------------------------------------------

class RelPosBlock(MessagePassing):
    def __init__(self, channels: int, dropout_p: float = 0.3):
        super().__init__(aggr='sum')
        self.message_mlp = Sequential(
            Linear(channels + 2, channels),
            ReLU(),
            Linear(channels, channels),
        )
        self.norm = LayerNorm(channels)
        self.dropout_p = dropout_p

    def forward(self, x: torch.Tensor, pos: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        out = self.propagate(edge_index, x=x, pos=pos)
        out = self.norm(F.relu(x + out))
        return F.dropout(out, p=self.dropout_p, training=self.training)

    def message(self, x_j: torch.Tensor, pos_i: torch.Tensor,
                pos_j: torch.Tensor) -> torch.Tensor:
        rel_pos = pos_j - pos_i
        return self.message_mlp(torch.cat([x_j, rel_pos], dim=-1))


class RelPosGNN(nn.Module):
    def __init__(self, in_dim: int = 7, hidden_dim: int = 64,
                 num_classes: int = 2, num_layers: int = 2,
                 dropout_p: float = 0.3, task: str = "multi-class", **kwargs):
        super().__init__()
        self.task = task
        self.proj = Sequential(
            Linear(in_dim, hidden_dim),
            LayerNorm(hidden_dim),
            ReLU(),
        )
        self.blocks = nn.ModuleList(
            [RelPosBlock(hidden_dim, dropout_p) for _ in range(num_layers)]
        )
        self.head = Linear(hidden_dim * 2, num_classes)

    def forward(self, data) -> torch.Tensor:
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch
        x = self.proj(x)
        for block in self.blocks:
            x = block(x, pos, edge_index)
        x = torch.cat([global_mean_pool(x, batch),
                        global_max_pool(x, batch)], dim=1)
        return self.head(x)

    @property
    def num_layers(self) -> int:
        return len(self.blocks)


# ---------------------------------------------------------------------------
# Level 3: Full model (Fourier rel-pos + virtual node + mean/max pool)
# ---------------------------------------------------------------------------

class FourierRelPos(nn.Module):
    def __init__(self, num_freq: int = 4):
        super().__init__()
        freqs = torch.pi * (2.0 ** torch.arange(num_freq))
        self.register_buffer('freqs', freqs)

    @property
    def out_dim(self) -> int:
        return 4 * self.freqs.shape[0]

    def forward(self, rel_pos: torch.Tensor) -> torch.Tensor:
        x = rel_pos.unsqueeze(-1) * self.freqs
        return torch.cat([x.sin(), x.cos()], dim=-1).flatten(1)


class FourierRelPosBlock(MessagePassing):
    def __init__(self, channels: int, num_freq: int = 4, dropout_p: float = 0.3):
        super().__init__(aggr='sum')
        self.fourier = FourierRelPos(num_freq)
        self.message_mlp = Sequential(
            Linear(channels + self.fourier.out_dim, channels),
            ReLU(),
            Linear(channels, channels),
        )
        self.norm = LayerNorm(channels)
        self.dropout_p = dropout_p

    def forward(self, x: torch.Tensor, pos: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        out = self.propagate(edge_index, x=x, pos=pos)
        out = self.norm(F.relu(x + out))
        return F.dropout(out, p=self.dropout_p, training=self.training)

    def message(self, x_j: torch.Tensor, pos_i: torch.Tensor,
                pos_j: torch.Tensor) -> torch.Tensor:
        rel_pos = pos_j - pos_i
        return self.message_mlp(torch.cat([x_j, self.fourier(rel_pos)], dim=-1))


class FullGNN(nn.Module):
    def __init__(self, in_dim: int = 7, hidden_dim: int = 64,
                 num_classes: int = 2, num_layers: int = 2,
                 dropout_p: float = 0.3, task: str = "multi-class", **kwargs):
        super().__init__()
        self.task = task
        self.proj = Sequential(
            Linear(in_dim, hidden_dim),
            LayerNorm(hidden_dim),
            ReLU(),
        )
        self.blocks = nn.ModuleList(
            [FourierRelPosBlock(hidden_dim, dropout_p=dropout_p)
             for _ in range(num_layers)]
        )
        self.vn_mlps = nn.ModuleList([
            Sequential(
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                LayerNorm(hidden_dim),
            ) for _ in range(num_layers)
        ])
        self.head = Linear(hidden_dim * 2, num_classes)

    def forward(self, data) -> torch.Tensor:
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch
        x = self.proj(x)
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


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "mlp": MLPPooling,
    "gcn": SimpleGCN,
    "relpos": RelPosGNN,
    "full": FullGNN,
}
