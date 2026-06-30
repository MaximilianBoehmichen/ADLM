"""ResNet-inspired graph classifier with relative-position message passing."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, ReLU, Sequential
from torch_geometric.nn import MessagePassing, global_mean_pool


class FourierEncoding(nn.Module):
    """Encodes a D-dim vector with L Fourier frequencies → 2*L*D dims.

    Frequencies are log-linearly spaced from 2^0 to 2^(L-1), following NeRF.
    The encoding is fixed (not learned) — no parameters.
    """

    def __init__(self, num_freqs: int = 4, pos_dim: int = 2):
        super().__init__()
        self.out_dim = 2 * num_freqs * pos_dim
        freqs = 2.0 ** torch.arange(num_freqs)  # [L]
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, D)  freqs: (L,)
        x_f = x.unsqueeze(-1) * self.freqs  # (N, D, L)
        return torch.cat([x_f.sin(), x_f.cos()], dim=-1).flatten(1)  # (N, 2*D*L)


class RelativeResNetBlock(MessagePassing):
    def __init__(self, channels: int, pos_enc_dim: int):
        super().__init__(aggr='mean')
        self.message_mlp = Sequential(
            Linear(channels + pos_enc_dim, channels),
            ReLU(),
            Linear(channels, channels),
            LayerNorm(channels),
        )

    def forward(self, x: torch.Tensor, pos_enc: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        aggr_out = self.propagate(edge_index, x=x, pos_enc=pos_enc)
        return F.relu(x + aggr_out)

    def message(self, x_j: torch.Tensor, pos_enc_i: torch.Tensor, pos_enc_j: torch.Tensor) -> torch.Tensor:
        return self.message_mlp(torch.cat([x_j, pos_enc_j - pos_enc_i], dim=-1))


class ResGCNClassifier(nn.Module):
    def __init__(self, in_dim: int = 5, hidden_dim: int = 64,
                 num_classes: int = 2, num_layers: int = 3,
                 task: str = "multi-class", pos_dim: int = 2,
                 fourier_freqs: int = 4):
        super().__init__()
        self.task = task
        self.pos_enc = FourierEncoding(num_freqs=fourier_freqs, pos_dim=pos_dim)
        pos_enc_dim = self.pos_enc.out_dim

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList(
            [RelativeResNetBlock(hidden_dim, pos_enc_dim) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch, pos = data.x, data.edge_index, data.batch, data.pos
        pos_enc = self.pos_enc(pos)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, pos_enc, edge_index)
        x = global_mean_pool(x, batch)
        return self.head(x)