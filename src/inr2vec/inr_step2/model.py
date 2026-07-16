"""Step-2 inr2vec encoders that classify images from their INR weights.

Three variants probe where the class signal lives in the weight space:

- :class:`Inr2vecPaper` — the paper-faithful encoder over the stacked inter-hidden
  ``L(H+1) × H`` matrix only (ignores the PE → hidden input layer).
- :class:`Inr2vecInput` — encodes only the first layer (``H`` per-neuron rows of
  ``[PE weights | bias]``), where most of a small INR's information lives.
- :class:`Inr2vecFull` — both branches, embeddings concatenated before the head.

All three reuse the baseline :class:`~baseline.models.base.BaseModel` interface so
they train through ``baseline.main`` exactly like the CNN baselines. Inputs are
the flat weight vectors produced by :mod:`inr2vec.inr_step2.data`; standardization
happens in the data pipeline, so the forward pass stays pure like ``ResNet8``.
"""

from __future__ import annotations

import torch
from torch import nn

from baseline.models.base import BaseModel, NormalizationStats
from inr2vec.inr_step2.layout import (
    INRLayout,
    build_layout,
    input_layer_rows,
    inter_hidden_matrix,
)


class RowEncoder(nn.Module):
    """Per-row MLP (Linear + BatchNorm + ReLU) with a max-pool over rows.

    Mirrors the inr2vec encoder: the same linear map is applied to every row of
    the stacked weight matrix, growing the feature dim, then a max-pool collapses
    the rows into one permutation-invariant embedding.
    """

    def __init__(
        self,
        row_dim: int,
        feature_dims: tuple[int, ...],
        dropout: float = 0.2,
    ) -> None:
        """Build the per-row encoder.

        Args:
            row_dim: Length of each input row (the matrix's column count).
            feature_dims: Output width of each per-row linear layer.
            dropout: Dropout applied after every block.
        """
        super().__init__()
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        prev = row_dim

        for dim in feature_dims:
            self.linears.append(nn.Linear(prev, dim))
            self.norms.append(nn.BatchNorm1d(dim))
            prev = dim

        self.dropout = nn.Dropout(dropout)
        self.embed_dim = feature_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of weight matrices.

        Args:
            x: Batched matrix of shape ``(B, R, row_dim)``.

        Returns:
            Embedding of shape ``(B, embed_dim)``.
        """
        batch, rows, _ = x.shape

        for linear, norm in zip(self.linears, self.norms):
            x = linear(x)
            x = norm(x.reshape(batch * rows, -1)).reshape(batch, rows, -1)
            x = self.dropout(torch.relu(x))

        return x.amax(dim=1)


class _Inr2vecBase(BaseModel):
    """Shared setup for the weight-space classifiers."""

    def __init__(
        self,
        num_classes: int,
        normalization: NormalizationStats,
        in_channels: int = 1,
    ) -> None:
        """Store the layout and expose the flat input length.

        Args:
            num_classes: Output logits required by the head.
            normalization: Recorded on the checkpoint; not applied in-model
                (the data pipeline standardizes the weights).
            in_channels: Unused; accepted for a uniform ``build_model`` signature.
        """
        super().__init__(num_classes=num_classes, normalization=normalization)
        self.layout: INRLayout = build_layout()
        self.input_numel = self.layout.total


class Inr2vecPaper(_Inr2vecBase):
    """Paper-faithful encoder over the inter-hidden ``L(H+1) × H`` matrix."""

    def __init__(
        self,
        num_classes: int,
        normalization: NormalizationStats,
        in_channels: int = 1,
        feature_dims: tuple[int, ...] = (32, 64, 64),
    ) -> None:
        super().__init__(num_classes, normalization, in_channels)
        self.backbone = RowEncoder(self.layout.hidden_dim, feature_dims)
        self.head = nn.Linear(self.backbone.embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map flat weights ``(B, total)`` to class logits ``(B, num_classes)``."""
        rows = inter_hidden_matrix(x, self.layout)

        return self.head(self.backbone(rows))


class Inr2vecInput(_Inr2vecBase):
    """Encoder over the first layer only (``H`` rows of ``[PE weights | bias]``)."""

    def __init__(
        self,
        num_classes: int,
        normalization: NormalizationStats,
        in_channels: int = 1,
        feature_dims: tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__(num_classes, normalization, in_channels)
        self.backbone = RowEncoder(self.layout.input_dim + 1, feature_dims)
        self.head = nn.Linear(self.backbone.embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map flat weights ``(B, total)`` to class logits ``(B, num_classes)``."""
        rows = input_layer_rows(x, self.layout)

        return self.head(self.backbone(rows))


class Inr2vecFull(_Inr2vecBase):
    """Both branches: inter-hidden + first-layer embeddings concatenated."""

    def __init__(
        self,
        num_classes: int,
        normalization: NormalizationStats,
        in_channels: int = 1,
        inter_dims: tuple[int, ...] = (32, 64, 64),
        input_dims: tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__(num_classes, normalization, in_channels)
        self.backbone = nn.ModuleDict(
            {
                "inter": RowEncoder(self.layout.hidden_dim, inter_dims),
                "input": RowEncoder(self.layout.input_dim + 1, input_dims),
            }
        )
        embed_dim = (
            self.backbone["inter"].embed_dim + self.backbone["input"].embed_dim
        )
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map flat weights ``(B, total)`` to class logits ``(B, num_classes)``."""
        inter = self.backbone["inter"](inter_hidden_matrix(x, self.layout))
        first = self.backbone["input"](input_layer_rows(x, self.layout))

        return self.head(torch.cat([inter, first], dim=-1))
