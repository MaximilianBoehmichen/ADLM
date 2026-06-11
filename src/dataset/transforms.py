from __future__ import annotations

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


def pos_normalization(
        data: Data,
        img_size: int,
        *,
        min_val: float = -1.0,
        max_val: float = 1.0,
) -> Data:
    """Normalizes the grid positions to a certain range.

    Modifies in place. Assumes data is already cloned.

    Note:
        Since some Gaussians may be placed outside the image, the given range may be exceeded.
    """
    pos = data.pos
    x = data.x
    assert pos is not None and x is not None

    data.pos = (pos / img_size) * (max_val - min_val) + min_val

    x[:, :2] = (x[:, :2] / img_size) * (max_val - min_val) + min_val
    data.x = x

    return data


def encode_rotation(data: Data) -> Data:
    """Encodes the theta to two params.

    Modifies in place. Assumes data is already cloned.
    """
    # I first wanted to just cap the theta between 0 and 2pi, but theoretically 0-pi also loses no information.
    # However, after some research, this still has the problem that both ends encode roughly the same, but an MLP
    # will likely not get that concept. But by encoding theta as cos(2 * theta) and sin(2 * theta), the distance between
    # the theta->0 and theta->pi vector is small.
    # However, this doesn't solve the problem of pi/2 and x-y interchangeability.
    x = data.x
    assert x is not None

    theta = x[:, 4:5]
    cos_t = torch.cos(2 * theta)  # treat theta, theta + pi, and theta + 2pi as the same ellipsis, because they are.
    sin_t = torch.sin(2 * theta)

    data.x = torch.cat([x[:, :4], cos_t, sin_t, x[:, 5:]], dim=1)
    return data


def basic_edge_attr(data: Data) -> Data:
    """Encodes the edge features.

    Modifies in place. Assumes data is already cloned.
    """
    if data.edge_attr is not None:
        raise ValueError("Data already has edge attribute vectors.")

    pos = data.pos
    assert pos is not None

    row, col = data.edge_index
    data.edge_attr = pos[row] - pos[col]

    return data


class FeatureNormalization:
    """Per-feature z-score standardization using precomputed training set statistics.

    Normalizes x columns to zero-mean unit-variance. Compute stats from the
    training set only, then apply to train/val/test with the same parameters.
    Must run after drop_pos_from_x so it only sees the 5 non-position features.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, data: Data) -> Data:
        data.x = (data.x - self.mean) / self.std
        return data

    @staticmethod
    def compute_stats(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-feature mean and std over all nodes in the dataset."""
        all_x = [dataset[i].x for i in range(len(dataset))]
        x = torch.cat(all_x, dim=0)
        mean = x.mean(dim=0)
        std = x.std(dim=0).clamp(min=1e-6)
        return mean, std


def drop_pos_from_x(data: Data) -> Data:
    """Remove absolute positions (x[:, :2]) from node features.

    Spatial information is instead captured via relative edge attributes
    (basic_edge_attr). Must run after encode_rotation and basic_edge_attr.
    """
    data.x = data.x[:, 2:]
    return data


def to_undirected_transform(data: Data) -> Data:
    """Symmetrize edge_index in place. Required for GCNConv on directed KNN graphs.

    Why:
        preprocess_dataset.py builds k directed edges per node (i -> j for each
        of i's k nearest neighbors), so in-degree varies wildly while out-degree
        is fixed at k. GCNConv assumes an undirected graph; passing a directed
        one breaks the symmetric Laplacian normalization.
    """
    edge_index = data.edge_index
    assert edge_index is not None
    data.edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)
    return data
