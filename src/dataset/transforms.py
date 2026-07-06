from __future__ import annotations

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, coalesce, to_undirected


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

    dim = pos.shape[1]
    scale = (max_val - min_val) / img_size

    data.pos = pos * scale + min_val
    x[:, :dim] = x[:, :dim] * scale + min_val
    x[:, dim:2 * dim] = x[:, dim:2 * dim] * scale
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


def limited_undirected_knn(data: Data, k: int = 9) -> Data:
    """Prune the stored directed KNN graph to `k` edges/node (self-loop included),
    add the self-loop, then symmetrize. Modifies in place.
    """
    edge_index = data.edge_index
    assert edge_index is not None

    num_nodes = data.num_nodes
    original_k = edge_index.size(1) // num_nodes
    keep_k = k - 1  # for the self-loop

    src, dst = edge_index
    src = src.view(num_nodes, original_k)[:, :keep_k].reshape(-1)
    dst = dst.view(num_nodes, original_k)[:, :keep_k].reshape(-1)

    src_sym = torch.cat([src, dst])
    dst_sym = torch.cat([dst, src])
    edge_index = torch.stack([src_sym, dst_sym], dim=0)

    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    data.edge_index = coalesce(edge_index, num_nodes=num_nodes)
    return data


def extract_layout(data: Data) -> Data:
    """Copies the layout features (effectively everything except the intensity) to a layout variable.

    This can be used to determine the weight of the message as it describes the node.

    Args:
        data: The data object to work on.

    Returns:
        The modified data.
    """
    x = data.x
    assert x is not None

    data.layout = x[:, :-1].clone()
    return data


def build_rotation(rot, dim):
    """Batched rotation matrices from a 2D angle or a 3D quaternion (wxyz)."""
    if dim == 2:
        c, s = torch.cos(rot[:, 0]), torch.sin(rot[:, 0])

        return torch.stack([c, -s, s, c], -1).view(-1, 2, 2)

    w, x, y, z = torch.nn.functional.normalize(rot, dim=1).unbind(-1)

    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).view(-1, 3, 3)


class BuildKNNGraph:
    """Recompute edge_index as a KNN graph with either L2 or mahalanobis neighbors.

    Also can build a reverse graph where the in degree is fixed.
    """

    def __init__(self, metric="l2", direction="propagate_to", eps=1e-6):
        assert metric in {"l2", "mahalanobis"}
        assert direction in {"propagate_to", "propagate_from"}
        self.metric = metric
        self.direction = direction
        self.eps = eps

    def __call__(self, data):
        if self.metric == "l2" and self.direction == "propagate_to":
            return data  # already the stored graph

        pos, x = data.pos, data.x
        n, dim = pos.shape
        k = min(data.edge_index.size(1) // n, n - 1)

        if self.metric == "l2":
            d2 = torch.cdist(pos, pos).square()

        else:
            s = x[:, dim:2 * dim]
            r = build_rotation(x[:, 2 * dim:2 * dim + (1 if dim == 2 else 4)], dim)
            diff = pos[None] - pos[:, None]
            white = torch.einsum("ied,ije->ijd", r, diff) / s.clamp(min=self.eps)[:, None]
            d2 = white.square().sum(-1)

        d2.fill_diagonal_(float("inf"))

        # propagate_from = pick per column instead of per row = same topk on the transpose
        mat = d2 if self.direction == "propagate_to" else d2.t()
        neigh = mat.topk(k, dim=1, largest=False).indices.reshape(-1)
        center = torch.arange(n).repeat_interleave(k)
        src, dst = (center, neigh) if self.direction == "propagate_to" else (neigh, center)
        data.edge_index = torch.stack([src, dst])

        return data
