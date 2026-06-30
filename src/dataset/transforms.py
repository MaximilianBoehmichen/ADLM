import torch
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
    """Encodes the theta to two params and strips absolute positions from x.

    Modifies in place. Assumes data is already cloned.

    Input x layout:  [mus(2) | scalings(2) | theta(1) | color(1)]  → 6 dims
    Output x layout: [scalings(2) | cos(2θ)(1) | sin(2θ)(1) | color(1)] → 5 dims

    Absolute positions are kept in data.pos for relative message passing (pos_j - pos_i)
    but removed from x so the classifier is position-invariant.
    """
    x = data.x
    assert x is not None

    theta = x[:, 4:5]
    cos_t = torch.cos(2 * theta)  # theta, theta+pi, theta+2pi are the same ellipse
    sin_t = torch.sin(2 * theta)

    # strip mus (x[:, :2]), keep scalings, cos/sin rotation, color
    data.x = torch.cat([x[:, 2:4], cos_t, sin_t, x[:, 5:]], dim=1)
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
