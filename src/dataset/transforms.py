import torch
from torch_geometric.data import Data


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
