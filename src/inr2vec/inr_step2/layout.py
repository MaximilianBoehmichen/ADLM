"""Layout of a fitted INR's learnable weights in a single flat vector.

Every INR shares the frozen ``BEST_CONFIG`` architecture, so its learnable
parameters always occupy the same slots of a fixed-length vector. This module is
the single source of truth for that layout: the dataset uses it to flatten each
``state_dict`` in a canonical order, and the step-2 encoders use it to slice the
flattened vector back into the per-layer weight matrices they consume.

The frozen ``pe.B`` buffer is excluded — it is identical across all images and
carries no per-image signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from inr2vec.inr_step1.defs import BEST_CONFIG


@dataclass(frozen=True)
class INRLayout:
    """Slot positions of the learnable INR weights within the flat vector.

    Attributes:
        keys: ``state_dict`` keys in flatten order (``pe.B`` excluded).
        shape: Original tensor shape for each key.
        offset: Start index of each key inside the flat vector.
        total: Length of the flat vector.
        hidden_dim: Hidden width ``H`` of the INR.
        input_dim: Width of the first layer's input (``2 * num_frequencies``).
        inter_keys: ``(weight, bias)`` key pairs of the inter-hidden ``H×H``
            maps — the substrate of the paper-faithful encoder.
        input_keys: ``(weight, bias)`` keys of the first layer (PE → hidden).
    """

    keys: tuple[str, ...]
    shape: dict[str, tuple[int, ...]]
    offset: dict[str, int]
    total: int
    hidden_dim: int
    input_dim: int
    inter_keys: tuple[tuple[str, str], ...]
    input_keys: tuple[str, str]


def build_layout(config: dict = BEST_CONFIG) -> INRLayout:
    """Derive the weight layout analytically from an INR config.

    Args:
        config: INR architecture config (defaults to the frozen ``BEST_CONFIG``).

    Returns:
        The :class:`INRLayout` describing the flat weight vector.
    """
    h = config["hidden_dim"]
    n_layers = config["hidden_layers"]
    in_dim = 2 * config["num_frequencies"]
    out_dim = config.get("out_dim", 1)

    specs: list[tuple[str, tuple[int, ...]]] = [
        ("net.0.weight", (h, in_dim)),
        ("net.0.bias", (h,)),
    ]
    inter: list[tuple[str, str]] = []
    for i in range(1, n_layers):
        idx = 2 * i
        specs += [(f"net.{idx}.weight", (h, h)), (f"net.{idx}.bias", (h,))]
        inter.append((f"net.{idx}.weight", f"net.{idx}.bias"))

    final = 2 * n_layers
    specs += [(f"net.{final}.weight", (out_dim, h)), (f"net.{final}.bias", (out_dim,))]

    keys: list[str] = []
    shape: dict[str, tuple[int, ...]] = {}
    offset: dict[str, int] = {}
    cursor = 0

    for key, tensor_shape in specs:
        keys.append(key)
        shape[key] = tensor_shape
        offset[key] = cursor
        cursor += math.prod(tensor_shape)

    return INRLayout(
        keys=tuple(keys),
        shape=shape,
        offset=offset,
        total=cursor,
        hidden_dim=h,
        input_dim=in_dim,
        inter_keys=tuple(inter),
        input_keys=("net.0.weight", "net.0.bias"),
    )


def flatten_state_dict(
    state_dict: dict[str, torch.Tensor], layout: INRLayout
) -> torch.Tensor:
    """Flatten an INR ``state_dict`` into one canonical-order weight vector.

    Args:
        state_dict: The fitted INR weights (may include the ``pe.B`` buffer).
        layout: The layout defining the key order.

    Returns:
        A 1-D ``(layout.total,)`` float tensor.
    """
    return torch.cat([state_dict[k].reshape(-1) for k in layout.keys])


def _take(flat: torch.Tensor, layout: INRLayout, key: str) -> torch.Tensor:
    """Slice and reshape one parameter out of a batched flat vector.

    Args:
        flat: Batched flat weights, shape ``(B, layout.total)``.
        layout: The layout giving the slot of ``key``.
        key: Parameter to extract.

    Returns:
        The parameter reshaped to ``(B, *layout.shape[key])``.
    """
    start = layout.offset[key]
    target_shape = layout.shape[key]
    end = start + math.prod(target_shape)

    return flat[:, start:end].reshape(flat.shape[0], *target_shape)


def inter_hidden_matrix(flat: torch.Tensor, layout: INRLayout) -> torch.Tensor:
    """Build the paper's ``L(H+1) × H`` stacked weight matrix.

    Each inter-hidden ``H×H`` weight matrix gets its bias appended as an extra
    row, and the resulting ``(H+1)×H`` blocks are stacked along the row axis.

    Args:
        flat: Batched flat weights, shape ``(B, layout.total)``.
        layout: The INR layout.

    Returns:
        A ``(B, L(H+1), H)`` tensor (e.g. ``(B, 52, 12)`` for ``BEST_CONFIG``).
    """
    blocks = []

    for weight_key, bias_key in layout.inter_keys:
        weight = _take(flat, layout, weight_key)
        bias = _take(flat, layout, bias_key)
        blocks.append(torch.cat([weight, bias.unsqueeze(1)], dim=1))

    return torch.cat(blocks, dim=1)


def input_layer_rows(flat: torch.Tensor, layout: INRLayout) -> torch.Tensor:
    """Build the first layer as ``H`` per-neuron rows of ``[weights | bias]``.

    The first layer is ``H × in_dim`` (not square), so the paper's bias-as-row
    trick does not apply; instead each hidden neuron becomes one row holding all
    its incoming PE weights plus its own bias.

    Args:
        flat: Batched flat weights, shape ``(B, layout.total)``.
        layout: The INR layout.

    Returns:
        A ``(B, H, in_dim + 1)`` tensor (e.g. ``(B, 12, 449)``).
    """
    weight_key, bias_key = layout.input_keys
    weight = _take(flat, layout, weight_key)
    bias = _take(flat, layout, bias_key)

    return torch.cat([weight, bias.unsqueeze(-1)], dim=-1)
