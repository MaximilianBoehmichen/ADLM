"""Hardware-aware device selection.

Single source of truth for which device the training loop runs on. CUDA
is preferred, then Apple's Metal Performance Shaders, then CPU.
"""

from __future__ import annotations

import torch


def resolve_device() -> torch.device:
    """Return the best available torch device.

    Returns:
        ``cuda:0`` if a CUDA GPU is visible, ``mps`` on Apple silicon
        with an available Metal backend, otherwise ``cpu``.
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")
