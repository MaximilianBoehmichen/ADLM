import torch


def resolve_device() -> torch.device:
    """Determine the best available torch device.

    Returns:
        ``cuda:0`` if a CUDA GPU is visible, ``mps`` on Apple silicon
        with an available Metal backend, otherwise ``cpu``.
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")
