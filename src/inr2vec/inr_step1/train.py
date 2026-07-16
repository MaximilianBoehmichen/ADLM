"""Step-1 training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchmetrics.functional.image import peak_signal_noise_ratio
from tqdm import tqdm

from inr2vec.inr_step1.model import INR


def train_one_epoch(
    model: INR,
    coords: torch.Tensor,
    target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Run one full-batch optimization step and return the MSE loss.

    Args:
        model: INR being fit.
        coords: Coords.
        target: Flattened pixel intensities.
        optimizer: Optimizer driving the fit.

    Returns:
        Scalar MSE loss for this epoch.
    """
    model.train()
    loss = F.mse_loss(model(coords), target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_inr(
    model: INR,
    coords: torch.Tensor,
    target: torch.Tensor,
    epochs: int,
    lr: float,
    patience: int = 0,
) -> float:
    """Fit ``model`` to one image.

    Args:
        model: INR.
        coords: Coord.
        target: Flattened pixel intensities.
        epochs: Maximum number of full-batch optimization steps.
        lr: Learning rate.
        patience: Epochs without improvement before stopping.

    Returns:
        Peak signal-to-noise ratio of the fitted reconstruction.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_loss = float("inf")
    epochs_without_improvement = 0

    epoch_iter = tqdm(range(epochs), desc="fit", leave=False)
    for _ in epoch_iter:
        loss = train_one_epoch(model, coords, target, optimizer)

        epoch_iter.set_postfix(loss=f"{loss:.6f}")

        if loss < best_loss:
            best_loss = loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if 0 < patience <= epochs_without_improvement:
                break

    model.eval()
    with torch.no_grad():
        pred = model(coords)
    return peak_signal_noise_ratio(pred, target, data_range=1.0).item()
