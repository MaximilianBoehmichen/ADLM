"""Entry point for baseline training.

Parses the command line, prints model statistics with ``torchinfo`` and
delegates to :func:`baseline.train.train`.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torchinfo import summary

from dataclasses import asdict

from baseline.cli import Config, parse_args
from baseline.data import build_dataloaders, dataset_info
from baseline.device import resolve_device
from baseline.models import build_model
from baseline.train import train


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducibility.

    Args:
        seed: Integer seed forwarded to every relevant RNG.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _print_config(config: Config) -> None:
    """Print the resolved configuration before training starts.

    Args:
        config: The fully-populated :class:`Config` dataclass instance,
            including any fields that were inferred at runtime (e.g.
            ``device``).
    """
    print("=" * 60)
    print("Training configuration")
    print("=" * 60)
    for key, value in asdict(config).items():
        print(f"  {key:<20s} {value}")
    print("=" * 60)


def main() -> None:
    """Parse the CLI, build everything and start the training run."""
    config = parse_args()
    device = resolve_device()
    config.device = str(device)
    if device.type == "mps" and config.num_workers > 0:
        print(
            f"MPS backend detected; forcing num_workers from "
            f"{config.num_workers} to 0 to avoid multiprocessing conflicts.",
        )
        config.num_workers = 0
    _print_config(config)
    _seed_everything(config.seed)

    info = dataset_info(config.dataset)
    model = build_model(
        name=config.model,
        num_classes=info.num_classes,
        pretrained=config.finetune,
    )
    loaders = build_dataloaders(config, model=model)

    print(
        summary(
            model,
            input_size=(1, 3, config.image_size, config.image_size),
            verbose=0,
        ),
    )

    train(config=config, model=model, loaders=loaders, dataset_info=info)


if __name__ == "__main__":
    main()
