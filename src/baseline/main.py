import os
import random
import sys
from dataclasses import asdict

import numpy as np
import torch
from torchinfo import summary

from baseline.data import build_dataloaders, dataset_info
from baseline.models import build_model
from baseline.train import train

if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

from baseline.cli import Config, parse_args


def _seed_everything(seed: int) -> None:
    """Set the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


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
    """Parse the CLI and start the training run."""
    argv = sys.argv[1:]
    config = parse_args(argv)

    _print_config(config)
    _seed_everything(config.seed)

    info = dataset_info(config.dataset)
    loaders, normalization = build_dataloaders(config)
    model = build_model(
        name=config.model,
        num_classes=info.num_classes,
        normalization=normalization,
    )

    if config.inr_root is not None:
        input_size = (1, model.input_numel)
    elif info.is_3d:
        input_size = (1, 1, config.image_size, config.image_size, config.image_size)
    else:
        input_size = (1, 1, config.image_size, config.image_size)

    print(summary(model, input_size=input_size, verbose=0))

    train(config, model, loaders, info)


if __name__ == "__main__":
    main()
