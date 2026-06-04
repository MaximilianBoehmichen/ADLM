import os
import random
import sys
from dataclasses import asdict

import numpy as np
import torch

from inr2vec.inr_step1.cli import Config, parse_args
from inr2vec.inr_step1.hparam_search import grid_search


if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")


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

    grid_search(config)


if __name__ == "__main__":
    main()
