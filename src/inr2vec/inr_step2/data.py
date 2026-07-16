"""Load fitted INR weights as flat vectors for step-2 classification.

Mirrors the baseline image pipeline so the weight-space models train through
``baseline.main`` unchanged: :class:`INRWeightDataset` yields ``(weights, label)``
pairs and :func:`build_inr_loaders` returns the train/val/test
:class:`~baseline.data.Loaders` plus the standardization stats recorded on the
checkpoint. Per-dimension standardization (estimated on the train split) plays
the same role here that ``transforms.Normalize`` plays for images.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from baseline.cli import Config
from baseline.data import Loaders
from baseline.models.base import NormalizationStats
from inr2vec.inr_step2.layout import INRLayout, build_layout, flatten_state_dict


@dataclass
class Standardizer:
    """Per-dimension ``(x - mean) / std`` normalization of a weight vector."""

    mean: torch.Tensor
    std: torch.Tensor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class INRWeightDataset(Dataset):
    """A torch dataset that loads our fitted INR weights and ground truth."""

    root: Path
    """The root directory of the dataset"""

    transforms: Callable[[torch.Tensor], torch.Tensor] | None
    """The transformations to apply to the flat weight vector."""

    in_memory: bool
    """Whether to keep the dataset in memory."""

    layout: INRLayout
    """Slot positions of the learnable weights within the flat vector."""

    files: list[Path] | list[str]
    """The actual files."""

    data: list[tuple[torch.Tensor, torch.Tensor]]
    """The in memory data."""

    def __init__(
        self,
        root: Path | str,
        split: Literal["train", "val", "test"],
        *,
        transforms: Callable[[torch.Tensor], torch.Tensor] | None = None,
        in_memory: bool = False,
    ) -> None:
        """Initializes the dataset.

        Args:
            root (Path | str): The root directory of the dataset, from which the files are read and are already located.
            split: Which data split to load.

        Keyword Args:
            transforms: The transformations to apply to the data. Applied after the flat weight vector is built.
            in_memory (bool): Whether to keep the dataset in memory.
        """

        self.root = Path(root) / split
        self.transforms = transforms
        self.in_memory = in_memory
        self.layout = build_layout()

        if not self.root.is_dir():
            raise NotADirectoryError(f"{self.root} is not a directory")

        self.files = sorted(list(self.root.rglob("*.pt")))

        if len(self.files) == 0:
            raise FileNotFoundError(f"Split {split} is empty")

        if self.in_memory:
            self.data = []

            for f in self.files:
                d = torch.load(f, weights_only=False, map_location="cpu")  # better to first load to RAM not VRAM
                self.data.append(self._extract(d))

    def _extract(self, record: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Flattens one loaded INR record into a ``(weights, label)`` pair.

        Args:
            record (dict): The dict written by step-1 pretraining.

        Returns:
            The flat weight vector and the float label tensor.
        """
        weights = flatten_state_dict(record["state_dict"], self.layout)
        label = record["y"].reshape(-1).float()

        return weights, label

    def __len__(self) -> int:
        """Gives the length of the dataset.

        Returns:
            The length of the dataset.
        """
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gives the data for a given selected example.

        Args:
            idx (int): The index of the data.

        Returns:
            A ``(weights, label)`` tuple for the index.
        """
        weights: torch.Tensor
        label: torch.Tensor

        if self.in_memory:
            weights, label = self.data[idx]
            weights = weights.clone()  # explicit clone outside the transforms (!)
        else:
            weights, label = self._extract(
                torch.load(self.files[idx], weights_only=False, map_location="cpu")
            )

        if self.transforms is not None:
            weights = self.transforms(weights)

        return weights, label


def _weight_stats(
    dataset: INRWeightDataset,
    dim: int,
    num_workers: int,
    max_samples: int = 2**32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate per-dimension mean/std over a capped subset of train weights.

    Args:
        dataset: The train dataset (without normalization applied).
        dim: Flat weight-vector length.
        num_workers: Worker count for the streaming loader.
        max_samples: Number of vectors to accumulate before stopping.

    Returns:
        A ``(mean, std)`` tuple of ``(dim,)`` tensors; std is floored to 1e-6.
    """
    loader = DataLoader(
        dataset, batch_size=256, shuffle=True, num_workers=num_workers
    )
    total = torch.zeros(dim)
    sq_total = torch.zeros(dim)
    count = 0
    for weights, _ in tqdm(loader, desc="weight stats"):
        total += weights.sum(dim=0)
        sq_total += (weights**2).sum(dim=0)
        count += weights.size(0)
        if count >= max_samples:
            break

    mean = total / count
    std = torch.sqrt(torch.clamp(sq_total / count - mean**2, min=1e-12))
    return mean, std.clamp_min(1e-6)


def build_inr_loaders(config: Config) -> tuple[Loaders, NormalizationStats]:
    """Build train/val/test loaders over fitted INR weights.

    Per-dimension standardization is estimated on the train split and applied to
    every split, so the encoder sees normalized weight coordinates.

    Args:
        config: Training configuration; ``inr_root`` is the directory holding the
            ``{split}/*.pt`` INR files.

    Returns:
        The loader triplet and the standardization stats to record on the
        checkpoint.
    """
    root = config.inr_root

    train_ds = INRWeightDataset(root, "train")
    mean, std = _weight_stats(train_ds, train_ds.layout.total, config.num_workers)
    normalize = Standardizer(mean, std)
    train_ds.transforms = normalize

    val_ds = INRWeightDataset(root, "val", transforms=normalize)
    test_ds = INRWeightDataset(root, "test", transforms=normalize)

    loader_kwargs = {
        "pin_memory": True,
        "persistent_workers": config.num_workers > 0,
        "num_workers": config.num_workers,
        "prefetch_factor": 2 if config.num_workers > 0 else None,
        # Workers start after wandb.init() spins up threads in the main process;
        # forking would inherit their locks and risk a deadlock.
        "multiprocessing_context": "spawn" if config.num_workers > 0 else None,
    }

    loaders = Loaders(
        train=DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        ),
        val=DataLoader(
            val_ds, batch_size=config.batch_size, shuffle=False, **loader_kwargs
        ),
        test=DataLoader(
            test_ds, batch_size=config.batch_size, shuffle=False, **loader_kwargs
        ),
    )

    return loaders, NormalizationStats(mean=mean.tolist(), std=std.tolist())
