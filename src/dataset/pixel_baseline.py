from pathlib import Path
from typing import Callable, Literal

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class PixelBaselineDataset(Dataset):
    """Loads pixel point-cloud .pt files produced by preprocess_pixel_baseline.py."""

    def __init__(
        self,
        root: Path | str,
        split: Literal["train", "val", "test"],
        *,
        transforms: Callable[[Data], Data] | None = None,
        in_memory: bool = False,
    ) -> None:
        self.root = Path(root) / split
        self.transforms = transforms
        self.in_memory = in_memory

        if not self.root.is_dir():
            raise NotADirectoryError(f"{self.root} is not a directory")

        self.files = sorted(self.root.rglob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No .pt files found in {self.root}")

        if in_memory:
            self.data = [
                torch.load(f, weights_only=False, map_location="cpu")
                for f in self.files
            ]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Data:
        d = self.data[idx].clone() if self.in_memory else \
            torch.load(self.files[idx], weights_only=False, map_location="cpu")
        if self.transforms is not None:
            d = self.transforms(d)
        return d