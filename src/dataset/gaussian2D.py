from pathlib import Path
from typing import Callable, Literal

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from dataset.transforms import pos_normalization


class Gaussian2DDataset(Dataset):
    """A torch dataset that loads our gaussian representations and ground truth."""

    root: Path
    """The root directory of the dataset"""

    transforms: Callable[[Data], Data] | None
    """The transformations to apply to the data."""

    in_memory: bool
    """Whether to keep the dataset in memory."""

    img_size: int = 224
    """The size of the images to load."""

    files: list[Path] | list[str]
    """The actual files."""

    data: list[Data | None]
    """The in memory data."""

    def __init__(
        self,
        root: Path | str,
        split: Literal["train", "val", "test"],
        *,
        transforms: Callable[[Data], Data] | None = None,
        in_memory: bool = False,
        img_size: int = 224,
    ) -> None:
        """Initializes the dataset.

        Args:
            root (Path | str): The root directory of the dataset, from which the files are read and are already located.
            split: Which data split to load.

        Keyword Args:
            transforms: The transformations to apply to the data. Applied after built-in -1..1 pos range normalization.
                Should use torch_geometric.transforms
            in_memory (bool): Whether to keep the dataset in memory. It lazily loads the dataset on first access,
                e.g. during the first epoch.
            img_size (int): The size of the images to load.

        Note:
            With num_workers > 0, every worker gets its own data list (independent).
            Without ``persistent_workers=True``, the cached list is discarded and reloaded!
        """

        self.root = Path(root) / split
        self.transforms = transforms
        self.in_memory = in_memory
        self.img_size = img_size

        if not self.root.is_dir():
            raise NotADirectoryError(f"{self.root} is not a directory")

        self.files = sorted(list(self.root.rglob("*.pt")))

        if len(self.files) == 0:
            raise FileNotFoundError(f"Split {split} is empty")

        if self.in_memory:
            self.data = [None] * len(self.files)

    def __len__(self) -> int:
        """Gives the length of the dataset.

        Returns:
            The length of the dataset.
        """
        return len(self.files)

    def __getitem__(self, idx: int) -> Data:
        """Gives the data for a given selected example.

        Args:
            idx (int): The index of the data.

        Returns:
            The data for the index (a torch_geometric object for the data).


        References:
            - https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.pool.knn_graph.html
            - https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.pool.radius_graph.html
        """
        d: Data

        if self.in_memory:
            dn: Data | None = self.data[idx]

            if dn is None:
                dn: Data = torch.load(self.files[idx], weights_only=False, map_location="cpu")
                self.data[idx] = dn

            d = dn.clone()

        else:
            d = torch.load(self.files[idx], weights_only=False, map_location="cpu")

        d = pos_normalization(d, self.img_size)  # explicit normalization outside the transforms (!)

        if self.transforms is not None:
            d = self.transforms(d)

        return d
