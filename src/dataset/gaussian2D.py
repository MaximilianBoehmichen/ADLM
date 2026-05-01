from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class Gaussian2DDataset(Dataset):
    """A torch dataset that loads our gaussian representations and ground truth."""

    root: Path | str
    """The root directory of the dataset. Is a Path or a string, whatever fits you.
    -> update the type accordingly
    """

    files: list[Path] | list[str]
    """The actual files/directories. Probably one per ground truth image, but this depends on the work of Dominik."""

    def __init__(self, root: Path | str) -> None:
        """Initializes the dataset, namely precomputes the attributes above.
        Depending on the size of the dataset, some in memory handling would be nice to have.

        Args:
            root (Path | str): The root directory of the dataset, from which the files are read and are already located.
        """
        pass

    def __len__(self) -> int:
        """Gives the length of the dataset.

        Should be either the number of different ground truth files (-> disrespecting the X alternative gaussian
        representations per file) or the total number of representations (-> #ground truth files * #runs per file).

        Returns:
            The length of the dataset.
        """
        return 0

    def __getitem__(self, idx: int) -> Data:
        """Gives the data for a given selected example.

        This depends on the same decision that has to be made for __len__:
            - either one (random?) selected representation of that ground truth file
            - or the index refers to a specific representation.

        The returned Data object is not modified inplace according to my short and not extensive research.

        Args:
            idx (int): The index of the data.

        Returns:
            The data for the index (a torch_geometric object for the data).


        References:
            - https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.pool.knn_graph.html
            - https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.pool.radius_graph.html
        """
        return Data(
            x=torch.empty(0),
            edge_index=torch.empty(0),
            edge_attr=torch.empty(
                0
            ),  # TODO: decide whether we actually want to use edge embeddings (like [x1 - x2, y1 - y2,
            # <other info...>])
            y=torch.empty(0),
            pos=torch.empty(
                0
            ),  # TODO: decide whether we want to use this, may be useful to determine the neighbors with other
            # metrics (see References),
        )
