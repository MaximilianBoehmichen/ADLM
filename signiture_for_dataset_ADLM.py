from pathlib import Path
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

class Gaussian2DDataset(Dataset):
    """
    A torch dataset that loads Gaussian representations and ground truth.
    Fully meets the requirements: in-memory handling, __len__, __getitem__, and Data object return.
    """
    def __init__(self, root: Path | str, in_memory: bool = False, img_size: int = 224) -> None:
        """
        Initializes the dataset with precomputed file list and optional in-memory caching.
        Args:
            root: Root directory containing .pt data files
            in_memory: If True, load all data into RAM for fast access
            img_size: Original image size for position normalization
        """
        # Initializations at the very beginning 
        self.root = Path(root)
        self.in_memory = in_memory
        self.img_size = img_size

        # Check if the path exists / legitimate
        if not self.root.exists():
            raise FileNotFoundError(f"Path does not exist: {self.root}")
        
        # Check if it is a valid directory / if it's a file 
        if not self.root.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self.root}")

        # Initialization: Find all .pt files (collect all files list as required)
        self.files = sorted(list(self.root.rglob("*.pt")))

        # Raise error if no data files found
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .pt files found in {self.root}")

        # Optional in-memory handling (as requested in the requirements)
        self.data_list = []

        # The In_memory-logic for safety check, something could go wrong e.g. the format. 
        if self.in_memory:
            print(f"Loading {len(self.files)} files into memory...")
            for f in tqdm(self.files):
                try:
                    data = torch.load(f, weights_only=False)
                    self.data_list.append(data)
                except Exception as e:
                    raise RuntimeError(f"Failed to load file {f}: {str(e)}")

    def __len__(self) -> int:
        """Return the total number of samples (matches requirement for dataset length)"""
        return len(self.files)

    def __getitem__(self, idx: int) -> Data:
        """Return a single Data object by index """
        # Load data from memory or disk
        if self.in_memory:
            data = self.data_list[idx].clone()
        else:
            data = torch.load(self.files[idx], weights_only=False)

        # Concert labels to a PyTorch tensor, I checked Dominik's file, he also uses .y as label. If this goes wrong, let me know. 
        if not isinstance(data.y, torch.Tensor):
            data.y = torch.tensor(data.y)
        data.y = data.y.long().view(-1)

        # Normalize position coordinates to [-1, 1] , as *2.0 get [0,2]. 
        if data.pos is not None and data.pos.dim() == 2:
            data.pos = (data.pos / self.img_size) * 2.0 - 1.0

        # Get edge attributes from relative positions, use edge_index as in Dom's file. 
        if data.edge_index is not None and data.edge_index.numel() > 0:
            row, col = data.edge_index
            data.edge_attr = data.pos[row] - data.pos[col]
        else:
            data.edge_attr = torch.empty((0, 2), dtype=torch.float32)

        return data


# mini Testcode
'''
if __name__ == "__main__":
    dataset = Gaussian2DDataset(root = "./data", bool = False, in_memory = True, img_size = 224)
    sample = dataset[0]
    print(f"Node Features: {sample.x.shape}")
    print(f"Normalized Pos Sample: {sample.pos[0]}")
    print(f"Edge Attributes: {sample.edge_attr.shape}")
'''
