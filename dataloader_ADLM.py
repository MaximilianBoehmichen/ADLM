from pathlib import Path
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

class Gaussian2DDataset(Dataset):
    """
    A specialized torch dataset for 2D Gaussian representations of medical images.
    Features: Optional in-memory caching, spatial normalization, and edge attribute computation.
    """

    def __init__(self, root: Path | str, in_memory: bool = False, img_size: int = 28) -> None:
        """
        Args:
            root: Path to the directory containing .pt files.
            in_memory: If True, loads all data into RAM during initialization. (Will be used only once when the boolian evaluates to True. )
            img_size: The resolution of the original image (used for pos normalization).
        """
        self.root = Path(root)
        self.in_memory = in_memory
        self.img_size = img_size
        
        # 1. Collect all processed .pt files

        if len(self.files) == 0:
            raise FileNotFoundError(f"No .pt files found in {self.root}")


        self.files = sorted(list(self.root.rglob("*.pt")))
        

        # 2. In-memory handling
        self.data_list = []
        if self.in_memory:
            print(f"Loading {len(self.files)} files into memory...")
            for f in tqdm(self.files):
                # We perform the basic loading here
                # Post-processing (normalization) still happens in __getitem__ to keep it flexible
                self.data_list.append(torch.load(f, weights_only=False))
                # WARNING: weights_only = False   is for safely loading PyG objects

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Data:

        """
        Read the idx'th graph, I choose the index to a specific representation.

        """
        # Read out the preprossed Data Object (We have x, pos, edge_index, y in the preprosessing file) 
        # WARNING: weights_only = False   is for safely loading PyG objects

        # 1. Data Retrieval (Memory vs Disk) :  Decide whether we store our data in memory or not. 
        if self.in_memory:
            data = self.data_list[idx].clone() # Clone to avoid modifying the cached data permanently, ensuring we get the primitive data
        else:
            data = torch.load(self.files[idx], weights_only=False)

        # 2. Label Processing 
        # Ensure y is a LongTensor for classification tasks, or float for regression (For what Cross Entropy Loss expects)
        if not isinstance(data.y, torch.Tensor):
            data.y = torch.tensor(data.y)
        data.y = data.y.long().view(-1) # Flattens [1] to [1] or [] to [1] for CrossEntropyLoss

        # 3. Data Normalization 
        # Mapping [0, img_size] to [-1, 1] for a better training. 
        if data.pos is not None:
            data.pos = (data.pos / self.img_size) * 2.0 - 1.0

        # 4. Edge Attribute Computation
        # Here we decide whether we need edge embeddings (compute edge_attr)--> I propose Yes. 
        # GNN need to know the relative position between connected nodes. 
        # data.edge_index's shape: [2, E]. First row : starting position; Second row: ending position 
        # Compute relative spatial displacement as edge features
        if data.edge_index is not None and data.edge_index.numel() > 0:
            row, col = data.edge_index
            
            # The relative position remains valid even after pos normalization
            relative_pos = data.pos[row] - data.pos[col]
            data.edge_attr = relative_pos
        else:
            # Fallback for isolated nodes
            data.edge_attr = torch.empty((0, 2), dtype=torch.float32)

        return data