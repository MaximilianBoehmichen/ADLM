
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import medmnist
from medmnist import INFO
import scipy.ndimage as ndimage

# Enforce the use of RFFPE as per project setup
from inr2vec.inr_step1.model import RFFPE

# =====================================================================
# Add 3D-native Positional Encoding & Update Model Architecture
# =====================================================================
class RFFPE_3D(nn.Module):
    def __init__(self, num_frequencies=256, sigma=10.0, seed=42):
        super().__init__()
        self.out_dim = num_frequencies * 2
        generator = torch.Generator().manual_seed(seed)
        # Random matrix calibrated specifically for 3D coordinates (x, y, z)
        B = 2 * torch.pi * torch.randn(3, num_frequencies, generator=generator) * sigma
        self.register_buffer("B", B)

    def forward(self, coords):
        projected = coords @ self.B
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)

class DeepSDF(nn.Module):
    """
    A single "super network" capable of learning multiple 3D shapes simultaneously.
    Concept: Instead of training multiple models for multiple shapes, we only train 1 model.
    """
    def __init__(self, num_shapes, latent_dim=64, hidden_dim=128, num_frequencies=8):
        super().__init__()
        
        # To let the network know WHICH shape's distance it is currently calculating,
        # we issue an "ID card" to each shape. This is the latent vector z_i.
        # =====================================================================
        self.latent_codes = nn.Embedding(num_shapes, latent_dim)
        # Initialize these multiple "ID cards" with completely random numbers
        nn.init.normal_(self.latent_codes.weight, mean=0.0, std=0.01)
        
        # 3D Coordinate Positional Encoding (x, y, z) -> High-dimensional features
        self.pe = RFFPE(num_frequencies, num_bands=4, sigma=10.0, seed=42)
        
        # Main Backbone Network
        in_dim = self.pe.out_dim + latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Output a scalar: SDF distance
        )

    def forward(self, coords, shape_indices):
        """
        THEORY: f(x, y, z, z_i) = Signed_Distance
        - Input 1 (coords): Spatial coordinates (x, y, z)
        - Input 2 (shape_indices): The index 'i' for retrieving the latent vector z_i
        """
        # Pull out the corresponding "ID card" z_i based on the index
        z = self.latent_codes(shape_indices)
        
        # Encode the 3D coordinates using Random Fourier Features
        encoded_coords = self.pe(coords)
        
        # =====================================================================
        # Concatenate the coordinate features and the latent vector
        # =====================================================================
        features = torch.cat([encoded_coords, z], dim=-1)
        
        return self.net(features)

def preprocess_voxel_to_sdf(voxel_grid):
    """
    Convert a 3D binary voxel grid into a Signed Distance Field (SDF).
    Inside object -> negative distance, Outside object -> positive distance.
    """
    binary_grid = voxel_grid > 0
    if not binary_grid.any():
        return np.ones_like(voxel_grid, dtype=np.float32) # Empty space
        
    # Calculate distance from outside to boundary
    dist_out = ndimage.distance_transform_edt(~binary_grid)
    # Calculate distance from inside to boundary
    dist_in = ndimage.distance_transform_edt(binary_grid)
    
    # Combine to form SDF
    sdf = dist_out - dist_in
    return sdf.astype(np.float32)

def load_medmnist_3d_dataset(num_shapes=5):
    """
    Load a small subset of the MedMNIST 3D dataset (OrganMNIST3D) and extract SDFs.
    """
    print("Downloading/Loading OrganMNIST3D dataset...")
    data_flag = 'organmnist3d'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    
    # Load the training split
    dataset = DataClass(split='train', download=True, size=28)
    
    all_coords, all_targets, all_indices = [], [], []
    
    # Generate a normalized 3D coordinate grid for 28x28x28 voxels
    res = 28
    ls = torch.linspace(-1, 1, res)
    z_grid, y_grid, x_grid = torch.meshgrid(ls, ls, ls, indexing='ij')
    base_coords = torch.stack([x_grid, y_grid, z_grid], dim=-1).reshape(-1, 3) # [28^3, 3]
    
    print(f"Extracting 3D SDFs for {num_shapes} organs...")
    for i in range(num_shapes):
        # voxel_grid shape is usually (1, 28, 28, 28). Squeeze to (28, 28, 28)
        voxel_grid, label = dataset[i]
        voxel_grid = voxel_grid.squeeze()
        
        # Compute SDF from the voxel grid
        sdf_grid = preprocess_voxel_to_sdf(voxel_grid)
        targets = torch.tensor(sdf_grid).reshape(-1, 1) # [28^3, 1]
        
        indices = torch.full((targets.shape[0],), i, dtype=torch.long)
        
        all_coords.append(base_coords)
        all_targets.append(targets)
        all_indices.append(indices)
        
    return (torch.cat(all_coords), torch.cat(all_targets), torch.cat(all_indices), num_shapes)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    num_shapes_to_test = 3 
    coords, targets, indices, num_shapes = load_medmnist_3d_dataset(num_shapes=num_shapes_to_test)
    
    # =====================================================================
    # FIX 2: Safeguard Memory with TensorDataset and Mini-Batched DataLoader
    # =====================================================================
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(coords, targets, indices)
    dataloader = DataLoader(dataset, batch_size=4096, shuffle=True)

    # 2. Initialize Updated Wide & Mid-Frequency Model (~73k params)
    model = DeepSDF(num_shapes=num_shapes, latent_dim=32, hidden_dim=256, num_frequencies=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 3. Safe Training Loop
    epochs = 300
    print("Starting Joint DeepSDF Training (Network Weights + Latent Codes)...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch_coords, batch_targets, batch_indices in dataloader:
            batch_coords = batch_coords.to(device)
            batch_targets = batch_targets.to(device)
            batch_indices = batch_indices.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_coords, batch_indices)
            loss = F.mse_loss(preds, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        if epoch % 50 == 0:
            print(f"Epoch [{epoch}/{epochs}] | MSE Loss: {avg_loss:.6f}")

    # 4. Visualization: Slicing the 3D organ at Z=0 (Middle of the volume)
    print("Training complete! Rendering 2D slice (Z=0) of the 3D volume...")
    model.eval()
    
    res = 28
    ls = torch.linspace(-1, 1, res)
    y_grid, x_grid = torch.meshgrid(ls, ls, indexing='ij')
    # Construct a slice where Z=0
    slice_coords = torch.stack([x_grid, y_grid, torch.zeros_like(x_grid)], dim=-1).reshape(-1, 3).to(device)

    

    fig, axes = plt.subplots(1, num_shapes, figsize=(4 * num_shapes, 4))
    if num_shapes == 1:
        axes = [axes]
        
    with torch.no_grad():
        for i in range(num_shapes):
            shape_idx = torch.full((res * res,), i, dtype=torch.long, device=device)
            pred_sdf = model(slice_coords, shape_idx).cpu().numpy().reshape(res, res)
            
            ax = axes[i]
            im = ax.imshow(pred_sdf, extent=[-1, 1, -1, 1], origin='lower', cmap='RdBu', vmin=-5, vmax=5)
            # Draw the boundary (SDF=0)
            ax.contour(pred_sdf, levels=[0], extent=[-1, 1, -1, 1], colors='black', linewidths=2)
            ax.set_title(f"MedMNIST Organ ID: {i}\n(Latent Vector $Z_{i}$)")
            ax.axis('off')
            
    plt.tight_layout()
    output_path = "medmnist_deepsdf_slices.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}. Look at how different latent vectors summon different organs!")