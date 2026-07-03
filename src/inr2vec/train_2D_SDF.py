import torch
import matplotlib.pyplot as plt

from inr2vec.inr_step1.model import INR
from inr2vec.inr_step1.train import train_inr
from inr2vec.inr_step1.hparam_search import load_split, to_tensor, make_coord_grid

# Smaller size = faster for local testing; use 224 on the cluster
IMAGE_SIZE = 64


def visualize_reconstruction(target, predicted, resolution, psnr):
    orig = target.reshape(resolution, resolution).cpu().numpy()
    pred = predicted.reshape(resolution, resolution).cpu().detach().numpy()
    error = abs(orig - pred)

    _, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(orig, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original (ChestMNIST)")
    axes[0].axis("off")

    axes[1].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("INR Reconstruction")
    axes[1].axis("off")

    axes[2].imshow(error, cmap="hot", vmin=0, vmax=0.3)
    axes[2].set_title("Absolute Error")
    axes[2].axis("off")

    plt.suptitle(f"ChestMNIST INR | PSNR: {psnr:.2f} dB")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # 1. Load one ChestMNIST image
    print("Loading ChestMNIST...")
    dataset = load_split("chestmnist", split="test", size=IMAGE_SIZE)
    img_np, label = dataset[0]
    coords = make_coord_grid(IMAGE_SIZE, IMAGE_SIZE, device)   # [H*W, 2]
    target = to_tensor(img_np).reshape(-1, 1).to(device)       # [H*W, 1]
    print(f"Image size: {IMAGE_SIZE}×{IMAGE_SIZE} | Device: {device}")

    # 2. INR with best hyperparameters from hparam search
    model = INR(
        hidden_dim=64,
        hidden_layers=2,
        out_dim=1,
        num_frequencies=512,
        num_bands=8,
        sigma=10.0,
    ).to(device)

    # 3. Train
    print("Training INR on ChestMNIST image...")
    psnr = train_inr(
        model=model,
        coords=coords,
        target=target,
        epochs=1000,
        lr=1e-3,
        patience=50,
    )
    print(f"Training done | PSNR: {psnr:.2f} dB")

    # 4. Visualize
    model.eval()
    with torch.no_grad():
        predicted = model(coords)

    visualize_reconstruction(target, predicted, IMAGE_SIZE, psnr)
