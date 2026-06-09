"""
Usage:
    python preprocess_dataset.py --dataset chestmnist --splits train val test
    python preprocess_dataset.py --dataset pneumoniamnist --splits test --max-epochs 50
"""

import argparse
import os
import sys
from pathlib import Path
import faiss
import numpy as np
import torch
from torch_geometric.data import Data
from optimize_static_repr_fast import (
    GaussianRepresentationND,
    TrainingConfig,
    load_medmnist_dataset,
    preprocess_medmnist_image,
    train_gs,
)

# Prevent OpenMP crash when faiss-cpu and torch both load libomp (Apple Silicon)
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def build_label_tensor(label_np: np.ndarray, task: str) -> torch.Tensor:
    """Build the graph-level y tensor according to the MedMNIST task type.

    multi-label, binary-class -> float (1, C) for BCEWithLogitsLoss
    everything else           -> long  (1,)   for CrossEntropyLoss
    """
    if task == "multi-label, binary-class":
        return torch.tensor(label_np, dtype=torch.float32).reshape(1, -1)
    return torch.tensor([int(np.asarray(label_np).squeeze())], dtype=torch.long)


def extract_pyg_data(gs: GaussianRepresentationND, y: torch.Tensor, k_graph: int,
                     psnr: float = None) -> Data:
    """Extract a PyG Data object from a trained GaussianRepresentationND model.

    Node features x = [mus(2) | scalings(2) | rotation(1) | color(1)]
    """
    with torch.no_grad():
        mus = gs.mus.detach().cpu()
        scalings = gs.scalings.detach().cpu()
        rotations = gs.rotations.detach().cpu()
        colors_mapped = (
            torch.sigmoid(gs.colors) * (gs.img_max - gs.img_min) + gs.img_min
        ).detach().cpu()

    x = torch.cat([mus, scalings, rotations, colors_mapped.unsqueeze(-1)], dim=-1)

    # KNN graph
    num_nodes = mus.shape[0]
    index = faiss.IndexFlatL2(mus.shape[1]) # L2 distance
    index.add(mus.numpy().astype(np.float32))
    # k+1 because the nearest neighbor of each point is itself
    _, knn_indices = index.search(mus.numpy().astype(np.float32), k_graph + 1)
    # Remove self-loops (first column) and build COO edge_index
    knn_indices = knn_indices[:, 1:]  # (N, k_graph)
    src = np.repeat(np.arange(num_nodes), k_graph) # [0,0,0, 1,1,1, 2,2,2, ...]
    dst = knn_indices.flatten() # [n1,n2,n3, n4,n5,n6, ...]
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    data = Data(
        x=x,
        pos=mus,
        edge_index=edge_index,
        y=y,
    )
    if psnr is not None:
        data.psnr = torch.tensor([psnr], dtype=torch.float32)
    return data


def process_split(dataset_flag: str, split: str, output_dir: Path,
                  k_graph: int, params: TrainingConfig, device: torch.device,
                  logging_dir: Path = None,
                  start_idx: int = 0, end_idx: int = None,
                  medmnist_root: str | None = None):
    """Process all images in one MedMNIST split."""
    dataset, info, D = load_medmnist_dataset(dataset_flag, split=split, root=medmnist_root)
    assert D == 2, "Only 2D datasets are supported for now."
    task = info["task"]

    split_dir = output_dir / dataset_flag / split
    split_dir.mkdir(parents=True, exist_ok=True)

    params_per_gauss = 2 + 2 + 1 + 1  # mus(2) + scalings(2) + rotation(1) + color(1)
    total = len(dataset)
    start = max(0, start_idx)
    end = total if end_idx is None else min(total, end_idx)

    for i in range(start, end):
        out_path = split_dir / f"{i:05d}.pt"
        if out_path.exists():
            continue

        img_pil, label_np = dataset[i]
        y = build_label_tensor(label_np, task)
        img_tensor = preprocess_medmnist_image(img_pil, D).to(device)

        num_gaussians = int(np.prod(img_tensor.shape) * params.compression_factor / params_per_gauss)

        gs = GaussianRepresentationND(num_gaussians, img_tensor.shape).to(device)
        gs.initialize_from_image(img_tensor, verbose=False)

        img_logging_dir = logging_dir / f"{i:05d}" if logging_dir else None
        gs, _, _, _, psnr = train_gs(gs, img_tensor, params, logging_dir=img_logging_dir)
        data = extract_pyg_data(gs, y, k_graph, psnr=psnr)
        torch.save(data, out_path)

        print(f"[{split}] {i + 1}/{total} (range {start}:{end}) | PSNR: {psnr:.1f} dB | {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess MedMNIST → Gaussian .pt files")
    parser.add_argument("--dataset", type=str, default="chestmnist")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--k-graph", type=int, default=15)
    parser.add_argument("--compression-factor", type=float, default=0.1)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--logging-dir", type=str, default=None,
                        help="Directory for per-image training logs (progress images, ellipses, etc.)")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="First image index to process (inclusive). Use with --end-idx to shard across jobs.")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="Last image index to process (exclusive). Defaults to end of split.")
    parser.add_argument("--medmnist-root", type=str, default=None,
                        help="Directory for medmnist raw data (default: ~/.medmnist). "
                             "On cluster: /vol/miltank/users/hdo/medmnist")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    params = TrainingConfig(
        compression_factor=args.compression_factor,
        max_epochs=args.max_epochs,
    )

    output_dir = Path(args.output_dir)
    logging_dir = Path(args.logging_dir) if args.logging_dir else None
    for split in args.splits:
        print(f"\n--- Processing {args.dataset} [{split}] ---")
        split_logging_dir = logging_dir / split if logging_dir else None
        if split_logging_dir:
            split_logging_dir.mkdir(parents=True, exist_ok=True)
        process_split(args.dataset, split, output_dir, args.k_graph, params, device, split_logging_dir,
                      start_idx=args.start_idx, end_idx=args.end_idx,
                      medmnist_root=args.medmnist_root)

    print("\nDone.")


if __name__ == "__main__":
    main()
