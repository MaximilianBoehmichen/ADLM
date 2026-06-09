"""Pixel point-cloud baseline preprocessor.

Loads MedMNIST images at native 28×28 resolution (784 nodes per image).
Every pixel becomes a node — no subsampling, no information loss.

    x: (784, 3) — [row_norm, col_norm, intensity]  all in [0, 1]
    pos: (784, 2) — [row_norm, col_norm]
    edge_index: KNN graph on pixel positions
    y: class label

Usage:
    python preprocess_pixel_baseline.py --dataset chestmnist --splits train val test
    python preprocess_pixel_baseline.py --dataset chestmnist --splits train val test --max-samples 1000
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
    load_medmnist_dataset,
    preprocess_medmnist_image,
)
from preprocess_dataset import build_label_tensor

if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

IMG_SIZE = 28


def image_to_pixel_graph(img: torch.Tensor, k: int) -> Data:
    """Convert a 2D (H, W) image tensor to a pixel point-cloud PyG Data.

    Every pixel is a node. Positions normalized to [0, 1].
    """
    H, W = img.shape
    rows = torch.arange(H)
    cols = torch.arange(W)
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")

    pos = torch.stack([grid_r.flatten(), grid_c.flatten()], dim=1).float()
    pos = pos / torch.tensor([H - 1, W - 1], dtype=torch.float32)  # (N, 2) in [0,1]

    intensities = img.flatten().unsqueeze(1)  # (N, 1)
    x = torch.cat([pos, intensities], dim=1)  # (N, 3)

    N = pos.shape[0]
    index = faiss.IndexFlatL2(2)
    index.add(pos.numpy().astype(np.float32))
    _, knn_idx = index.search(pos.numpy().astype(np.float32), k + 1)
    knn_idx = knn_idx[:, 1:]  # remove self-loop
    src = np.repeat(np.arange(N), k)
    dst = knn_idx.flatten()
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    return Data(x=x, pos=pos, edge_index=edge_index)


def process_split(dataset_flag: str, split: str, output_dir: Path,
                  k: int, max_samples: int | None = None, medmnist_root: str | None = None):
    dataset, info, D = load_medmnist_dataset(dataset_flag, split=split, size=IMG_SIZE,
                                             root=medmnist_root)
    assert D == 2, "Only 2D datasets are supported."
    task = info["task"]

    split_dir = output_dir / dataset_flag / split
    split_dir.mkdir(parents=True, exist_ok=True)

    total = len(dataset)
    end = total if max_samples is None else min(total, max_samples)

    for i in range(end):
        out_path = split_dir / f"{i:05d}.pt"
        if out_path.exists():
            continue

        img_pil, label_np = dataset[i]
        y = build_label_tensor(label_np, task)
        img = preprocess_medmnist_image(img_pil, D)  # (28, 28) in [0, 1]

        graph = image_to_pixel_graph(img, k=k)
        graph.y = y
        torch.save(graph, out_path)

        if (i + 1) % 500 == 0 or i == 0:
            print(f"[{split}] {i + 1}/{end} — nodes: {graph.x.shape[0]}")

    print(f"[{split}] done — {end} samples in {split_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="chestmnist",
                   choices=["pneumoniamnist", "chestmnist"])
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--output-dir", default="data_pixel")
    p.add_argument("--k-graph", type=int, default=8,
                   help="KNN degree (default 8 — each pixel connects to its 8 spatial neighbours)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap per-split sample count (e.g. 1000 for a quick signal run)")
    p.add_argument("--medmnist-root", type=str, default=None,
                   help="Directory for medmnist raw data (default: ~/.medmnist). "
                        "On cluster: /vol/miltank/users/hdo/medmnist")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    for split in args.splits:
        print(f"\n--- {args.dataset} [{split}] (28×28, {IMG_SIZE*IMG_SIZE} nodes) ---")
        process_split(args.dataset, split, output_dir,
                      k=args.k_graph, max_samples=args.max_samples,
                      medmnist_root=args.medmnist_root)
    print("\nDone.")


if __name__ == "__main__":
    main()