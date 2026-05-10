# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ADLM (Advanced Deep Learning Methods) — SS2026 team project investigating image representation via Gaussian Splatting combined with Graph Neural Networks. The core idea: compress images into anisotropic Gaussian primitives, then classify using GNNs on the compressed representation rather than full-resolution pixels. Dataset: MedMNIST (medical images, 224×224).

## Build & Run Commands

```bash
uv sync                              # Install/sync all dependencies from uv.lock
python optimize_static_repr_fast.py   # Run single-image Gaussian splatting optimization
python scripts/presentation1_eda.py   # Run EDA analysis on MedMNIST

# Preprocessing pipeline: convert MedMNIST → Gaussian graph .pt files
python preprocess_dataset.py --dataset pneumoniamnist --splits train val test
python preprocess_dataset.py --dataset pneumoniamnist --splits test --max-epochs 50  # quick test
```

No test suite exists yet. Validation is done visually via output images and PSNR metrics.

## Architecture

### Core Pipeline (`optimize_static_repr_fast.py`)

**GaussianRepresentationND** — the central model. Represents an image as K anisotropic Gaussians, each parameterized by:
- `mus`: positions (D-dim)
- `scalings_inv_`: inverse scales (log-space for positivity)
- `rotations`: angle (2D) or quaternion (3D)
- `colors`: intensity logits (sigmoid-mapped)

Initialization is content-adaptive: places Gaussians at high-gradient regions using NMS on gradient magnitude maps, with anisotropic scaling derived from local image structure.

Rendering uses FAISS KNN lookups (not brute-force per-pixel) for efficiency. The FAISS index is rebuilt periodically (every 10 steps) as Gaussians move during optimization.

**TrainingConfig** — dataclass controlling hyperparameters (epochs, LR schedule, compression factor, visualization settings).

**train_gs()** — training loop using Adam + cosine annealing. Loss = L1 reconstruction + position/scale regularization. When `logging=False`, skips all prints, file saves, and the post-training forward pass.

**Data loading** — `load_medmnist_dataset()` returns a full MedMNIST dataset for any dataset flag + split. `preprocess_medmnist_image()` handles RGB→grayscale and [0,1] normalization. `load_medmnist()` is a legacy wrapper for single-image loading.

### Preprocessing Pipeline (`preprocess_dataset.py`)

Converts an entire MedMNIST dataset into per-image `.pt` files, each containing a `torch_geometric.data.Data` object:
- `x` (N, 6): node features [mus(2) | scalings(2) | rotation(1) | color(1)]
- `pos` (N, 2): Gaussian positions
- `edge_index` (2, N×k): KNN edges between Gaussians (built via FAISS)
- `y` (1,): label
- `psnr` (1,): reconstruction quality

Output layout: `data/{dataset}/{split}/{index:05d}.pt`. Resumable — skips existing files on restart.

### Visualization (`vis_utils.py`)

- `save_progress_figure()` — prediction vs GT with error heatmaps
- `visualize_2d_gaussians()` — Gaussian ellipses overlaid on image (from live model)
- `visualize_pt_file()` — visualize a saved `.pt` file (ellipses + KNN edges)

### Data

Git LFS tracks `data/**`. MedMNIST datasets are used (currently PneumoniaMNIST, ChestMNIST).

## Key Dependencies

- **torch** — model and optimization
- **faiss-cpu** — fast KNN for Gaussian neighborhood queries
- **kornia** — image gradients, PSNR metrics
- **medmnist** — medical image datasets
- **torch-geometric** — GNN framework, `Data` class used for graph storage

## Environment

- Python >=3.12 (pinned in `.python-version`)
- Package manager: **uv** (not pip)
- Apple Silicon: `preprocess_dataset.py` sets `OMP_NUM_THREADS=1` to prevent faiss-cpu/torch OpenMP conflict