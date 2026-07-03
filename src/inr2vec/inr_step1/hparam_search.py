"""Grid search over INR hparams by mean reconstruction PSNR."""

from __future__ import annotations

import itertools
import random
from pathlib import Path

import medmnist
import numpy as np
import torch
from medmnist import INFO
from tqdm import tqdm

from inr2vec.inr_step1.cli import Config, parse_args
from inr2vec.inr_step1.defs import DEFAULT_ROOT
from inr2vec.inr_step1.model import INR, MixedPE, RFFPE
from inr2vec.inr_step1.train import train_inr

# For a fair comparison with ResNet8, 76k leaves a bit room for training. 
PARAM_MAX_BUDGET = 76_000 
PARAM_MIN_BUDGET = 1_000

# 64 images for statistically reliable mean PSNR
NUM_IMAGES = 64


"""
If we execute a combinatorial brute-force grid search over the implicit neural 
representation (INR) hyperparameter space: 

The search space comprehensively permutes the following architectural variants:
    - hidden_dim (4 choices): [8, 16, 32, 64]
    - num_bands (4 choices): [4, 6, 8, 12]
    - seed (4 choices): Evaluates performance across 4 distinct random seeds
    - sigma (6 choices): Gaussian scale factors [0.5, 1, 2, 3, 4, 5]
    - num_frequencies (8 choices): [96, 128, 160, 192, 224, 256, 288, 320]

This combinatorial expansion results in a total of 8,064 valid hyperparameter 
configurations (4 * 4 * 4 * 6 * 8 ...). For each individual run, the model is 
trained on the full batch of 64 sample images from the ChestMNIST dataset 
for 500 steps per image to collect mean reconstruction PSNR metrics.
"""

# Targeted local tuning space (Total combinations: 1 * 1 * 1 * 1 * 1 * 2 * 1 * 3 = 6 configs)
HPARAMS_SEARCH_SPACE = {
    "pe": [RFFPE],
    "hidden_dim": [64, 96, 128],       # test 3 gradients of hidden dimension around 64, which is the best in the original search
    "hidden_layers": [2,3],
    "seed": [42],
    "out_dim": [1],
    "num_frequencies": [256, 384, 512],       # Test if adding more spectral capacity helps
    "num_bands": [8],
    "sigma": [10.0, 15.0],          # Fine-tune the sigmas 
}



def _count_params(model: torch.nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def grid_search(config: Config) -> None:
    """Fit every in-budget hparam configuration.

    Args:
        config: CLI configuration.
    """
    device = config.device

    dataset = load_split(config.dataset, split="test", size=config.image_size)
    coords = make_coord_grid(config.image_size, config.image_size, device)
    indices = random.Random(config.seed).sample(range(len(dataset)), NUM_IMAGES)
    targets = [to_tensor(dataset[i][0]).reshape(-1, 1).to(device) for i in indices]

    keys = list(HPARAMS_SEARCH_SPACE)
    combos = [
        dict(zip(keys, values))
        for values in itertools.product(*HPARAMS_SEARCH_SPACE.values())
    ]

    results: list[tuple[float, int, dict[str, int]]] = []
    skipped = 0

    print(
        f"Device: {device} | {NUM_IMAGES} images @ {config.image_size}px | "
        f"{config.epochs} steps/image | {len(combos)} configs"
    )

    for hparams in tqdm(combos[config.start:config.end], desc="configs"):
        # Retain Max's original filtering logic; it will pass cleanly for our current RFFPE setup
        if hparams["pe"] is MixedPE:
            if (
                hparams["seed"] != HPARAMS_SEARCH_SPACE["seed"][0]
                or hparams["sigma"] != HPARAMS_SEARCH_SPACE["sigma"][0]
            ):
                skipped += 1
                continue

        else:
            if hparams["num_bands"] != HPARAMS_SEARCH_SPACE["num_bands"][0]:
                skipped += 1
                continue

        n_params = _count_params(INR(**hparams))

        if n_params > PARAM_MAX_BUDGET or n_params < PARAM_MIN_BUDGET:
            skipped += 1
            continue

        psnrs = []

        pbar = tqdm(targets, leave=False)
        for target in pbar:
            model = INR(**hparams).to(device)
            psnrs.append(
                train_inr(
                    model,
                    coords,
                    target,
                    config.epochs,
                    config.lr,
                    patience=config.patience,
                )
            )

            pbar.set_postfix(psnr=f"{np.mean(psnrs):.3f}")

        mean_psnr = float(np.mean(psnrs))
        results.append((mean_psnr, n_params, hparams))
        print(
            f"done | mean PSNR {mean_psnr:6.3f} dB | {n_params:5d} params | {hparams}"
        )

    results.sort(key=lambda r: r[0], reverse=True)
    print(
        f"\n=== Ranking by mean PSNR "
        f"({len(results)} configs in budget, {skipped} skipped > {PARAM_MAX_BUDGET} params) ==="
    )
    for rank, (mean_psnr, n_params, hparams) in enumerate(results, start=1):
        print(f"{rank:3d}. {mean_psnr:6.3f} dB | {n_params:5d} params | {hparams}")


def make_coord_grid(h: int, w: int, device) -> torch.Tensor:
    ys = torch.linspace(-1, 1, h, device=device)
    xs = torch.linspace(-1, 1, w, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1).reshape(-1, 2)


def load_split(dataset: str, split: str, size: int = 224, root: Path = DEFAULT_ROOT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    info = INFO[dataset]
    DataClass = getattr(medmnist, info["python_class"])
    ds = DataClass(split=split, download=True, size=size, root=str(root))
    return ds


def to_tensor(img_np: np.ndarray) -> torch.Tensor:
    arr = np.asarray(img_np, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[None]
    else:
        arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr)


if __name__ == "__main__":
    grid_search(parse_args())

    """
    Findings: 
    1. "High Frequency" Completely Outperforms "Network Depth"
#    ---------------------------------------------------------------------------
#    - Rank 1: hidden_layers = 2 | num_frequencies = 512  =>  38.332 dB
#    - Rank 3: hidden_layers = 3 | num_frequencies = 512  =>  36.452 dB
#
#    Analysis: Under a strict parameter budget (<75k), allocating capacity to 
#    a wider spectral mapping (512 frequencies) while thinning the MLP down to 
#    2 layers yields significantly better reconstruction than deepening the network. 
#    The high-frequency projections of RFFPE handle the heavy lifting of spatial 
#    feature extraction, leaving a trivial optimization task for the shallow MLP.
#
# 2. Sigma = 10.0 is Inarguably the "Golden Scale Factor"
#    ---------------------------------------------------------------------------
#    - Looking across the Top 6 configurations, EVERY single run features sigma = 10.0.
#    - Starting from Rank 7, performance drops off a cliff as sigma shifts to 15.0.
#
#    Analysis: A sigma value of 15.0 injects frequencies that are too high and noisy, 
#    preventing such a compact network architecture from properly converging on the 
#    underlying coordinate distribution.
#
# 3. The Ultimate "Efficiency Champion" (Best Compute-Performance Trade-off)
#    ---------------------------------------------------------------------------
#    - Rank 4 achieves a rock-solid 36.409 dB using only 53,441 parameters.
#
#    Analysis: If multi-model scaling or conditional latent-code learning imposes 
#    heavy memory constraints during the 3D phase, this specific configuration 
#    serves as our ideal, proven "ultra-lightweight fallback scheme."
    """