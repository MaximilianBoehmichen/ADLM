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
from inr2vec.inr_step1.defs import DEFAULT_ROOT, HPARAMS_SEARCH_SPACE
from inr2vec.inr_step1.model import INR, MixedPE
from inr2vec.inr_step1.train import train_inr

PARAM_MAX_BUDGET = 6_400
PARAM_MIN_BUDGET = 4_800
NUM_IMAGES = 64


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

    for hparams in tqdm(combos, desc="configs"):
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
        print(f"{n_params} -> {str(hparams)}")

        if n_params > PARAM_MAX_BUDGET or n_params < PARAM_MIN_BUDGET:
            skipped += 1
            continue

        psnrs = []

        pbar = tqdm(targets)
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
