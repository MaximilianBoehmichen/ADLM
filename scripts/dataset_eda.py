import math
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt


def load_measurements(p: Path) -> np.ndarray:
    files = list(p.rglob("*.pt"))
    psnrs: list = []

    for file in files:
        d = torch.load(file, map_location="cpu", weights_only=False)
        psnrs.append(float(d.psnr))
        del d

    print(f"loaded {p}")
    return np.array(psnrs)


def dataset_eda(root: Path) -> None:
    """Plot overview statistics for the dataset."""
    train_psnrs = load_measurements(root / "train")
    val_psnrs = load_measurements(root / "val")
    test_psnrs = load_measurements(root / "test")

    bin_width = 0.125
    all_data = np.concatenate([train_psnrs, val_psnrs, test_psnrs])
    bins = np.arange(math.floor(min(all_data)), math.ceil(max(all_data)), bin_width)

    fig, ax = plt.subplots(figsize=(8, 6))  # Explicitly set an aspect ratio

    ax.hist(
        train_psnrs,
        bins=bins,
        histtype="step",
        label=f"train",
        weights=np.ones_like(train_psnrs) / len(train_psnrs),
    )
    ax.hist(
        val_psnrs,
        bins=bins,
        histtype="step",
        label=f"val",
        weights=np.ones_like(val_psnrs) / len(val_psnrs),
    )
    ax.hist(
        test_psnrs,
        bins=bins,
        histtype="step",
        label=f"test",
        weights=np.ones_like(test_psnrs) / len(test_psnrs),
    )

    ax.set_xlabel("PSNR", labelpad=12)
    ax.set_ylabel("Relative Bin Frequency")
    ax.set_title("Normalized Histogram of PSNR Distribution")
    ax.legend()

    lower_train, upper_train = np.percentile(train_psnrs, [2.5, 97.5])
    lower_val, upper_val = np.percentile(val_psnrs, [2.5, 97.5])
    lower_test, upper_test = np.percentile(test_psnrs, [2.5, 97.5])

    stats_text = (
        f"train:  N={len(train_psnrs):<8} μ={train_psnrs.mean():<6.2f} σ={train_psnrs.std():<5.2f} 95%=[{lower_train:.2f}, {upper_train:.2f}]\n"
        f"val:    N={len(val_psnrs):<8} μ={val_psnrs.mean():<6.2f} σ={val_psnrs.std():<5.2f} 95%=[{lower_val:.2f}, {upper_val:.2f}]\n"
        f"test:   N={len(test_psnrs):<8} μ={test_psnrs.mean():<6.2f} σ={test_psnrs.std():<5.2f} 95%=[{lower_test:.2f}, {upper_test:.2f}]"
    )

    plt.figtext(
        0.5, 0.02,
        stats_text,
        fontsize=9,
        family='monospace',
        horizontalalignment='center',
        verticalalignment='bottom',
        linespacing=1.4
    )

    plt.subplots_adjust(bottom=0.22)

    plt.savefig('psnr_hist.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    root = Path("../data/chestmnist")
    dataset_eda(root)
