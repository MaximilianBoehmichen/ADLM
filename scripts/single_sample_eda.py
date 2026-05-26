from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
import statsmodels.api as sm

def load_measurements(p: Path) -> list[float]:
    files = list(p.rglob("*.pt"))
    psnrs: list = []

    for file in files:
        d = torch.load(file, map_location="cpu", weights_only=False)
        psnrs.append(float(d.psnr))

    return psnrs

def single_sample_dist(p: Path) -> None:
    """Plots the distribution of PSNR for one sample."""
    psnrs = load_measurements(p)

    fig, ax = plt.subplots(1, 4, squeeze=False, figsize=(15, 4))
    ax[0, 0].hist(psnrs, bins=np.arange(37, 43, 0.125))
    ax[0, 1].violinplot([psnrs], [1], showextrema=False)
    ax[0, 2].boxplot([psnrs])
    sm.qqplot(np.array(psnrs), line="45", fit=True, ax=ax[0, 3])

    ax[0, 0].set_box_aspect(1)
    ax[0, 1].set_box_aspect(1)
    ax[0, 2].set_box_aspect(1)
    ax[0, 3].set_box_aspect(1)

    plt.suptitle("PSNR distribution for train sample <sample>")
    plt.show()


if __name__ == "__main__":
    p = Path("<path>")
    single_sample_dist(p)