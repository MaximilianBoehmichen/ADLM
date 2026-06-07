from pathlib import Path

from inr2vec.inr_step1.model import MixedPE, RFFPE

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "medmnist_cache"

HPARAMS_SEARCH_SPACE: dict[str, list] = {
    "hidden_dim": [8, 12, 16, 24, 32, 48],
    "hidden_layers": [2, 3, 4, 5, 6, 7, 8],
    "num_frequencies": [96, 128, 160, 192, 224, 256, 288, 320],
    "num_bands": [6],
    "seed": [848577, 839156007, 1306993369, 9738130],
    "sigma": [0.5, 1, 2, 3, 4, 5],
    "pe": [RFFPE],
}
