from pathlib import Path

from inr2vec.inr_step1.model import MixedPE, RFFPE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "medmnist_cache"

HPARAMS_SEARCH_SPACE: dict[str, list] = {
    "hidden_dim": [16, 24, 32, 48, 64, 96],
    "hidden_layers": [2, 3, 4, 5, 6, 7, 8],
    "num_frequencies": [16, 24, 32, 48, 64, 96, 128, 160, 192],
    "num_bands": [6, 7, 8],
    "seed": [848577, 839156007, 1306993369, 9738130],
    "sigma": [6, 8, 10, 12],
    "pe": [MixedPE, RFFPE],
}
