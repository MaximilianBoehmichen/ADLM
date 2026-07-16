import shutil
from pathlib import Path

from tqdm import tqdm

DATASET_SOURCE = "./data/chestmnist"
DATASET_TARGET = "./data/chestmnistNEW"

DIGITS_PER_SUBDIR = 3


def reorg_dataset(root: str) -> None:
    path = Path(root)
    target = Path(DATASET_TARGET)
    split_dirs = [x for x in path.iterdir() if x.is_dir()]

    print(split_dirs)

    for split in split_dirs:
        print(split.stem)
        print(target / split.stem)
        print(f"processing {split}")

        for x in tqdm(split.iterdir()):
            if x.suffix != ".pt":
                continue

            index: str = x.stem
            subdir_name = f"{"".join(index[i] if len(index) - i > DIGITS_PER_SUBDIR else "0" for i in range(len(index)))}"
            file_name = f"{index[-3:]}.pt"

            new_dir = target / split.stem / subdir_name
            new_path = new_dir / file_name

            new_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(x), str(new_path))


if __name__ == "__main__":
    reorg_dataset(DATASET_SOURCE)
