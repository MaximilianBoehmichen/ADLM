"""Step-1 pretraining: fit one INR per MedMNIST image and save its weights.

Fits the frozen best architecture found by the hparam search (``BEST_CONFIG``)
to every image of the requested split(s) and stores each fitted INR as a
``.pt`` file::

    {output_dir}/inr2vec/{dataset}/{split}/{index:05d}.pt

Each file holds the model ``state_dict`` plus the label, reconstruction PSNR and
the config used, so step 2 (inr2vec encoder) can load the weights directly.

The run is **resumable**: any index whose output file already exists is skipped,
so re-launching (or sharding across array jobs with ``--start-idx/--end-idx``)
only fits what is missing.

Run:
    PYTHONPATH=src python -m inr2vec.inr_step1.pretrain \\
        --dataset chestmnist --splits train val test
    # quick smoke test
    PYTHONPATH=src python -m inr2vec.inr_step1.pretrain \\
        --dataset chestmnist --splits test --end-idx 4 --epochs 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from medmnist import INFO

from inr2vec.inr_step1.defs import BEST_CONFIG, PROJECT_ROOT
from inr2vec.inr_step1.hparam_search import load_split, make_coord_grid, to_tensor
from inr2vec.inr_step1.model import INR
from inr2vec.inr_step1.train import train_inr

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data"


def build_label_tensor(label_np: np.ndarray, task: str) -> torch.Tensor:
    """Build the per-image label tensor according to the MedMNIST task type.

    multi-label, binary-class -> float (1, C) for BCEWithLogitsLoss
    everything else           -> long  (1,)   for CrossEntropyLoss
    """
    if task == "multi-label, binary-class":
        return torch.tensor(label_np, dtype=torch.float32).reshape(1, -1)
    return torch.tensor([int(np.asarray(label_np).squeeze())], dtype=torch.long)


def _config_meta() -> dict:
    """JSON/torch-friendly copy of BEST_CONFIG (``pe`` class -> its name)."""
    meta = dict(BEST_CONFIG)
    meta["pe"] = BEST_CONFIG["pe"].__name__
    return meta


def process_split(
    dataset_flag: str,
    split: str,
    output_dir: Path,
    epochs: int,
    lr: float,
    patience: int,
    image_size: int,
    device: torch.device,
    start_idx: int = 0,
    end_idx: int | None = None,
) -> None:
    """Fit and save one INR for every image in a split."""
    dataset = load_split(dataset_flag, split=split, size=image_size)
    task = INFO[dataset_flag]["task"]

    split_dir = output_dir / "inr2vec" / dataset_flag / split
    split_dir.mkdir(parents=True, exist_ok=True)

    coords = make_coord_grid(image_size, image_size, device)
    config_meta = _config_meta()

    total = len(dataset)
    start = max(0, start_idx)
    end = total if end_idx is None else min(total, end_idx)

    for i in range(start, end):
        out_path = split_dir / f"{i:05d}.pt"
        if out_path.exists():
            continue

        img_np, label_np = dataset[i]
        y = build_label_tensor(label_np, task)
        target = to_tensor(img_np).reshape(-1, 1).to(device)

        # A fresh INR per image; INR.__init__ re-seeds, so every image starts
        # from the same shared initialization and the same RFFPE basis.
        model = INR(**BEST_CONFIG).to(device)
        psnr = train_inr(model, coords, target, epochs, lr, patience=patience)

        torch.save(
            {
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "y": y,
                "psnr": float(psnr),
                "index": i,
                "config": config_meta,
            },
            out_path,
        )

        print(
            f"[{split}] {i + 1}/{total} (range {start}:{end}) | "
            f"PSNR: {psnr:5.2f} dB | {out_path}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit one INR per MedMNIST image and save its weights."
    )
    parser.add_argument("--dataset", default="chestmnist")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root output dir; files go to "
        "{output_dir}/inr2vec/{dataset}/{split}/{index:05d}.pt",
    )
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Stop a fit after this many epochs without MSE improvement "
        "(0 disables early stopping). Matches the hparam search.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="First image index (inclusive). Use with --end-idx to shard jobs.",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="Last image index (exclusive). Defaults to end of split.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)

    print(f"Device: {device} | config: {_config_meta()}")
    for split in args.splits:
        print(f"\n--- Fitting INRs for {args.dataset} [{split}] ---")
        process_split(
            args.dataset,
            split,
            args.output_dir,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            image_size=args.image_size,
            device=device,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
