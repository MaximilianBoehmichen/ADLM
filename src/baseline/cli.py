"""CLI and configuration for baseline training."""

import argparse
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

import torch

from baseline.defs import DEFAULT_DATA_DIR
from baseline.device import resolve_device
from baseline.models import MODELS


@dataclass(slots=True)
class Config:
    model: str
    dataset: str
    epochs: int
    patience: int
    lr: float
    image_size: int
    batch_size: int
    accum_batch_size: int
    output_dir: Path
    seed: int
    device: torch.device
    num_workers: int
    gaussian_root: Path | None = None
    gaussian_k: int = 15
    gaussian_cache: bool = False
    run_name: str = field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    wandb: bool = False
    wandb_project: str = "ADLM-baseline"
    wandb_tags: tuple[str, ...] = ()

    @property
    def accum_steps(self) -> int:
        """Number of minibatches per optimizer step."""
        if self.accum_batch_size % self.batch_size != 0:
            raise ValueError(
                "accum_batch_size must be a multiple of batch_size "
                f"(got {self.accum_batch_size} and {self.batch_size}).",
            )

        return self.accum_batch_size // self.batch_size

    @property
    def model_dir(self) -> Path:
        """Directory under which checkpoints for this run are stored."""
        return self.output_dir / "models" / self.model / self.run_name


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser used by the CLI.

    Returns:
        A parser whose options match the fields of :class:`Config`.
    """
    parser = argparse.ArgumentParser(
        description="Baseline training of CNN/ViT models on MedMNIST.",
    )

    parser.add_argument(
        "--model",
        choices=MODELS.keys(),
        required=True,
        help="Architecture to train.",
    )
    parser.add_argument(
        "--dataset",
        default="chestmnist",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Early-stopping patience on validation AUC.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Constant learning rate for AdamW.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Dataset image size.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accum-batch-size", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--gaussian-root",
        type=Path,
        default=None,
        help="Dir holding {split}/*.pt Gaussian graphs. When set, the model "
        "trains on rendered graphs instead of the raw images.",
    )
    parser.add_argument(
        "--gaussian-k",
        type=int,
        default=15,
        help="Nearest Gaussians blended per pixel when rendering.",
    )
    parser.add_argument(
        "--gaussian-cache",
        action="store_true",
        help="Cache each rendered image in memory as uint8 after first access "
        "(per-worker; peak memory scales with --num-workers).",
    )
    parser.add_argument("--seed", type=int, default=848577)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional override for the run name (default: current timestamp).",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
    )
    parser.add_argument(
        "--wandb-project",
        default="ADLM-baseline",
        help="W&B project name (used when --wandb is set).",
    )
    parser.add_argument(
        "--wandb-tag",
        action="append",
        default=[],
        dest="wandb_tags",
        help="Optional W&B tag; repeat to add multiple.",
    )

    return parser


def parse_args(argv: list[str] | None = None) -> Config:
    """Parse command-line arguments into a :class:`Config`.

    Args:
        argv: Optional explicit argument list.

    Returns:
        A fully-populated :class:`Config` ready to drive a training run.
    """
    args = build_parser().parse_args(argv)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")

    return Config(
        model=args.model,
        dataset=args.dataset,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        device=resolve_device(),
        image_size=args.image_size,
        batch_size=args.batch_size,
        accum_batch_size=args.accum_batch_size,
        output_dir=args.output_dir,
        seed=args.seed,
        num_workers=args.num_workers,
        gaussian_root=args.gaussian_root,
        gaussian_k=args.gaussian_k,
        gaussian_cache=args.gaussian_cache,
        run_name=run_name,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_tags=tuple(args.wandb_tags),
    )
