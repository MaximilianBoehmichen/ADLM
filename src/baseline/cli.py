"""Command-line interface and configuration for baseline training.

Defines the :class:`Config` dataclass that bundles every option exposed by the
training entry point and a :func:`parse_args` function that constructs it from
``sys.argv``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

MODEL_CHOICES: tuple[str, ...] = (
    "resnet8",
    "resnet18",
    "densenet121",
    "mobilenetv3",
    "efficientnetb0",
    "deit_tiny",
)

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(slots=True)
class Config:
    """Container for every option that drives a training run.

    Attributes:
        model: Name of the architecture to instantiate; one of
            :data:`MODEL_CHOICES`.
        dataset: Name of the MedMNIST dataset flag (e.g. ``"chestmnist"``).
        epochs: Maximum number of training epochs.
        patience: Number of consecutive epochs without an improvement in the
            per-epoch validation AUC after which training is stopped.
        finetune: If ``True``, start from pretrained ImageNet weights and use
            the freeze schedule. If ``False``, weights are randomly
            initialised and ``freeze_epochs`` is ignored.
        freeze_epochs: Number of initial epochs during which every parameter
            outside of the classification head stays frozen.
        lr: Constant learning rate used by AdamW.
        batch_size: Mini-batch size handed to the data loader.
        accum_batch_size: Effective batch size after gradient accumulation.
            Must be a multiple of ``batch_size``.
        rotation_degrees: Maximum rotation (in degrees) used for the random
            rotation augmentation.
        jitter: Strength of the brightness/contrast/saturation augmentation
            applied via ``ColorJitter``.
        image_size: Spatial size of the square inputs fed to the network.
        num_workers: Number of subprocesses used by the data loader.
        output_dir: Root directory under which models, TensorBoard logs and
            history files are stored.
        seed: Seed used for ``torch``, ``numpy`` and Python's ``random``.
        run_name: Identifier used to namespace this run's artefacts.
        device: String identifier of the resolved device. Populated by
            :func:`baseline.main.main` after the CLI has been parsed; the
            empty string means "not yet resolved".
        wandb: Either ``False`` (disabled) or a positive integer ``N``
            meaning "mirror TensorBoard scalars to Weights & Biases and
            push ``loss/train`` once every ``N`` optimiser steps". Passing
            ``--wandb`` without a value defaults to ``64``.
        wandb_project: W&B project name. Ignored unless ``wandb`` is set.
        wandb_entity: W&B entity (team or user). ``None`` falls back to
            the wandb default.
        wandb_group: Optional W&B run group, useful for SLURM sweeps.
        wandb_tags: Optional list of W&B tags for the run.
    """

    model: str
    dataset: str
    epochs: int
    patience: int
    finetune: bool
    freeze_epochs: int
    lr: float
    batch_size: int
    accum_batch_size: int
    rotation_degrees: float
    jitter: float
    image_size: int
    num_workers: int
    output_dir: Path
    seed: int
    run_name: str = field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    device: str = ""
    wandb: int | bool = False
    wandb_project: str = "adlm-baseline"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_tags: tuple[str, ...] = ()

    @property
    def accumulation_steps(self) -> int:
        """Number of mini-batches per optimiser step."""
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

    @property
    def tensorboard_dir(self) -> Path:
        """Directory under which TensorBoard event files for this run live."""
        return self.output_dir / "tensorboard" / self.model / self.run_name


def build_parser() -> argparse.ArgumentParser:
    """Construct the :mod:`argparse` parser used by the CLI.

    Returns:
        A parser whose options match the fields of :class:`Config`.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune or train baseline CNN/ViT models on MedMNIST.",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        required=True,
        help=(
            "Architecture to train. CNNs: resnet8 (custom, no pretrained), "
            "resnet18, densenet121, mobilenetv3 (large), efficientnetb0. "
            "ViT: deit_tiny (DeiT-Tiny/16, ~5.7M params, via timm)."
        ),
    )
    parser.add_argument("--dataset", default="chestmnist")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early-stopping patience on validation AUC.",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Train from random initialisation rather than fine-tuning.",
    )
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=100,
        help="Epochs to keep the backbone frozen while fine-tuning.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Constant learning rate for AdamW.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accum-batch-size", type=int, default=128)
    parser.add_argument("--rotation-degrees", type=float, default=15.0)
    parser.add_argument("--jitter", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--seed", type=int, default=848577)
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional override for the run name (default: current timestamp).",
    )
    parser.add_argument(
        "--wandb",
        nargs="?",
        type=int,
        const=64,
        default=False,
        metavar="N",
        help=(
            "Enable W&B logging. Optionally pass a positive integer N to "
            "throttle per-step loss logging to once every N optimiser "
            "steps (default 64 when --wandb is given alone)."
        ),
    )
    parser.add_argument(
        "--wandb-project",
        default="ADLM-baseline",
        help="W&B project name (used when --wandb is set).",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="W&B entity (team/user). Defaults to the wandb-configured entity.",
    )
    parser.add_argument(
        "--wandb-group",
        default=None,
        help="Optional W&B run group (e.g. to bundle SLURM array jobs).",
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
        argv: Optional explicit argument list. ``None`` falls back to
            ``sys.argv[1:]``, which is what :mod:`argparse` does by
            default.

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
        finetune=not args.from_scratch,
        freeze_epochs=args.freeze_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        accum_batch_size=args.accum_batch_size,
        rotation_degrees=args.rotation_degrees,
        jitter=args.jitter,
        image_size=args.image_size,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        seed=args.seed,
        run_name=run_name,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_tags=tuple(args.wandb_tags),
    )
