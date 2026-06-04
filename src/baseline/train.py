import math
import os
import time
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseline.cli import Config
from baseline.data import DatasetInfo, Loaders
from baseline.evaluation import evaluate
from baseline.models import BaseModel
from baseline.models.base import CheckpointMetrics
from baseline.plot import save_roc_plot


class _WandbLogger:
    """Thin wrapper around wandb, no-op when disabled or import fails."""

    def __init__(self, config: Config, hparams: dict[str, Any]) -> None:
        self._run = None
        if not config.wandb:
            return

        self._wandb = wandb
        wandb_dir = config.output_dir / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        slurm_meta = {
            key: os.environ[key]
            for key in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NAME",
                "SLURM_NODELIST",
                "SLURMD_NODENAME",
                "SLURM_GPUS_ON_NODE",
                "SLURM_ARRAY_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
            )
            if key in os.environ
        }
        self._run = wandb.init(
            project=config.wandb_project,
            name=config.run_name,
            tags=list(config.wandb_tags) or None,
            config={**hparams, **slurm_meta},
            dir=str(wandb_dir),
            save_code=True,
        )

        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

    def log(self, payload: dict[str, float], step: int | None = None) -> None:
        if self._run is None:
            return

        self._wandb.log(payload, step=step)

    def summary_update(self, payload: dict[str, Any]) -> None:
        if self._run is None:
            return

        self._wandb.summary.update(payload)

    def finish(self) -> None:
        if self._run is None:
            return
        self._wandb.finish()


def _gpu_metrics(device: torch.device) -> dict[str, float]:
    """Return per-epoch CUDA memory metrics; empty on CPU."""
    if device.type != "cuda":
        return {}

    idx = device.index if device.index is not None else torch.cuda.current_device()
    peak = torch.cuda.max_memory_allocated(idx) / (1024**3)
    reserved = torch.cuda.memory_reserved(idx) / (1024**3)
    torch.cuda.reset_peak_memory_stats(idx)

    return {
        "resources/gpu_peak_mem_gib": peak,
        "resources/gpu_reserved_mem_gib": reserved,
    }


def _build_criterion(is_multilabel: bool) -> nn.Module:
    """Pick the loss matching the dataset task.

    Args:
        is_multilabel: Whether the targets are multi-label binary
            vectors (in which case ``BCEWithLogitsLoss`` is used).

    Returns:
        A ready-to-use loss module.

    Raises:
        NotImplementedError: For multi-class data; only the multi-label
            branch is wired up.
    """
    if not is_multilabel:
        raise NotImplementedError(
            "Only multi-label loss (BCEWithLogitsLoss) is wired up.",
        )

    return nn.BCEWithLogitsLoss()


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup over the first 10% of optimizer steps, then cosine.

    Args:
        optimizer: Optimizer whose learning rate is scheduled.
        total_steps: Total number of optimizer steps across the whole run.

    Returns:
        A per-step scheduler; call ``.step()`` once after each ``optimizer.step()``.
    """
    warmup_steps = max(1, round(0.05 * total_steps))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
    )

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


def train(
    config: Config,
    model: BaseModel,
    loaders: Loaders,
    dataset_info: DatasetInfo,
) -> None:
    """Run the full training procedure.

    Trains from scratch with early stopping on validation AUC. The best
    checkpoint so far is written to ``model.pt`` in the model dir and
    overwritten whenever validation AUC improves. After training, the best
    checkpoint is evaluated on the test split and a square ROC plot is
    saved beside it. The wandb run is always closed, even on a crash or
    interrupt.

    Args:
        config: Parsed CLI configuration.
        model: Architecture wrapper to train.
        loaders: Train/val/test loaders.
        dataset_info: Static metadata describing the dataset.
    """
    device = config.device
    print(f"Using device: {device}")

    model.to(device)
    criterion = _build_criterion(dataset_info.is_multilabel)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    steps_per_epoch = math.ceil(
        len(loaders.train) / config.accum_steps
    )  # optimizer steps/epoch
    scheduler = _build_scheduler(optimizer, steps_per_epoch * config.epochs)

    config.model_dir.mkdir(parents=True, exist_ok=True)

    wandb_logger = _WandbLogger(config, asdict(config))
    best_path = config.model_dir / "model.pt"

    best_auc = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    training_start = time.monotonic()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(
            device.index if device.index is not None else torch.cuda.current_device(),
        )

    try:
        for epoch in range(1, config.epochs + 1):
            train_loss = train_one_epoch(
                model,
                loaders.train,
                criterion,
                optimizer,
                scheduler,
                device,
                epoch,
                config,
            )
            val = evaluate(
                model,
                loaders.val,
                device,
                config,
                dataset_info,
                split="val",
                criterion=criterion,
            )

            payload: dict[str, float] = {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/lr": float(scheduler.get_last_lr()[0]),
                "val/loss": val.loss,
                "val/auc": val.auc,
                "val/acc": val.acc,
                **_gpu_metrics(device),
            }

            for name, class_auc, class_acc in zip(
                dataset_info.class_names,
                val.per_class_auc,
                val.per_class_acc,
            ):
                payload[f"val_auc/{name}"] = class_auc
                payload[f"val_acc/{name}"] = class_acc

            wandb_logger.log(payload)
            print(
                f"epoch {epoch} | train loss {train_loss:.6f} | val loss {val.loss:.6f} "
                f"| val AUC {val.auc:.4f} | val ACC {val.acc:.4f}"
            )

            if val.auc > best_auc:
                best_auc, best_epoch = val.auc, epoch
                epochs_without_improvement = 0
                model.save(
                    best_path,
                    CheckpointMetrics(
                        epoch=epoch,
                        val_auc=val.auc,
                        val_accuracy=val.acc,
                        training_time_seconds=time.monotonic() - training_start,
                        device=str(device),
                        extra={
                            "per_class_auc": dict(
                                zip(dataset_info.class_names, val.per_class_auc)
                            ),
                            "per_class_acc": dict(
                                zip(dataset_info.class_names, val.per_class_acc)
                            ),
                        },
                    ),
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.patience:
                    print(
                        f"Early stopping at epoch {epoch} (best AUC {best_auc:.4f} @ epoch {best_epoch})."
                    )
                    break

        # Final evaluation on the test split using the best checkpoint.
        best_state = torch.load(best_path, map_location=device, weights_only=False)[
            "state_dict"
        ]
        model.load_state_dict(best_state)
        test = evaluate(
            model,
            loaders.test,
            device,
            config,
            dataset_info,
            split="test",
            criterion=criterion,
        )

        print(f"TEST | AUC {test.auc:.4f} | ACC {test.acc:.4f}")
        for name, class_auc, class_acc in zip(
            dataset_info.class_names,
            test.per_class_auc,
            test.per_class_acc,
        ):
            print(f"  {name:<20} AUC {class_auc:.4f} | ACC {class_acc:.4f}")

        save_roc_plot(
            test.y_true,
            test.y_score,
            dataset_info.class_names,
            test.per_class_auc,
            test.auc,
            config.model_dir / "roc_auc.png",
        )

        wandb_logger.summary_update(
            {"test/auc": test.auc, "test/acc": test.acc, "best_epoch": best_epoch},
        )
    finally:  # close wandb correctly on any crash/interrupt
        wandb_logger.finish()


def train_one_epoch(
    model: BaseModel,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    config: Config,
) -> float:
    model.train()
    running_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"epoch {epoch}")
    for step, (images, labels) in enumerate(pbar, start=1):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_function(logits, labels)
        (loss / config.accum_steps).backward()

        if step % config.accum_steps == 0:  # apply accumulated grads
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        running_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{running_loss / n_batches:.6f}")

    if n_batches % config.accum_steps != 0:  # flush leftover micro-batches
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    return running_loss / max(n_batches, 1)
