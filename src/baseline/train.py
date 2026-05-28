"""Training loop for baseline MedMNIST models.

Features:
- Gradient accumulation with a fractional-epoch x-axis for TensorBoard, so
  runs with different ``accum_batch_size`` line up visually.
- Two-phase fine-tuning: the backbone is frozen for the first
  ``freeze_epochs`` epochs and unfrozen afterwards. The optimiser is rebuilt
  at the unfreeze boundary.
- Per-epoch evaluation on the ``val`` split, early stopping on the overall
  AUC, final evaluation on the ``test`` split once on the best checkpoint.
- TensorBoard logs and a ``history.json`` snapshot for offline plotting.
- Optional Weights & Biases mirroring of every scalar plus per-epoch
  GPU/CPU resource metrics.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from baseline.device import resolve_device
from baseline.metrics import EvalMetrics, compute_metrics
from baseline.models.base import CheckpointMetrics
from baseline.plotting import plot_history


if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from baseline.cli import Config
    from baseline.data import DatasetInfo, Loaders
    from baseline.models.base import BaseModel


@dataclass(slots=True)
class History:
    """Time series collected over the course of training.

    All entries with key ``train_loss`` are recorded at sub-epoch
    resolution (one per optimiser step); every other entry is recorded
    once per epoch. The ``epoch`` field of each list holds the
    fractional-epoch x-coordinate (1.0 = end of epoch 1, 1.5 = halfway
    through epoch 2, ...).
    """

    train_loss: list[tuple[float, float]] = field(default_factory=list)
    val_loss: list[tuple[float, float]] = field(default_factory=list)
    val_accuracy: list[tuple[float, float]] = field(default_factory=list)
    val_auc: list[tuple[float, float]] = field(default_factory=list)
    val_per_class_accuracy: list[tuple[float, list[float]]] = field(default_factory=list)
    val_per_class_auc: list[tuple[float, list[float]]] = field(default_factory=list)
    test: dict[str, float | list[float]] = field(default_factory=dict)

    def dump(self, path: Path) -> None:
        """Persist the history as JSON.

        Args:
            path: Destination file. Parent directories are created on
                demand.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


WEIGHT_DECAY = 1e-4


def _build_optimizer(params, lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)


class _WandbLogger:
    """Thin wrapper around wandb, no-op when disabled or import fails."""

    def __init__(self, config: Config, hparams: dict[str, Any]) -> None:
        self._run = None
        if not config.wandb:
            return
        try:
            import wandb
        except ImportError:
            print("wandb not installed; skipping W&B logging.")
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
            entity=config.wandb_entity,
            name=config.run_name,
            group=config.wandb_group,
            tags=list(config.wandb_tags) or None,
            config={**hparams, **slurm_meta},
            dir=str(wandb_dir),
            save_code=True,
        )

    @property
    def enabled(self) -> bool:
        return self._run is not None

    def log(self, payload: dict[str, float], step: int | None = None) -> None:
        if self._run is None:
            return
        if step is None:
            self._wandb.log(payload)
        else:
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
    peak = torch.cuda.max_memory_allocated(idx) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(idx) / (1024 ** 3)
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


@torch.no_grad()
def _evaluate(
    model: BaseModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    is_multilabel: bool,
    desc: str,
) -> EvalMetrics:
    """Run ``model`` over ``loader`` and return aggregate metrics.

    Args:
        model: Model to evaluate.
        loader: Dataloader producing ``(image, label)`` tuples.
        criterion: Loss module used to compute the reported mean loss.
        device: Device on which inputs and targets are placed.
        is_multilabel: Whether to use multi-label metrics.
        desc: Description shown by the ``tqdm`` progress bar.

    Returns:
        The aggregated :class:`EvalMetrics` for this split.
    """
    model.eval()
    losses: list[float] = []
    weights: list[int] = []
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        losses.append(loss.item())
        weights.append(images.size(0))
        all_logits.append(logits.detach().cpu().numpy())
        all_targets.append(labels.detach().cpu().numpy())
    weighted_loss = float(np.average(losses, weights=weights))
    logits_np = np.concatenate(all_logits, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    return compute_metrics(
        logits_np,
        targets_np,
        loss=weighted_loss,
        is_multilabel=is_multilabel,
    )


def _log_eval(
    writer: SummaryWriter,
    wandb_logger: _WandbLogger,
    metrics: EvalMetrics,
    epoch: float,
    class_names: tuple[str, ...],
    split: str,
) -> None:
    """Write a full set of evaluation metrics to TensorBoard and W&B."""
    step = _to_step(epoch)
    payload: dict[str, float] = {
        f"loss/{split}": metrics.loss,
        f"acc/overall/{split}": metrics.accuracy,
        f"auc/overall/{split}": metrics.auc,
    }
    for name, value in zip(class_names, metrics.per_class_accuracy, strict=True):
        payload[f"acc/class_{name}/{split}"] = float(value)
    for name, value in zip(class_names, metrics.per_class_auc, strict=True):
        if not np.isnan(value):
            payload[f"auc/class_{name}/{split}"] = float(value)
    for tag, value in payload.items():
        writer.add_scalar(tag, value, step)
    wandb_logger.log({**payload, "epoch": epoch}, step=step)


def _save_checkpoint(
    model: BaseModel,
    config: Config,
    epoch: int,
    metrics: EvalMetrics,
    training_time_seconds: float,
    device: torch.device,
) -> Path:
    """Persist a new best-AUC checkpoint and return its path.

    Args:
        model: Model whose state dict is to be saved.
        config: Training configuration (used for the output directory).
        epoch: Epoch number at which the checkpoint is being saved.
        metrics: Validation metrics that motivated saving.
        training_time_seconds: Wall-clock seconds elapsed since training
            started.
        device: Device the model is currently on, stored in the
            checkpoint for later inspection.
    """
    path = config.model_dir / f"best.pt"
    ckpt = CheckpointMetrics(
        epoch=epoch,
        val_auc=metrics.auc,
        val_accuracy=metrics.accuracy,
        training_time_seconds=training_time_seconds,
        device=str(device),
        extra={
            "per_class_auc": metrics.per_class_auc,
            "per_class_accuracy": metrics.per_class_accuracy,
            "config": asdict(config),
        },
    )
    model.save(path, ckpt)
    return path


def train(
    config: Config,
    model: BaseModel,
    loaders: Loaders,
    dataset_info: DatasetInfo,
) -> None:
    """Run the full training procedure.

    Args:
        config: Parsed CLI configuration.
        model: Architecture wrapper to train.
        loaders: Train/val/test loaders.
        dataset_info: Static metadata describing the dataset.
    """
    device = resolve_device()
    print(f"Using device: {device}")
    model.to(device)
    criterion = _build_criterion(dataset_info.is_multilabel)

    writer = SummaryWriter(log_dir=str(config.tensorboard_dir))
    history = History()
    config.model_dir.mkdir(parents=True, exist_ok=True)

    hparams = {
        "model": config.model,
        "dataset": config.dataset,
        "lr": config.lr,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": config.batch_size,
        "accum_batch_size": config.accum_batch_size,
        "finetune": int(config.finetune),
        "freeze_epochs": config.freeze_epochs,
        "rotation": config.rotation_degrees,
        "jitter": config.jitter,
        "image_size": config.image_size,
        "epochs": config.epochs,
        "seed": config.seed,
        "device": str(device),
    }
    wandb_logger = _WandbLogger(config, hparams)
    wandb_log_every = int(config.wandb) if config.wandb else 0

    steps_per_epoch = len(loaders.train)
    optimizer_steps_per_epoch = max(1, steps_per_epoch // config.accumulation_steps)

    if config.finetune:
        model.freeze_backbone()
    else:
        model.unfreeze_backbone()
    optimizer = _build_optimizer(model.trainable_parameters(), config.lr)

    best_auc = -float("inf")
    best_path: Path | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    last_epoch_run = 0
    global_opt_step = 0
    training_start = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(
            device.index if device.index is not None else torch.cuda.current_device(),
        )

    for epoch in range(1, config.epochs + 1):
        if config.finetune and epoch == config.freeze_epochs + 1:
            model.unfreeze_backbone()
            optimizer = _build_optimizer(model.trainable_parameters(), config.lr)

        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        microbatch_in_step = 0
        optimizer_step_in_epoch = 0
        epoch_start = time.monotonic()
        samples_this_epoch = 0

        pbar = tqdm(
            loaders.train,
            desc=f"epoch {epoch}/{config.epochs}",
            leave=False,
        )
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            (loss / config.accumulation_steps).backward()
            running_loss += loss.item()
            microbatch_in_step += 1
            samples_this_epoch += images.size(0)

            if microbatch_in_step == config.accumulation_steps:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step_loss = running_loss / microbatch_in_step
                optimizer_step_in_epoch += 1
                global_opt_step += 1
                frac_epoch = (
                    epoch
                    - 1
                    + optimizer_step_in_epoch / optimizer_steps_per_epoch
                )
                step = _to_step(frac_epoch)
                writer.add_scalar("loss/train", step_loss, step)
                if wandb_log_every and (
                    global_opt_step == 1 or global_opt_step % wandb_log_every == 0
                ):
                    wandb_logger.log(
                        {"loss/train": step_loss, "epoch": frac_epoch},
                        step=step,
                    )
                history.train_loss.append((frac_epoch, step_loss))
                pbar.set_postfix(loss=f"{step_loss:.4f}")
                running_loss = 0.0
                microbatch_in_step = 0

        # Flush a trailing partial accumulation, if any.
        if microbatch_in_step > 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step_loss = running_loss / microbatch_in_step
            optimizer_step_in_epoch += 1
            global_opt_step += 1
            frac_epoch = float(epoch)
            step = _to_step(frac_epoch)
            writer.add_scalar("loss/train", step_loss, step)
            if wandb_log_every and (
                    global_opt_step == 1 or global_opt_step % wandb_log_every == 0
                ):
                wandb_logger.log(
                    {"loss/train": step_loss, "epoch": frac_epoch},
                    step=step,
                )
            history.train_loss.append((frac_epoch, step_loss))

        val_metrics = _evaluate(
            model,
            loaders.val,
            criterion,
            device,
            dataset_info.is_multilabel,
            desc=f"val {epoch}",
        )
        _log_eval(
            writer, wandb_logger, val_metrics,
            float(epoch), dataset_info.class_names, "val",
        )
        history.val_loss.append((float(epoch), val_metrics.loss))
        history.val_accuracy.append((float(epoch), val_metrics.accuracy))
        history.val_auc.append((float(epoch), val_metrics.auc))
        history.val_per_class_accuracy.append((float(epoch), val_metrics.per_class_accuracy))
        history.val_per_class_auc.append((float(epoch), val_metrics.per_class_auc))

        epoch_seconds = time.monotonic() - epoch_start
        resource_payload: dict[str, float] = {
            "resources/epoch_seconds": epoch_seconds,
            "resources/samples_per_second": (
                samples_this_epoch / epoch_seconds if epoch_seconds > 0 else 0.0
            ),
            **_gpu_metrics(device),
        }
        step = _to_step(float(epoch))
        for tag, value in resource_payload.items():
            writer.add_scalar(tag, value, step)
        wandb_logger.log({**resource_payload, "epoch": float(epoch)}, step=step)

        print(
            f"[epoch {epoch}] val loss={val_metrics.loss:.4f} "
            f"acc={val_metrics.accuracy:.4f} auc={val_metrics.auc:.4f}",
        )

        last_epoch_run = epoch

        if val_metrics.auc > best_auc:
            best_auc = val_metrics.auc
            best_epoch = epoch
            best_path = _save_checkpoint(
                model,
                config,
                epoch,
                val_metrics,
                training_time_seconds=time.monotonic() - training_start,
                device=device,
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"Early stopping at epoch {epoch} (no AUC improvement for {config.patience} epochs).")
                break

        history.dump(config.model_dir / "history.json")

    if best_path is None:
        raise RuntimeError("Training finished without ever improving val AUC.")

    best_model, best_ckpt = type(model).load(best_path, map_location=device)
    best_model.to(device)
    test_metrics = _evaluate(
        best_model,
        loaders.test,
        criterion,
        device,
        dataset_info.is_multilabel,
        desc="test",
    )
    _log_eval(
        writer, wandb_logger, test_metrics,
        float(best_epoch), dataset_info.class_names, "test",
    )
    history.test = {
        "loss": test_metrics.loss,
        "accuracy": test_metrics.accuracy,
        "auc": test_metrics.auc,
        "per_class_accuracy": test_metrics.per_class_accuracy,
        "per_class_auc": test_metrics.per_class_auc,
        "epoch": best_epoch,
    }
    history.dump(config.model_dir / "history.json")

    final_ckpt = CheckpointMetrics(
        epoch=best_ckpt.epoch,
        val_auc=best_ckpt.val_auc,
        val_accuracy=best_ckpt.val_accuracy,
        training_time_seconds=best_ckpt.training_time_seconds,
        device=best_ckpt.device,
        test_auc=test_metrics.auc,
        test_accuracy=test_metrics.accuracy,
        extra={
            **best_ckpt.extra,
            "test_per_class_auc": test_metrics.per_class_auc,
            "test_per_class_accuracy": test_metrics.per_class_accuracy,
        },
    )
    final_path = config.model_dir / f"best_epoch_{best_epoch:03d}_with_test.pt"
    best_model.save(final_path, final_ckpt)

    total_train_seconds = time.monotonic() - training_start
    summary = {
        "hparam/test_auc": test_metrics.auc,
        "hparam/test_accuracy": test_metrics.accuracy,
        "hparam/best_val_auc": best_auc,
        "hparam/best_epoch": best_epoch,
        "hparam/train_time_seconds": total_train_seconds,
        "resources/epochs_run": last_epoch_run,
    }
    writer.add_hparams({**hparams, "epochs_run": last_epoch_run}, summary)
    writer.close()
    wandb_logger.summary_update(summary)
    wandb_logger.finish()

    print(
        f"Training complete. Best epoch {best_epoch}, val AUC={best_auc:.4f}, "
        f"test AUC={test_metrics.auc:.4f}, test acc={test_metrics.accuracy:.4f}.",
    )

    plot_history(config.model_dir / "history.json", config.model_dir / "plots")


def _to_step(frac_epoch: float, resolution: int = 1000) -> int:
    """Convert a fractional epoch into the integer step TensorBoard needs.

    ``SummaryWriter.add_scalar`` only accepts integer global steps, so we
    multiply by a fixed resolution to preserve fractional ordering while
    keeping the x-axis intuitive (1000 = end of epoch 1).

    Args:
        frac_epoch: Fractional epoch coordinate.
        resolution: Multiplier applied before rounding to an integer.

    Returns:
        The integer global step passed to TensorBoard.
    """
    return int(round(frac_epoch * resolution))
