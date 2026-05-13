"""Training loop for baseline MedMNIST models.

Features:
- Gradient accumulation with a fractional-epoch x-axis for TensorBoard, so
  runs with different ``accum_batch_size`` line up visually.
- Two-phase fine-tuning: the backbone is frozen for the first
  ``freeze_epochs`` epochs and unfrozen afterwards. The training strategy
  (optimiser + LR schedule) is shared across both phases; the optimiser is
  rebuilt at the unfreeze boundary while the LR schedule continues over the
  global optimiser-step counter.
- Per-epoch evaluation on the ``val`` split, early stopping on the overall
  AUC, final evaluation on the ``test`` split once on the best checkpoint.
- TensorBoard logs and a ``history.json`` snapshot for offline plotting.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

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


@dataclass(slots=True, frozen=True)
class Strategy:
    """Optimiser + LR schedule bundle selected via ``--strategy``."""

    name: str
    optimizer_factory: Callable[[Iterable[nn.Parameter], float], torch.optim.Optimizer]
    weight_decay: float
    warmup_epochs: int
    start_lr_ref: float
    min_lr_ref: float

    def lr_at_step(
        self,
        step: int,
        warmup_steps: int,
        total_steps: int,
        lr: float,
    ) -> float:
        """Compute the LR for optimiser step index ``step`` (0-based)."""
        peak = lr
        start = self.start_lr_ref
        end = self.min_lr_ref
        if warmup_steps > 0 and step < warmup_steps:
            return start + (peak - start) * (step / warmup_steps)
        cosine_len = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / cosine_len)
        return end + 0.5 * (peak - end) * (1.0 + math.cos(math.pi * progress))


_STRATEGIES: dict[str, Strategy] = {
    "strategy1": Strategy(
        name="strategy1",
        optimizer_factory=lambda params, lr: torch.optim.AdamW(
            params, lr=lr, weight_decay=0.05,
        ),
        weight_decay=0.05,
        warmup_epochs=10,
        start_lr_ref=1e-6,
        min_lr_ref=1e-5,
    ),
}


def _get_strategy(name: str) -> Strategy:
    try:
        return _STRATEGIES[name]
    except KeyError as err:
        raise ValueError(f"Unknown training strategy {name!r}.") from err


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


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
    metrics: EvalMetrics,
    epoch: float,
    class_names: tuple[str, ...],
    split: str,
) -> None:
    """Write a full set of evaluation metrics to TensorBoard.

    Args:
        writer: TensorBoard summary writer.
        metrics: The metrics returned by :func:`_evaluate`.
        epoch: Fractional epoch used as the x-coordinate.
        class_names: Class labels used to tag per-class scalars.
        split: Split name appended to the TensorBoard tag (e.g.
            ``"val"`` or ``"test"``).
    """
    step = _to_step(epoch)
    writer.add_scalar(f"loss/{split}", metrics.loss, step)
    writer.add_scalar(f"acc/overall/{split}", metrics.accuracy, step)
    writer.add_scalar(f"auc/overall/{split}", metrics.auc, step)
    for name, value in zip(class_names, metrics.per_class_accuracy, strict=True):
        writer.add_scalar(f"acc/class_{name}/{split}", value, step)
    for name, value in zip(class_names, metrics.per_class_auc, strict=True):
        if not np.isnan(value):
            writer.add_scalar(f"auc/class_{name}/{split}", value, step)


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

    strategy = _get_strategy(config.strategy)

    steps_per_epoch = len(loaders.train)
    optimizer_steps_per_epoch = max(1, steps_per_epoch // config.accumulation_steps)
    total_opt_steps = config.epochs * optimizer_steps_per_epoch
    warmup_opt_steps = strategy.warmup_epochs * optimizer_steps_per_epoch

    if config.finetune:
        model.freeze_backbone()
    else:
        model.unfreeze_backbone()
    optimizer = strategy.optimizer_factory(model.trainable_parameters(), config.lr)
    global_opt_step = 0
    _set_lr(
        optimizer,
        strategy.lr_at_step(
            global_opt_step, warmup_opt_steps, total_opt_steps, config.lr,
        ),
    )

    best_auc = -float("inf")
    best_path: Path | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    last_epoch_run = 0
    training_start = time.monotonic()

    for epoch in range(1, config.epochs + 1):
        if config.finetune and epoch == config.freeze_epochs + 1:
            model.unfreeze_backbone()
            optimizer = strategy.optimizer_factory(
                model.trainable_parameters(),
                strategy.lr_at_step(
                    global_opt_step, warmup_opt_steps, total_opt_steps,
                    config.lr,
                ),
            )

        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        microbatch_in_step = 0
        optimizer_step_in_epoch = 0

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

            if microbatch_in_step == config.accumulation_steps:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step_loss = running_loss / microbatch_in_step
                optimizer_step_in_epoch += 1
                global_opt_step += 1
                current_lr = strategy.lr_at_step(
                    global_opt_step, warmup_opt_steps, total_opt_steps,
                    config.lr,
                )
                _set_lr(optimizer, current_lr)
                frac_epoch = (
                    epoch
                    - 1
                    + optimizer_step_in_epoch / optimizer_steps_per_epoch
                )
                writer.add_scalar("loss/train", step_loss, _to_step(frac_epoch))
                writer.add_scalar("lr", current_lr, _to_step(frac_epoch))
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
            current_lr = strategy.lr_at_step(
                global_opt_step, warmup_opt_steps, total_opt_steps,
                config.lr,
            )
            _set_lr(optimizer, current_lr)
            frac_epoch = float(epoch)
            writer.add_scalar("loss/train", step_loss, _to_step(frac_epoch))
            writer.add_scalar("lr", current_lr, _to_step(frac_epoch))
            history.train_loss.append((frac_epoch, step_loss))

        val_metrics = _evaluate(
            model,
            loaders.val,
            criterion,
            device,
            dataset_info.is_multilabel,
            desc=f"val {epoch}",
        )
        _log_eval(writer, val_metrics, float(epoch), dataset_info.class_names, "val")
        history.val_loss.append((float(epoch), val_metrics.loss))
        history.val_accuracy.append((float(epoch), val_metrics.accuracy))
        history.val_auc.append((float(epoch), val_metrics.auc))
        history.val_per_class_accuracy.append((float(epoch), val_metrics.per_class_accuracy))
        history.val_per_class_auc.append((float(epoch), val_metrics.per_class_auc))

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
    writer.add_hparams(
        {
            "model": config.model,
            "strategy": config.strategy,
            "lr": config.lr,
            "weight_decay": strategy.weight_decay,
            "warmup_epochs": strategy.warmup_epochs,
            "min_lr": strategy.min_lr_ref,
            "batch_size": config.batch_size,
            "accum_batch_size": config.accum_batch_size,
            "finetune": int(config.finetune),
            "freeze_epochs": config.freeze_epochs,
            "rotation": config.rotation_degrees,
            "jitter": config.jitter,
            "image_size": config.image_size,
            "epochs": config.epochs,
            "epochs_run": last_epoch_run,
            "seed": config.seed,
            "device": str(device),
        },
        {
            "hparam/test_auc": test_metrics.auc,
            "hparam/test_accuracy": test_metrics.accuracy,
            "hparam/best_val_auc": best_auc,
            "hparam/best_epoch": best_epoch,
            "hparam/train_time_seconds": total_train_seconds,
        },
    )
    writer.close()

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
