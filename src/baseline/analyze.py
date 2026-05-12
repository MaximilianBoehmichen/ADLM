"""Inspect a saved checkpoint and render its training history.

Loads the ``.pt`` file produced by :meth:`baseline.models.base.BaseModel.save`,
prints the stored :class:`~baseline.models.base.CheckpointMetrics` and, if a
sibling ``history.json`` exists, regenerates the loss/accuracy/AUC PNGs into
a ``plots/`` directory next to the checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import torch

from baseline.plotting import plot_history


def _format_seconds(seconds: float) -> str:
    """Render a duration as ``HH:MM:SS``.

    Args:
        seconds: Wall-clock seconds to format.

    Returns:
        Zero-padded ``HH:MM:SS`` string.
    """
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def analyze(checkpoint_path: Path) -> None:
    """Print checkpoint statistics and render plots from ``history.json``.

    Args:
        checkpoint_path: Path to a ``.pt`` file written by
            :meth:`baseline.models.base.BaseModel.save`. The function
            looks for ``history.json`` next to the checkpoint and writes
            plots into ``<run_dir>/plots/``.

    Raises:
        FileNotFoundError: If ``checkpoint_path`` does not exist.
    """
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metrics = payload["metrics"]
    norm = payload["normalization"]

    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 60)
    print(f"  class_name           {payload.get('class_name', '?')}")
    print(f"  num_classes          {payload.get('num_classes', '?')}")
    print(f"  epoch                {metrics['epoch']}")
    print(f"  val_auc              {metrics['val_auc']:.4f}")
    print(f"  val_accuracy         {metrics['val_accuracy']:.4f}")
    test_auc = metrics.get("test_auc")
    test_acc = metrics.get("test_accuracy")
    print(f"  test_auc             {test_auc if test_auc is None else f'{test_auc:.4f}'}")
    print(f"  test_accuracy        {test_acc if test_acc is None else f'{test_acc:.4f}'}")
    elapsed = float(metrics.get("training_time_seconds", 0.0))
    print(f"  training_time        {_format_seconds(elapsed)} ({elapsed:.1f}s)")
    print(f"  device               {metrics.get('device', '?')}")
    print(f"  normalization.mean   {tuple(round(float(v), 4) for v in norm['mean'])}")
    print(f"  normalization.std    {tuple(round(float(v), 4) for v in norm['std'])}")
    if metrics.get("extra"):
        print(f"  extra                {metrics['extra']}")
    print("=" * 60)

    run_dir = checkpoint_path.parent
    history_path = run_dir / "history.json"
    if history_path.is_file():
        plots_dir = run_dir / "plots"
        plot_history(history_path, plots_dir)
        print(f"Wrote plots to {plots_dir}")
    else:
        print(f"No history.json next to checkpoint ({history_path}); skipping plots.")
