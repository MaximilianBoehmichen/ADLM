"""MedMNIST data loading and augmentation.

For the moment only ``chestmnist`` (multi-label, 14 classes, grayscale) is
supported. The structure leaves the obvious hooks (task type, label dtype,
loss selection) in place; non-chestmnist datasets raise
:class:`NotImplementedError` until they are wired in explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import medmnist
import numpy as np
import torch
from medmnist import INFO
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

if TYPE_CHECKING:
    from baseline.cli import Config
    from baseline.models.base import BaseModel


CHESTMNIST_CLASSES: tuple[str, ...] = (
    "atelectasis",
    "cardiomegaly",
    "effusion",
    "infiltration",
    "mass",
    "nodule",
    "pneumonia",
    "pneumothorax",
    "consolidation",
    "edema",
    "emphysema",
    "fibrosis",
    "pleural_thickening",
    "hernia",
)


@dataclass(slots=True)
class DatasetInfo:
    """Static metadata for the dataset currently being trained on.

    Attributes:
        name: MedMNIST flag name (e.g. ``"chestmnist"``).
        task: MedMNIST task string (``"multi-label, binary-class"`` for
            chestmnist).
        num_classes: Number of output logits required by the head.
        class_names: Human-readable class names used for TensorBoard tags
            and final reporting.
        is_multilabel: ``True`` when each sample can carry several labels.
        in_channels_original: Number of channels in the raw images (1 for
            grayscale MedMNIST flags, 3 for colour ones).
    """

    name: str
    task: str
    num_classes: int
    class_names: tuple[str, ...]
    is_multilabel: bool
    in_channels_original: int


@dataclass(slots=True)
class Loaders:
    """The three loaders produced by :func:`build_dataloaders`."""

    train: DataLoader
    val: DataLoader
    test: DataLoader


def dataset_info(name: str) -> DatasetInfo:
    """Return :class:`DatasetInfo` for the given MedMNIST flag.

    Args:
        name: MedMNIST flag (e.g. ``"chestmnist"``).

    Returns:
        Static dataset metadata used by the rest of the pipeline.

    Raises:
        NotImplementedError: For any dataset other than ``chestmnist``;
            the scaffolding to extend support is in place but
            unverified.
    """
    if name != "chestmnist":
        raise NotImplementedError(
            f"Only 'chestmnist' is supported right now; got {name!r}. "
            "Extend dataset_info() and the label/loss handling in train.py "
            "before enabling other MedMNIST flags.",
        )
    raw = INFO[name]
    return DatasetInfo(
        name=name,
        task=raw["task"],
        num_classes=len(raw["label"]),
        class_names=CHESTMNIST_CLASSES,
        is_multilabel=True,
        in_channels_original=raw["n_channels"],
    )


class _GrayscaleToRGB:
    """Repeat a single-channel tensor along the channel dimension."""

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Replicate the channel dim if there is only one channel.

        Args:
            image: Tensor shaped ``(C, H, W)``.

        Returns:
            Tensor shaped ``(3, H, W)``; passed through unchanged if it
            already has 3 channels.
        """
        if image.shape[0] == 1:
            return image.repeat(3, 1, 1)
        return image


def _compute_dataset_stats(dataset: Dataset) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Compute per-channel mean/std over a dataset's raw images.

    Used when training from scratch so that the network sees normalised
    inputs matched to the data rather than to ImageNet.

    Args:
        dataset: The underlying MedMNIST training dataset (already
            converted to 3-channel float tensors in [0, 1]).

    Returns:
        A ``(mean, std)`` tuple of three-element tuples, ready to be
        handed to :class:`torchvision.transforms.Normalize`.
    """
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    total = torch.zeros(3)
    sq_total = torch.zeros(3)
    count = 0
    for images, _ in loader:
        batch = images.view(images.size(0), 3, -1)
        total += batch.mean(dim=2).sum(dim=0)
        sq_total += (batch**2).mean(dim=2).sum(dim=0)
        count += images.size(0)
    mean = total / count
    var = sq_total / count - mean**2
    std = torch.sqrt(torch.clamp(var, min=1e-8))
    return tuple(mean.tolist()), tuple(std.tolist())


class _MedMNISTWrapper(Dataset):
    """Apply a transform pipeline on top of a MedMNIST dataset.

    The underlying MedMNIST dataset returns ``(PIL.Image, np.ndarray)``;
    this wrapper applies the supplied transform to the image and converts
    the label to a ``float32`` tensor (multi-label) or ``int64`` tensor
    (multi-class).
    """

    def __init__(
        self,
        base: Dataset,
        transform: transforms.Compose,
        is_multilabel: bool,
    ) -> None:
        """Store the underlying dataset and the transform pipeline.

        Args:
            base: The raw MedMNIST dataset for one split.
            transform: Image preprocessing applied to every sample.
            is_multilabel: Whether labels should be returned as
                ``float32`` vectors (multi-label) or scalar ``int64``
                indices (multi-class).
        """
        self._base = base
        self._transform = transform
        self._is_multilabel = is_multilabel

    def __len__(self) -> int:
        """Return the number of samples in the underlying split."""
        return len(self._base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one transformed sample.

        Args:
            idx: Sample index within the underlying split.

        Returns:
            A ``(image, label)`` tuple where ``image`` is the
            transformed tensor and ``label`` is a ``float32`` vector
            (multi-label) or ``int64`` scalar (multi-class).
        """
        image, label = self._base[idx]
        image_t = self._transform(image)
        if self._is_multilabel:
            label_t = torch.from_numpy(np.asarray(label, dtype=np.float32))
        else:
            label_t = torch.tensor(int(np.asarray(label).item()), dtype=torch.long)
        return image_t, label_t


def build_dataloaders(config: Config, model: BaseModel) -> Loaders:
    """Build train/val/test loaders for the configured dataset.

    When fine-tuning, the loaders normalise inputs with the
    ImageNet-style stats baked into the model. When training from
    scratch, the channel-wise mean/std are computed on the training
    split and written back to ``model.normalization`` so that the saved
    checkpoint records the exact stats the model was trained with.

    Args:
        config: Training configuration.
        model: The model that will consume these loaders; its
            ``normalization`` attribute is read in fine-tune mode and
            overwritten in from-scratch mode.

    Returns:
        A :class:`Loaders` triplet (train/val/test).
    """
    info = dataset_info(config.dataset)
    data_root = config.output_dir / "medmnist_cache"
    data_root.mkdir(parents=True, exist_ok=True)
    dataset_cls = getattr(medmnist, INFO[config.dataset]["python_class"])

    base_transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            _GrayscaleToRGB(),
        ],
    )

    raw_train = dataset_cls(
        split="train",
        download=True,
        root=str(data_root),
        size=224,
    )
    raw_val = dataset_cls(split="val", download=True, root=str(data_root), size=224)
    raw_test = dataset_cls(split="test", download=True, root=str(data_root), size=224)

    if config.finetune:
        mean, std = model.normalization.mean, model.normalization.std
    else:
        stats_dataset = _MedMNISTWrapper(raw_train, base_transform, info.is_multilabel)
        mean, std = _compute_dataset_stats(stats_dataset)
        from baseline.models.base import NormalizationStats

        model.normalization = NormalizationStats(mean=mean, std=std)  # type: ignore[arg-type]

    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomRotation(config.rotation_degrees),
            transforms.ColorJitter(
                brightness=config.jitter,
                contrast=config.jitter,
                saturation=config.jitter,
            ),
            transforms.ToTensor(),
            _GrayscaleToRGB(),
            normalize,
        ],
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            _GrayscaleToRGB(),
            normalize,
        ],
    )

    train_ds = _MedMNISTWrapper(raw_train, train_transform, info.is_multilabel)
    val_ds = _MedMNISTWrapper(raw_val, eval_transform, info.is_multilabel)
    test_ds = _MedMNISTWrapper(raw_test, eval_transform, info.is_multilabel)

    loader_kwargs = {
        "num_workers": config.num_workers,
        "pin_memory": True,
        "persistent_workers": config.num_workers > 0,
    }
    return Loaders(
        train=DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        ),
        val=DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            **loader_kwargs,
        ),
        test=DataLoader(
            test_ds,
            batch_size=config.batch_size,
            shuffle=False,
            **loader_kwargs,
        ),
    )
