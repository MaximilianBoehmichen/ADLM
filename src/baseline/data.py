from dataclasses import dataclass

import medmnist
import numpy as np
import torch
from medmnist import INFO
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from baseline.cli import Config
from baseline.models.base import NormalizationStats


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
    class_names: list[str]
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
        )
    info = INFO[name]

    return DatasetInfo(
        name=name,
        task=info["task"],
        num_classes=len(info["label"]),
        class_names=[info["label"][str(i)] for i in range(len(info["label"]))],
        is_multilabel="multi-label" in info["task"],
        in_channels_original=info["n_channels"],
    )


def _compute_dataset_stats(
    dataset: Dataset, in_channels: int = 1
) -> tuple[list[float], list[float]]:
    """Compute per-channel mean/std over a dataset's raw images.

    Used when training from scratch so that the network sees normalized
    inputs matched to the data rather than to ImageNet.

    Args:
        dataset: The underlying MedMNIST training dataset

    Returns:
        A ``(mean, std)`` tuple of a list with one entry per dimension.
    """
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    total = torch.zeros(in_channels, dtype=torch.float32)
    sq_total = torch.zeros(in_channels, dtype=torch.float32)
    count = 0

    for images, _ in loader:
        batch = images.view(images.size(0), in_channels, -1)
        total += batch.mean(dim=2).sum(dim=0)
        sq_total += (batch**2).mean(dim=2).sum(dim=0)
        count += images.size(0)

    mean = total / count
    var = sq_total / count - mean**2
    std = torch.sqrt(torch.clamp(var, min=1e-8))

    return mean.tolist(), std.tolist()


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


def build_dataloaders(config: Config) -> tuple[Loaders, NormalizationStats]:
    """Build train/val/test loaders for the configured dataset.

    Models are always trained from scratch: the channel-wise mean/std are
    computed on the training split and written back to
    ``model.normalization`` so that the saved checkpoint records the exact
    stats the model was trained with. Images keep their native channel
    count (1 for grayscale MedMNIST flags such as chestmnist).

    Args:
        config: Training configuration.

    Returns:
        A :class:`Loaders` triplet (train/val/test).
    """
    if config.gaussian_root is not None:
        from baseline.gaussian import build_gaussian_loaders

        return build_gaussian_loaders(config)

    if config.inr_root is not None:
        from inr2vec.inr_step2.data import build_inr_loaders

        return build_inr_loaders(config)

    info = dataset_info(config.dataset)
    data_root = config.output_dir / "medmnist_cache"
    data_root.mkdir(parents=True, exist_ok=True)
    dataset_cls = getattr(medmnist, INFO[config.dataset]["python_class"])

    raw_train = dataset_cls(
        split="train",
        download=True,
        root=str(data_root),
        size=config.image_size,
    )
    raw_val = dataset_cls(
        split="val", download=True, root=str(data_root), size=config.image_size
    )
    raw_test = dataset_cls(
        split="test", download=True, root=str(data_root), size=config.image_size
    )

    # Per-channel mean/std on the raw training images.
    stats_transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
        ],
    )
    stats_dataset = _MedMNISTWrapper(raw_train, stats_transform, info.is_multilabel)
    mean, std = _compute_dataset_stats(stats_dataset, info.in_channels_original)
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            normalize,
        ],
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            normalize,
        ],
    )

    train_ds = _MedMNISTWrapper(raw_train, train_transform, info.is_multilabel)
    val_ds = _MedMNISTWrapper(raw_val, eval_transform, info.is_multilabel)
    test_ds = _MedMNISTWrapper(raw_test, eval_transform, info.is_multilabel)

    loader_kwargs = {
        "pin_memory": True,
        "persistent_workers": True if config.num_workers > 0 else False,
        "num_workers": config.num_workers,
        "prefetch_factor": 2 if config.num_workers > 0 else None,
    }
    loaders = Loaders(
        train=DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,  # avoids problems with BatchNorm for too small remaining batches
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

    return loaders, NormalizationStats(mean=mean, std=std)
