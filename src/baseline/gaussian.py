"""On-the-fly rendering of preprocessed Gaussian graphs back to images.

Each ``.pt`` written by ``preprocess_dataset.py`` stores a torch_geometric
``Data`` whose node features are ``[mu(2) | scaling(2) | theta(1) | colour(1)]``.
:class:`GaussianRenderDataset` rasterizes those primitives to a single-channel
image whenever a sample is accessed and discards it afterwards, so the baseline
can be trained on the compressed representation without materializing a second
copy of the dataset on disk.
"""

from pathlib import Path

import faiss
import torch
from medmnist import INFO
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from baseline.cli import Config
from baseline.data import Loaders
from baseline.models.base import NormalizationStats

faiss.omp_set_num_threads(1)


def _init_worker(worker_id: int) -> None:
    """Pin each dataloader worker to one thread.

    Rendering runs faiss and torch inside every worker; left unbounded they
    each spawn a full thread pool and oversubscribe the machine.

    Args:
        worker_id: Index supplied by the dataloader (unused).
    """
    faiss.omp_set_num_threads(1)
    torch.set_num_threads(1)


def _render_gaussians(features: torch.Tensor, image_size: int, k: int) -> torch.Tensor:
    """Rasterize one Gaussian graph to a single-channel image.

    Mirrors ``GaussianRepresentationND.forward``: each pixel is the
    neighbour-normalized, Gaussian-weighted average of the colours of its ``k``
    nearest Gaussians.

    Args:
        features: Node features ``[mu(2) | scaling(2) | theta(1) | colour(1)]``
            of shape ``(N, 6)`` exactly as written by ``extract_pyg_data``.
        image_size: Side length of the square image to render.
        k: Number of nearest Gaussians blended per pixel.

    Returns:
        A ``(1, image_size, image_size)`` float tensor in the colour range the
        Gaussians were fit on (``~[0, 1]``).
    """
    mus = features[:, :2]
    scalings = features[:, 2:4]
    theta = features[:, 4]
    colors = features[:, 5]

    cos, sin = torch.cos(theta), torch.sin(theta)
    rotation = torch.stack(
        [torch.stack([cos, -sin], dim=-1), torch.stack([sin, cos], dim=-1)],
        dim=-2,
    )
    scale_inv_sq = torch.diag_embed(1.0 / scalings**2)
    sigma_inv = rotation @ scale_inv_sq @ rotation.transpose(-1, -2)

    axis = torch.arange(image_size, dtype=torch.float32)
    coords = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1).reshape(
        -1, 2
    )

    index = faiss.IndexFlatL2(2)
    index.add(mus.contiguous().numpy())
    _, neighbours = index.search(coords.numpy(), k)
    neighbours = torch.from_numpy(neighbours).long()

    offset = coords.unsqueeze(1) - mus[neighbours]
    mahalanobis = (
        offset * torch.einsum("pkij,pkj->pki", sigma_inv[neighbours], offset)
    ).sum(-1)
    weights = torch.exp(-0.5 * mahalanobis)
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

    pixels = (weights * colors[neighbours]).sum(dim=1)

    return pixels.reshape(1, image_size, image_size)


class GaussianRenderDataset(Dataset):
    """Renders one MedMNIST split from its Gaussian graphs on access.

    The split's file order is the official MedMNIST order (zero-padded shards
    sorted lexicographically), so an unshuffled loader feeds the official
    evaluator rows in the right order.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        dataset_name: str,
        image_size: int,
        k: int,
        normalize: transforms.Normalize | None = None,
        cache: bool = False,
    ) -> None:
        """Discover the graph files for one split.

        Args:
            root: Directory containing ``{split}/**/*.pt``.
            split: MedMNIST split name.
            dataset_name: MedMNIST flag, used to assert split completeness.
            image_size: Side length passed to the renderer.
            k: Number of nearest Gaussians blended per pixel.
            normalize: Optional channel normalization applied after rendering.
            cache: Keep each rendered image in memory as ``uint8`` after its
                first access so later epochs skip the render. The cache is
                per-worker, so peak memory scales with the worker count.

        Raises:
            FileNotFoundError: If the split directory holds no files.
            ValueError: If the file count differs from the official split
                size; a partial run would silently misalign the evaluator.
        """
        self._files = sorted((root / split).rglob("*.pt"))
        if not self._files:
            raise FileNotFoundError(f"No .pt files under {root / split}")

        expected = INFO[dataset_name]["n_samples"][split]
        if len(self._files) != expected:
            raise ValueError(
                f"{split}: found {len(self._files)} graphs but the official split "
                f"has {expected}."
            )

        self._image_size = image_size
        self._k = k
        self.normalize = normalize
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = (
            {} if cache else None
        )

    def __len__(self) -> int:
        """Return the number of graphs in the split."""
        return len(self._files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Render one graph and return it with its label.

        Args:
            idx: Sample index within the split.

        Returns:
            A ``(image, label)`` tuple: a ``(1, H, W)`` float image and a
            ``float32`` multi-label target vector.
        """
        if self._cache is not None and idx in self._cache:
            quantized, label = self._cache[idx]
            image = quantized.float() / 255.0

        else:
            data = torch.load(self._files[idx], weights_only=False, map_location="cpu")
            image = _render_gaussians(data.x.float(), self._image_size, self._k)
            label = data.y.reshape(-1).float()

            if self._cache is not None:
                quantized = (image * 255.0).round().clamp(0, 255).to(torch.uint8)
                self._cache[idx] = (quantized, label)
                image = quantized.float() / 255.0

        if self.normalize is not None:
            image = self.normalize(image)

        return image, label


def _render_stats(
    dataset: GaussianRenderDataset,
    num_workers: int,
    max_images: int = 4096,
) -> tuple[list[float], list[float]]:
    """Estimate the channel mean/std over rendered training images.

    Rendering the full split only to gather statistics would dominate startup,
    so a parallel loader streams a capped, shuffled subset instead.

    Args:
        dataset: The training :class:`GaussianRenderDataset` (without
            normalization applied).
        num_workers: Worker count for the streaming loader.
        max_images: Number of rendered images to accumulate before stopping.

    Returns:
        A ``(mean, std)`` tuple, each a single-element list.
    """
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=_init_worker,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )

    total = torch.zeros(1)
    sq_total = torch.zeros(1)
    count = 0

    for images, _ in tqdm(loader):
        batch = images.view(images.size(0), 1, -1)
        total += batch.mean(dim=2).sum(dim=0)
        sq_total += (batch**2).mean(dim=2).sum(dim=0)
        count += images.size(0)

        if count >= max_images:
            break

    mean = total / count
    std = torch.sqrt(torch.clamp(sq_total / count - mean**2, min=1e-8))

    return mean.tolist(), std.tolist()


def build_gaussian_loaders(config: Config) -> tuple[Loaders, NormalizationStats]:
    """Build train/val/test loaders that render Gaussian graphs to images.

    Normalization is estimated on the rendered training split so the network
    sees inputs matched to the rasterized data rather than to the raw MedMNIST
    images.

    Args:
        config: Training configuration; ``gaussian_root`` is the directory
            holding the ``{split}/*.pt`` graphs and ``image_size`` the render
            resolution.

    Returns:
        The loader triplet and the normalization statistics to record on the
        checkpoint.
    """
    root = config.gaussian_root

    train_ds = GaussianRenderDataset(
        root,
        "train",
        config.dataset,
        config.image_size,
        config.gaussian_k,
        cache=config.gaussian_cache,
    )
    mean, std = _render_stats(train_ds, config.num_workers)
    normalize = transforms.Normalize(mean=mean, std=std)
    train_ds.normalize = normalize

    val_ds = GaussianRenderDataset(
        root,
        "val",
        config.dataset,
        config.image_size,
        config.gaussian_k,
        normalize,
        cache=config.gaussian_cache,
    )
    test_ds = GaussianRenderDataset(
        root,
        "test",
        config.dataset,
        config.image_size,
        config.gaussian_k,
        normalize,
        cache=config.gaussian_cache,
    )

    loader_kwargs = {
        "pin_memory": True,
        "persistent_workers": config.num_workers > 0,
        "num_workers": config.num_workers,
        "prefetch_factor": 2 if config.num_workers > 0 else None,
        "worker_init_fn": _init_worker,
        # Workers start after wandb.init() (and CUDA on GPU) spin up threads in
        # the main process; forking would inherit their locks and deadlock.
        "multiprocessing_context": "spawn" if config.num_workers > 0 else None,
    }

    loaders = Loaders(
        train=DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        ),
        val=DataLoader(
            val_ds, batch_size=config.batch_size, shuffle=False, **loader_kwargs
        ),
        test=DataLoader(
            test_ds, batch_size=config.batch_size, shuffle=False, **loader_kwargs
        ),
    )

    return loaders, NormalizationStats(mean=mean, std=std)
