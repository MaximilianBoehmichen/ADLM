import torch
from torch import nn


class PositionalEncoder(nn.Module):
    """Base encoder."""

    def __init__(
        self, num_frequencies: int, num_bands: int, sigma: float = 10.0, seed: int = 42
    ) -> None:
        super().__init__()
        self.register_buffer(
            "B", self._build_B(num_frequencies, num_bands, sigma, seed)
        )

    def _build_B(
        self, num_frequencies: int, num_bands: int, sigma: float, seed: int
    ) -> torch.Tensor:
        raise NotImplementedError

    @property
    def out_dim(self) -> int:
        return 2 * self.B.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor: # for projection
        proj = x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class MixedPE(PositionalEncoder):
    """Deterministic octave-band Fourier Features.

    Combines NeRF's octave magnitude with RFF-style coordinate mixing.
    This mixture with deterministic frequencies and no sigma and a dot-product instead of element-wise sin/cos,
    Should be easier to tune while also using fewer parameters.

    But this means it is now neither NeRF nor RFF.
    """

    def _build_B(
        self, num_frequencies: int, num_bands: int, sigma: float, seed: int
    ) -> torch.Tensor:
        golden = torch.pi * (3.0 - 5.0**0.5)
        j = torch.arange(num_frequencies)

        angles = (j * golden) % torch.pi
        magnitudes = 2.0 ** torch.linspace(0, num_bands - 1, num_frequencies) * torch.pi
        return torch.stack([torch.cos(angles), torch.sin(angles)]) * magnitudes


class RFFPE(PositionalEncoder):
    """Random Gaussian Fourier features."""

    def _build_B(
        self, num_frequencies: int, num_bands: int, sigma: float, seed: int
    ) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        return (
            2 * torch.pi * torch.randn(2, num_frequencies, generator=generator) * sigma
        )


class INR(nn.Module): # main architecture
    def __init__(
        self,
        *,
        pe: type[PositionalEncoder] = RFFPE, 
        hidden_dim: int = 64,
        hidden_layers: int = 1,
        seed: int = 42,
        out_dim: int = 1,
        num_frequencies: int = 16,
        num_bands: int = 8,
        sigma: float = 10.0,
    ) -> None:
        torch.manual_seed(seed)
        super().__init__()

        self.pe = pe(num_frequencies, num_bands, sigma, seed)
        layers: list[nn.Module] = [nn.Linear(self.pe.out_dim, hidden_dim), nn.ReLU()]

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.pe(x))
