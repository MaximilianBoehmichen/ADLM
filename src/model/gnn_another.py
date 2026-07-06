import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops
import torch.nn.functional as F

from dataset.transforms import build_rotation


def prune_knn_edges(edge_index: Tensor, num_nodes: int, original_k: int = 15, keep_k: int = 8) -> Tensor:
    """Original neighbors to specified number of neighbors."""
    src, dst = edge_index

    src_pruned = src.view(num_nodes, original_k)[:, :keep_k].flatten()
    dst_pruned = dst.view(num_nodes, original_k)[:, :keep_k].flatten()

    return torch.stack([src_pruned, dst_pruned], dim=0)


class SumConv(MessagePassing):
    """MessagePassing equivalent to KNNConv. Name no longer accurate :)."""
    NUM_WEIGHTING_FEATURES: int
    num_bases = 2
    hidden_dim = 8
    out_channels: int

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_bases: int = 2,
        hidden_dim: int = 8,
        num_layers: int = 8,
        d: int = 2,
        mahalanobis: bool = False,
        rff_features: int = 0,
        rff_sigma: float = 1.0,
        rotate: bool = False,
    ) -> None:
        super().__init__(aggr=["sum"])
        self.out_channels = out_channels
        self.num_bases = num_bases
        self.hidden_dim = hidden_dim
        self.d = d
        self.rotations = 1 if d == 2 else 4
        self.mahalanobis = mahalanobis
        self.use_rff = rff_features > 0
        self.rotate = rotate
        self.rot_feat_dim = d * d if rotate else 2 * self.rotations

        if self.use_rff:
            self.register_buffer("rff_B", torch.randn(d, rff_features, generator=torch.Generator().manual_seed(848577)) * rff_sigma)

        pos_dim = 2 * rff_features if self.use_rff else d
        self.NUM_WEIGHTING_FEATURES = pos_dim + 2 * d + 1 + self.rot_feat_dim

        self.weighting = nn.Sequential(
            nn.Linear(self.NUM_WEIGHTING_FEATURES, self.hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
        )
        for _ in range(num_layers - 2):
            self.weighting.extend([
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
            ])

        self.weighting.extend([
            nn.Linear(self.hidden_dim, self.num_bases),
            nn.LeakyReLU(0.1),
        ])

        self.weighting.apply(self._init_weights)

        self.bases = nn.Linear(in_channels, num_bases * out_channels, bias=False)

    def _init_weights(self, m) -> None:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity="leaky_relu")

            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: Tensor, layout: Tensor, edge_index: Tensor) -> Tensor:
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))  # Why did we drop them in the preprocessing?
        h = self.bases(x)

        return self.propagate(edge_index, h=h, layout=layout)

    def message(self, h_j: Tensor, layout_i: Tensor, layout_j: Tensor) -> Tensor:
        pos_i, pos_j = layout_i[:, :self.d], layout_j[:, :self.d]
        scaling_i, scaling_j = layout_i[:, self.d:2*self.d], layout_j[:, self.d:2*self.d]
        rot_i, rot_j = layout_i[:, 2*self.d:2*self.d + self.rotations], layout_j[:, 2*self.d:2*self.d + self.rotations]

        rel_pos = pos_j - pos_i
        R_i = build_rotation(rot_i, self.d) if (self.rotate or self.mahalanobis) else None

        if self.rotate:
            rel_pos = torch.einsum("eij,ej->ei", R_i.mT, rel_pos)

        pos_feat = self._fourier(rel_pos) if self.use_rff else rel_pos

        if self.mahalanobis:
            frame = rel_pos if self.rotate else torch.einsum("eij,ej->ei", R_i.mT, rel_pos)
            dist = (frame / scaling_i.clamp(min=1e-6)).norm(dim=-1, keepdim=True)

        else:
            dist = rel_pos.norm(dim=-1, keepdim=True)

        if self.rotate:
            R_j = build_rotation(rot_j, self.d)
            rot_feat = (R_i.mT @ R_j).flatten(1)
        else:
            delta = rot_j - rot_i
            rot_feat = torch.cat([torch.cos(2 * delta), torch.sin(2 * delta)], dim=-1)

        weighting_features = torch.cat([
                pos_feat,
                dist,
                scaling_i,
                scaling_j,
                rot_feat,
            ],
            dim=-1,
        )
        coeff = self.weighting(weighting_features)
        h_j = h_j.view(-1, self.num_bases, self.out_channels)

        return torch.einsum("eb,ebo->eo", coeff, h_j)

    def update(self, aggr_out: Tensor) -> Tensor:
        return aggr_out

    def _fourier(self, x):
        proj = 2 * torch.pi * x @ self.rff_B

        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class ResNetBasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_bases: int = 8,
        hidden_dim: int = 8,
        num_layers: int = 3,
        d: int = 2,
        mahalanobis: bool = False,
        rff_features: int = 0,
        rff_sigma: float = 1.0,
        rotate: bool = False,
    ):
        super().__init__()

        self.conv1 = SumConv(
            in_channels,
            out_channels,
            num_bases,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            d=d,
            mahalanobis=mahalanobis,
            rff_features=rff_features,
            rff_sigma=rff_sigma,
            rotate=rotate,
        )
        self.norm1 = nn.LayerNorm(out_channels)

        self.conv2 = SumConv(
            out_channels,
            out_channels,
            num_bases,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            d=d,
            mahalanobis=mahalanobis,
            rff_features=rff_features,
            rff_sigma=rff_sigma,
            rotate=rotate,
        )
        self.norm2 = nn.LayerNorm(out_channels)

        self.pad = (out_channels - in_channels) // 2

    def forward(self, x: Tensor, layout: Tensor, edge_index: Tensor) -> Tensor:
        shortcut = x
        if self.pad:
            shortcut = F.pad(shortcut, (self.pad, self.pad))

        out = self.conv1(x, layout, edge_index)
        out = self.norm1(out)
        out = F.relu(out)

        out = self.conv2(out, layout, edge_index)
        out = self.norm2(out)

        return F.relu(out + shortcut)


class ResNetLikePYGGNN(nn.Module):
    CHANNELS = [16, 32, 64]

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        k: int = 9,
        num_bases: int = 8,
        hidden_dim: int = 8,
        num_layers: int = 8,
        d: int = 2,
        mahalanobis: bool = False,
        rff_features: int = 0,
        rff_sigma: float = 1.0,
        rotate: bool = False,
    ) -> None:
        super().__init__()
        self.k = k

        sumconv_kwargs = {
            "num_bases": num_bases,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "d": d,
            "mahalanobis": mahalanobis,
            "rff_features": rff_features,
            "rff_sigma": rff_sigma,
            "rotate": rotate,
        }

        self.stem_conv = SumConv(
            in_channels,
            self.CHANNELS[0],
            **sumconv_kwargs,
        )
        self.stem_norm = nn.LayerNorm(self.CHANNELS[0])

        self.stages = nn.ModuleList([
            ResNetBasicBlock(
                self.CHANNELS[0],
                self.CHANNELS[0],
                **sumconv_kwargs,
            ),
            ResNetBasicBlock(
                self.CHANNELS[0],
                self.CHANNELS[1],
                **sumconv_kwargs,
            ),
            ResNetBasicBlock(
                self.CHANNELS[1],
                self.CHANNELS[2],
                **sumconv_kwargs,
            ),
        ])

        self.head = nn.Linear(self.CHANNELS[-1], num_classes)

    def forward(self, data: Data) -> Tensor:
        x, layout, edge_index, batch = data.x, data.layout, data.edge_index, data.batch

        assert x is not None
        assert edge_index is not None
        assert layout is not None, "data.layout is missing — run the ExtractLayout transform in the pipeline."
        num_nodes = x.size(0)

        edge_index = prune_knn_edges(
            edge_index,
            num_nodes=num_nodes,
            original_k=data.num_edges // num_nodes,
            keep_k=self.k - 1  # to make space for removed self loop of the preprocessing
        )

        x = self.stem_conv(x, layout, edge_index)
        x = self.stem_norm(x)
        x = F.relu(x)

        for block in self.stages:
            x = block(x, layout, edge_index)

        x = global_mean_pool(x, batch)

        return self.head(x)
