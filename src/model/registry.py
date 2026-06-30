"""Model registry: build_model(arch, **kwargs) → nn.Module."""
from __future__ import annotations

import torch.nn as nn

ARCHS = ("resgcn", "resgcn_v2", "edgeconv", "gat")


def build_model(arch: str, **kwargs) -> nn.Module:
    if arch == "resgcn":
        from model.gcn_classifier import ResGCNClassifier
        return ResGCNClassifier(**kwargs)
    if arch == "resgcn_v2":
        from model.resgcn_v2 import ResGCNClassifierV2
        return ResGCNClassifierV2(**kwargs)
    if arch == "edgeconv":
        from model.edgeconv_classifier import EdgeConvClassifier
        return EdgeConvClassifier(**kwargs)
    if arch == "gat":
        from model.gat_classifier import GATClassifier
        return GATClassifier(**kwargs)
    raise ValueError(f"Unknown arch {arch!r}. Choose from: {ARCHS}")