"""Small trainable descriptor network for local lunar image patches."""

from __future__ import annotations

from typing import Any


def _torch() -> Any:
    """Import torch lazily so the classical installation remains usable."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Training requires: pip install 'lunamatch[learned]'") from exc
    return torch, nn


def build_descriptor_class() -> Any:
    """Build the module class after loading the optional torch dependency."""
    torch, nn = _torch()

    class _Descriptor(nn.Module):
        """Compact normalized embedding network for one-channel patches."""

        def __init__(self, embedding_dim: int = 128) -> None:
            super().__init__()
            if embedding_dim < 8:
                raise ValueError("embedding_dim must be at least 8")
            self.embedding_dim = embedding_dim
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
                nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.GELU(),
                nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.projection = nn.Linear(128, embedding_dim)

        def forward(self, patches: Any) -> Any:
            """Return L2-normalized embeddings for ``N×1×H×W`` patches."""
            if patches.ndim != 4 or patches.shape[1] != 1:
                raise ValueError("Expected patches shaped N x 1 x H x W")
            features = self.encoder(patches).flatten(1)
            return torch.nn.functional.normalize(self.projection(features), dim=1)

    _Descriptor.__name__ = "LunarPatchDescriptor"
    return _Descriptor


try:
    LunarPatchDescriptor = build_descriptor_class()
except RuntimeError:  # pragma: no cover - exercised without the optional extra
    class LunarPatchDescriptor:  # type: ignore[no-redef]
        """Placeholder that reports the optional dependency clearly."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            _torch()
