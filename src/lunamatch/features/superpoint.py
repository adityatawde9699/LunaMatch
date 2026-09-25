"""Optional pretrained SuperPoint feature adapter."""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def superpoint_tensor(image: np.ndarray, device: str) -> "torch.Tensor":
    """Convert normalized 8-bit grayscale to SuperPoint's float tensor."""
    import torch

    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("SuperPoint needs an 8-bit grayscale image")
    return torch.from_numpy(image.copy()).float()[None, None].to(device) / 255
