"""Inspection preview generation. Display scaling never modifies input pixels."""

from pathlib import Path

import numpy as np
from PIL import Image

from lunamatch.ingestion.metadata import LunarImage


def preview_array(product: LunarImage, *, band: int = 0,
                  max_side: int = 1600) -> np.ndarray:
    """Create an 8-bit contrast-stretched grayscale inspection preview."""
    if not 0 <= band < product.bands:
        raise ValueError(f"Band must be in 0..{product.bands - 1}")
    plane = product.image if product.image.ndim == 2 else product.image[:, :, band]
    valid = np.isfinite(plane)
    if product.valid_mask is not None:
        valid &= product.valid_mask
    output = np.zeros(plane.shape, dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(plane[valid], (2, 98))
        if high > low:
            output[valid] = np.clip((plane[valid].astype(np.float64) - low) * 255 / (high - low), 0, 255).astype(np.uint8)
    if max(output.shape) > max_side:
        scale = max_side / max(output.shape)
        size = (max(1, round(output.shape[1] * scale)), max(1, round(output.shape[0] * scale)))
        output = np.asarray(Image.fromarray(output).resize(size, Image.Resampling.BILINEAR))
    return output


def save_preview(product: LunarImage, path: str | Path, *, band: int = 0,
                 max_side: int = 1600) -> Path:
    """Write a PNG preview with bounded dimensions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview_array(product, band=band, max_side=max_side)).save(path)
    return path
