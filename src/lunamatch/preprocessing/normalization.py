"""Conservative display and matching intensity normalization."""

import cv2
import numpy as np

from lunamatch.ingestion.metadata import LunarImage


def matching_gray(product: LunarImage, *, band: int = 0,
                  clahe: bool = False) -> np.ndarray:
    """Convert one band to masked 8-bit grayscale using robust percentiles.

    The image is a derived matching representation, never calibrated data.
    """
    if not 0 <= band < product.bands:
        raise ValueError(f"Band must be in 0..{product.bands - 1}")
    plane = product.image if product.image.ndim == 2 else product.image[:, :, band]
    valid = np.isfinite(plane)
    if product.valid_mask is not None:
        valid &= product.valid_mask
    result = np.zeros(plane.shape, dtype=np.uint8)
    if not np.any(valid):
        raise ValueError("Image has no valid pixels")
    low, high = np.percentile(plane[valid], (1, 99))
    if high <= low:
        raise ValueError("Image has no usable intensity variation")
    result[valid] = np.clip((plane[valid].astype(np.float64) - low) * 255 / (high - low),
                            0, 255).astype(np.uint8)
    if clahe:
        result = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(result)
        result[~valid] = 0
    return result
