"""Warp source image into reference pixel coordinates."""

import cv2
import numpy as np


def warp_source(source: np.ndarray, matrix: np.ndarray,
                reference_shape: tuple[int, int]) -> np.ndarray:
    """Warp a source plane or band cube to reference height and width."""
    height, width = reference_shape
    if source.ndim == 3:
        return np.stack([warp_source(source[:, :, band], matrix, reference_shape)
                         for band in range(source.shape[2])], axis=-1)
    if source.dtype not in (np.uint8, np.uint16, np.int16, np.float32, np.float64):
        raise ValueError(f"Warping does not support pixel dtype {source.dtype}")
    return cv2.warpPerspective(source, matrix, (width, height),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
