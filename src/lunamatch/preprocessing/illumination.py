"""Local contrast representation for differing solar illumination."""

import cv2
import numpy as np


def local_normalize(image: np.ndarray, *, sigma: float = 15.0,
                    epsilon: float = 1.0) -> np.ndarray:
    """Remove slow intensity variation with local mean and deviation.

    Returns an 8-bit matching representation; no radiometric calibration is
    inferred. This can reduce useful albedo signal, so it stays optional.
    """
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("Local normalization needs 8-bit grayscale")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    data = image.astype(np.float32)
    mean = cv2.GaussianBlur(data, (0, 0), sigma)
    variance = cv2.GaussianBlur(data * data, (0, 0), sigma) - mean * mean
    z = (data - mean) / np.sqrt(np.maximum(variance, 0) + epsilon)
    return np.clip(z * 38 + 127, 0, 255).astype(np.uint8)
