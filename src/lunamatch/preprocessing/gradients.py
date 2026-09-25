"""Gradient magnitude as an optional terrain structure representation."""

import cv2
import numpy as np


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Compute robustly stretched Sobel gradient magnitude."""
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("Gradient input needs 8-bit grayscale")
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    high = float(np.percentile(magnitude, 99))
    if high <= 0:
        return np.zeros_like(image)
    return np.clip(magnitude * (255 / high), 0, 255).astype(np.uint8)
