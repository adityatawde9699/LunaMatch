"""Image pyramid with coordinates measured relative to original pixels."""

import cv2
import numpy as np


def build_pyramid(image: np.ndarray, levels: int = 4) -> list[tuple[np.ndarray, float, float]]:
    """Return (image, original_x_scale, original_y_scale) from fine to coarse."""
    if levels < 1:
        raise ValueError("Pyramid levels must be positive")
    result = [(image, 1.0, 1.0)]
    current = image
    for _ in range(1, levels):
        if min(current.shape[:2]) < 32:
            break
        next_image = cv2.pyrDown(current)
        result.append((next_image, image.shape[1] / next_image.shape[1],
                       image.shape[0] / next_image.shape[0]))
        current = next_image
    return result
