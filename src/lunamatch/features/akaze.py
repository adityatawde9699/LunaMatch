"""AKAZE feature extraction."""

import cv2
import numpy as np

from .feature_interface import Features


def extract_akaze(image: np.ndarray, mask: np.ndarray | None = None,
                  max_features: int = 6000) -> Features:
    """Detect AKAZE binary descriptors on an 8-bit grayscale image."""
    detector = cv2.AKAZE_create(max_points=max_features)
    mask_u8 = mask.astype(np.uint8) * 255 if mask is not None else None
    points, descriptors = detector.detectAndCompute(image, mask_u8)
    return Features(list(points), descriptors)
