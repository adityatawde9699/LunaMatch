"""SIFT image features for the classical baseline."""

import cv2
import numpy as np

from .feature_interface import Features


def extract_sift(image: np.ndarray, mask: np.ndarray | None = None,
                 max_features: int = 6000) -> Features:
    """Detect SIFT points on an 8-bit grayscale representation."""
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("SIFT requires an 8-bit grayscale image")
    detector = cv2.SIFT_create(nfeatures=max_features)
    mask_u8 = mask.astype(np.uint8) * 255 if mask is not None else None
    points, descriptors = detector.detectAndCompute(image, mask_u8)
    return Features(list(points), descriptors)
