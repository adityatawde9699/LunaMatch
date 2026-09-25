"""Shared feature extraction result."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class Features:
    """Keypoints and their descriptor matrix."""

    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray | None
