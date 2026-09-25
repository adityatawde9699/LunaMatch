"""Matcher output shared by classical and future learned adapters."""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Correspondences:
    """Matched source/reference pixel coordinates and confidence values."""

    source_points: np.ndarray
    reference_points: np.ndarray
    confidence: np.ndarray

    def __len__(self) -> int:
        return len(self.source_points)
