"""Optional dark-region rejection for uncertain cast shadows."""

import numpy as np


def shadow_mask(image: np.ndarray, valid_mask: np.ndarray | None = None,
                percentile: float = 5.0) -> np.ndarray:
    """Mask the darkest valid pixels using a within-image percentile.

    This is a heuristic, not physical shadow classification.
    """
    valid = np.ones(image.shape, dtype=bool) if valid_mask is None else valid_mask.copy()
    if not np.any(valid):
        return valid
    threshold = np.percentile(image[valid], percentile)
    return valid & (image > threshold)
