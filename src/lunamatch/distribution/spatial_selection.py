"""Confidence-ranked, per-cell selection of verified correspondences."""

import numpy as np


def select_uniform(points: np.ndarray, confidence: np.ndarray, inliers: np.ndarray,
                   image_shape: tuple[int, int], *, rows: int = 8, cols: int = 8,
                   max_per_cell: int = 10) -> np.ndarray:
    """Return indices of high-confidence inliers with a grid cell cap."""
    if rows < 1 or cols < 1 or max_per_cell < 1:
        raise ValueError("Grid dimensions and max_per_cell must be positive")
    if len(points) != len(confidence) or len(points) != len(inliers):
        raise ValueError("Points, confidence, and inlier arrays must have equal length")
    height, width = image_shape
    grid = np.zeros((rows, cols), dtype=int)
    selected = []
    for index in np.argsort(-confidence, kind="stable"):
        if not inliers[index]:
            continue
        x, y = points[index]
        if not (0 <= x < width and 0 <= y < height):
            continue
        row = min(int(y * rows / height), rows - 1)
        col = min(int(x * cols / width), cols - 1)
        if grid[row, col] < max_per_cell:
            selected.append(int(index))
            grid[row, col] += 1
    return np.asarray(selected, dtype=int)
