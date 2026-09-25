"""Grid occupancy metrics for image correspondences."""

import numpy as np


def coverage_metrics(points: np.ndarray, inliers: np.ndarray,
                     image_shape: tuple[int, int], rows: int = 8,
                     cols: int = 8) -> dict[str, float | int | list[list[int]]]:
    """Measure reference-image occupancy by geometrically verified matches."""
    if rows < 1 or cols < 1:
        raise ValueError("Grid dimensions must be positive")
    height, width = image_shape
    if height < 1 or width < 1:
        raise ValueError("Image dimensions must be positive")
    grid = np.zeros((rows, cols), dtype=np.int64)
    selected = points[inliers]
    for x, y in selected:
        if 0 <= x < width and 0 <= y < height:
            row = min(int(y * rows / height), rows - 1)
            col = min(int(x * cols / width), cols - 1)
            grid[row, col] += 1
    occupied = int(np.count_nonzero(grid))
    return {
        "spatial_coverage": occupied / (rows * cols),
        "occupied_cells": occupied,
        "grid_rows": rows,
        "grid_cols": cols,
        "matches_per_cell": grid.tolist(),
        "distribution_variance": float(np.var(grid)),
    }
