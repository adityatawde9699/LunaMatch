import numpy as np

from lunamatch.distribution.spatial_selection import select_uniform


def test_grid_cap_keeps_confident_inliers_in_each_cell() -> None:
    points = np.array([[1, 1], [2, 2], [3, 3], [8, 1], [8, 8]], dtype=float)
    confidence = np.array([0.1, 0.9, 0.5, 0.7, 1.0])
    inliers = np.array([True, True, True, True, False])
    selected = select_uniform(points, confidence, inliers, (10, 10), rows=2,
                              cols=2, max_per_cell=1)
    assert selected.tolist() == [1, 3]
