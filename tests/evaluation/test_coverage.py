import numpy as np

from lunamatch.distribution.coverage import coverage_metrics


def test_coverage_uses_inliers_and_reference_coordinates() -> None:
    points = np.array([[1, 1], [9, 1], [1, 9], [9, 9], [1, 1]], dtype=float)
    inliers = np.array([True, True, True, False, False])
    metrics = coverage_metrics(points, inliers, (10, 10), rows=2, cols=2)
    assert metrics["occupied_cells"] == 3
    assert metrics["spatial_coverage"] == 0.75
    assert metrics["matches_per_cell"] == [[1, 1], [1, 0]]
