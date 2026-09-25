"""Tests for raster warping, including OpenCV's image-size boundary."""

import cv2
import numpy as np

from lunamatch.geometry import warping
from lunamatch.geometry.warping import warp_source


def test_large_axis_warp_is_tiled_without_changing_identity_image():
    """Tiled warping should preserve all pixels across the 32767 limit."""
    for shape in ((3, 32768), (32768, 3)):
        source = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
        warped = warp_source(source, np.eye(3, dtype=np.float64), shape)

        np.testing.assert_array_equal(warped, source)


def test_tiled_fractional_warp_has_no_tile_boundary_seams(monkeypatch):
    """Overlapping source tiles preserve subpixel interpolation at seams."""
    monkeypatch.setattr(warping, "_MAX_WARP_AXIS", 10)
    source = np.arange(23 * 29, dtype=np.uint8).reshape(23, 29)
    matrix = np.array([[1, 0, 0.35], [0, 1, 0.45], [0, 0, 1]], dtype=np.float64)

    actual = warping._warp_tiled(source, matrix, *source.shape)
    expected = cv2.warpPerspective(source, matrix, (source.shape[1], source.shape[0]))

    np.testing.assert_array_equal(actual, expected)


def test_small_warp_keeps_standard_opencv_behavior():
    """Ordinary image dimensions continue through OpenCV's normal path."""
    source = np.arange(20, dtype=np.uint8).reshape(4, 5)
    matrix = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]], dtype=np.float64)

    actual = warp_source(source, matrix, source.shape)
    expected = cv2.warpPerspective(source, matrix, (5, 4))

    np.testing.assert_array_equal(actual, expected)
