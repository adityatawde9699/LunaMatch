"""Robust image-to-image geometric verification."""

from dataclasses import dataclass

import cv2
import numpy as np

from lunamatch.matching.matcher_interface import Correspondences


class GeometryError(ValueError):
    """Raised when a transform cannot be estimated reliably."""


@dataclass(slots=True)
class GeometryResult:
    """Source-to-reference transform and pixel residuals."""

    matrix: np.ndarray
    model: str
    inliers: np.ndarray
    residuals: np.ndarray


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Map floating-point (x,y) points through a 3x3 homography."""
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    projected = (matrix @ homogeneous.T).T
    denominator = projected[:, 2:3]
    return np.divide(projected[:, :2], denominator,
                     out=np.full((len(points), 2), np.nan), where=np.abs(denominator) > 1e-12)


def estimate_transform(matches: Correspondences, *, model: str = "homography",
                       threshold: float = 3.0, confidence: float = 0.99) -> GeometryResult:
    """Estimate affine or homography by RANSAC in reference-pixel coordinates."""
    minimum = 4 if model == "homography" else 3
    if model not in ("homography", "affine"):
        raise GeometryError(f"Unsupported geometry model: {model}")
    if len(matches) < minimum:
        raise GeometryError(f"Insufficient matches: need {minimum}, got {len(matches)}")
    if threshold <= 0 or not 0 < confidence < 1:
        raise GeometryError("RANSAC threshold must be positive and confidence in (0,1)")
    src, dst = matches.source_points, matches.reference_points
    if model == "homography":
        matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, threshold,
                                          confidence=confidence)
    else:
        affine, mask = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC,
                                            ransacReprojThreshold=threshold,
                                            confidence=confidence)
        matrix = np.vstack((affine, [0, 0, 1])) if affine is not None else None
    if matrix is None or mask is None or not np.all(np.isfinite(matrix)):
        raise GeometryError("RANSAC failed to estimate a finite transformation")
    inliers = mask.ravel().astype(bool)
    if inliers.sum() < minimum:
        raise GeometryError("RANSAC found too few inliers")
    residuals = np.linalg.norm(transform_points(src, matrix) - dst, axis=1)
    return GeometryResult(matrix, model, inliers, residuals)
