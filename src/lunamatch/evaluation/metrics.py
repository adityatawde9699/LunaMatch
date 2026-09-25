"""Metrics derived from geometric inlier residuals."""

import numpy as np

from lunamatch.geometry.ransac import GeometryResult


def registration_metrics(geometry: GeometryResult, *, source_keypoints: int | None,
                         reference_keypoints: int | None, candidate_matches: int,
                         runtime_seconds: float) -> dict[str, float | int | None | str]:
    """Summarize measured residuals; these are not ground-truth errors."""
    errors = geometry.residuals[geometry.inliers]
    return {
        "source_keypoints": source_keypoints,
        "reference_keypoints": reference_keypoints,
        "candidate_matches": candidate_matches,
        "valid_matches": len(geometry.residuals),
        "inliers": int(geometry.inliers.sum()),
        "inlier_ratio": float(geometry.inliers.mean()),
        "registration_residual_rmse_px": float(np.sqrt(np.mean(errors**2))),
        "registration_residual_median_px": float(np.median(errors)),
        "registration_residual_p95_px": float(np.percentile(errors, 95)),
        "ground_truth_rmse_px": None,
        "error_type": "RANSAC inlier reprojection residual; ground truth unavailable",
        "runtime_seconds": runtime_seconds,
    }
