"""Registration inspection images."""

import cv2
import numpy as np


def match_visualization(source: np.ndarray, reference: np.ndarray,
                        source_points: np.ndarray, reference_points: np.ndarray,
                        inliers: np.ndarray, max_lines: int = 300) -> np.ndarray:
    """Draw corresponding inliers side by side in green."""
    h = max(source.shape[0], reference.shape[0])
    w = source.shape[1] + reference.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:source.shape[0], :source.shape[1]] = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    canvas[:reference.shape[0], source.shape[1]:] = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
    indices = np.flatnonzero(inliers)[:max_lines]
    for index in indices:
        a = tuple(np.rint(source_points[index]).astype(int))
        b = tuple(np.rint(reference_points[index]).astype(int) + [source.shape[1], 0])
        cv2.line(canvas, a, b, (70, 220, 70), 1, cv2.LINE_AA)
        cv2.circle(canvas, a, 3, (70, 220, 70), 1, cv2.LINE_AA)
        cv2.circle(canvas, b, 3, (70, 220, 70), 1, cv2.LINE_AA)
    return canvas
