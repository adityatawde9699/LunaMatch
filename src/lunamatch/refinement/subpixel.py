"""Sub-pixel local normalized cross-correlation refinement."""

import cv2
import numpy as np


def _patch(image: np.ndarray, center: tuple[float, float], size: int) -> np.ndarray:
    return cv2.getRectSubPix(image.astype(np.float32), (size, size), center)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    left = a - float(a.mean())
    right = b - float(b.mean())
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.sum(left * right) / denominator) if denominator > 1e-6 else -1.0


def _parabolic(left: float, center: float, right: float) -> float:
    denominator = left - 2 * center + right
    if abs(denominator) < 1e-8:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))


def refine_points(source: np.ndarray, reference: np.ndarray,
                  source_points: np.ndarray, reference_points: np.ndarray,
                  *, window_size: int = 11, search_radius: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Refine reference points by local NCC and quadratic peak interpolation.

    Returns refined float coordinates and a success mask. This estimates
    fractional coordinates; only independent ground truth can validate error.
    """
    if window_size < 5 or window_size % 2 != 1 or search_radius < 1:
        raise ValueError("window_size must be odd and >=5; search_radius must be >=1")
    if len(source_points) != len(reference_points):
        raise ValueError("Source and reference point counts differ")
    refined = reference_points.astype(np.float32).copy()
    success = np.zeros(len(refined), dtype=bool)
    margin = window_size // 2 + search_radius + 2
    for index, (src, ref) in enumerate(zip(source_points, reference_points)):
        if not (margin <= src[0] < source.shape[1] - margin and
                margin <= src[1] < source.shape[0] - margin and
                margin <= ref[0] < reference.shape[1] - margin and
                margin <= ref[1] < reference.shape[0] - margin):
            continue
        template = _patch(source, (float(src[0]), float(src[1])), window_size)
        width = 2 * search_radius + 1
        scores = np.empty((width, width), dtype=np.float32)
        for row, dy in enumerate(range(-search_radius, search_radius + 1)):
            for col, dx in enumerate(range(-search_radius, search_radius + 1)):
                candidate = _patch(reference, (float(ref[0] + dx), float(ref[1] + dy)),
                                   window_size)
                scores[row, col] = _ncc(template, candidate)
        row, col = np.unravel_index(np.argmax(scores), scores.shape)
        if scores[row, col] <= 0 or row in (0, width - 1) or col in (0, width - 1):
            continue
        dx = col - search_radius + _parabolic(*scores[row, col - 1:col + 2])
        dy = row - search_radius + _parabolic(*scores[row - 1:row + 2, col])
        refined[index] = ref + [dx, dy]
        success[index] = True
    return refined, success
