import cv2
import numpy as np

from lunamatch.refinement.subpixel import refine_points


def test_fractional_shift_refinement() -> None:
    rng = np.random.default_rng(42)
    source = rng.integers(0, 255, (100, 100), dtype=np.uint8)
    reference = cv2.warpAffine(source, np.float32([[1, 0, 0.35], [0, 1, -0.4]]),
                               (100, 100), flags=cv2.INTER_LINEAR)
    refined, success = refine_points(source, reference,
                                     np.array([[50., 50.]], np.float32),
                                     np.array([[50., 50.]], np.float32),
                                     window_size=15)
    assert success[0]
    assert abs(refined[0, 0] - 50.35) < 0.25
    assert abs(refined[0, 1] - 49.6) < 0.25
