"""Window-local output must map correctly into native image coordinates."""

import numpy as np

from experiments.process_tmc2_science_windows import global_transform
from lunamatch.geometry.ransac import transform_points


def test_window_transform_preserves_full_image_coordinates() -> None:
    local = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]])
    full = global_transform(local, (100, 200, 64, 64), (300, 400, 64, 64))
    mapped = transform_points(np.array([[110.0, 210.0]]), full)
    np.testing.assert_allclose(mapped, [[315.0, 407.0]])
