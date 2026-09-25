import numpy as np

from lunamatch.matching.cross_modal import combine_correspondences
from lunamatch.matching.matcher_interface import Correspondences


def test_hybrid_fusion_deduplicates_pairs() -> None:
    classical = Correspondences(np.array([[10, 10], [30, 30]], np.float32),
                                np.array([[20, 20], [40, 40]], np.float32),
                                np.array([0.2, 0.1], np.float32))
    learned = Correspondences(np.array([[10.2, 10.1], [60, 60]], np.float32),
                              np.array([[20.1, 20.1], [70, 70]], np.float32),
                              np.array([0.9, 0.8], np.float32))
    combined = combine_correspondences(classical, learned)
    assert len(combined) == 3
    assert combined.confidence[0] == np.float32(0.9)
