"""Descriptor matching with Lowe ratio filtering."""

import cv2
import numpy as np

from lunamatch.features.feature_interface import Features
from .matcher_interface import Correspondences


def match_features(source: Features, reference: Features,
                   ratio: float = 0.75) -> Correspondences:
    """Match float or binary descriptors using the appropriate distance metric."""
    empty = Correspondences(np.empty((0, 2), np.float32),
                            np.empty((0, 2), np.float32), np.empty(0, np.float32))
    if source.descriptors is None or reference.descriptors is None or len(reference.descriptors) < 2:
        return empty
    if source.descriptors.dtype != reference.descriptors.dtype:
        raise ValueError("Descriptor types differ")
    norm = cv2.NORM_HAMMING if source.descriptors.dtype == np.uint8 else cv2.NORM_L2
    pairs = cv2.BFMatcher(norm).knnMatch(source.descriptors,
                                                reference.descriptors, k=2)
    accepted = [(a, b) for pair in pairs if len(pair) == 2
                for a, b in [pair] if a.distance < ratio * b.distance]
    if not accepted:
        return empty
    return Correspondences(
        np.asarray([source.keypoints[a.queryIdx].pt for a, _ in accepted], dtype=np.float32),
        np.asarray([reference.keypoints[a.trainIdx].pt for a, _ in accepted], dtype=np.float32),
        np.asarray([max(0.0, 1.0 - a.distance / max(b.distance, 1e-9))
                    for a, b in accepted], dtype=np.float32),
    )


def match_sift(source: Features, reference: Features,
               ratio: float = 0.75) -> Correspondences:
    """Compatibility wrapper for the verified SIFT baseline."""
    return match_features(source, reference, ratio)
