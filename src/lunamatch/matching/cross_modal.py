"""Candidate fusion for classical and learned point correspondences."""

import numpy as np

from .matcher_interface import Correspondences


def combine_correspondences(classical: Correspondences,
                            learned: Correspondences,
                            *, min_confidence: float = 0.05,
                            deduplication_pixels: float = 2.0) -> Correspondences:
    """Merge confidence-filtered matches and suppress nearby duplicate pairs."""
    if deduplication_pixels <= 0:
        raise ValueError("deduplication_pixels must be positive")
    sources = np.vstack((classical.source_points, learned.source_points))
    references = np.vstack((classical.reference_points, learned.reference_points))
    # Classical Lowe-ratio separation has a different scale from model confidence.
    confidence = np.concatenate((np.clip(classical.confidence / 0.25, 0, 1),
                                 learned.confidence))
    seen = set()
    indices = []
    for index in np.argsort(-confidence, kind="stable"):
        if confidence[index] < min_confidence:
            continue
        key = tuple(np.rint(np.r_[sources[index], references[index]] /
                             deduplication_pixels).astype(int))
        if key in seen:
            continue
        seen.add(key)
        indices.append(index)
    return Correspondences(sources[indices], references[indices], confidence[indices])
