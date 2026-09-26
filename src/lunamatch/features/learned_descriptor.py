"""Inference adapter for a locally trained LunaPatchDescriptor checkpoint."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from .feature_interface import Features


class DescriptorUnavailable(RuntimeError):
    """Raised when the optional descriptor checkpoint cannot be used."""


def extract_learned(image: np.ndarray, mask: np.ndarray | None = None,
                    max_features: int = 6000,
                    checkpoint: str | Path | None = None) -> Features:
    """Extract normalized learned descriptors at Shi-Tomasi points.

    Set ``LUNAMATCH_DESCRIPTOR_CHECKPOINT`` to a checkpoint produced by
    ``lunamatch train-descriptor``. The checkpoint is never downloaded or
    silently replaced by a classical descriptor.
    """
    checkpoint_name = str(checkpoint) if checkpoint is not None else os.environ.get(
        "LUNAMATCH_DESCRIPTOR_CHECKPOINT", ""
    ).strip()
    if not checkpoint_name:
        raise DescriptorUnavailable("Set LUNAMATCH_DESCRIPTOR_CHECKPOINT to a trained checkpoint")
    checkpoint_path = Path(checkpoint_name)
    if not checkpoint_path.is_file():
        raise DescriptorUnavailable(f"Descriptor checkpoint does not exist: {checkpoint_path}")
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("Learned descriptor requires 8-bit grayscale images")
    try:
        import torch
        from lunamatch.learning.descriptor import LunarPatchDescriptor
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = LunarPatchDescriptor(embedding_dim=int(checkpoint["embedding_dim"]))
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
    except (ImportError, OSError, KeyError, RuntimeError, ValueError) as exc:
        raise DescriptorUnavailable(f"Unable to load descriptor checkpoint: {exc}") from exc
    patch_size = int(checkpoint.get("patch_size", 32))
    half = patch_size / 2.0
    mask_u8 = mask.astype(np.uint8) * 255 if mask is not None else None
    points = cv2.goodFeaturesToTrack(image, maxCorners=max_features, qualityLevel=0.01,
                                     minDistance=max(3, patch_size // 2), mask=mask_u8)
    if points is None:
        return Features([], None)
    keypoints = [cv2.KeyPoint(float(point[0, 0]), float(point[0, 1]), patch_size)
                 for point in points]
    patches = []
    valid_keypoints = []
    for keypoint in keypoints:
        x, y = keypoint.pt
        if x < half or y < half or x >= image.shape[1] - half or y >= image.shape[0] - half:
            continue
        patch = cv2.getRectSubPix(image, (patch_size, patch_size), (x, y))
        patches.append(torch.from_numpy(patch.copy()).float()[None] / 255.0)
        valid_keypoints.append(keypoint)
    if not patches:
        return Features([], None)
    with torch.inference_mode():
        descriptors = model(torch.stack(patches)).cpu().numpy().astype(np.float32)
    return Features(valid_keypoints, descriptors)
