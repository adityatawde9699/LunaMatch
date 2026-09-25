"""Optional pretrained LoFTR dense correspondence adapter."""

import cv2
import numpy as np

from lunamatch.utils.model_cache import set_model_cache
from .matcher_interface import Correspondences


class ModelUnavailable(RuntimeError):
    """Raised when the optional model or its pretrained weights are unavailable."""


def _resize_for_model(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float, float]:
    height, width = image.shape
    scale = min(1.0, max_side / max(height, width))
    new_width = max(8, int(width * scale) // 8 * 8)
    new_height = max(8, int(height * scale) // 8 * 8)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, width / new_width, height / new_height


def match_loftr(source: np.ndarray, reference: np.ndarray, *,
                max_side: int = 1024, device: str = "auto") -> tuple[Correspondences, str]:
    """Run Kornia's pretrained outdoor LoFTR with CPU/CUDA selection.

    Weights are fetched by Kornia on first use. Unavailable weights fail
    explicitly, so the classical baseline is never silently substituted.
    """
    if source.dtype != np.uint8 or reference.dtype != np.uint8:
        raise ValueError("LoFTR inputs must be normalized 8-bit grayscale")
    if max_side < 8:
        raise ValueError("LoFTR max_side must be at least 8")
    try:
        import torch
        import kornia.feature as KF
    except ImportError as exc:
        raise ModelUnavailable("LoFTR requires: pip install 'lunamatch[learned]'") from exc
    selected = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    if selected == "cuda" and not torch.cuda.is_available():
        raise ModelUnavailable("CUDA requested but no CUDA GPU is available")
    if selected not in ("cpu", "cuda"):
        raise ValueError(f"Unsupported inference device: {selected}")
    try:
        set_model_cache()
    except OSError as exc:
        raise ModelUnavailable(f"Model cache is not writable: {exc}") from exc
    src_small, sx, sy = _resize_for_model(source, max_side)
    ref_small, rx, ry = _resize_for_model(reference, max_side)
    try:
        model = KF.LoFTR(pretrained="outdoor").eval().to(selected)
        inputs = {
            "image0": torch.from_numpy(src_small.copy()).float()[None, None].to(selected) / 255,
            "image1": torch.from_numpy(ref_small.copy()).float()[None, None].to(selected) / 255,
        }
        with torch.inference_mode():
            output = model(inputs)
        source_points = output["keypoints0"].detach().cpu().numpy().astype(np.float32)
        reference_points = output["keypoints1"].detach().cpu().numpy().astype(np.float32)
        confidence = output["confidence"].detach().cpu().numpy().astype(np.float32)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ModelUnavailable(f"LoFTR inference or pretrained weights unavailable: {exc}") from exc
    source_points *= np.array([sx, sy], dtype=np.float32)
    reference_points *= np.array([rx, ry], dtype=np.float32)
    return Correspondences(source_points, reference_points, confidence), selected
