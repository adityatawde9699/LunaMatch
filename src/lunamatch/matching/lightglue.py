"""Optional official SuperPoint + LightGlue correspondence adapter."""

import numpy as np

from lunamatch.features.superpoint import superpoint_tensor
from lunamatch.utils.model_cache import set_model_cache
from .loftr import ModelUnavailable
from .matcher_interface import Correspondences


def match_lightglue(source: np.ndarray, reference: np.ndarray, *,
                    max_side: int = 1024, max_keypoints: int = 2048,
                    device: str = "auto") -> tuple[Correspondences, str, int, int]:
    """Run pretrained SuperPoint extraction and LightGlue matching.

    The official CVG package and public weights are optional. First use may
    download two model files into LUNAMATCH_MODEL_CACHE.
    """
    try:
        import torch
        from lightglue import LightGlue, SuperPoint
    except ImportError as exc:
        raise ModelUnavailable("LightGlue requires the official CVG package and torchvision") from exc
    selected = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    if selected == "cuda" and not torch.cuda.is_available():
        raise ModelUnavailable("CUDA requested but no CUDA GPU is available")
    if selected not in ("cpu", "cuda"):
        raise ValueError(f"Unsupported inference device: {selected}")
    if max_side < 32 or max_keypoints < 16:
        raise ValueError("Invalid LightGlue input limits")
    source_tensor = superpoint_tensor(source, selected)
    reference_tensor = superpoint_tensor(reference, selected)
    try:
        set_model_cache()
        extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(selected)
        matcher = LightGlue(features="superpoint").eval().to(selected)
        with torch.inference_mode():
            source_features = extractor.extract(source_tensor, resize=min(max_side, max(source.shape)))
            reference_features = extractor.extract(reference_tensor, resize=min(max_side, max(reference.shape)))
            output = matcher({"image0": source_features, "image1": reference_features})
        indices = output["matches"][0].detach().cpu().numpy()
        source_points = source_features["keypoints"][0].detach().cpu().numpy()[indices[:, 0]]
        reference_points = reference_features["keypoints"][0].detach().cpu().numpy()[indices[:, 1]]
        confidence = output["scores"][0].detach().cpu().numpy()
        return (Correspondences(source_points.astype(np.float32),
                                reference_points.astype(np.float32),
                                confidence.astype(np.float32)), selected,
                int(source_features["keypoints"].shape[1]),
                int(reference_features["keypoints"].shape[1]))
    except (OSError, RuntimeError, ValueError, IndexError) as exc:
        raise ModelUnavailable(f"SuperPoint + LightGlue unavailable: {exc}") from exc
