import cv2
import numpy as np

from lunamatch.ingestion import load_image
from lunamatch.pipeline.config import RegistrationConfig
from lunamatch.pipeline.registration import register
from lunamatch.preprocessing.illumination import local_normalize
from lunamatch.preprocessing.pyramid import build_pyramid

from tests.matching.test_sift_registration import synthetic_terrain


def test_local_normalization_reduces_brightness_offset() -> None:
    rng = np.random.default_rng(1)
    image = rng.integers(35, 170, (100, 120), dtype=np.uint8)
    brighter = np.clip(image.astype(np.int16) + 35, 0, 255).astype(np.uint8)
    a = local_normalize(image)
    b = local_normalize(brighter)
    assert np.mean(np.abs(a.astype(int) - b.astype(int))) < 2


def test_pyramid_and_scale_registration() -> None:
    source = synthetic_terrain()
    reference = cv2.resize(source, (240, 180), interpolation=cv2.INTER_AREA)
    assert [level[0].shape for level in build_pyramid(source, 3)] == [
        (360, 480), (180, 240), (90, 120)]
    result = register(load_image(source), load_image(reference),
                      RegistrationConfig(geometry_model="affine", pyramid_levels=3))
    assert result.metrics["inliers"] > 15
    assert abs(result.geometry.matrix[0, 0] - 0.5) < 0.05
