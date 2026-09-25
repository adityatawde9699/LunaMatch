"""Opt-in smoke tests require public pretrained weights in the model cache."""

import os

import cv2
import pytest

from lunamatch.ingestion import load_image
from lunamatch.pipeline.config import RegistrationConfig
from lunamatch.pipeline.registration import register

from tests.matching.test_sift_registration import synthetic_terrain


@pytest.mark.skipif(os.environ.get("LUNAMATCH_TEST_LEARNED") != "1",
                    reason="Set LUNAMATCH_TEST_LEARNED=1 after downloading model weights")
@pytest.mark.parametrize("matcher", ["loftr", "lightglue", "hybrid"])
def test_pretrained_matcher_on_generated_pair(matcher: str) -> None:
    source = synthetic_terrain()[:256, :256]
    reference = cv2.warpAffine(source, cv2.getRotationMatrix2D((128, 128), 0, 1),
                               (256, 256))
    result = register(load_image(source), load_image(reference),
                      RegistrationConfig(matcher=matcher, geometry_model="affine"))
    assert result.metrics["inliers"] >= 10
    assert result.metrics["device"] in ("cpu", "cuda")
