"""Generated image pairs exercise software behavior, not lunar accuracy."""

import json

import cv2
import numpy as np
import pytest

from lunamatch.geometry.ransac import GeometryError
from lunamatch.ingestion import load_image
from lunamatch.pipeline.config import RegistrationConfig
from lunamatch.pipeline.registration import register


def synthetic_terrain() -> np.ndarray:
    """Make deterministic texture with crater-like rings for registration tests."""
    rng = np.random.default_rng(26166)
    image = rng.normal(105, 25, (360, 480)).clip(0, 255).astype(np.uint8)
    for _ in range(90):
        center = (int(rng.integers(15, 465)), int(rng.integers(15, 345)))
        radius = int(rng.integers(3, 16))
        cv2.circle(image, center, radius, int(rng.integers(150, 235)), 2)
        cv2.circle(image, (center[0] + 2, center[1] + 2), max(1, radius - 2),
                   int(rng.integers(30, 85)), 1)
    return cv2.GaussianBlur(image, (3, 3), 0)


def test_sift_end_to_end(tmp_path) -> None:
    source = synthetic_terrain()
    shift = np.array([[1, 0, 21], [0, 1, -13]], dtype=np.float32)
    reference = cv2.warpAffine(source, shift, (480, 360))
    result = register(load_image(source, sensor="OHRC"),
                      load_image(reference, sensor="OHRC"),
                      RegistrationConfig(geometry_model="affine"))
    assert result.metrics["inliers"] > 30
    assert result.metrics["inlier_ratio"] > 0.7
    assert result.metrics["ground_truth_rmse_px"] is None
    assert abs(result.geometry.matrix[0, 2] - 21) < 1
    assert abs(result.geometry.matrix[1, 2] + 13) < 1
    result.save(tmp_path)
    for name in ("registered_image.tif", "matches.csv", "matches.json",
                 "transformation.json", "metrics.json", "job_log.json", "overlay.png",
                 "match_visualization.png"):
        assert (tmp_path / name).is_file()
    assert json.loads((tmp_path / "metrics.json").read_text())["inliers"] > 30


def test_featureless_pair_has_useful_error() -> None:
    image = np.full((100, 100), 127, np.uint8)
    with pytest.raises(ValueError, match="intensity variation"):
        register(load_image(image), load_image(image))


@pytest.mark.parametrize("matcher", ["orb", "akaze"])
def test_other_classical_matchers(matcher: str) -> None:
    source = synthetic_terrain()
    reference = cv2.warpAffine(source, np.float32([[1, 0, 12], [0, 1, -8]]),
                               (480, 360))
    result = register(load_image(source), load_image(reference),
                      RegistrationConfig(matcher=matcher, geometry_model="affine"))
    assert result.metrics["inliers"] > 10
    assert abs(result.geometry.matrix[0, 2] - 12) < 2


def test_export_preserves_source_bit_depth(tmp_path) -> None:
    import rasterio

    source = synthetic_terrain().astype(np.uint16) * 200
    reference = cv2.warpAffine(source, np.float32([[1, 0, 7], [0, 1, -4]]),
                               (480, 360))
    result = register(load_image(source), load_image(reference),
                      RegistrationConfig(geometry_model="affine"))
    result.save(tmp_path)
    with rasterio.open(tmp_path / "registered_image.tif") as dataset:
        assert dataset.dtypes[0] == "uint16"
        assert dataset.read(1).max() > 255


def test_registered_tiff_uses_reference_georeferencing(tmp_path) -> None:
    import rasterio
    from rasterio.transform import from_origin

    source = synthetic_terrain()
    reference = cv2.warpAffine(source, np.float32([[1, 0, 7], [0, 1, -4]]),
                               (480, 360))
    transform = from_origin(1000, 2000, 2, 2)
    for name, image in (("source.tif", source), ("reference.tif", reference)):
        with rasterio.open(tmp_path / name, "w", driver="GTiff", width=480,
                           height=360, count=1, dtype="uint8", crs="EPSG:3857",
                           transform=transform) as dataset:
            dataset.write(image, 1)
    result = register(load_image(tmp_path / "source.tif"),
                      load_image(tmp_path / "reference.tif"),
                      RegistrationConfig(geometry_model="affine"))
    result.save(tmp_path / "output")
    with rasterio.open(tmp_path / "output/registered_image.tif") as dataset:
        assert dataset.crs.to_epsg() == 3857
        assert dataset.transform == transform
