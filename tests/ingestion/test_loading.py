"""Ingestion tests use generated pixels, not lunar validation data."""

import json

import numpy as np
import pytest
from PIL import Image

from lunamatch.ingestion import load_image
from lunamatch.ingestion.validators import ImageLoadError
from lunamatch.utils.image import preview_array, save_preview


def test_numpy_keeps_unknown_metadata() -> None:
    pixels = np.arange(100, dtype=np.uint16).reshape(10, 10)
    product = load_image(pixels)
    assert product.image is pixels
    assert product.sensor is None
    assert product.pixel_scale is None
    assert product.data_level == "unknown"
    assert product.summary()["bands"] == 1
    assert json.dumps(product.summary())


def test_png_load_and_preview(tmp_path) -> None:
    pixels = np.arange(600, dtype=np.uint8).reshape(20, 30)
    path = tmp_path / "sample.png"
    Image.fromarray(pixels).save(path)
    product = load_image(path, sensor="OHRC", data_level="raw")
    assert product.image.shape == (20, 30)
    assert product.sensor == "OHRC"
    assert product.data_level == "raw"
    assert product.valid_mask.all()
    assert preview_array(product, max_side=10).shape == (7, 10)
    assert save_preview(product, tmp_path / "preview.png").is_file()


def test_multiband_geotiff_load(tmp_path) -> None:
    import rasterio
    from rasterio.transform import from_origin

    path = tmp_path / "cube.tif"
    with rasterio.open(path, "w", driver="GTiff", width=9, height=7,
                       count=3, dtype="uint16", transform=from_origin(0, 0, 2, 2)) as ds:
        ds.write(np.ones((3, 7, 9), dtype=np.uint16))
        ds.update_tags(instrument_id="TMC-2")
    product = load_image(path)
    assert product.image.shape == (7, 9, 3)
    assert product.sensor == "TMC-2"
    assert product.pixel_scale is None
    assert product.metadata["format"] == "GTiff"


def test_projected_raster_scale_and_sun_tags(tmp_path) -> None:
    import rasterio
    from rasterio.transform import from_origin

    path = tmp_path / "tagged.tif"
    with rasterio.open(path, "w", driver="GTiff", width=9, height=7,
                       count=1, dtype="uint16", crs="EPSG:3857",
                       transform=from_origin(0, 0, 2, 2)) as dataset:
        dataset.write(np.ones((7, 9), dtype=np.uint16), 1)
        dataset.update_tags(sun_azimuth="103.2", sun_elevation="12.4",
                            acquisition_time="2026-01-01T00:00:00Z")
    product = load_image(path)
    assert product.pixel_scale == 2.0
    assert product.sun_azimuth == 103.2
    assert product.sun_elevation == 12.4
    assert product.acquisition_time == "2026-01-01T00:00:00Z"


def test_invalid_inputs(tmp_path) -> None:
    with pytest.raises(ImageLoadError, match="Unsupported"):
        load_image(tmp_path / "bad.exe")
    with pytest.raises(ImageLoadError, match="does not exist"):
        load_image(tmp_path / "missing.png")
    with pytest.raises(ImageLoadError, match="exceeds"):
        load_image(np.zeros((11, 11)), max_pixels=100)
    with pytest.raises(ImageLoadError, match="Invalid data level"):
        load_image(np.zeros((2, 2)), data_level="invented")


def test_pds4_array_and_label(tmp_path) -> None:
    (tmp_path / "image.dat").write_bytes(np.arange(24, dtype=np.uint8).tobytes())
    label = """<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
    <Identification_Area><logical_identifier>urn:nasa:pds:test:test</logical_identifier>
    <version_id>1.0</version_id><title>Test</title><information_model_version>1.10.0.0</information_model_version>
    <product_class>Product_Observational</product_class></Identification_Area>
    <File_Area_Observational><File><file_name>image.dat</file_name></File>
    <Array_3D_Image><local_identifier>test_image</local_identifier><offset unit="byte">0</offset>
    <axes>3</axes><axis_index_order>Last Index Fastest</axis_index_order>
    <Element_Array><data_type>UnsignedByte</data_type></Element_Array>
    <Axis_Array><axis_name>Line</axis_name><elements>3</elements><sequence_number>1</sequence_number></Axis_Array>
    <Axis_Array><axis_name>Sample</axis_name><elements>4</elements><sequence_number>2</sequence_number></Axis_Array>
    <Axis_Array><axis_name>Band</axis_name><elements>2</elements><sequence_number>3</sequence_number></Axis_Array>
    </Array_3D_Image></File_Area_Observational></Product_Observational>"""
    path = tmp_path / "image.xml"
    path.write_text(label)
    product = load_image(path, sensor="IIRS")
    assert product.image.shape == (3, 4, 2)
    assert product.bands == 2
    assert product.sensor == "IIRS"
    assert product.metadata["format"] == "PDS4"
