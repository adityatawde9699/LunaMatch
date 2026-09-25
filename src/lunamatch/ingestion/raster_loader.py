"""Raster, common image, and NumPy array ingestion."""

from pathlib import Path
from typing import Any
import importlib.util

import numpy as np
from PIL import Image, UnidentifiedImageError

from .metadata import LunarImage
from .validators import ImageLoadError, MAX_ARRAY_BYTES, MAX_PIXELS, validate_array, validate_path


def _rasterio_load(path: Path, max_pixels: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        import rasterio
    except ImportError as exc:
        raise ImageLoadError("GeoTIFF/TIFF support requires: pip install 'lunamatch[planetary]'") from exc
    try:
        with rasterio.open(path) as ds:
            if ds.width * ds.height > max_pixels:
                raise ImageLoadError(f"Image exceeds {max_pixels} pixels; use a tile")
            if ds.count == 0:
                raise ImageLoadError("Raster has no bands")
            estimated_bytes = ds.width * ds.height * sum(np.dtype(dtype).itemsize for dtype in ds.dtypes)
            if estimated_bytes > MAX_ARRAY_BYTES:
                raise ImageLoadError(f"Raster exceeds {MAX_ARRAY_BYTES} uncompressed bytes; use a tile")
            bands = ds.read(masked=True)
            image = bands[0] if ds.count == 1 else np.moveaxis(bands, 0, -1)
            mask = ~np.ma.getmaskarray(image)
            if image.ndim == 3:
                mask = mask.all(axis=-1)
            image = np.asarray(np.ma.filled(image, 0))
            mask &= np.isfinite(image).all(axis=-1) if image.ndim == 3 else np.isfinite(image)
            metadata = {
                "format": ds.driver,
                "crs": str(ds.crs) if ds.crs else None,
                "transform": list(ds.transform)[:6],
                "tags": ds.tags(),
                "nodata": ds.nodata,
            }
            if ds.crs and ds.crs.is_projected and (ds.crs.linear_units or "").lower() in ("metre", "meter", "m"):
                scale_x = float(np.hypot(ds.transform.a, ds.transform.d))
                scale_y = float(np.hypot(ds.transform.b, ds.transform.e))
                if scale_x > 0 and abs(scale_x - scale_y) / scale_x < 0.01:
                    metadata["pixel_scale_m"] = (scale_x + scale_y) / 2
            return image, mask, metadata
    except ImageLoadError:
        raise
    except Exception as exc:
        raise ImageLoadError(f"Unable to read raster: {exc}") from exc


def _pillow_load(path: Path, max_pixels: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        with Image.open(path) as im:
            if im.width * im.height > max_pixels:
                raise ImageLoadError(f"Image exceeds {max_pixels} pixels; use a tile")
            if im.mode in ("P", "1", "LA"):
                im = im.convert("RGBA" if im.mode == "LA" else "RGB")
            array = np.asarray(im)
            image_format = im.format or path.suffix.lstrip(".").upper()
        mask = np.isfinite(array).all(axis=-1) if array.ndim == 3 else np.isfinite(array)
        return array, mask, {"format": image_format}
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageLoadError(f"Unable to read image: {exc}") from exc


def load_image(source: str | Path | np.ndarray, *, sensor: str | None = None,
               data_level: str = "unknown", max_pixels: int = MAX_PIXELS) -> LunarImage:
    """Load an image without inventing sensor, calibration, or geometry metadata.

    Accepted sources: PDS4 XML, GeoTIFF/TIFF, PNG/JPEG, and NumPy arrays.
    The caller labels raw/calibrated/processed status if it is known.
    """
    if data_level not in ("raw", "calibrated", "processed", "unknown"):
        raise ImageLoadError(f"Invalid data level: {data_level}")
    if isinstance(source, np.ndarray):
        image = validate_array(source, max_pixels)
        mask = np.isfinite(image).all(axis=-1) if image.ndim == 3 else np.isfinite(image)
        return LunarImage(image=image, sensor=sensor, data_level=data_level,
                          valid_mask=mask, metadata={"format": "NumPy"})
    path = Path(source)
    validate_path(path)
    if path.suffix.lower() == ".xml":
        from .pds4_loader import load_pds4
        return load_pds4(path, sensor=sensor, data_level=data_level, max_pixels=max_pixels)
    if path.suffix.lower() in (".tif", ".tiff") and importlib.util.find_spec("rasterio"):
        image, mask, metadata = _rasterio_load(path, max_pixels)
    else:
        image, mask, metadata = _pillow_load(path, max_pixels)
    image = validate_array(image, max_pixels)
    tags = {str(k).lower(): v for k, v in metadata.get("tags", {}).items()}
    sensor = sensor or tags.get("instrument_id") or tags.get("sensor")
    def numeric(*keys: str) -> float | None:
        for key in keys:
            if key in tags:
                try:
                    return float(tags[key])
                except ValueError:
                    return None
        return None
    return LunarImage(image=image, sensor=sensor, data_level=data_level,
                      source_path=path, metadata=metadata, valid_mask=mask,
                      pixel_scale=metadata.get("pixel_scale_m"),
                      acquisition_time=tags.get("start_date_time") or tags.get("acquisition_time"),
                      sun_azimuth=numeric("sun_azimuth", "solar_azimuth"),
                      sun_elevation=numeric("sun_elevation", "solar_elevation"))
