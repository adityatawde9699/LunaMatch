"""Input limits and image-array validation."""

from pathlib import Path

import numpy as np

SUPPORTED_SUFFIXES = {".xml", ".tif", ".tiff", ".png", ".jpg", ".jpeg"}
MAX_FILE_BYTES = 2 * 1024**3
MAX_PIXELS = 150_000_000
MAX_ARRAY_BYTES = 1024**3


class ImageLoadError(ValueError):
    """Raised when a product cannot be safely interpreted as an image."""


def validate_path(path: Path, max_bytes: int = MAX_FILE_BYTES) -> None:
    """Check extension, regular-file status, and compressed size."""
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ImageLoadError(f"Unsupported image extension: {path.suffix}")
    if not path.is_file():
        raise ImageLoadError(f"Image file does not exist: {path}")
    if path.stat().st_size > max_bytes:
        raise ImageLoadError(f"Image file exceeds {max_bytes} bytes: {path}")


def validate_array(image: np.ndarray, max_pixels: int = MAX_PIXELS) -> np.ndarray:
    """Validate shape and numeric dtype without copying pixel data."""
    image = np.asarray(image)
    if image.ndim not in (2, 3) or min(image.shape[:2]) == 0:
        raise ImageLoadError("Expected a nonempty 2D image or 3D band cube")
    if image.shape[0] * image.shape[1] > max_pixels:
        raise ImageLoadError(f"Image exceeds {max_pixels} pixels; use a tile")
    if image.nbytes > MAX_ARRAY_BYTES:
        raise ImageLoadError(f"Image exceeds {MAX_ARRAY_BYTES} uncompressed bytes; use a tile")
    if not np.issubdtype(image.dtype, np.number):
        raise ImageLoadError("Image pixels must be numeric")
    return image
