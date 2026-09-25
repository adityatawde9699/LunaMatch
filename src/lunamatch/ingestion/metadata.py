"""Common image representation with explicit unknown metadata."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

DataLevel = Literal["raw", "calibrated", "derived", "processed", "unknown"]


@dataclass(slots=True)
class LunarImage:
    """Pixel data and metadata retained from its source product.

    Array axes are (height, width) or (height, width, bands). Missing values
    remain ``None``; no geometric or radiometric calibration is implied.
    """

    image: np.ndarray
    sensor: str | None = None
    pixel_scale: float | None = None
    acquisition_time: str | None = None
    sun_azimuth: float | None = None
    sun_elevation: float | None = None
    data_level: DataLevel = "unknown"
    source_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    valid_mask: np.ndarray | None = None

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return int(self.image.shape[1])

    @property
    def bands(self) -> int:
        """Number of image bands."""
        return 1 if self.image.ndim == 2 else int(self.image.shape[2])

    def summary(self) -> dict[str, Any]:
        """Return JSON-compatible inspection information."""
        return {
            "source_path": str(self.source_path) if self.source_path else None,
            "sensor": self.sensor,
            "data_level": self.data_level,
            "width": self.width,
            "height": self.height,
            "bands": self.bands,
            "dtype": str(self.image.dtype),
            "pixel_scale": self.pixel_scale,
            "acquisition_time": self.acquisition_time,
            "sun_azimuth": self.sun_azimuth,
            "sun_elevation": self.sun_elevation,
            "valid_fraction": float(np.mean(self.valid_mask)) if self.valid_mask is not None else None,
            "metadata": self.metadata,
        }
