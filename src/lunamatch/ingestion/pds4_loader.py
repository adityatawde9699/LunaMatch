"""PDS4 label and array loading via optional pds4-tools."""

from pathlib import Path
from math import prod
import xml.etree.ElementTree as ET

import numpy as np

from .metadata import LunarImage
from .validators import ImageLoadError, MAX_ARRAY_BYTES, validate_array, validate_path


def _label_fields(path: Path) -> dict[str, str]:
    """Read selected public PDS4 label fields without guessing missing values."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ImageLoadError(f"Invalid PDS4 XML label: {exc}") from exc
    wanted = {"instrument_id", "instrument_name", "start_date_time", "solar_azimuth",
              "solar_elevation", "sun_azimuth_angle", "sun_elevation_angle"}
    fields: dict[str, str] = {}
    for element in root.iter():
        name = element.tag.rsplit("}", 1)[-1].lower()
        if name in wanted and element.text and name not in fields:
            fields[name] = element.text.strip()
    return fields


def _numeric_field(fields: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        if key in fields:
            try:
                return float(fields[key])
            except ValueError:
                return None
    return None


def load_pds4(path: str | Path, *, sensor: str | None = None,
              data_level: str = "unknown", max_pixels: int = 150_000_000) -> LunarImage:
    """Load the first numeric 2D/3D PDS4 array, preserving a band cube.

    Products with no image array are rejected. pds4-tools resolves external
    data files relative to the XML label; both must be available locally.
    """
    path = Path(path)
    validate_path(path)
    try:
        from pds4_tools import pds4_read
    except ImportError as exc:
        raise ImageLoadError("PDS4 support requires: pip install 'lunamatch[planetary]'") from exc
    try:
        product = pds4_read(str(path), quiet=True)
        candidates = []
        for structure in product:
            if not hasattr(structure, "meta_data") or "Axis_Array" not in structure.meta_data:
                continue
            axes = structure.meta_data["Axis_Array"]
            if not isinstance(axes, list) or len(axes) not in (2, 3):
                continue
            dimensions = {str(axis["axis_name"]).lower(): int(axis["elements"]) for axis in axes}
            if dimensions.get("line", 0) * dimensions.get("sample", 0) > max_pixels:
                raise ImageLoadError(f"PDS4 image exceeds {max_pixels} pixels; use a tile")
            if prod(dimensions.values()) * 8 > MAX_ARRAY_BYTES:
                raise ImageLoadError("PDS4 array may exceed 1 GiB in memory; use a tile")
            array = np.asanyarray(structure.data)
            if array.ndim in (2, 3) and np.issubdtype(array.dtype, np.number):
                candidates.append((structure, array))
        if not candidates:
            raise ImageLoadError("PDS4 product contains no supported 2D/3D numeric image array")
        structure, image = candidates[0]
        if image.ndim == 3:
            axes = sorted(structure.meta_data["Axis_Array"],
                          key=lambda axis: int(axis["sequence_number"]))
            names = [str(axis["axis_name"]).lower() for axis in axes]
            try:
                image = np.transpose(image, (names.index("line"), names.index("sample"),
                                             names.index("band")))
            except ValueError as exc:
                raise ImageLoadError("3D PDS4 array needs Line, Sample, and Band axes") from exc
        image = validate_array(image, max_pixels)
        mask = ~np.ma.getmaskarray(image).any(axis=-1) if image.ndim == 3 else ~np.ma.getmaskarray(image)
        image = np.asarray(np.ma.filled(image, np.nan if np.issubdtype(image.dtype, np.floating) else 0))
        mask &= np.all(np.isfinite(image), axis=-1) if image.ndim == 3 else np.isfinite(image)
        fields = _label_fields(path)
        return LunarImage(image=image, sensor=sensor or fields.get("instrument_id") or fields.get("instrument_name"),
                          acquisition_time=fields.get("start_date_time"),
                          sun_azimuth=_numeric_field(fields, "solar_azimuth", "sun_azimuth_angle"),
                          sun_elevation=_numeric_field(fields, "solar_elevation", "sun_elevation_angle"),
                          data_level=data_level,
                          source_path=path, valid_mask=mask,
                          metadata={"format": "PDS4", "array_id": getattr(structure, "id", None), "label_fields": fields})
    except ImageLoadError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ImageLoadError(f"Unable to read PDS4 image product: {exc}") from exc
