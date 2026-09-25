"""PDS4 label and array loading via optional pds4-tools."""

from pathlib import Path, PurePosixPath, PureWindowsPath
from math import prod
import xml.etree.ElementTree as ET

import numpy as np

from .metadata import LunarImage
from .validators import ImageLoadError, MAX_ARRAY_BYTES, validate_array, validate_path


_WINDOW_DTYPES = {
    "UnsignedByte": "u1", "SignedByte": "i1",
    "UnsignedLSB2": "<u2", "SignedLSB2": "<i2",
    "UnsignedMSB2": ">u2", "SignedMSB2": ">i2",
    "UnsignedLSB4": "<u4", "SignedLSB4": "<i4",
    "UnsignedMSB4": ">u4", "SignedMSB4": ">i4",
    "IEEE754LSBSingle": "<f4", "IEEE754MSBSingle": ">f4",
    "IEEE754LSBDouble": "<f8", "IEEE754MSBDouble": ">f8",
}


def _child_text(parent: ET.Element, name: str) -> str | None:
    """Return direct child text by local XML name."""
    for child in parent:
        if child.tag.rsplit("}", 1)[-1] == name:
            return child.text.strip() if child.text else None
    return None


def _direct_child(parent: ET.Element, name: str) -> ET.Element | None:
    for child in parent:
        if child.tag.rsplit("}", 1)[-1] == name:
            return child
    return None


def load_pds4_window(path: str | Path, window: tuple[int, int, int, int], *,
                     sensor: str | None = None, data_level: str = "unknown",
                     max_pixels: int = 150_000_000) -> LunarImage:
    """Read a bounded window from an uncompressed row-major PDS4 2D image.

    ``window`` is ``(x, y, width, height)`` in full-image pixel coordinates.
    Other PDS4 array layouts are rejected explicitly and can use the regular
    loader when they fit memory. The binary is mapped read-only; only the crop
    is copied into memory.
    """
    label_path = Path(path)
    validate_path(label_path)
    x, y, width, height = window
    if any(not isinstance(value, int) for value in window):
        raise ImageLoadError("PDS4 window coordinates must be integers")
    if min(x, y) < 0 or min(width, height) < 1:
        raise ImageLoadError("PDS4 window needs nonnegative origin and positive size")
    if width * height > max_pixels:
        raise ImageLoadError(f"PDS4 window exceeds {max_pixels} pixels")
    try:
        root = ET.parse(label_path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ImageLoadError(f"Invalid PDS4 XML label: {exc}") from exc

    candidates: list[tuple[ET.Element, ET.Element]] = []
    for file_area in root.iter():
        if not file_area.tag.rsplit("}", 1)[-1].startswith("File_Area"):
            continue
        file_element = _direct_child(file_area, "File")
        if file_element is None:
            continue
        for array in file_area:
            if array.tag.rsplit("}", 1)[-1] == "Array_2D_Image":
                candidates.append((file_element, array))
    if not candidates:
        raise ImageLoadError("PDS4 window loading requires an Array_2D_Image")
    file_element, array = candidates[0]
    file_name = _child_text(file_element, "file_name")
    if not file_name or "\\" in file_name:
        raise ImageLoadError("PDS4 image file_name is missing or unsafe")
    relative = PurePosixPath(file_name)
    windows_path = PureWindowsPath(file_name)
    if relative.is_absolute() or windows_path.is_absolute() or windows_path.drive or ".." in relative.parts:
        raise ImageLoadError("PDS4 image file_name is unsafe")
    binary_path = (label_path.parent / Path(*relative.parts)).resolve()
    if not binary_path.is_relative_to(label_path.parent.resolve()) or not binary_path.is_file():
        raise ImageLoadError(f"PDS4 image binary is unavailable: {file_name}")
    if _child_text(array, "axis_index_order") != "Last Index Fastest":
        raise ImageLoadError("PDS4 window loader supports Last Index Fastest arrays only")
    axes: dict[str, tuple[int, int]] = {}
    for axis in array:
        if axis.tag.rsplit("}", 1)[-1] == "Axis_Array":
            name = (_child_text(axis, "axis_name") or "").lower()
            try:
                axes[name] = (int(_child_text(axis, "sequence_number") or ""),
                              int(_child_text(axis, "elements") or ""))
            except ValueError as exc:
                raise ImageLoadError("PDS4 image has invalid axis dimensions") from exc
    if set(axes) != {"line", "sample"} or axes["line"][0] != 1 or axes["sample"][0] != 2:
        raise ImageLoadError("PDS4 window loader requires Line then Sample axes")
    lines, samples = axes["line"][1], axes["sample"][1]
    if min(lines, samples) < 1 or x + width > samples or y + height > lines:
        raise ImageLoadError("PDS4 window extends outside the labeled image")
    element = _direct_child(array, "Element_Array")
    data_type = _child_text(element, "data_type") if element is not None else None
    if data_type not in _WINDOW_DTYPES:
        raise ImageLoadError(f"Unsupported PDS4 window data type: {data_type}")
    dtype = np.dtype(_WINDOW_DTYPES[data_type])
    try:
        offset = int(_child_text(array, "offset") or "")
    except ValueError as exc:
        raise ImageLoadError("PDS4 image has invalid byte offset") from exc
    if offset < 0 or binary_path.stat().st_size < offset + lines * samples * dtype.itemsize:
        raise ImageLoadError("PDS4 image binary is shorter than the labeled array")
    if width * height * dtype.itemsize > MAX_ARRAY_BYTES:
        raise ImageLoadError("PDS4 window exceeds the uncompressed memory limit")

    mapped = np.memmap(binary_path, dtype=dtype, mode="r", offset=offset,
                       shape=(lines, samples), order="C")
    image = np.array(mapped[y:y + height, x:x + width], copy=True)
    del mapped
    mask = np.isfinite(image)
    constants = _direct_child(array, "Special_Constants")
    if constants is not None:
        for constant in constants:
            if constant.text:
                try:
                    mask &= image != np.asarray(constant.text.strip(), dtype=dtype)
                except (ValueError, OverflowError):
                    continue
    fields = _label_fields(label_path)
    return LunarImage(
        image=image, sensor=sensor or fields.get("instrument_id") or fields.get("instrument_name"),
        acquisition_time=fields.get("start_date_time"),
        sun_azimuth=_numeric_field(fields, "solar_azimuth", "sun_azimuth_angle"),
        sun_elevation=_numeric_field(fields, "solar_elevation", "sun_elevation_angle"),
        data_level=data_level, source_path=label_path, valid_mask=mask,
        metadata={"format": "PDS4", "representation": "science image window",
                  "label_fields": fields, "data_type": data_type,
                  "full_width": samples, "full_height": lines,
                  "window": {"x": x, "y": y, "width": width, "height": height}},
    )


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
