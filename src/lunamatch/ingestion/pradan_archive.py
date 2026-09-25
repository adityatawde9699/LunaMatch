"""Safe inspection and staging of PRADAN product ZIP archives."""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import stat
import tempfile
from typing import Any
from zipfile import BadZipFile, ZipFile

from .validators import ImageLoadError

_SENSOR_PATTERNS = {
    "OHRC": re.compile(r"(?:^|[^a-z0-9])(?:ohrc|ohr)(?:[^a-z0-9]|$)", re.I),
    "TMC-2": re.compile(r"(?:^|[^a-z0-9])tmc(?:-?2)?(?:[^a-z0-9]|$)", re.I),
    "IIRS": re.compile(r"(?:^|[^a-z0-9])(?:iirs|iir)(?:[^a-z0-9]|$)", re.I),
}
_DATA_LEVELS = {"raw", "calibrated", "derived", "processed"}
_IMAGE_SUFFIXES = {".img", ".tif", ".tiff", ".jp2", ".png", ".jpg", ".jpeg", ".qub"}


def _classify_sensor(name: str) -> str | None:
    """Infer a sensor only from an explicit recognized name token."""
    for sensor, pattern in _SENSOR_PATTERNS.items():
        if pattern.search(name):
            return sensor
    return None


def _classify_level(names: list[str]) -> str:
    """Infer product level from explicit archive path components."""
    tokens = {part.casefold() for name in names for part in PurePosixPath(name).parts}
    for level in ("calibrated", "processed", "derived", "raw"):
        if level in tokens:
            return level
    return "unknown"


def inspect_pradan_product(path: str | Path) -> dict[str, Any]:
    """Inventory a PRADAN ZIP without extracting its contents."""
    archive = Path(path)
    if not archive.is_file() or archive.suffix.lower() != ".zip":
        raise ImageLoadError("Expected an existing PRADAN product .zip archive")
    try:
        with ZipFile(archive) as zipped:
            files = [entry for entry in zipped.infolist() if not entry.is_dir()]
            names = [entry.filename for entry in files]
            label_names = [name for name in names if name.lower().endswith(".xml")]
            return {
                "archive": str(archive),
                "archive_bytes": archive.stat().st_size,
                "sensor": _classify_sensor(" ".join([archive.name, *names[:100]])),
                "data_level": _classify_level(names),
                "member_count": len(files),
                "uncompressed_bytes": sum(entry.file_size for entry in files),
                "image_members": [name for name in names
                                  if PurePosixPath(name).suffix.lower() in _IMAGE_SUFFIXES],
                "has_data_directory": any(
                    part.casefold() == "data" for name in names
                    for part in PurePosixPath(name).parts[:-1]),
                "xml_labels": label_names,
                "directories": sorted({part for name in names
                                        for part in PurePosixPath(name).parts[:-1]}),
                "members": names,
            }
    except BadZipFile as exc:
        raise ImageLoadError(f"Invalid ZIP archive: {archive}") from exc


def _safe_member_path(name: str) -> Path:
    """Convert a ZIP member to a relative path or reject traversal names."""
    if "\x00" in name or "\\" in name:
        raise ImageLoadError(f"Unsafe ZIP member path: {name!r}")
    posix = PurePosixPath(name)
    windows = PureWindowsPath(name)
    if posix.is_absolute() or windows.is_absolute() or windows.drive or ".." in posix.parts:
        raise ImageLoadError(f"Unsafe ZIP member path: {name!r}")
    return Path(*posix.parts)


def stage_pradan_product(path: str | Path, output_dir: str | Path, *,
                         max_uncompressed_bytes: int = 20 * 1024**3,
                         max_members: int = 100_000) -> dict[str, Any]:
    """Safely extract a product ZIP into a sensor/level-specific directory.

    Extraction is staged beside its destination and renamed into place only
    after every member has been copied. Existing destinations are never
    overwritten. ZIP symlinks, encrypted members, path traversal, and archives
    exceeding configured expansion limits are rejected.
    """
    archive = Path(path)
    inventory = inspect_pradan_product(archive)
    if not inventory["image_members"] and not inventory["has_data_directory"]:
        raise ImageLoadError("Archive has no raster members or product data directory; it may be documentation only")
    if inventory["member_count"] > max_members:
        raise ImageLoadError(f"Archive contains more than {max_members} files")
    if inventory["uncompressed_bytes"] > max_uncompressed_bytes:
        raise ImageLoadError("Archive exceeds the configured uncompressed size limit")
    sensor = inventory["sensor"] or "UNKNOWN"
    level = inventory["data_level"]
    destination = Path(output_dir) / level / sensor / archive.stem
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ImageLoadError(f"Product destination already exists: {destination}")

    staging = Path(tempfile.mkdtemp(prefix=f".{archive.stem}-", dir=destination.parent))
    try:
        with ZipFile(archive) as zipped:
            files = [entry for entry in zipped.infolist() if not entry.is_dir()]
            for entry in files:
                relative = _safe_member_path(entry.filename)
                mode = entry.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ImageLoadError(f"ZIP symlink is not allowed: {entry.filename}")
                if entry.flag_bits & 0x1:
                    raise ImageLoadError(f"Encrypted ZIP member is not supported: {entry.filename}")
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(entry) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        os.replace(staging, destination)
    except (BadZipFile, OSError, RuntimeError) as exc:
        shutil.rmtree(staging, ignore_errors=True)
        if isinstance(exc, BadZipFile):
            raise ImageLoadError(f"Invalid or corrupt ZIP archive: {archive}") from exc
        raise ImageLoadError(f"Unable to stage PRADAN product: {exc}") from exc
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    manifest = {**inventory, "staged_at": str(destination),
                "source_archive": str(archive.resolve())}
    (destination / "lunamatch_product_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
