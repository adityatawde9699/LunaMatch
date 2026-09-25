"""Count downloaded and staged Chandrayaan-2 image products by sensor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile


ROOT = Path(__file__).resolve().parents[1]
SENSORS = ("OHRC", "TMC-2", "IIRS")
SCIENCE_SUFFIXES = {".img", ".qub", ".jp2", ".tif", ".tiff"}
QUICKLOOK_SUFFIXES = {".png", ".jpg", ".jpeg"}


def inventory_sensor_data(raw_root: Path, data_root: Path) -> dict[str, Any]:
    """Inventory valid product archives, raster members, partials, and staged products."""
    output: dict[str, Any] = {"sensors": {}, "scope_note": (
        "Science products are counted by raster members (.img/.qub/.jp2/TIFF); "
        "PNG/JPEG browse quicklooks are reported separately."
    )}
    for sensor in SENSORS:
        sensor_dir = raw_root / sensor
        archives = sorted(sensor_dir.rglob("*.zip")) if sensor_dir.exists() else []
        complete: list[dict[str, Any]] = []
        invalid: list[str] = []
        for archive_path in archives:
            try:
                with ZipFile(archive_path) as archive:
                    names = [name for name in archive.namelist() if not name.endswith("/")]
                science = [name for name in names if Path(name).suffix.lower() in SCIENCE_SUFFIXES]
                quicklooks = [name for name in names if Path(name).suffix.lower() in QUICKLOOK_SUFFIXES]
                complete.append({
                    "path": str(archive_path.relative_to(raw_root)),
                    "bytes": archive_path.stat().st_size,
                    "science_raster_members": len(science),
                    "browse_image_members": len(quicklooks),
                    "science_members": science,
                })
            except (BadZipFile, OSError):
                invalid.append(str(archive_path.relative_to(raw_root)))
        partials = sorted(sensor_dir.rglob("*.part")) if sensor_dir.exists() else []
        staged_root = data_root.rglob("lunamatch_product_manifest.json")
        manifests = [path for path in staged_root if f"/{sensor}/" in path.as_posix()]
        output["sensors"][sensor] = {
            "downloaded_product_archives": len(complete),
            "archives_containing_science_rasters": sum(
                bool(record["science_raster_members"]) for record in complete
            ),
            "science_raster_members": sum(
                record["science_raster_members"] for record in complete
            ),
            "browse_image_members": sum(
                record["browse_image_members"] for record in complete
            ),
            "staged_products": len(manifests),
            "incomplete_downloads": len(partials),
            "invalid_archives": invalid,
            "products": complete,
        }
    return output


def main() -> int:
    """Write a current sensor inventory to data/processed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/raw2")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/processed/sensor_product_inventory.json")
    args = parser.parse_args()
    report = inventory_sensor_data(args.raw_root, args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for sensor, details in report["sensors"].items():
        print(f"{sensor}: {details['downloaded_product_archives']} product ZIPs, "
              f"{details['staged_products']} staged, "
              f"{details['incomplete_downloads']} partial")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
