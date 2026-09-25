"""Index PRADAN TMC geolocation grids and extract sensor footprint files."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any
from xml.etree import ElementTree
from zipfile import ZipFile

from process_pradan_tmc_browse import BrowseProduct, catalog_archive


GEOMETRY_PATTERN = re.compile(
    r"(?:^|/)ch2_tmc_n([cr])([fna])_(\d{8}T\d+)_g_grd_([^/.]+)\.csv$",
    re.IGNORECASE,
)
SHAPE_COMPONENTS = {".shp", ".shx", ".dbf", ".prj", ".cpg"}


def _first(root: ElementTree.Element, field: str) -> str | None:
    """Read the first non-empty PDS XML value for a local element name."""
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] == field and element.text:
            text = element.text.strip()
            if text:
                return text
    return None


def _browse_index(archive_dir: Path) -> dict[tuple[str, str], list[BrowseProduct]]:
    """Index available TMC browse products by date and fore/nadir/aft view."""
    archives = [
        ("raw", archive_dir / f"browse_raw_{year}.zip")
        for year in (2019, 2020, 2021)
    ] + [("calibrated", archive_dir / "browse_calibrated_2019.zip")]
    index: dict[tuple[str, str], list[BrowseProduct]] = {}
    for level, archive in archives:
        if not archive.is_file():
            continue
        for product in catalog_archive(archive, level):
            index.setdefault((product.start.date().isoformat(), product.sensor_view), []).append(product)
    for products in index.values():
        products.sort(key=lambda product: product.start)
    return index


def index_geolocation(archive_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Create a compact manifest for each PDS4 calibrated TMC geolocation grid."""
    browse = _browse_index(archive_dir)
    grids = []
    for archive_name in ("geometry_calibrated_2019_2020.zip",
                         "geometry_calibrated_2022.zip"):
        archive_path = archive_dir / archive_name
        if not archive_path.is_file():
            continue
        with ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            for member in sorted(names):
                match = GEOMETRY_PATTERN.search(member)
                if not match:
                    continue
                level_code, view, _stamp, resolution = match.groups()
                label = str(Path(member).with_suffix(".xml")).replace("\\", "/")
                if label not in names:
                    continue
                root = ElementTree.fromstring(archive.read(label))
                start = _first(root, "start_date_time")
                if start is None:
                    continue
                acquisition = datetime.fromisoformat(start.replace("Z", "+00:00"))
                candidates = browse.get((acquisition.date().isoformat(), view.lower()), [])
                matched = [product for product in candidates
                           if abs((product.start - acquisition).total_seconds()) <= 1.0]
                record_count = _first(root, "records")
                file_name = _first(root, "file_name")
                grids.append({
                    "geometry_archive": archive_name,
                    "geometry_csv_member": member,
                    "pds_label_member": label,
                    "product_id": _first(root, "logical_identifier"),
                    "instrument": "TMC-2",
                    "data_level": "calibrated geometry grid",
                    "view": {"f": "fore", "n": "nadir", "a": "aft"}[view.lower()],
                    "calibration_code": level_code,
                    "resolution_code": resolution,
                    "acquisition_start_utc": start,
                    "declared_grid_records": int(record_count) if record_count else None,
                    "csv_file_name": file_name,
                    "matched_browse_products": [
                        {"archive": product.archive.name, "member": product.member,
                         "data_level": product.data_level,
                         "product_id": product.product_id}
                        for product in matched
                    ],
                    "browse_products_found": len(matched),
                })
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "tmc_geolocation_index.json"
    path.write_text(json.dumps(grids, indent=2) + "\n", encoding="utf-8")
    columns = list(grids[0]) if grids else []
    with (output_dir / "tmc_geolocation_index.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        if columns:
            writer.writeheader()
            for row in grids:
                writer.writerow({**row,
                                 "matched_browse_products": json.dumps(
                                     row["matched_browse_products"]),
                                 "browse_products_found": row["browse_products_found"]})
    return {
        "geometry_products_indexed": len(grids),
        "products_with_browse_scene_match": sum(row["browse_products_found"] > 0 for row in grids),
        "declared_grid_records": sum(row["declared_grid_records"] or 0 for row in grids),
        "archives": sorted({row["geometry_archive"] for row in grids}),
        "output_json": str(path),
    }


def extract_footprints(archive_dir: Path, output_root: Path) -> dict[str, Any]:
    """Extract SHP/DBF/SHX/projection components as usable GIS source assets."""
    archives = {
        "OHRC": "OHRC_ShapeFiles.zip",
        "TMC-2": "TMC2_ShapeFiles.zip",
        "IIRS": "IIRS_ShapeFiles.zip",
    }
    summary: dict[str, Any] = {}
    for sensor, archive_name in archives.items():
        archive_path = archive_dir / archive_name
        if not archive_path.is_file():
            summary[sensor] = {"status": "archive missing", "files": 0}
            continue
        extracted = []
        destination = output_root / sensor
        with ZipFile(archive_path) as archive:
            for member in archive.namelist():
                source_path = Path(member)
                if source_path.suffix.lower() not in SHAPE_COMPONENTS:
                    continue
                if source_path.is_absolute() or ".." in source_path.parts:
                    continue
                # Preserve each archive's layer directory and component basename.
                safe_member = Path(source_path.parent.name) / source_path.name
                target = destination / safe_member
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
                extracted.append({"member": member, "output": str(target),
                                  "bytes": target.stat().st_size})
        summary[sensor] = {"status": "extracted", "archive": archive_name,
                           "files": len(extracted), "components": extracted}
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "footprint_extraction.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    """Prepare local footprint and geolocation references from PRADAN archives."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/spatial"))
    args = parser.parse_args()
    result = {
        "geolocation": index_geolocation(args.archive_dir, args.output_dir),
        "footprints": extract_footprints(args.archive_dir,
                                          args.output_dir / "footprints"),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
