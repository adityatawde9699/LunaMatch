"""Register all usable PRADAN TMC-2 browse stereo pairs.

Products are streamed from ZIP archives and paired from PDS4 acquisition
timestamps. This is a browse quicklook experiment, not science-resolution
validation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import csv
from dataclasses import dataclass, replace
from datetime import datetime
from io import BytesIO
import json
from pathlib import Path
import re
import statistics
import threading
import time
from typing import Any
from xml.etree import ElementTree
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image

from lunamatch.ingestion.metadata import LunarImage
from lunamatch.pipeline.config import RegistrationConfig
from lunamatch.pipeline.registration import register
from lunamatch.pipeline.result import RegistrationResult


PRODUCT_PATTERN = re.compile(
    r"(?:^|/)ch2_tmc_n[cr]([fna])_(\d{8}T\d+)_b_brw_([^/.]+)\.png$",
    re.IGNORECASE,
)
VIEWS = {"f": "fore", "n": "nadir", "a": "aft"}
VIEW_PAIRS = (("f", "n"), ("a", "n"))
WORKER_LOCAL = threading.local()


@dataclass(frozen=True, slots=True)
class BrowseProduct:
    """One labeled browse image stored in a PRADAN ZIP archive."""

    archive: Path
    member: str
    sensor_view: str
    acquisition_start: str
    acquisition_stop: str | None
    product_id: str | None
    variant: str
    data_level: str
    year: int

    @property
    def start(self) -> datetime:
        """Acquisition start parsed from the PDS4 label."""
        return datetime.fromisoformat(self.acquisition_start.replace("Z", "+00:00"))


@dataclass(frozen=True, slots=True)
class RegistrationPair:
    """An identified fore/nadir or aft/nadir product pair."""

    pair_id: str
    pair_type: str
    source: BrowseProduct
    reference: BrowseProduct


def _xml_value(root: ElementTree.Element, name: str) -> str | None:
    """Return the first non-empty PDS label value matching a local name."""
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] == name and element.text:
            value = element.text.strip()
            if value:
                return value
    return None


def catalog_archive(archive: Path, data_level: str) -> list[BrowseProduct]:
    """Catalog TMC browse images that have readable PDS4 sidecar labels."""
    products: list[BrowseProduct] = []
    with ZipFile(archive) as source:
        names = set(source.namelist())
        for member in sorted(names):
            match = PRODUCT_PATTERN.search(member)
            if not match:
                continue
            view, _stamp, variant = match.groups()
            label_member = str(Path(member).with_suffix(".xml")).replace("\\", "/")
            if label_member not in names:
                continue
            root = ElementTree.fromstring(source.read(label_member))
            start = _xml_value(root, "start_date_time")
            if start is None:
                continue
            acquisition = datetime.fromisoformat(start.replace("Z", "+00:00"))
            products.append(BrowseProduct(
                archive=archive,
                member=member,
                sensor_view=view.lower(),
                acquisition_start=start,
                acquisition_stop=_xml_value(root, "stop_date_time"),
                product_id=_xml_value(root, "logical_identifier"),
                variant=variant,
                data_level=data_level,
                year=acquisition.year,
            ))
    return products


def pair_views(products: list[BrowseProduct], max_time_delta_s: float = 1.0
               ) -> list[tuple[BrowseProduct, BrowseProduct]]:
    """Pair each adjacent stereo view with nadir using unique nearest times."""
    grouped: dict[tuple[str, str], dict[str, list[BrowseProduct]]] = {}
    for product in products:
        key = (product.start.date().isoformat(), product.variant)
        grouped.setdefault(key, {"f": [], "n": [], "a": []})
        grouped[key][product.sensor_view].append(product)

    result: list[tuple[BrowseProduct, BrowseProduct]] = []
    for views in grouped.values():
        for source_view, reference_view in VIEW_PAIRS:
            candidates = sorted(
                (abs((source.start - reference.start).total_seconds()),
                 source.member, reference.member, source, reference)
                for source in views[source_view]
                for reference in views[reference_view]
            )
            used_source: set[str] = set()
            used_reference: set[str] = set()
            for delta, _source_name, _reference_name, source, reference in candidates:
                if delta > max_time_delta_s:
                    break
                if source.member in used_source or reference.member in used_reference:
                    continue
                result.append((source, reference))
                used_source.add(source.member)
                used_reference.add(reference.member)
    return sorted(result, key=lambda pair: (pair[0].data_level, pair[0].year,
                                             pair[0].start, pair[0].sensor_view))


def _load_product(product: BrowseProduct) -> LunarImage:
    """Read a ZIP-contained PNG and preserve PDS label provenance."""
    archives: dict[Path, ZipFile] = getattr(WORKER_LOCAL, "archives", {})
    archive = archives.get(product.archive)
    if archive is None:
        archive = ZipFile(product.archive)
        archives[product.archive] = archive
        WORKER_LOCAL.archives = archives
    pixels = np.asarray(Image.open(BytesIO(archive.read(product.member))).convert("L"))
    return LunarImage(
        image=pixels,
        sensor="TMC-2",
        acquisition_time=product.acquisition_start,
        data_level=product.data_level,  # type: ignore[arg-type]
        metadata={
            "format": "PNG",
            "product_id": product.product_id,
            "archive": product.archive.name,
            "archive_member": product.member,
            "product_view": VIEWS[product.sensor_view],
            "product_representation": "PDS4 browse quicklook",
            "acquisition_stop": product.acquisition_stop,
        },
        valid_mask=np.ones(pixels.shape, dtype=bool),
    )


def _register_pair(pair: RegistrationPair,
                   config: RegistrationConfig) -> tuple[RegistrationPair, RegistrationResult | None, str | None]:
    """Run one pair, returning an explicit error string on failure."""
    try:
        result = register(_load_product(pair.source), _load_product(pair.reference), config)
        return pair, result, None
    except Exception as exc:  # failures are recorded and do not stop the batch
        return pair, None, f"{type(exc).__name__}: {exc}"


def _metadata(pair: RegistrationPair) -> dict[str, Any]:
    """Build provenance fields shared by pair manifest and metrics."""
    return {
        "pair_id": pair.pair_id,
        "data_level": pair.source.data_level,
        "year": pair.source.year,
        "pair_type": pair.pair_type,
        "source_view": VIEWS[pair.source.sensor_view],
        "reference_view": VIEWS[pair.reference.sensor_view],
        "source_product_id": pair.source.product_id,
        "reference_product_id": pair.reference.product_id,
        "source_archive": pair.source.archive.name,
        "reference_archive": pair.reference.archive.name,
        "source_member": pair.source.member,
        "reference_member": pair.reference.member,
        "acquisition_start_utc": pair.source.acquisition_start,
        "source_reference_time_delta_seconds": abs(
            (pair.source.start - pair.reference.start).total_seconds()),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a collection of mappings as UTF-8 CSV."""
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    """Read a previously written result CSV when retrying failed pairs."""
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _typed_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Restore numeric columns read from a prior CSV for JSON summaries."""
    integer_fields = {"year", "source_keypoints", "reference_keypoints",
                      "candidate_matches", "valid_matches", "inliers",
                      "selected_correspondences", "occupied_cells"}
    float_fields = {"runtime_seconds", "inlier_ratio", "registration_residual_rmse_px",
                    "registration_residual_median_px", "registration_residual_p95_px",
                    "selected_coverage"}
    typed = []
    for row in rows:
        converted = dict(row)
        for field in integer_fields | float_fields:
            value = converted.get(field)
            if value not in (None, ""):
                converted[field] = int(float(value)) if field in integer_fields else float(value)
        typed.append(converted)
    return typed


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize registration outcomes by year, archive level, and view pair."""
    keys = sorted({(row["year"], row["data_level"], row["pair_type"]) for row in rows})
    summary: dict[str, Any] = {}
    for year, level, pair_type in keys:
        group = [row for row in rows if (row["year"], row["data_level"],
                                         row["pair_type"]) == (year, level, pair_type)]
        successful = [row for row in group if row.get("status") == "completed"]
        key = f"{year}_{level}_{pair_type}"
        summary[key] = {
            "pairs_attempted": len(group),
            "pairs_completed": len(successful),
            "success_rate": len(successful) / len(group) if group else None,
            "median_inliers": statistics.median(row["inliers"] for row in successful)
                if successful else None,
            "median_inlier_ratio": statistics.median(
                row["inlier_ratio"] for row in successful) if successful else None,
            "median_registration_residual_rmse_px": statistics.median(
                row["registration_residual_rmse_px"] for row in successful)
                if successful else None,
            "median_selected_coverage": statistics.median(
                row["selected_coverage"] for row in successful) if successful else None,
            "ground_truth_accuracy": "not measured; no independent tie-point ground truth",
        }
    return summary


def run(archive_dir: Path, output_dir: Path, max_pairs: int = 0,
        exemplar_count: int = 1, workers: int = 4,
        retry_failures_from: Path | None = None) -> dict[str, Any]:
    """Process all complete raw 2019–2021 and calibrated 2019 browse pairs."""
    sources = [
        ("raw", archive_dir / f"browse_raw_{year}.zip")
        for year in (2019, 2020, 2021)
    ] + [("calibrated", archive_dir / "browse_calibrated_2019.zip")]
    missing = [str(path) for _, path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required browse archives are unavailable: {missing}")

    previous_rows = _read_csv(retry_failures_from) if retry_failures_from else []
    previous_summary_path = output_dir / "summary.json"
    previous_summary = (json.loads(previous_summary_path.read_text(encoding="utf-8"))
                        if retry_failures_from and previous_summary_path.is_file() else {})
    previous_by_members = {
        (row["source_member"], row["reference_member"]): row for row in previous_rows
    }
    retry_members = {
        key for key, row in previous_by_members.items() if row.get("status") == "failed"
    }

    pairs: list[RegistrationPair] = []
    for level, archive in sources:
        products = catalog_archive(archive, level)
        for source, reference in pair_views(products):
            pair_type = f"{source.sensor_view}n"
            pair_id = (f"tmc2_{level}_{source.year}_{pair_type}_"
                       f"{source.start:%Y%m%dT%H%M%S}_{len(pairs) + 1:05d}")
            pair = RegistrationPair(pair_id, pair_type, source, reference)
            old_row = previous_by_members.get((source.member, reference.member))
            if old_row:
                pair = replace(pair, pair_id=old_row["pair_id"])
            if retry_failures_from is None or (source.member, reference.member) in retry_members:
                pairs.append(pair)
    if retry_failures_from and not retry_members:
        raise ValueError(f"No failed pair rows found in {retry_failures_from}")
    if max_pairs > 0:
        pairs = pairs[:max_pairs]

    output_dir.mkdir(parents=True, exist_ok=True)
    config = RegistrationConfig(matcher="sift", geometry_model="homography")
    cv2.setNumThreads(1)
    started = time.perf_counter()
    metric_rows: list[dict[str, Any]] = []
    manifest_rows = [_metadata(pair) for pair in pairs]
    match_rows: list[dict[str, Any]] = []
    exemplar_counts: dict[tuple[str, str], int] = {}
    exemplar_dir = output_dir / "exemplars"

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        pending: dict[Future, RegistrationPair] = {}
        pair_iter = iter(pairs)
        for _ in range(min(len(pairs), max(1, workers) * 2)):
            pair = next(pair_iter)
            pending[pool.submit(_register_pair, pair, config)] = pair
        done_count = 0
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                done_count += 1
                pending.pop(future)
                pair, result, error = future.result()
                row = _metadata(pair)
                row["status"] = "completed" if result is not None else "failed"
                row["error"] = error
                if result is not None:
                    row.update(result.metrics)
                    group = (pair.source.data_level, pair.pair_type)
                    if exemplar_counts.get(group, 0) < exemplar_count:
                        target = exemplar_dir / pair.source.data_level / pair.pair_type / pair.pair_id
                        result.save(target)
                        row["exemplar_result"] = str(target)
                        exemplar_counts[group] = exemplar_counts.get(group, 0) + 1
                    for selected_id in result.selected_indices:
                        index = int(selected_id)
                        match_rows.append({
                            "pair_id": pair.pair_id,
                            "data_level": pair.source.data_level,
                            "year": pair.source.year,
                            "pair_type": pair.pair_type,
                            "source_product_id": pair.source.product_id,
                            "reference_product_id": pair.reference.product_id,
                            "source_x": float(result.matches.source_points[index, 0]),
                            "source_y": float(result.matches.source_points[index, 1]),
                            "reference_x": float(result.matches.reference_points[index, 0]),
                            "reference_y": float(result.matches.reference_points[index, 1]),
                            "confidence": float(result.matches.confidence[index]),
                            "inlier": bool(result.geometry.inliers[index]),
                            "registration_residual_px": float(result.geometry.residuals[index]),
                        })
                metric_rows.append(row)
                if done_count % 50 == 0 or done_count == len(pairs):
                    elapsed = time.perf_counter() - started
                    print(f"Processed {done_count}/{len(pairs)} pairs in {elapsed:.1f}s",
                          flush=True)
                try:
                    pair = next(pair_iter)
                except StopIteration:
                    continue
                pending[pool.submit(_register_pair, pair, config)] = pair

    if retry_failures_from:
        retry_keys = {(pair.source.member, pair.reference.member) for pair in pairs}
        metric_rows = [row for row in previous_rows
                       if (row["source_member"], row["reference_member"]) not in retry_keys] + metric_rows
        old_manifest = _read_csv(output_dir / "pair_manifest.csv")
        manifest_rows = [row for row in old_manifest
                         if (row["source_member"], row["reference_member"]) not in retry_keys] + manifest_rows
        match_rows = _read_csv(output_dir / "selected_matches.csv") + match_rows
        metric_rows = _typed_metric_rows(metric_rows)
    metric_rows.sort(key=lambda row: row["pair_id"])
    match_rows.sort(key=lambda row: (row["pair_id"], float(row["source_x"]),
                                      float(row["source_y"])))
    _write_csv(output_dir / "pair_manifest.csv", manifest_rows)
    _write_csv(output_dir / "pair_metrics.csv", metric_rows)
    _write_csv(output_dir / "selected_matches.csv", match_rows)
    summary = {
        "dataset": "ISRO/ISSDC Chandrayaan-2 TMC-2 browse PNG products",
        "input_archives": [str(path) for _, path in sources],
        "pairing": "PDS4 acquisition timestamps; unique nearest fore/nadir or aft/nadir within 1 second",
        "processing": "SIFT descriptors + homography RANSAC + 8x8 spatial selection",
        "pairs_total": len(metric_rows),
        "pairs_completed": sum(row["status"] == "completed" for row in metric_rows),
        "pairs_failed": sum(row["status"] == "failed" for row in metric_rows),
        "summary_by_year_level_view_pair": _aggregate(metric_rows),
        "runtime_seconds": (time.perf_counter() - started
                            + previous_summary.get("runtime_seconds", 0.0)),
        "ground_truth_rmse": None,
        "limitations": [
            "Browse quicklooks are display products, not science-resolution imagery.",
            "Fitted reprojection residual is not independent lunar surface accuracy.",
            "Raw and calibrated here refer to browse archive product classes; both are PNG renders.",
            "No photometric or geometric ground-truth validation was available.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                             encoding="utf-8")
    return summary


def main() -> None:
    """CLI entry point for processing available TMC-2 browse stereo pairs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/pradan_tmc2_browse_all"))
    parser.add_argument("--max-pairs", type=int, default=0,
                        help="Limit processing for a smoke run; zero means all pairs")
    parser.add_argument("--exemplars", type=int, default=1,
                        help="Number of full result bundles per data-level/view-pair")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retry-failures-from", type=Path,
                        help="Retry failed rows from an existing pair_metrics.csv and merge results")
    args = parser.parse_args()
    print(json.dumps(run(args.archive_dir, args.output, args.max_pairs,
                         args.exemplars, args.workers, args.retry_failures_from), indent=2))


if __name__ == "__main__":
    main()
