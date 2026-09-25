"""Register bounded, native-resolution windows from five staged TMC-2 products.

The browse registrations are used only to locate overlapping stereo windows.
Native-resolution correspondences and geometry are estimated from IMG pixels.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from lunamatch.ingestion import load_image
from lunamatch.pipeline.config import RegistrationConfig
from lunamatch.pipeline.registration import register


ROOT = Path(__file__).resolve().parents[1]
DATE = "20260823"
STEMS = {
    "calibrated_nadir": "ch2_tmc_ncn_20260823T1657513854_d_img_d18",
    "calibrated_fore": "ch2_tmc_ncf_20260823T1657513886_d_img_d18",
    "calibrated_aft": "ch2_tmc_nca_20260823T1657513854_d_img_d18",
    "raw_nadir": "ch2_tmc_nrn_20260823T1657513854_d_img_d18",
}
DEFAULT_WINDOW = (1000, 20000, 2048, 2048)


def _paths(product: str) -> tuple[Path, Path]:
    """Locate the PDS4 science label and its browse PNG for a staged product."""
    level = product.split("_", 1)[0]
    stem = STEMS[product]
    folder = ROOT / "data" / level / "TMC-2" / stem
    label = folder / "data" / level / DATE / f"{stem}.xml"
    browse_stem = stem.replace("_d_img_", "_b_brw_")
    browse = folder / "browse" / level / DATE / f"{browse_stem}.png"
    return label, browse


def global_transform(matrix: np.ndarray, source_window: tuple[int, int, int, int],
                     reference_window: tuple[int, int, int, int]) -> np.ndarray:
    """Convert a crop-local homography into full-image pixel coordinates."""
    sx, sy = source_window[:2]
    rx, ry = reference_window[:2]
    source_to_crop = np.array([[1, 0, -sx], [0, 1, -sy], [0, 0, 1]], dtype=np.float64)
    crop_to_reference = np.array([[1, 0, rx], [0, 1, ry], [0, 0, 1]], dtype=np.float64)
    result = crop_to_reference @ matrix @ source_to_crop
    return result / result[2, 2]


def _source_window(source_label: Path, source_browse: Path,
                   reference_label: Path, reference_browse: Path,
                   reference_window: tuple[int, int, int, int],
                   browse_transform: Path) -> tuple[int, int, int, int]:
    """Locate a source science crop using a measured browse homography."""
    source_info = load_image(source_label, sensor="TMC-2", window=(0, 0, 1, 1))
    reference_info = load_image(reference_label, sensor="TMC-2", window=(0, 0, 1, 1))
    source_width = source_info.metadata["full_width"]
    source_height = source_info.metadata["full_height"]
    reference_width = reference_info.metadata["full_width"]
    reference_height = reference_info.metadata["full_height"]
    with Image.open(source_browse) as image:
        browse_source_width, browse_source_height = image.size
    with Image.open(reference_browse) as image:
        browse_reference_width, browse_reference_height = image.size
    matrix = np.asarray(json.loads(browse_transform.read_text(encoding="utf-8"))["matrix"],
                        dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid browse transformation: {browse_transform}")
    rx, ry, width, height = reference_window
    reference_centre_browse = np.array([
        (rx + width / 2) * browse_reference_width / reference_width,
        (ry + height / 2) * browse_reference_height / reference_height,
        1.0,
    ])
    source_centre_browse = np.linalg.inv(matrix) @ reference_centre_browse
    if source_centre_browse[2] == 0:
        raise ValueError("Browse transformation maps the crop centre to infinity")
    source_centre_browse /= source_centre_browse[2]
    source_centre = np.array([
        source_centre_browse[0] * source_width / browse_source_width,
        source_centre_browse[1] * source_height / browse_source_height,
    ])
    source_x = int(np.clip(round(source_centre[0] - width / 2), 0, source_width - width))
    source_y = int(np.clip(round(source_centre[1] - height / 2), 0, source_height - height))
    return source_x, source_y, width, height


def _write_global_matches(path: Path, matches_path: Path,
                          source_window: tuple[int, int, int, int],
                          reference_window: tuple[int, int, int, int]) -> None:
    """Export selected tie points in full-image pixel coordinates."""
    with matches_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    fields = ["id", "source_x", "source_y", "reference_x", "reference_y",
              "confidence", "inlier", "error"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "id": row["id"],
                "source_x": float(row["source_x"]) + source_window[0],
                "source_y": float(row["source_y"]) + source_window[1],
                "reference_x": float(row["reference_x"]) + reference_window[0],
                "reference_y": float(row["reference_y"]) + reference_window[1],
                "confidence": row["confidence"], "inlier": row["inlier"],
                "error": row["error"],
            })


def run(reference_window: tuple[int, int, int, int] = DEFAULT_WINDOW,
        output_dir: Path = ROOT / "results/tmc2_science_prototype") -> dict[str, Any]:
    """Register same-pass science windows and save native and global results."""
    if len(reference_window) != 4 or min(reference_window[:2]) < 0 or min(reference_window[2:]) < 1:
        raise ValueError("Reference window must be X Y WIDTH HEIGHT with positive dimensions")
    pairs = (
        ("calibrated_fore_to_nadir", "calibrated_fore", "calibrated_nadir",
         ROOT / "results/tmc2_prototype/calibrated_nadir_to_fore/transformation.json"),
        ("calibrated_aft_to_nadir", "calibrated_aft", "calibrated_nadir",
         ROOT / "results/tmc2_prototype/calibrated_nadir_to_aft/transformation.json"),
        ("raw_to_calibrated_nadir", "raw_nadir", "calibrated_nadir", None),
    )
    config = RegistrationConfig(matcher="sift", geometry_model="homography", clahe=True)
    rows = []
    for name, source_name, reference_name, browse_transform in pairs:
        source_label, source_browse = _paths(source_name)
        reference_label, reference_browse = _paths(reference_name)
        for needed in (source_label, source_browse, reference_label, reference_browse):
            if not needed.is_file():
                raise FileNotFoundError(f"Selected staged product is missing: {needed}")
        if browse_transform is not None and not browse_transform.is_file():
            raise FileNotFoundError(
                f"Browse registration is required first: {browse_transform}"
            )
        source_window = (
            _source_window(source_label, source_browse, reference_label,
                           reference_browse, reference_window, browse_transform)
            if browse_transform is not None else reference_window
        )
        print(f"Registering {name}: source={source_window}, reference={reference_window}",
              flush=True)
        source = load_image(source_label, sensor="TMC-2",
                            data_level=source_name.split("_", 1)[0], window=source_window)
        reference = load_image(reference_label, sensor="TMC-2",
                               data_level=reference_name.split("_", 1)[0],
                               window=reference_window)
        result = register(source, reference, config)
        target = output_dir / name
        result.save(target)
        full_matrix = global_transform(result.geometry.matrix, source_window, reference_window)
        (target / "full_image_transformation.json").write_text(json.dumps({
            "model": config.geometry_model, "matrix": full_matrix.tolist(),
            "direction": "source_to_reference_full_image_pixels",
            "source_window": source_window, "reference_window": reference_window,
        }, indent=2) + "\n", encoding="utf-8")
        _write_global_matches(target / "matches_full_image.csv", target / "matches.csv",
                              source_window, reference_window)
        rows.append({
            "pair": name,
            "source_label": str(source_label.relative_to(ROOT)),
            "reference_label": str(reference_label.relative_to(ROOT)),
            "source_window": source_window, "reference_window": reference_window,
            "browse_transform_for_crop_selection": (
                str(browse_transform.relative_to(ROOT)) if browse_transform else None),
            "output": str(target.relative_to(ROOT)),
            **result.metrics,
        })
    report = {
        "dataset": "ISRO/ISSDC Chandrayaan-2 TMC-2 science image windows",
        "representation": "native IMG pixels from bounded PDS4 windows",
        "method": "SIFT + homography RANSAC + 8x8 spatial selection",
        "ground_truth_rmse": None,
        "ground_truth_note": "No independent tie-point ground truth; residuals are fitted reprojection errors.",
        "pairs": rows,
    }
    report_path = ROOT / "data/processed/tmc2_science_window_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    selection_path = ROOT / "data/processed/tmc2_prototype_selection.json"
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selection["processing_status"] = (
            "five products staged; browse and native-resolution window registrations "
            "completed; full-strip and independent accuracy evaluation pending"
        )
        selection["science_window_report"] = str(report_path.relative_to(ROOT))
        selection["science_window_registrations"] = [
            {"pair": row["pair"], "result_dir": row["output"]} for row in rows
        ]
        selection_path.write_text(json.dumps(selection, indent=2) + "\n",
                                  encoding="utf-8")
    return report


def main() -> int:
    """Run the bounded science-image experiment from the project root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-window", type=int, nargs=4,
                        metavar=("X", "Y", "WIDTH", "HEIGHT"), default=DEFAULT_WINDOW)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/tmc2_science_prototype")
    args = parser.parse_args()
    report = run(tuple(args.reference_window), args.output)
    for row in report["pairs"]:
        print(f"{row['pair']}: {row['inliers']} inliers, "
              f"{row['registration_residual_rmse_px']:.3f} px fitted residual, "
              f"{row['selected_coverage']:.1%} coverage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
