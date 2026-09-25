"""Reproduce the small five-product PRADAN TMC-2 prototype evaluation.

Only the three calibrated and two raw products selected in
``prepare_tmc2_prototype.py`` are used. Registrations run on PDS4 browse PNGs,
not the larger science-resolution IMG arrays.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from lunamatch.ingestion import load_image
from lunamatch.pipeline.config import RegistrationConfig
from lunamatch.pipeline.registration import register


ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "data/calibrated/TMC-2"
RAW = ROOT / "data/raw/TMC-2"
OUTPUT = ROOT / "results/tmc2_prototype"
SELECTION = ROOT / "data/processed/tmc2_prototype_selection.json"
REPORT_JSON = ROOT / "data/processed/tmc2_prototype_report.json"
REPORT_CSV = ROOT / "data/processed/tmc2_prototype_metrics.csv"

PAIRS = {
    "calibrated_nadir_to_aft": (
        CAL / "ch2_tmc_nca_20260823T1657513854_d_img_d18/browse/calibrated/20260823/ch2_tmc_nca_20260823T1657513854_b_brw_d18.png",
        CAL / "ch2_tmc_ncn_20260823T1657513854_d_img_d18/browse/calibrated/20260823/ch2_tmc_ncn_20260823T1657513854_b_brw_d18.png",
    ),
    "calibrated_nadir_to_fore": (
        CAL / "ch2_tmc_ncf_20260823T1657513886_d_img_d18/browse/calibrated/20260823/ch2_tmc_ncf_20260823T1657513886_b_brw_d18.png",
        CAL / "ch2_tmc_ncn_20260823T1657513854_d_img_d18/browse/calibrated/20260823/ch2_tmc_ncn_20260823T1657513854_b_brw_d18.png",
    ),
    "raw_nadir_to_fore": (
        RAW / "ch2_tmc_nrf_20260823T1657513886_d_img_d18/browse/raw/20260823/ch2_tmc_nrf_20260823T1657513886_b_brw_d18.png",
        RAW / "ch2_tmc_nrn_20260823T1657513854_d_img_d18/browse/raw/20260823/ch2_tmc_nrn_20260823T1657513854_b_brw_d18.png",
    ),
    "raw_to_calibrated_nadir": (
        RAW / "ch2_tmc_nrn_20260823T1657513854_d_img_d18/browse/raw/20260823/ch2_tmc_nrn_20260823T1657513854_b_brw_d18.png",
        CAL / "ch2_tmc_ncn_20260823T1657513854_d_img_d18/browse/calibrated/20260823/ch2_tmc_ncn_20260823T1657513854_b_brw_d18.png",
    ),
}


def run(output_dir: Path = OUTPUT) -> dict[str, Any]:
    """Register the selected quicklook pairs and write reproducible reports."""
    absent = sorted({str(path) for pair in PAIRS.values() for path in pair if not path.is_file()})
    if absent:
        raise FileNotFoundError(f"Selected browse images are missing: {absent}")

    output_dir.mkdir(parents=True, exist_ok=True)
    config = RegistrationConfig(matcher="sift", geometry_model="homography", clahe=True)
    rows: list[dict[str, Any]] = []
    for pair_name, (source_path, reference_path) in PAIRS.items():
        print(f"Registering {pair_name}", flush=True)
        source = load_image(source_path, sensor="TMC-2")
        reference = load_image(reference_path, sensor="TMC-2")
        result = register(source, reference, config)
        result_dir = output_dir / pair_name
        result.save(result_dir)
        row: dict[str, Any] = {
            "pair": pair_name,
            "source": str(source_path.relative_to(ROOT)),
            "reference": str(reference_path.relative_to(ROOT)),
            "result_dir": str(result_dir.relative_to(ROOT)),
            "status": "completed",
            **result.metrics,
        }
        rows.append(row)

    report = {
        "dataset": "ISRO/ISSDC Chandrayaan-2 TMC-2 prototype products",
        "selected_product_count": 5,
        "selected_products_manifest": str(SELECTION.relative_to(ROOT)),
        "representation": "PDS4 browse PNG quicklooks; science-resolution IMG registration not run",
        "method": {
            "matcher": config.matcher,
            "geometry": config.geometry_model,
            "clahe": config.clahe,
            "spatial_grid": [config.grid_rows, config.grid_cols],
        },
        "ground_truth_rmse": None,
        "ground_truth_note": "No independent tie-point reference is available; residuals are fitted RANSAC errors.",
        "pairs": rows,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    if SELECTION.is_file():
        selection = json.loads(SELECTION.read_text(encoding="utf-8"))
        selection["processing_status"] = (
            "five products staged; four official browse-pair registrations completed; "
            "science-resolution registration pending"
        )
        selection["prototype_report"] = str(REPORT_JSON.relative_to(ROOT))
        selection["registrations"] = [
            {"pair": row["pair"], "result_dir": row["result_dir"]}
            for row in rows
        ]
        SELECTION.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    """Run the prototype evaluation from the project root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
