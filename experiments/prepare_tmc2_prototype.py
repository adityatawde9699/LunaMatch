"""Stage a five-product TMC-2 same-pass prototype subset from data/raw2."""

from __future__ import annotations

import json
from pathlib import Path

from lunamatch.ingestion.pradan_archive import stage_pradan_product

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw2/TMC-2"
OUTPUT = ROOT / "data"
MANIFEST = ROOT / "data/processed/tmc2_prototype_selection.json"

# One calibrated stereo triplet plus raw nadir/fore counterparts from the same
# acquisition. This supports multi-view matching and a small raw/calibrated
# representation comparison over one footprint.
PRODUCTS = [
    "data/calibrated/20260823/ch2_tmc_ncn_20260823T1657513854_d_img_d18.zip",
    "data/calibrated/20260823/ch2_tmc_nca_20260823T1657513854_d_img_d18.zip",
    "data/calibrated/20260823/ch2_tmc_ncf_20260823T1657513886_d_img_d18.zip",
    "data/raw/20260823/ch2_tmc_nrn_20260823T1657513854_d_img_d18.zip",
    "data/raw/20260823/ch2_tmc_nrf_20260823T1657513886_d_img_d18.zip",
]


def main() -> int:
    previous = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else {}
    staged = []
    for relative in PRODUCTS:
        archive = RAW / relative
        if not archive.is_file():
            raise FileNotFoundError(f"Selected prototype product is missing: {archive}")
        level = relative.split("/", 2)[1]
        destination = OUTPUT / level / "TMC-2" / archive.stem
        manifest_path = destination / "lunamatch_product_manifest.json"
        if manifest_path.is_file():
            item = json.loads(manifest_path.read_text())
            print(f"Already staged: {archive.name}")
        else:
            item = stage_pradan_product(archive, OUTPUT)
            print(f"Staged: {archive.name} -> {item['staged_at']}")
        staged.append({**item, "archive": str(archive.relative_to(ROOT))})

    selection = {
        **previous,
        "sensor": "TMC-2",
        "product_count": len(staged),
        "selection_reason": "one calibrated fore/nadir/aft stereo triplet plus two same-pass raw views",
        "processing_status": previous.get(
            "processing_status", "staged; registration validation pending"),
        "products": staged,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote selection manifest: {MANIFEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
