from __future__ import annotations

import json
from zipfile import ZipFile

from experiments.inventory_sensor_data import inventory_sensor_data


def test_inventory_separates_product_rasters_quicklooks_and_staging(tmp_path) -> None:
    raw_root = tmp_path / "raw2"
    data_root = tmp_path / "data"
    product_dir = raw_root / "TMC-2/data/calibrated/20260823"
    product_dir.mkdir(parents=True)
    with ZipFile(product_dir / "tmc_product.zip", "w") as archive:
        archive.writestr("data/calibrated/product.img", b"science")
        archive.writestr("browse/calibrated/product.png", b"quicklook")
    (product_dir / "partial.zip.part").write_bytes(b"partial")
    staged = data_root / "calibrated/TMC-2/product"
    staged.mkdir(parents=True)
    (staged / "lunamatch_product_manifest.json").write_text(json.dumps({"sensor": "TMC-2"}))

    result = inventory_sensor_data(raw_root, data_root)

    assert result["sensors"]["TMC-2"]["downloaded_product_archives"] == 1
    assert result["sensors"]["TMC-2"]["science_raster_members"] == 1
    assert result["sensors"]["TMC-2"]["browse_image_members"] == 1
    assert result["sensors"]["TMC-2"]["staged_products"] == 1
    assert result["sensors"]["TMC-2"]["incomplete_downloads"] == 1
    assert result["sensors"]["IIRS"]["downloaded_product_archives"] == 0
