"""Safety and classification checks for local PRADAN product ZIPs."""

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from lunamatch.ingestion.pradan_archive import (
    inspect_pradan_product,
    stage_pradan_product,
)
from lunamatch.ingestion.validators import ImageLoadError


def _product_zip(path: Path, entries: dict[str, bytes]) -> Path:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return path


def test_inspect_and_stage_calibrated_ohrc_product(tmp_path: Path) -> None:
    archive = _product_zip(tmp_path / "ch2_ohr_product.zip", {
        "data/calibrated/image.img": b"pixels",
        "data/calibrated/image.xml": b"<Product_Observational />",
        "geometry/geometry.xml": b"<geometry />",
        "browse/preview.png": b"preview",
    })

    inventory = inspect_pradan_product(archive)
    assert inventory["sensor"] == "OHRC"
    assert inventory["data_level"] == "calibrated"
    assert len(inventory["xml_labels"]) == 2

    manifest = stage_pradan_product(archive, tmp_path / "data")
    destination = Path(manifest["staged_at"])
    assert destination == tmp_path / "data/calibrated/OHRC/ch2_ohr_product"
    assert (destination / "data/calibrated/image.img").read_bytes() == b"pixels"
    assert json.loads((destination / "lunamatch_product_manifest.json").read_text())["sensor"] == "OHRC"


def test_reject_path_traversal_and_clean_staging(tmp_path: Path) -> None:
    archive = _product_zip(tmp_path / "ch2_tmc_product.zip", {
        "../escape.img": b"no",
        "data/raw/label.xml": b"<label />",
    })
    output = tmp_path / "data"
    with pytest.raises(ImageLoadError, match="Unsafe ZIP member path"):
        stage_pradan_product(archive, output)
    assert not (tmp_path / "escape.img").exists()
    assert not list(output.rglob("*.img"))


def test_refuse_oversized_and_document_only_archives(tmp_path: Path) -> None:
    product = _product_zip(tmp_path / "ch2_iir_calibrated.zip", {
        "data/calibrated/cube.qub": b"123456",
    })
    with pytest.raises(ImageLoadError, match="uncompressed size limit"):
        stage_pradan_product(product, tmp_path / "data", max_uncompressed_bytes=2)

    docs = _product_zip(tmp_path / "ohr.zip", {"document/readme.txt": b"docs"})
    with pytest.raises(ImageLoadError, match="documentation only"):
        stage_pradan_product(docs, tmp_path / "data")


def test_existing_product_is_not_overwritten(tmp_path: Path) -> None:
    archive = _product_zip(tmp_path / "ch2_tmc2_raw.zip", {
        "raw/image.tif": b"image",
    })
    first = stage_pradan_product(archive, tmp_path / "data")
    with pytest.raises(ImageLoadError, match="already exists"):
        stage_pradan_product(archive, tmp_path / "data")
    assert Path(first["staged_at"]).is_dir()
