from lunamatch.ingestion.classification import classify_sensor


def test_classifies_authoritative_product_names() -> None:
    assert classify_sensor("data/calibrated/OHRC/ch2_ohr_ncp.zip")["sensor"] == "OHRC"
    assert classify_sensor("data/raw/TMC-2/ch2_tmc_ncn.zip")["sensor"] == "TMC-2"
    assert classify_sensor("data/derived/IIRS/ch2_iir_ndi.qub")["sensor"] == "IIRS"


def test_does_not_guess_synthetic_sensor() -> None:
    result = classify_sensor("data/samples/synthetic_iirs_cube.tif")
    assert result["sensor"] == "UNKNOWN"
