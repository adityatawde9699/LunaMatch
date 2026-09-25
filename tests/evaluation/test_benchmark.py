import csv
import json

import cv2
import numpy as np
from PIL import Image
import yaml

from lunamatch.evaluation.benchmark import run_benchmark


def test_benchmark_writes_measured_rows(tmp_path) -> None:
    rng = np.random.default_rng(17)
    source = rng.integers(0, 255, (200, 200), dtype=np.uint8)
    reference = cv2.warpAffine(source, np.float32([[1, 0, 8], [0, 1, -5]]), (200, 200))
    Image.fromarray(source).save(tmp_path / "source.png")
    Image.fromarray(reference).save(tmp_path / "reference.png")
    experiment = {
        "name": "generated_test",
        "pairs": [{"name": "translation", "source": "source.png",
                   "reference": "reference.png", "synthetic": True}],
        "variants": [{"name": "SIFT", "parameters": {"matcher": "sift",
                                                  "geometry_model": "affine"}}],
    }
    config = tmp_path / "experiment.yaml"
    config.write_text(yaml.safe_dump(experiment))
    rows = run_benchmark(config, tmp_path / "output")
    assert rows[0]["status"] == "completed"
    assert rows[0]["synthetic"] is True
    assert rows[0]["ground_truth_rmse_px"] is None
    assert rows[0]["inliers"] > 10
    assert len(list(csv.DictReader((tmp_path / "output/results.csv").open()))) == 1
    assert json.loads((tmp_path / "output/results.json").read_text())["rows"][0]["matcher"] == "sift"
