"""Known geometry and independent point evaluation checks."""

import csv
import json

import numpy as np
import pytest
from PIL import Image

from lunamatch.evaluation.ground_truth import (
    evaluate_points, load_points_csv, save_evaluation,
)


def test_independent_points_measure_fixed_transform(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    matrix = [[1, 0, 10], [0, 1, -5], [0, 0, 1]]
    (run / "transformation.json").write_text(json.dumps({
        "model": "affine", "matrix": matrix, "direction": "source_to_reference_pixels"}))
    Image.new("L", (100, 100)).save(run / "source_preview.png")
    Image.new("L", (100, 100)).save(run / "reference_preview.png")
    truth = tmp_path / "truth.csv"
    with truth.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("id", "source_x", "source_y", "reference_x", "reference_y"))
        writer.writeheader()
        writer.writerows([
            {"id": "A", "source_x": 20, "source_y": 20, "reference_x": 30, "reference_y": 15},
            {"id": "B", "source_x": 40, "source_y": 30, "reference_x": 51, "reference_y": 25},
        ])
    report = save_evaluation(run, load_points_csv(truth))
    assert report["ground_truth_rmse_px"] == pytest.approx(np.sqrt(0.5))
    assert report["ground_truth_max_px"] == pytest.approx(1)
    assert report["annotation_provenance"] == "user_supplied_unverified"
    assert (run / "ground_truth_points.csv").is_file()
    assert (run / "ground_truth_evaluation.json").is_file()


def test_reject_duplicate_and_nonfinite_points() -> None:
    row = {"id": "1", "source_x": 5, "source_y": 4, "reference_x": 6, "reference_y": 3}
    with pytest.raises(ValueError, match="unique"):
        evaluate_points(np.eye(3), [row, row])
    with pytest.raises(ValueError, match="finite"):
        evaluate_points(np.eye(3), [{**row, "source_x": float("nan")}])
