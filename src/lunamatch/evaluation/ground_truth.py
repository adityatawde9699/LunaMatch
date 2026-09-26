"""Evaluate a saved registration against independently supplied tie points."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from lunamatch.geometry.ransac import transform_points

FIELDS = ("id", "source_x", "source_y", "reference_x", "reference_y")


def validate_points(rows: list[dict[str, Any]]) -> list[dict[str, float | str]]:
    """Validate finite, unique point pairs in image pixel coordinates."""
    if not rows:
        raise ValueError("At least one independent tie point is required")
    if len(rows) > 10000:
        raise ValueError("At most 10000 tie points can be evaluated")
    normalized: list[dict[str, float | str]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        identifier = str(row.get("id", index)).strip()
        if not identifier or identifier in seen:
            raise ValueError(f"Tie-point id must be nonempty and unique: {identifier!r}")
        seen.add(identifier)
        point: dict[str, float | str] = {"id": identifier}
        for key in FIELDS[1:]:
            try:
                value = float(row[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Tie point {identifier}: {key} must be a number") from exc
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Tie point {identifier}: {key} must be finite and nonnegative")
            point[key] = value
        normalized.append(point)
    return normalized


def load_points_csv(path: str | Path) -> list[dict[str, float | str]]:
    """Read independent source/reference coordinate pairs from CSV."""
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not set(FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"Tie-point CSV requires columns: {', '.join(FIELDS)}")
        return validate_points(list(reader))


def evaluate_points(matrix: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute point errors using a fixed transform; never refit on these points."""
    points = validate_points(rows)
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (3, 3) or not np.all(np.isfinite(transform)):
        raise ValueError("Transformation must be a finite 3 x 3 matrix")
    source = np.array([[point["source_x"], point["source_y"]] for point in points], dtype=np.float64)
    reference = np.array([[point["reference_x"], point["reference_y"]] for point in points], dtype=np.float64)
    predicted = transform_points(source, transform)
    if not np.all(np.isfinite(predicted)):
        raise ValueError("Transformation projects a tie point outside finite coordinates")
    errors = np.linalg.norm(predicted - reference, axis=1)
    detailed = [{**point, "predicted_reference_x": float(predicted[i, 0]),
                 "predicted_reference_y": float(predicted[i, 1]),
                 "error_px": float(errors[i])} for i, point in enumerate(points)]
    return {
        "point_count": len(points),
        "ground_truth_rmse_px": float(np.sqrt(np.mean(errors ** 2))),
        "ground_truth_median_px": float(np.median(errors)),
        "ground_truth_p95_px": float(np.percentile(errors, 95)),
        "ground_truth_max_px": float(np.max(errors)),
        "evaluation_type": "fixed_transform_vs_independent_user_supplied_points",
        "annotation_provenance": "user_supplied_unverified",
        "points": detailed,
    }


def evaluate_run(run_dir: str | Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate a saved registration without changing its fitted metrics."""
    run = Path(run_dir)
    payload = json.loads((run / "transformation.json").read_text())
    if payload.get("direction") != "source_to_reference_pixels":
        raise ValueError("Saved transformation has an unsupported direction")
    report = evaluate_points(np.asarray(payload["matrix"]), rows)
    for side in ("source", "reference"):
        preview = run / f"{side}_preview.png"
        if preview.is_file():
            with Image.open(preview) as image:
                width, height = image.size
            for point in report["points"]:
                if point[f"{side}_x"] >= width or point[f"{side}_y"] >= height:
                    raise ValueError(f"Tie point {point['id']} lies outside the {side} preview")
    report["transformation_model"] = payload.get("model")
    return report


def save_evaluation(run_dir: str | Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Persist the supplied points and a separate validation report."""
    run = Path(run_dir)
    report = evaluate_run(run, rows)
    with (run / "ground_truth_points.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows([{key: point[key] for key in FIELDS} for point in report["points"]])
    (run / "ground_truth_evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
