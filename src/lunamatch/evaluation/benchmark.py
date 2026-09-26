"""Reproducible pair/variant experiments without invented results."""

import csv
from dataclasses import asdict
import json
from pathlib import Path
from time import time

import yaml

from lunamatch.ingestion import load_image
from lunamatch.pipeline.config import RegistrationConfig
from lunamatch.pipeline.registration import register


def run_benchmark(config_path: str | Path, output_dir: str | Path) -> list[dict]:
    """Execute declared variants on declared pairs and write CSV/JSON results."""
    config_path = Path(config_path)
    with config_path.open() as stream:
        experiment = yaml.safe_load(stream)
    if not isinstance(experiment, dict) or not experiment.get("pairs") or not experiment.get("variants"):
        raise ValueError("Benchmark config needs nonempty pairs and variants")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pair in experiment["pairs"]:
        source_path = (config_path.parent / pair["source"]).resolve()
        reference_path = (config_path.parent / pair["reference"]).resolve()
        source = reference = None
        load_error = None
        try:
            source = load_image(source_path, sensor=pair.get("source_sensor"))
            reference = load_image(reference_path, sensor=pair.get("reference_sensor"))
        except (RuntimeError, ValueError, MemoryError, OSError) as exc:
            load_error = exc
        for variant in experiment["variants"]:
            name = str(variant["name"])
            parameters = variant.get("parameters", {})
            baseline = asdict(RegistrationConfig())
            unknown = set(parameters) - set(baseline)
            if unknown:
                raise ValueError(f"Unknown benchmark parameters: {sorted(unknown)}")
            settings = RegistrationConfig(**{**baseline, **parameters})
            row = {"experiment": experiment.get("name", config_path.stem),
                   "pair": pair.get("name", source_path.name), "method": name,
                   "matcher": settings.matcher, "synthetic": bool(pair.get("synthetic", False)),
                   "source": str(source_path), "reference": str(reference_path),
                   "status": "completed", "error": None}
            try:
                if load_error is not None:
                    raise ValueError(f"Unable to load pair: {load_error}") from load_error
                result = register(source, reference, settings)
                row.update({
                    "matches": result.metrics["candidate_matches"],
                    "inliers": result.metrics["inliers"],
                    "inlier_ratio": result.metrics["inlier_ratio"],
                    "registration_residual_rmse_px": result.metrics["registration_residual_rmse_px"],
                    "ground_truth_rmse_px": None,
                    "coverage": result.metrics["selected_coverage"],
                    "runtime_seconds": result.metrics["runtime_seconds"],
                    "device": result.metrics["device"],
                })
            except (RuntimeError, ValueError, MemoryError, OSError) as exc:
                row.update({"status": "failed", "error": str(exc), "matches": None,
                            "inliers": None, "inlier_ratio": None,
                            "registration_residual_rmse_px": None,
                            "ground_truth_rmse_px": None, "coverage": None,
                            "runtime_seconds": None, "device": None})
            rows.append(row)
    (output_dir / "results.json").write_text(json.dumps({
        "config": str(config_path.resolve()), "created_unix_time": time(),
        "rows": rows}, indent=2))
    with (output_dir / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows
