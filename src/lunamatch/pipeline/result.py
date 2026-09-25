"""Registration result and reproducible baseline exports."""

from dataclasses import dataclass
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from lunamatch.geometry.ransac import GeometryResult
from lunamatch.matching.matcher_interface import Correspondences


@dataclass(slots=True)
class RegistrationResult:
    """Pixels, geometry, and measured metrics from one registration."""

    source_gray: np.ndarray
    reference_gray: np.ndarray
    registered: np.ndarray
    registered_preview: np.ndarray
    matches: Correspondences
    geometry: GeometryResult
    metrics: dict
    visualization: np.ndarray
    selected_indices: np.ndarray
    provenance: dict

    def save(self, directory: str | Path) -> Path:
        """Write registered image, correspondences, transform, and visuals."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        try:
            import rasterio
            reference_meta = self.provenance["reference"].get("metadata", {})
            geospatial = {}
            if reference_meta.get("crs") and reference_meta.get("transform"):
                geospatial = {
                    "crs": reference_meta["crs"],
                    "transform": rasterio.Affine(*reference_meta["transform"]),
                }
            bands = 1 if self.registered.ndim == 2 else self.registered.shape[2]
            with rasterio.open(directory / "registered_image.tif", "w", driver="GTiff",
                               width=self.registered.shape[1], height=self.registered.shape[0],
                               count=bands, dtype=str(self.registered.dtype), **geospatial) as dataset:
                if bands == 1:
                    dataset.write(self.registered, 1)
                else:
                    dataset.write(np.moveaxis(self.registered, -1, 0))
        except ImportError:
            if self.registered.ndim == 3 and self.registered.shape[2] not in (3, 4):
                raise OSError("Multiband TIFF export requires rasterio")
            if not cv2.imwrite(str(directory / "registered_image.tif"), self.registered):
                raise OSError("Failed to save registered image")
        cv2.imwrite(str(directory / "registered_preview.png"), self.registered_preview)
        cv2.imwrite(str(directory / "source_preview.png"), self.source_gray)
        cv2.imwrite(str(directory / "reference_preview.png"), self.reference_gray)
        cv2.imwrite(str(directory / "match_visualization.png"), self.visualization)
        overlay = cv2.addWeighted(self.registered_preview, 0.5, self.reference_gray, 0.5, 0)
        cv2.imwrite(str(directory / "overlay.png"), overlay)
        errors = cv2.cvtColor(self.reference_gray, cv2.COLOR_GRAY2BGR)
        confidences = errors.copy()
        distribution = errors.copy()
        height, width = self.reference_gray.shape
        for row in range(1, 8):
            cv2.line(distribution, (0, round(row * height / 8)),
                     (width - 1, round(row * height / 8)), (120, 120, 120), 1)
        for col in range(1, 8):
            cv2.line(distribution, (round(col * width / 8), 0),
                     (round(col * width / 8), height - 1), (120, 120, 120), 1)
        for index in self.selected_indices:
            point = tuple(np.rint(self.matches.reference_points[index]).astype(int))
            error = float(self.geometry.residuals[index])
            confidence = float(self.matches.confidence[index])
            cv2.circle(errors, point, 3, (0, int(max(0, 255 - error * 50)),
                                           int(min(255, error * 50))), -1)
            cv2.circle(confidences, point, 3, (0, int(255 * confidence),
                                                int(255 * (1 - confidence))), -1)
            cv2.circle(distribution, point, 2, (0, 220, 110), -1)
        cv2.imwrite(str(directory / "error_map.png"), errors)
        cv2.imwrite(str(directory / "confidence_map.png"), confidences)
        cv2.imwrite(str(directory / "distribution.png"), distribution)
        columns = ("id", "source_x", "source_y", "reference_x", "reference_y",
                   "confidence", "inlier", "selected", "error")
        selected_set = set(self.selected_indices.tolist())
        rows = [dict(zip(columns, (i + 1, float(s[0]), float(s[1]), float(r[0]),
                                   float(r[1]), float(c), bool(ok), i in selected_set,
                                   float(e))))
                for i, (s, r, c, ok, e) in enumerate(zip(
                    self.matches.source_points, self.matches.reference_points,
                    self.matches.confidence, self.geometry.inliers,
                    self.geometry.residuals))]
        with (directory / "candidate_matches.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        (directory / "candidate_matches.json").write_text(json.dumps(rows, indent=2))
        selected_rows = [rows[index] for index in self.selected_indices]
        with (directory / "matches.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(selected_rows)
        (directory / "matches.json").write_text(json.dumps(selected_rows, indent=2))
        (directory / "transformation.json").write_text(json.dumps({
            "model": self.geometry.model,
            "matrix": self.geometry.matrix.tolist(),
            "direction": "source_to_reference_pixels",
        }, indent=2))
        (directory / "metrics.json").write_text(json.dumps(self.metrics, indent=2))
        (directory / "job_log.json").write_text(json.dumps({
            **self.provenance,
            "transformation": self.geometry.matrix.tolist(),
            "metrics": self.metrics,
        }, indent=2, default=str))
        return directory
