"""Synchronous registration and durable result retrieval endpoints."""

import json
import os
from pathlib import Path
import re
import tempfile
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import ValidationError

from api.schemas import RegistrationOptions
from lunamatch.ingestion import load_image
from lunamatch.ingestion.validators import ImageLoadError
from lunamatch.pipeline.registration import register as register_images

router = APIRouter(prefix="/api/v1")
UPLOAD_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
ARTIFACTS = {"registered_image.tif", "registered_preview.png", "source_preview.png",
             "reference_preview.png", "overlay.png",
             "match_visualization.png", "error_map.png", "confidence_map.png",
             "distribution.png",
             "matches.csv", "matches.json", "candidate_matches.csv", "candidate_matches.json",
             "metrics.json", "transformation.json", "job_log.json"}


def _results_root() -> Path:
    return Path(os.environ.get("LUNAMATCH_RESULTS_DIR", "results/jobs"))


def _job_dir(job_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise HTTPException(404, "Job not found")
    directory = _results_root() / job_id
    if not (directory / "metrics.json").is_file():
        raise HTTPException(404, "Job not found")
    return directory


async def _save_upload(upload: UploadFile, directory: Path, stem: str) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in UPLOAD_SUFFIXES:
        raise HTTPException(415, f"Unsupported upload type: {suffix or 'none'}")
    max_bytes = int(os.environ.get("LUNAMATCH_MAX_UPLOAD_BYTES", str(100 * 1024**2)))
    path = directory / f"{stem}{suffix}"
    total = 0
    with path.open("wb") as output:
        while chunk := await upload.read(1024 * 1024):
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(413, f"Upload exceeds {max_bytes} bytes")
            output.write(chunk)
    if total == 0:
        raise HTTPException(422, "Uploaded image is empty")
    return path


@router.post("/register")
async def register(source: UploadFile = File(...), reference: UploadFile = File(...),
                   source_sensor: str = Form(...), reference_sensor: str = Form(...),
                   matcher: str = Form("sift"), preprocessing: str = Form("{}")) -> dict:
    """Register two uploaded raster images and persist the result artifacts."""
    if source_sensor not in ("OHRC", "TMC-2", "IIRS") or reference_sensor not in ("OHRC", "TMC-2", "IIRS"):
        raise HTTPException(422, "Sensors must be OHRC, TMC-2, or IIRS")
    if matcher not in ("sift", "orb", "akaze", "loftr", "hybrid"):
        raise HTTPException(422, "Unsupported matcher")
    try:
        options = RegistrationOptions.model_validate_json(preprocessing)
    except ValidationError as exc:
        raise HTTPException(422, f"Invalid preprocessing configuration: {exc}") from exc
    try:
        with tempfile.TemporaryDirectory(prefix="lunamatch-upload-") as temporary:
            temp = Path(temporary)
            source_path = await _save_upload(source, temp, "source")
            reference_path = await _save_upload(reference, temp, "reference")
            source_image = load_image(source_path, sensor=source_sensor)
            reference_image = load_image(reference_path, sensor=reference_sensor)
            result = register_images(source_image, reference_image, options.to_config(matcher))
    except (ImageLoadError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    job_id = uuid4().hex
    result.save(_results_root() / job_id)
    return {
        "job_id": job_id,
        "status": "completed",
        "matches": result.metrics["candidate_matches"],
        "inliers": result.metrics["inliers"],
        "inlier_ratio": result.metrics["inlier_ratio"],
        "rmse": result.metrics["registration_residual_rmse_px"],
        "rmse_type": "RANSAC inlier reprojection residual, pixels",
        "coverage": result.metrics["selected_coverage"],
        "runtime_seconds": result.metrics["runtime_seconds"],
        "device": result.metrics["device"],
    }


@router.get("/results/{job_id}")
async def get_result(job_id: str) -> dict:
    """Return metrics and discoverable artifact paths for a completed job."""
    directory = _job_dir(job_id)
    return {"job_id": job_id, "status": "completed",
            "metrics": json.loads((directory / "metrics.json").read_text()),
            "artifacts": {name: f"/api/v1/results/{job_id}/artifacts/{name}"
                          for name in ARTIFACTS if (directory / name).is_file()}}


@router.get("/results/{job_id}/matches")
async def get_matches(job_id: str) -> list[dict]:
    """Return selected, geometrically verified correspondences."""
    return json.loads((_job_dir(job_id) / "matches.json").read_text())


@router.get("/results/{job_id}/candidates")
async def get_candidates(job_id: str) -> list[dict]:
    """Return all candidate pairs with RANSAC and grid-selection flags."""
    return json.loads((_job_dir(job_id) / "candidate_matches.json").read_text())


@router.get("/results/{job_id}/metrics")
async def get_metrics(job_id: str) -> dict:
    """Return measured metrics with explicit residual semantics."""
    return json.loads((_job_dir(job_id) / "metrics.json").read_text())


@router.get("/results/{job_id}/registered-image")
async def get_registered_image(job_id: str) -> Response:
    """Download the source image warped into reference coordinates."""
    return Response((_job_dir(job_id) / "registered_image.tif").read_bytes(),
                    media_type="image/tiff")


@router.get("/results/{job_id}/artifacts/{name}")
async def get_artifact(job_id: str, name: str) -> Response:
    """Retrieve a whitelisted visualization or export artifact."""
    if name not in ARTIFACTS:
        raise HTTPException(404, "Artifact not found")
    path = _job_dir(job_id) / name
    media_type = "image/png" if name.endswith(".png") else (
        "image/tiff" if name.endswith(".tif") else "application/octet-stream")
    return Response(path.read_bytes(), media_type=media_type)
