"""API integration tests use generated pixels only."""

from io import BytesIO
import asyncio

import cv2
import numpy as np
import httpx
from PIL import Image

from api.main import app


def _png(image: np.ndarray) -> bytes:
    stream = BytesIO()
    Image.fromarray(image).save(stream, format="PNG")
    return stream.getvalue()


def test_upload_register_and_fetch(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LUNAMATCH_RESULTS_DIR", str(tmp_path / "jobs"))
    rng = np.random.default_rng(5)
    source = rng.integers(0, 255, (240, 240), dtype=np.uint8)
    reference = cv2.warpAffine(source, np.float32([[1, 0, 9], [0, 1, -6]]),
                               (240, 240))
    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://test") as client:
            response = await client.post("/api/v1/register", data={
                "source_sensor": "OHRC", "reference_sensor": "OHRC",
                "matcher": "sift", "preprocessing": '{"geometry_model":"affine"}'},
                files={"source": ("source.png", _png(source), "image/png"),
                       "reference": ("reference.png", _png(reference), "image/png")})
            assert response.status_code == 200, response.text
            job = response.json()["job_id"]
            assert response.json()["inliers"] > 10
            assert (await client.get(f"/api/v1/results/{job}")).status_code == 200
            assert (await client.get(f"/api/v1/results/{job}/matches")).json()
            metrics = (await client.get(f"/api/v1/results/{job}/metrics")).json()
            assert metrics["ground_truth_rmse_px"] is None
            assert (await client.get(f"/api/v1/results/{job}/registered-image")).status_code == 200
            assert (await client.get(f"/api/v1/results/{job}/artifacts/overlay.png")).status_code == 200
            assert (await client.get(f"/api/v1/results/{job}/artifacts/nope.py")).status_code == 404
    asyncio.run(exercise())


def test_reject_executable_upload() -> None:
    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://test") as client:
            response = await client.post("/api/v1/register", data={
                "source_sensor": "OHRC", "reference_sensor": "OHRC"},
                files={"source": ("code.py", b"print(1)", "text/plain"),
                       "reference": ("image.png", b"x", "image/png")})
            assert response.status_code == 415
    asyncio.run(exercise())
