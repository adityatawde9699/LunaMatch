"""Generate a software-test pair; it is not Chandrayaan-2 imagery."""

from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    """Write a deterministic crater-like translated pair for a local smoke test."""
    output = Path("data/samples")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(26166)
    image = rng.normal(110, 22, (512, 640)).clip(0, 255).astype(np.uint8)
    for _ in range(160):
        center = (int(rng.integers(20, 620)), int(rng.integers(20, 492)))
        radius = int(rng.integers(3, 20))
        cv2.circle(image, center, radius, 225, 2)
        cv2.circle(image, (center[0] + 2, center[1] + 2), radius, 45, 1)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    reference = cv2.warpAffine(image, np.float32([[1, 0, 23], [0, 1, -14]]),
                               (640, 512))
    cv2.imwrite(str(output / "synthetic_source.png"), image)
    cv2.imwrite(str(output / "synthetic_reference.png"), reference)
    try:
        import rasterio
        cube = np.stack((image, np.clip(image.astype(np.float32) * 0.8 + 20, 0, 255).astype(np.uint8),
                         np.clip(image.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)))
        with rasterio.open(output / "synthetic_iirs_cube.tif", "w", driver="GTiff",
                           width=640, height=512, count=3, dtype="uint8") as dataset:
            dataset.write(cube)
    except ImportError:
        print("Rasterio unavailable: skipped synthetic IIRS cube")
    print(f"Wrote synthetic software-test pair to {output}")


if __name__ == "__main__":
    main()
