"""Generate software-test pairs and exact synthetic tie points."""

import csv
from pathlib import Path

import cv2
import numpy as np


def generate_samples(output: Path) -> None:
    """Write deterministic synthetic pairs and known-transform control points."""
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
    truth = output / "synthetic_translation_tie_points.csv"
    with truth.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("id", "source_x", "source_y", "reference_x", "reference_y"))
        for identifier, (x, y) in enumerate(((90, 100), (200, 100), (320, 100),
                                             (500, 100), (90, 250), (200, 250),
                                             (320, 250), (500, 250), (90, 400),
                                             (200, 400), (320, 400), (500, 400)), start=1):
            writer.writerow((identifier, x, y, x + 23, y - 14))

    # A moderate affine change exercises scale and viewpoint handling.
    affine_matrix = cv2.getRotationMatrix2D((320, 256), 7.0, 1.08)
    affine_matrix[:, 2] += (12.0, -9.0)
    affine_reference = cv2.warpAffine(image, affine_matrix, (640, 512),
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cv2.imwrite(str(output / "synthetic_affine_source.png"), image)
    cv2.imwrite(str(output / "synthetic_affine_reference.png"), affine_reference)

    # A photometric pair exercises illumination normalization.
    illumination_reference = np.clip(
        np.power(reference.astype(np.float32) / 255.0, 0.72) * 255.0 * 0.82 + 18.0,
        0, 255,
    ).astype(np.uint8)
    cv2.imwrite(str(output / "synthetic_illumination_source.png"), image)
    cv2.imwrite(str(output / "synthetic_illumination_reference.png"), illumination_reference)

    # A degraded pair exercises noise and blur handling.
    noisy_reference = cv2.GaussianBlur(reference, (5, 5), 1.2)
    noise = rng.normal(0, 9, noisy_reference.shape).astype(np.float32)
    noisy_reference = np.clip(noisy_reference.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(str(output / "synthetic_noisy_source.png"), image)
    cv2.imwrite(str(output / "synthetic_noisy_reference.png"), noisy_reference)
    try:
        import rasterio
        cube = np.stack((image, np.clip(image.astype(np.float32) * 0.8 + 20, 0, 255).astype(np.uint8),
                         np.clip(image.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)))
        with rasterio.open(output / "synthetic_iirs_cube.tif", "w", driver="GTiff",
                           width=640, height=512, count=3, dtype="uint8") as dataset:
            dataset.write(cube)
    except ImportError:
        print("Rasterio unavailable: skipped synthetic IIRS cube")
    print(f"Wrote synthetic software-test pairs to {output}")


def main() -> None:
    """Generate local sample files in the default development directory."""
    generate_samples(Path("data/samples"))


if __name__ == "__main__":
    main()
