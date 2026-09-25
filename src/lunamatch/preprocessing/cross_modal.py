"""Spatial representations for OHRC, TMC-2, and IIRS products."""

import numpy as np

from lunamatch.ingestion.metadata import LunarImage


def iirs_pca_plane(product: LunarImage, *, sample_pixels: int = 50_000) -> LunarImage:
    """Project a valid IIRS band cube onto its first spectral principal axis.

    PCA is estimated from a bounded deterministic pixel sample. It produces
    a derived spatial representation, not calibrated reflectance.
    """
    image = product.image
    if image.ndim != 3 or image.shape[2] < 2:
        raise ValueError("IIRS PCA requires at least two bands")
    valid = np.isfinite(image).all(axis=2)
    if product.valid_mask is not None:
        valid &= product.valid_mask
    locations = np.flatnonzero(valid.ravel())
    if len(locations) < 3:
        raise ValueError("IIRS cube has too few valid spectra")
    sample = locations[np.linspace(0, len(locations) - 1,
                                   min(len(locations), sample_pixels), dtype=int)]
    pixels = image.reshape(-1, image.shape[2])
    spectra = pixels[sample].astype(np.float64)
    mean = spectra.mean(axis=0)
    _, _, vt = np.linalg.svd(spectra - mean, full_matrices=False)
    axis = vt[0]
    if axis.sum() < 0:
        axis = -axis
    output = np.zeros(image.shape[:2], dtype=np.float32)
    flattened = output.ravel()
    for chunk in np.array_split(locations, max(1, int(np.ceil(len(locations) / 100_000)))):
        flattened[chunk] = (pixels[chunk].astype(np.float64) - mean) @ axis
    return LunarImage(image=output, sensor=product.sensor, data_level="processed",
                      source_path=product.source_path, valid_mask=valid,
                      metadata={**product.metadata, "spatial_representation": "IIRS first PCA component",
                                "source_data_level": product.data_level})
