import numpy as np

from lunamatch.ingestion import load_image
from lunamatch.preprocessing.cross_modal import iirs_pca_plane


def test_iirs_pca_produces_derived_spatial_plane() -> None:
    texture = np.arange(100, dtype=np.float32).reshape(10, 10)
    cube = np.stack((texture, 2 * texture, 3 * texture), axis=-1)
    source = load_image(cube, sensor="IIRS", data_level="raw")
    result = iirs_pca_plane(source)
    assert result.image.shape == (10, 10)
    assert result.data_level == "processed"
    assert source.data_level == "raw"
    assert np.corrcoef(result.image.ravel(), texture.ravel())[0, 1] > 0.99
