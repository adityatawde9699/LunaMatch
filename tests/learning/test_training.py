"""Training tests use small generated pixels and never claim lunar validation."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lunamatch.learning.descriptor import LunarPatchDescriptor
from lunamatch.learning.training import train_descriptor
from lunamatch.features.learned_descriptor import extract_learned


def test_descriptor_forward_and_checkpoint(tmp_path, monkeypatch) -> None:
    rng = np.random.default_rng(11)
    image = rng.integers(0, 255, (96, 96), dtype=np.uint8)
    model = LunarPatchDescriptor(embedding_dim=16)
    output = model(torch.rand(4, 1, 32, 32))
    assert output.shape == (4, 16)
    assert torch.allclose(output.norm(dim=1), torch.ones(4), atol=1e-5)

    report = train_descriptor(
        image, tmp_path / "descriptor.pt", epochs=1, batch_size=8,
        samples=16, embedding_dim=16, patch_size=16, seed=3, device="cpu",
    )
    assert report["device"] == "cpu"
    assert (tmp_path / "descriptor.pt").is_file()
    log = json.loads((tmp_path / "descriptor.json").read_text())
    assert log["validation"].startswith("augmentation-pair retrieval")
    assert 0.0 <= report["alignment_top1"] <= 1.0

    monkeypatch.setenv("LUNAMATCH_DESCRIPTOR_CHECKPOINT", str(tmp_path / "descriptor.pt"))
    features = extract_learned(image, max_features=20)
    assert features.descriptors is not None
    assert features.descriptors.shape[1] == 16


def test_multi_scene_training_reports_scene_count(tmp_path) -> None:
    rng = np.random.default_rng(12)
    images = [rng.integers(0, 255, (96, 96), dtype=np.uint8),
              rng.integers(0, 255, (96, 96), dtype=np.uint8)]
    report = train_descriptor(images, tmp_path / "multi.pt", epochs=1,
                              batch_size=8, samples=16, embedding_dim=16,
                              patch_size=16, device="cpu")
    assert report["training_images"] == 2
