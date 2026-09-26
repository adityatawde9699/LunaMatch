"""Training tests use small generated pixels and never claim lunar validation."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lunamatch.learning.descriptor import LunarPatchDescriptor
from lunamatch.learning.training import train_descriptor
from lunamatch.features.learned_descriptor import _load_model, extract_learned


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


def test_short_training_run_takes_steps_and_reuses_checkpoint(tmp_path, monkeypatch) -> None:
    image = np.random.default_rng(13).integers(0, 255, (96, 96), dtype=np.uint8)
    checkpoint = tmp_path / "short.pt"
    report = train_descriptor(image, checkpoint, epochs=1, batch_size=64,
                              samples=8, embedding_dim=16, patch_size=16, device="cpu",
                              input_paths=[tmp_path / "source.png"])
    assert report["training_steps"] == 1
    assert report["effective_batch_size"] == 8
    assert report["evaluation_split"] == "training samples only; no held-out scenes"
    assert len(report["training_image_sha256"]) == 1
    assert report["training_input_paths"] == [str((tmp_path / "source.png").resolve())]
    monkeypatch.setenv("LUNAMATCH_DESCRIPTOR_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("LUNAMATCH_DESCRIPTOR_DEVICE", "cpu")
    _load_model.cache_clear()
    extract_learned(image, max_features=20)
    extract_learned(image, max_features=20)
    assert _load_model.cache_info().misses == 1
    assert _load_model.cache_info().hits == 1


def test_training_rejects_single_sample_without_checkpoint(tmp_path) -> None:
    with pytest.raises(ValueError, match="samples"):
        train_descriptor(np.full((96, 96), 128, dtype=np.uint8), tmp_path / "bad.pt",
                         samples=1, epochs=1, batch_size=64)
    assert not (tmp_path / "bad.pt").exists()
