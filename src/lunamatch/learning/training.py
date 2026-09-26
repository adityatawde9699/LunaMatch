"""Contrastive training for a local lunar patch descriptor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .descriptor import LunarPatchDescriptor


def _imports() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Training requires: pip install 'lunamatch[learned]'") from exc
    return torch, nn, (DataLoader, Dataset)


class PatchAugmentationDataset:
    """Deterministic patch sampler with illumination and noise augmentations."""

    def __init__(self, image: np.ndarray, *, patch_size: int = 32,
                 samples: int = 2048, seed: int = 26166) -> None:
        if image.ndim != 2:
            raise ValueError("Training image must be a grayscale 2D array")
        if patch_size < 16 or patch_size % 2:
            raise ValueError("patch_size must be an even value of at least 16")
        if min(image.shape) < patch_size + 2:
            raise ValueError("Training image is smaller than patch_size")
        self.image = image.astype(np.float32) / 255.0
        self.patch_size = patch_size
        self.samples = samples
        self.rng = np.random.default_rng(seed)
        margin = patch_size // 2
        self.y = self.rng.integers(margin, image.shape[0] - margin, samples)
        self.x = self.rng.integers(margin, image.shape[1] - margin, samples)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        torch, _, _ = _imports()
        half = self.patch_size // 2
        y, x = int(self.y[index]), int(self.x[index])
        patch = self.image[y - half:y + half, x - half:x + half]
        first = torch.from_numpy(patch.copy()).float()[None]
        second = first.clone()
        if index % 2:
            second = torch.flip(second, dims=(1,))
        gain = 0.75 + 0.5 * ((index * 1103515245 + 12345) % 1000) / 1000
        bias = -0.12 + 0.24 * ((index * 214013 + 2531011) % 1000) / 1000
        noise = torch.randn_like(second) * 0.025
        second = (second * gain + bias + noise).clamp(0.0, 1.0)
        return first, second


def _info_nce(z1: Any, z2: Any, temperature: float) -> Any:
    """Symmetric in-batch contrastive loss."""
    torch, _, _ = _imports()
    logits = z1 @ z2.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (torch.nn.functional.cross_entropy(logits, labels) +
            torch.nn.functional.cross_entropy(logits.T, labels)) / 2


def train_descriptor(image: np.ndarray, output: str | Path, *, epochs: int = 5,
                     batch_size: int = 64, learning_rate: float = 1e-3,
                     embedding_dim: int = 128, patch_size: int = 32,
                     samples: int = 2048, temperature: float = 0.1,
                     seed: int = 26166, device: str = "auto") -> dict[str, Any]:
    """Train and save a descriptor checkpoint on declared training pixels.

    This is representation pretraining. If the input is synthetic, the
    checkpoint must not be presented as Chandrayaan-2 validation.
    """
    torch, _, data = _imports()
    if epochs < 1 or batch_size < 2 or learning_rate <= 0 or temperature <= 0:
        raise ValueError("epochs, batch_size, learning_rate, and temperature are invalid")
    torch.manual_seed(seed)
    np.random.seed(seed)
    selected = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if selected == "auto":
        selected = "cpu"
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    if selected not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    DataLoader, _ = data
    dataset = PatchAugmentationDataset(image, patch_size=patch_size, samples=samples, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = LunarPatchDescriptor(embedding_dim=embedding_dim).to(selected)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(epochs):
        total = 0.0
        batches = 0
        for first, second in loader:
            first, second = first.to(selected), second.to(selected)
            loss = _info_nce(model(first), model(second), temperature)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1
        history.append({"epoch": epoch + 1, "loss": total / max(batches, 1)})
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "embedding_dim": embedding_dim,
                "patch_size": patch_size, "model": "LunarPatchDescriptor"}, destination)
    report = {
        "checkpoint": str(destination), "model": "LunarPatchDescriptor",
        "training_objective": "symmetric in-batch InfoNCE under photometric/flip augmentation",
        "epochs": epochs, "batch_size": batch_size, "samples": samples,
        "embedding_dim": embedding_dim, "patch_size": patch_size,
        "device": selected, "seed": seed, "history": history,
        "validation": "representation pretraining only; registration accuracy not evaluated",
    }
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report
