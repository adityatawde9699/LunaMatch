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
    """Deterministic patch sampler with lunar viewpoint and illumination views."""

    def __init__(self, image: np.ndarray | list[np.ndarray], *, patch_size: int = 32,
                 samples: int = 2048, seed: int = 26166) -> None:
        images = image if isinstance(image, list) else [image]
        if not images or any(item.ndim != 2 for item in images):
            raise ValueError("Training images must be non-empty grayscale 2D arrays")
        if patch_size < 16 or patch_size % 2:
            raise ValueError("patch_size must be an even value of at least 16")
        if any(min(item.shape) < patch_size + 2 for item in images):
            raise ValueError("Training image is smaller than patch_size")
        self.images = [item.astype(np.float32) / 255.0 for item in images]
        self.patch_size = patch_size
        self.samples = samples
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        margin = patch_size // 2
        self.scene = self.rng.integers(0, len(self.images), samples)
        self.y = np.array([self.rng.integers(margin, self.images[s].shape[0] - margin)
                           for s in self.scene])
        self.x = np.array([self.rng.integers(margin, self.images[s].shape[1] - margin)
                           for s in self.scene])

    def __len__(self) -> int:
        return self.samples

    def _view(self, patch: np.ndarray, seed: int) -> np.ndarray:
        """Create a small geometric and photometric variant of one patch."""
        rng = np.random.default_rng(seed)
        size = self.patch_size
        center = (size / 2.0 - 0.5, size / 2.0 - 0.5)
        angle = float(rng.uniform(-25.0, 25.0))
        scale = float(rng.uniform(0.88, 1.12))
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        matrix[:, 2] += rng.uniform(-2.0, 2.0, size=2)
        view = cv2.warpAffine(patch, matrix, (size, size), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT101)
        if rng.random() < 0.5:
            view = np.flip(view, axis=1)
        if rng.random() < 0.25:
            view = cv2.GaussianBlur(view, (3, 3), 0.0)
        gain = float(rng.uniform(0.70, 1.30))
        bias = float(rng.uniform(-0.15, 0.15))
        gamma = float(rng.uniform(0.85, 1.20))
        view = np.power(np.clip(view * gain + bias, 0.0, 1.0), gamma)
        view += rng.normal(0.0, 0.025, view.shape).astype(np.float32)
        return np.clip(view, 0.0, 1.0).copy()

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        torch, _, _ = _imports()
        half = self.patch_size // 2
        scene = int(self.scene[index])
        y, x = int(self.y[index]), int(self.x[index])
        patch = self.images[scene][y - half:y + half, x - half:x + half]
        first = torch.from_numpy(self._view(patch, self.seed + index * 2 + 1)).float()[None]
        second = torch.from_numpy(self._view(patch, self.seed + index * 2 + 2)).float()[None]
        return first, second


def _info_nce(z1: Any, z2: Any, temperature: float) -> Any:
    """Symmetric in-batch contrastive loss."""
    torch, _, _ = _imports()
    logits = z1 @ z2.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (torch.nn.functional.cross_entropy(logits, labels) +
            torch.nn.functional.cross_entropy(logits.T, labels)) / 2


def train_descriptor(image: np.ndarray | list[np.ndarray], output: str | Path, *, epochs: int = 5,
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
    model.eval()
    correct = total_pairs = 0
    with torch.inference_mode():
        for first, second in loader:
            z1, z2 = model(first.to(selected)), model(second.to(selected))
            predictions = (z1 @ z2.T).argmax(dim=1)
            correct += int((predictions == torch.arange(len(predictions), device=predictions.device)).sum())
            total_pairs += len(predictions)
    alignment_top1 = correct / max(total_pairs, 1)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "embedding_dim": embedding_dim,
                "patch_size": patch_size, "model": "LunarPatchDescriptor"}, destination)
    report = {
        "checkpoint": str(destination), "model": "LunarPatchDescriptor",
        "training_objective": "symmetric in-batch InfoNCE under geometric/photometric lunar-view augmentation",
        "epochs": epochs, "batch_size": batch_size, "samples": samples,
        "training_images": len(image) if isinstance(image, list) else 1,
        "embedding_dim": embedding_dim, "patch_size": patch_size,
        "device": selected, "seed": seed, "history": history,
        "alignment_top1": alignment_top1,
        "validation": "augmentation-pair retrieval only; registration accuracy not evaluated",
    }
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report
