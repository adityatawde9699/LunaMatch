"""Train the optional LunaMatch local descriptor on declared image pixels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lunamatch.ingestion import load_image
from lunamatch.preprocessing.normalization import matching_gray
from lunamatch.learning.training import train_descriptor


def main() -> int:
    """Train on a local image and write a checkpoint plus JSON training log."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, action="append",
                        help="Training image; repeat for multiple scenes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--sensor", default=None)
    args = parser.parse_args()
    products = [load_image(path, sensor=args.sensor) for path in args.input]
    gray = [matching_gray(product) for product in products]
    report = train_descriptor(
        gray, args.output, epochs=args.epochs, batch_size=args.batch_size,
        samples=args.samples, patch_size=args.patch_size,
        embedding_dim=args.embedding_dim, learning_rate=args.learning_rate,
        temperature=args.temperature, device=args.device,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
