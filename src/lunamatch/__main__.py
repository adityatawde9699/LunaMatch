"""Command line interface for image inspection."""

import argparse
import json
from pathlib import Path

from lunamatch.ingestion import load_image
from lunamatch.ingestion.pradan_archive import inspect_pradan_product, stage_pradan_product
from lunamatch.ingestion.validators import ImageLoadError
from lunamatch.utils.image import save_preview


def main() -> int:
    """Run the LunaMatch command line interface."""
    parser = argparse.ArgumentParser(prog="lunamatch")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Inspect image metadata and optional preview")
    inspect.add_argument("--input", required=True, type=Path)
    inspect.add_argument("--sensor", default=None)
    inspect.add_argument("--data-level", choices=("raw", "calibrated", "derived", "processed", "unknown"), default="unknown")
    inspect.add_argument("--preview", type=Path)
    inspect.add_argument("--band", type=int, default=0)
    inspect.add_argument("--window", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    classify = commands.add_parser("classify", help="Classify sensor from product provenance")
    classify.add_argument("--input", required=True, action="append", type=Path,
                          help="Product/image path; repeat for multiple inputs")
    inspect_product = commands.add_parser(
        "inspect-product", help="Inventory a PRADAN product ZIP without extracting it")
    inspect_product.add_argument("--input", required=True, type=Path)
    stage_product = commands.add_parser(
        "stage-product", help="Safely extract a PRADAN product ZIP by sensor and data level")
    stage_product.add_argument("--input", required=True, type=Path)
    stage_product.add_argument("--output", type=Path, default=Path("data"))
    stage_product.add_argument("--max-uncompressed-gb", type=float, default=20.0)
    register_cmd = commands.add_parser("register", help="Register an image pair with SIFT and RANSAC")
    register_cmd.add_argument("--source", required=True, type=Path)
    register_cmd.add_argument("--reference", required=True, type=Path)
    register_cmd.add_argument("--source-window", type=int, nargs=4,
                              metavar=("X", "Y", "WIDTH", "HEIGHT"))
    register_cmd.add_argument("--reference-window", type=int, nargs=4,
                              metavar=("X", "Y", "WIDTH", "HEIGHT"))
    register_cmd.add_argument("--source-sensor", default=None)
    register_cmd.add_argument("--reference-sensor", default=None)
    register_cmd.add_argument("--matcher", choices=("sift", "orb", "akaze", "descriptor", "loftr", "lightglue", "hybrid"))
    register_cmd.add_argument("--geometry", choices=("homography", "affine"))
    register_cmd.add_argument("--clahe", action="store_true")
    register_cmd.add_argument("--config", type=Path)
    register_cmd.add_argument("--output", required=True, type=Path)
    benchmark_cmd = commands.add_parser("benchmark", help="Run a declared experiment matrix")
    benchmark_cmd.add_argument("--config", required=True, type=Path)
    benchmark_cmd.add_argument("--output", required=True, type=Path)
    evaluate_cmd = commands.add_parser("evaluate-ground-truth", help="Evaluate a saved transform against independent tie points")
    evaluate_cmd.add_argument("--run", required=True, type=Path, help="Saved registration result directory")
    evaluate_cmd.add_argument("--points", required=True, type=Path, help="CSV with id,source_x,source_y,reference_x,reference_y")
    train_cmd = commands.add_parser("train-descriptor", help="Train the optional local patch descriptor")
    train_cmd.add_argument("--input", required=True, action="append", type=Path,
                           help="Training image; repeat for multiple scenes")
    train_cmd.add_argument("--output", required=True, type=Path)
    train_cmd.add_argument("--epochs", type=int, default=5)
    train_cmd.add_argument("--batch-size", type=int, default=64)
    train_cmd.add_argument("--samples", type=int, default=2048)
    train_cmd.add_argument("--patch-size", type=int, default=32)
    train_cmd.add_argument("--embedding-dim", type=int, default=128)
    train_cmd.add_argument("--learning-rate", type=float, default=1e-3)
    train_cmd.add_argument("--temperature", type=float, default=0.1)
    train_cmd.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train_cmd.add_argument("--sensor", default=None)
    args = parser.parse_args()
    try:
        if args.command == "classify":
            from lunamatch.ingestion.classification import classify_sensor
            print(json.dumps([classify_sensor(path) for path in args.input], indent=2))
            return 0
        if args.command == "inspect":
            product = load_image(args.input, sensor=args.sensor, data_level=args.data_level,
                                 window=tuple(args.window) if args.window else None)
            if args.preview:
                save_preview(product, args.preview, band=args.band)
            print(json.dumps(product.summary(), indent=2, default=str))
            return 0
        if args.command == "inspect-product":
            print(json.dumps(inspect_pradan_product(args.input), indent=2))
            return 0
        if args.command == "stage-product":
            if args.max_uncompressed_gb <= 0:
                raise ValueError("--max-uncompressed-gb must be positive")
            manifest = stage_pradan_product(
                args.input, args.output,
                max_uncompressed_bytes=int(args.max_uncompressed_gb * 1024**3))
            print(json.dumps(manifest, indent=2))
            return 0
        if args.command == "register":
            from lunamatch.pipeline.config import RegistrationConfig, load_config
            from lunamatch.pipeline.registration import register

            source = load_image(args.source, sensor=args.source_sensor,
                                window=tuple(args.source_window) if args.source_window else None)
            reference = load_image(args.reference, sensor=args.reference_sensor,
                                   window=tuple(args.reference_window) if args.reference_window else None)
            config = load_config(args.config) if args.config else RegistrationConfig()
            if args.matcher:
                config.matcher = args.matcher
            if args.geometry:
                config.geometry_model = args.geometry
            if args.clahe:
                config.clahe = True
            result = register(source, reference, config)
            result.save(args.output)
            print(json.dumps({"output": str(args.output), **result.metrics}, indent=2))
            return 0
        if args.command == "benchmark":
            from lunamatch.evaluation.benchmark import run_benchmark

            rows = run_benchmark(args.config, args.output)
            print(json.dumps({"output": str(args.output), "runs": len(rows),
                              "completed": sum(row["status"] == "completed" for row in rows)}, indent=2))
            return 0
        if args.command == "evaluate-ground-truth":
            from lunamatch.evaluation.ground_truth import load_points_csv, save_evaluation

            report = save_evaluation(args.run, load_points_csv(args.points))
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "train-descriptor":
            from lunamatch.learning.training import train_descriptor
            from lunamatch.preprocessing.normalization import matching_gray

            products = [load_image(path, sensor=args.sensor) for path in args.input]
            report = train_descriptor(
                [matching_gray(product) for product in products], args.output, epochs=args.epochs,
                batch_size=args.batch_size, samples=args.samples,
                patch_size=args.patch_size, embedding_dim=args.embedding_dim,
                learning_rate=args.learning_rate, temperature=args.temperature,
                device=args.device,
            )
            print(json.dumps(report, indent=2))
            return 0
    except (ImageLoadError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"lunamatch: {exc}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
