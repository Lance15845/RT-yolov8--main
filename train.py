from __future__ import annotations

import argparse
import shutil

from project_config import (
    DATASET_KEYS,
    DEFAULT_TRAIN_ARGS,
    MODEL_KEYS,
    bootstrap_ultralytics_path,
    dataset_yaml_path,
    ensure_workspace_dirs,
    experiment_dir,
    model_yaml_path,
    pick_sample_image,
    resolve_device,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one VOC YOLOv8-Light experiment variant.")
    parser.add_argument("--dataset", required=True, choices=DATASET_KEYS)
    parser.add_argument("--model", required=True, choices=MODEL_KEYS)
    parser.add_argument("--weights", help="Optional weights file used for fine-tuning or resume.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAIN_ARGS["epochs"])
    parser.add_argument("--batch", type=int, default=DEFAULT_TRAIN_ARGS["batch"])
    parser.add_argument("--imgsz", type=int, default=DEFAULT_TRAIN_ARGS["imgsz"])
    parser.add_argument("--optimizer", default=DEFAULT_TRAIN_ARGS["optimizer"])
    parser.add_argument("--lr0", type=float, default=DEFAULT_TRAIN_ARGS["lr0"])
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAIN_ARGS["seed"])
    parser.add_argument("--workers", type=int, default=DEFAULT_TRAIN_ARGS["workers"])
    parser.add_argument("--patience", type=int, default=DEFAULT_TRAIN_ARGS["patience"])
    parser.add_argument(
        "--cache",
        default=DEFAULT_TRAIN_ARGS["cache"],
        help="Image caching mode: ram, disk, or off.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Dataset fraction to use for smoke tests. 1.0 means the full dataset.",
    )
    parser.add_argument("--device", help="CUDA-only. Omit or use cuda:0.")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TRAIN_ARGS["amp"],
        help="Enable or disable CUDA Automatic Mixed Precision.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from the provided --weights checkpoint.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_workspace_dirs()
    bootstrap_ultralytics_path()

    from ultralytics import YOLO

    if args.resume and not args.weights:
        raise ValueError("--resume requires --weights pointing to an existing checkpoint.")

    data_path = dataset_yaml_path(args.dataset)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {data_path}. Run prepare_data.py for dataset '{args.dataset}' first."
        )

    model_source = args.weights or str(model_yaml_path(args.model))
    model = YOLO(model_source)
    run_dir = experiment_dir(args.dataset, args.model)
    run_dir.mkdir(parents=True, exist_ok=True)
    ensure_amp_check_asset()

    result = model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        optimizer=args.optimizer,
        lr0=args.lr0,
        seed=args.seed,
        workers=args.workers,
        patience=args.patience,
        cache=normalize_cache_mode(args.cache),
        fraction=args.fraction,
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
        device=resolve_device(args.device),
        amp=bool(args.amp),
        verbose=True,
        resume=args.resume,
    )

    print(f"Training finished for dataset={args.dataset} model={args.model} results_dir={result.save_dir}")
    return 0


def ensure_amp_check_asset() -> None:
    from ultralytics.utils import ASSETS

    target = ASSETS / "bus.jpg"
    if target.exists():
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pick_sample_image("voc"), target)


def normalize_cache_mode(value: object) -> str | bool:
    normalized = str(value).strip().lower()
    if normalized in {"off", "false", "0", "none", ""}:
        return False
    if normalized in {"ram", "disk"}:
        return normalized
    raise ValueError("Unsupported cache mode. Use 'ram', 'disk', or 'off'.")


if __name__ == "__main__":
    raise SystemExit(main())
