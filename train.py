from __future__ import annotations

import argparse

from project_config import (
    DATASET_KEYS,
    DEFAULT_TRAIN_ARGS,
    MODEL_KEYS,
    bootstrap_ultralytics_path,
    dataset_yaml_path,
    ensure_workspace_dirs,
    experiment_dir,
    model_yaml_path,
    resolve_device,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one YOLOv8-Light experiment variant.")
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
        "--fraction",
        type=float,
        default=1.0,
        help="Dataset fraction to use for smoke tests. 1.0 means the full dataset.",
    )
    parser.add_argument("--device", help="Explicit device string. Defaults to mps when available, otherwise cpu.")
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
        fraction=args.fraction,
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
        device=resolve_device(args.device),
        verbose=True,
        resume=args.resume,
    )

    print(f"Training finished for dataset={args.dataset} model={args.model} results_dir={result.save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
