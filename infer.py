from __future__ import annotations

import argparse

from project_config import (
    DATASET_KEYS,
    MODEL_KEYS,
    bootstrap_ultralytics_path,
    ensure_workspace_dirs,
    figures_dir_for,
    find_experiment_weights,
    pick_sample_image,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate qualitative prediction figures for trained variants.")
    parser.add_argument("--dataset", default="face_mask", choices=DATASET_KEYS)
    parser.add_argument("--model", choices=MODEL_KEYS, help="Run inference for a single model key. Default runs all.")
    parser.add_argument("--weights", help="Optional explicit weights path for single-model inference.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", help="Device string forwarded to Ultralytics predict().")
    parser.add_argument("--source", help="Optional explicit source image. Defaults to the first validation image.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_workspace_dirs()
    bootstrap_ultralytics_path()

    from ultralytics import YOLO

    source = args.source or str(pick_sample_image(args.dataset))
    selected_models = [args.model] if args.model else list(MODEL_KEYS)
    output_dir = figures_dir_for(args.dataset)

    for model_key in selected_models:
        try:
            weights_path = find_experiment_weights(args.dataset, model_key, override=args.weights if args.model else None)
        except FileNotFoundError:
            if args.model:
                raise
            continue

        model = YOLO(str(weights_path))
        prediction = model.predict(source=source, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        output_path = output_dir / f"{model_key}_predictions.png"
        prediction.save(filename=str(output_path))
        print(f"Saved prediction figure for model={model_key} to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
