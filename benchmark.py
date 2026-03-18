from __future__ import annotations

import argparse
import time

import numpy as np
from PIL import Image

from project_config import (
    BENCHMARK_FIELDS,
    COMPARISON_FIELDS,
    DATASET_KEYS,
    MODEL_KEYS,
    benchmark_csv_path,
    bootstrap_ultralytics_path,
    build_comparison_rows,
    comparison_csv_path,
    ensure_workspace_dirs,
    find_experiment_weights,
    metrics_csv_path,
    pick_sample_image,
    read_csv_rows,
    resolve_device,
    write_csv_rows,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark inference latency for trained experiment variants.")
    parser.add_argument("--dataset", default="face_mask", choices=DATASET_KEYS)
    parser.add_argument("--model", choices=MODEL_KEYS, help="Benchmark a single model key. Default benchmarks all.")
    parser.add_argument("--weights", help="Optional explicit weights path for single-model benchmarking.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", help="Explicit device string. Defaults to mps when available, otherwise cpu.")
    parser.add_argument("--source", help="Optional explicit source image. Defaults to the first validation image.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_workspace_dirs()
    bootstrap_ultralytics_path()

    from ultralytics import YOLO

    device = resolve_device(args.device)
    source_path = args.source or str(pick_sample_image(args.dataset))
    source_image = Image.open(source_path).convert("RGB")
    source_array = np.asarray(source_image)
    selected_models = [args.model] if args.model else list(MODEL_KEYS)
    rows = []

    for model_key in selected_models:
        try:
            weights_path = find_experiment_weights(args.dataset, model_key, override=args.weights if args.model else None)
        except FileNotFoundError:
            if args.model:
                raise
            continue

        model = YOLO(str(weights_path))
        for _ in range(args.warmup):
            model.predict(source=source_array, imgsz=args.imgsz, device=device, verbose=False)
            synchronize_device(device)

        start = time.perf_counter()
        for _ in range(args.iters):
            model.predict(source=source_array, imgsz=args.imgsz, device=device, verbose=False)
            synchronize_device(device)
        elapsed = time.perf_counter() - start
        latency_ms = (elapsed / args.iters) * 1000.0
        fps = 0.0 if latency_ms == 0 else 1000.0 / latency_ms

        rows.append(
            {
                "dataset": args.dataset,
                "model_key": model_key,
                "weights_path": str(weights_path),
                "device": device,
                "source_image": source_path,
                "latency_ms": round(latency_ms, 6),
                "fps": round(fps, 6),
            }
        )

    if not rows:
        raise RuntimeError(f"No models could be benchmarked for dataset '{args.dataset}'.")

    existing_rows = {row["model_key"]: row for row in read_csv_rows(benchmark_csv_path(args.dataset))}
    for row in rows:
        existing_rows[row["model_key"]] = row
    merged_rows = [existing_rows[key] for key in MODEL_KEYS if key in existing_rows]
    write_csv_rows(benchmark_csv_path(args.dataset), BENCHMARK_FIELDS, merged_rows)

    if args.dataset == "face_mask":
        metric_rows = []
        for model_key in MODEL_KEYS:
            metric_rows.extend(read_csv_rows(metrics_csv_path(args.dataset, model_key)))
        if metric_rows:
            comparison_rows = build_comparison_rows(metric_rows, merged_rows)
            write_csv_rows(comparison_csv_path(args.dataset), COMPARISON_FIELDS, comparison_rows)

    print(f"Benchmark finished for dataset={args.dataset} models={','.join(row['model_key'] for row in rows)}")
    return 0


def synchronize_device(device: str) -> None:
    if device == "mps":
        import torch

        if getattr(torch, "mps", None):
            torch.mps.synchronize()
    elif device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


if __name__ == "__main__":
    raise SystemExit(main())
