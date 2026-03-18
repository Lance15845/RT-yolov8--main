from __future__ import annotations

import argparse

from project_config import (
    COMPARISON_FIELDS,
    DATASET_KEYS,
    METRIC_FIELDS,
    MODEL_KEYS,
    benchmark_csv_path,
    bootstrap_ultralytics_path,
    build_comparison_rows,
    comparison_csv_path,
    dataset_yaml_path,
    ensure_workspace_dirs,
    find_experiment_weights,
    metrics_csv_path,
    read_csv_rows,
    resolve_device,
    write_csv_rows,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate one or more trained experiment variants.")
    parser.add_argument("--dataset", required=True, choices=DATASET_KEYS)
    parser.add_argument("--model", choices=MODEL_KEYS, help="Evaluate a single model key. Default evaluates all.")
    parser.add_argument("--weights", help="Optional explicit weights path for single-model evaluation.")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", help="Explicit device string. Defaults to mps when available, otherwise cpu.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_workspace_dirs()
    bootstrap_ultralytics_path()

    from ultralytics import YOLO

    data_path = dataset_yaml_path(args.dataset)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {data_path}. Run prepare_data.py for dataset '{args.dataset}' first."
        )

    selected_models = [args.model] if args.model else list(MODEL_KEYS)
    metric_rows = []

    for model_key in selected_models:
        try:
            weights_path = find_experiment_weights(args.dataset, model_key, override=args.weights if args.model else None)
        except FileNotFoundError:
            if args.model:
                raise
            continue

        model = YOLO(str(weights_path))
        metrics = model.val(
            data=str(data_path),
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=resolve_device(args.device),
            verbose=False,
        )
        _, params, _, flops = model.info(verbose=False)
        row = {
            "dataset": args.dataset,
            "model_key": model_key,
            "weights_path": str(weights_path),
            "map50": round(float(metrics.box.map50), 6),
            "map5095": round(float(metrics.box.map), 6),
            "params_m": round(float(params) / 1e6, 6),
            "gflops": round(float(flops), 6),
        }
        metric_rows.append(row)
        write_csv_rows(metrics_csv_path(args.dataset, model_key), METRIC_FIELDS, [row])

    if not metric_rows:
        raise RuntimeError(f"No models could be evaluated for dataset '{args.dataset}'.")

    if args.dataset == "face_mask":
        all_metric_rows = []
        for model_key in MODEL_KEYS:
            all_metric_rows.extend(read_csv_rows(metrics_csv_path(args.dataset, model_key)))
        benchmark_rows = read_csv_rows(benchmark_csv_path(args.dataset))
        comparison_rows = build_comparison_rows(all_metric_rows, benchmark_rows)
        write_csv_rows(comparison_csv_path(args.dataset), COMPARISON_FIELDS, comparison_rows)

    print(f"Evaluation finished for dataset={args.dataset} models={','.join(row['model_key'] for row in metric_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
