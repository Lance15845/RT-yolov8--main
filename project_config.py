from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Iterable, Optional

import yaml


ROOT = Path(__file__).resolve().parent
ULTRALYTICS_ROOT = ROOT / "ultralytics"
CONFIGS_ROOT = ROOT / "configs"
DATASET_CONFIG_DIR = CONFIGS_ROOT / "datasets"
DATASETS_DIR = ROOT / "datasets"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = ROOT / "runs"

DATASET_KEYS = ("voc",)
MODEL_KEYS = ("baseline", "mobilenetv2", "pcg_ghost")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}

VOC_NAMES = {
    0: "aeroplane",
    1: "bicycle",
    2: "bird",
    3: "boat",
    4: "bottle",
    5: "bus",
    6: "car",
    7: "cat",
    8: "chair",
    9: "cow",
    10: "diningtable",
    11: "dog",
    12: "horse",
    13: "motorbike",
    14: "person",
    15: "pottedplant",
    16: "sheep",
    17: "sofa",
    18: "train",
    19: "tvmonitor",
}

DATASET_CONFIGS = {
    "voc": DATASET_CONFIG_DIR / "voc.yaml",
}

MODEL_CONFIGS = {
    "baseline": ULTRALYTICS_ROOT / "ultralytics" / "cfg" / "models" / "custom" / "yolov8n_voc_baseline.yaml",
    "mobilenetv2": ULTRALYTICS_ROOT / "ultralytics" / "cfg" / "models" / "custom" / "yolov8n_voc_mobilenetv2.yaml",
    "pcg_ghost": ULTRALYTICS_ROOT / "ultralytics" / "cfg" / "models" / "custom" / "yolov8n_voc_pcg_ghost.yaml",
}

DEFAULT_TRAIN_ARGS = {
    "imgsz": 640,
    "batch": 24,
    "epochs": 100,
    "optimizer": "auto",
    "lr0": 0.01,
    "seed": 42,
    "workers": 2,
    "patience": 20,
    "cache": "disk",
    "amp": True,
}

METRIC_FIELDS = ["dataset", "model_key", "weights_path", "map50", "map5095", "params_m", "gflops"]
BENCHMARK_FIELDS = ["dataset", "model_key", "weights_path", "device", "source_image", "latency_ms", "fps"]
COMPARISON_FIELDS = [
    "dataset",
    "model_key",
    "weights_path",
    "map50",
    "map5095",
    "params_m",
    "gflops",
    "latency_ms",
    "fps",
    "delta_map50_vs_baseline",
    "meets_lt3pct_drop",
]


def bootstrap_ultralytics_path() -> None:
    ultralytics_path = str(ULTRALYTICS_ROOT)
    if ultralytics_path not in sys.path:
        sys.path.insert(0, ultralytics_path)


def ensure_workspace_dirs() -> None:
    for path in (CONFIGS_ROOT, DATASET_CONFIG_DIR, DATASETS_DIR, RESULTS_DIR, RUNS_DIR):
        path.mkdir(parents=True, exist_ok=True)
    figures_dir_for("voc")


def dataset_root(dataset_key: str) -> Path:
    if dataset_key != "voc":
        raise KeyError(f"Unknown dataset key: {dataset_key}")
    return DATASETS_DIR / "VOC_subset"


def dataset_yaml_path(dataset_key: str) -> Path:
    return DATASET_CONFIGS[dataset_key]


def model_yaml_path(model_key: str) -> Path:
    return MODEL_CONFIGS[model_key]


def results_dir_for(dataset_key: str) -> Path:
    path = RESULTS_DIR / dataset_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir_for(dataset_key: str) -> Path:
    figures_dir = results_dir_for(dataset_key) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def metrics_csv_path(dataset_key: str, model_key: str) -> Path:
    return results_dir_for(dataset_key) / f"{model_key}_metrics.csv"


def benchmark_csv_path(dataset_key: str) -> Path:
    return results_dir_for(dataset_key) / "latency_benchmark.csv"


def comparison_csv_path(dataset_key: str) -> Path:
    return results_dir_for(dataset_key) / "model_comparison.csv"


def experiment_dir(dataset_key: str, model_key: str) -> Path:
    return RUNS_DIR / dataset_key / model_key


def weights_dir(dataset_key: str, model_key: str) -> Path:
    return experiment_dir(dataset_key, model_key) / "weights"


def find_experiment_weights(dataset_key: str, model_key: str, override: Optional[str] = None) -> Path:
    if override:
        path = Path(override).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Weights file not found: {path}")
        return path

    candidates = [
        weights_dir(dataset_key, model_key) / "best.pt",
        weights_dir(dataset_key, model_key) / "last.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No weights found for dataset='{dataset_key}', model='{model_key}'. "
        f"Expected one of: {', '.join(str(path) for path in candidates)}"
    )


def resolve_device(requested: Optional[str] = None) -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to resolve the CUDA device.") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("This project is CUDA-only. No CUDA-capable GPU is available to PyTorch.")

    if requested is None:
        return "cuda:0"

    normalized = str(requested).strip().lower()
    if normalized not in {"cuda", "cuda:0", "0"}:
        raise ValueError("This project is CUDA-only. Use --device cuda:0.")
    return "cuda:0"


def iter_image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and not path.name.startswith("._") and path.suffix.lower() in IMAGE_SUFFIXES
    )


def pick_sample_image(dataset_key: str, split_order: Iterable[str] = ("val", "train")) -> Path:
    root = dataset_root(dataset_key) / "images"
    for split in split_order:
        candidates = iter_image_files(root / split)
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"No images found for dataset '{dataset_key}' under {root}")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def dump_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_comparison_rows(metric_rows: list[dict], benchmark_rows: list[dict]) -> list[dict]:
    metric_by_model = {row["model_key"]: row for row in metric_rows}
    benchmark_by_model = {row["model_key"]: row for row in benchmark_rows}
    baseline_row = metric_by_model.get("baseline")
    baseline_map50 = _to_float(baseline_row["map50"]) if baseline_row else None

    rows = []
    for model_key in MODEL_KEYS:
        metric_row = metric_by_model.get(model_key)
        if not metric_row:
            continue

        bench_row = benchmark_by_model.get(model_key, {})
        current_map50 = _to_float(metric_row["map50"])
        delta_pct = ""
        meets = ""
        if baseline_map50 and current_map50 is not None:
            delta_pct = round(((baseline_map50 - current_map50) / baseline_map50) * 100.0, 4)
            meets = str(delta_pct < 3.0)

        rows.append(
            {
                "dataset": metric_row["dataset"],
                "model_key": model_key,
                "weights_path": metric_row["weights_path"],
                "map50": metric_row["map50"],
                "map5095": metric_row["map5095"],
                "params_m": metric_row["params_m"],
                "gflops": metric_row["gflops"],
                "latency_ms": bench_row.get("latency_ms", ""),
                "fps": bench_row.get("fps", ""),
                "delta_map50_vs_baseline": delta_pct,
                "meets_lt3pct_drop": meets,
            }
        )
    return rows


def _to_float(value: object) -> float | None:
    if value in ("", None):
        return None
    return float(value)
