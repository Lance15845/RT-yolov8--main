from __future__ import annotations

import argparse
from pathlib import Path

from project_config import (
    DATASET_KEYS,
    VOC_NAMES,
    dataset_root,
    dataset_yaml_path,
    dump_yaml,
    ensure_workspace_dirs,
    iter_image_files,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the VOC dataset and generate the formal dataset config.")
    parser.add_argument("--dataset", required=True, choices=DATASET_KEYS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_workspace_dirs()
    prepare_voc()
    return 0


def prepare_voc() -> None:
    root = dataset_root("voc")
    split_counts = {split: validate_split(root, split) for split in ("train", "val")}
    payload = {
        "path": "datasets/VOC_subset",
        "train": "images/train",
        "val": "images/val",
        "names": VOC_NAMES,
    }
    dump_yaml(dataset_yaml_path("voc"), payload)

    print(
        "Prepared VOC dataset config:",
        f"train_images={split_counts['train']}",
        f"val_images={split_counts['val']}",
        f"config={dataset_yaml_path('voc')}",
        sep=" ",
    )


def validate_split(root: Path, split: str) -> int:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    required = [image_dir, label_dir]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"VOC dataset is incomplete for split '{split}'. Missing: {', '.join(missing)}")

    images = iter_image_files(image_dir)
    if not images:
        raise FileNotFoundError(f"No images were found in {image_dir}")

    missing_labels = [str(label_dir / f"{image_path.stem}.txt") for image_path in images if not (label_dir / f"{image_path.stem}.txt").exists()]
    if missing_labels:
        preview = ", ".join(missing_labels[:10])
        suffix = " ..." if len(missing_labels) > 10 else ""
        raise FileNotFoundError(f"Missing YOLO labels for split '{split}': {preview}{suffix}")

    return len(images)


if __name__ == "__main__":
    raise SystemExit(main())
