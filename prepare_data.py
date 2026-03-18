from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

from project_config import (
    DATASET_KEYS,
    DEFAULT_FACE_MASK_DATASET_SLUG,
    FACE_MASK_NAMES,
    VOC_SMOKE_NAMES,
    DATASETS_DIR,
    dataset_root,
    dataset_yaml_path,
    dump_yaml,
    ensure_workspace_dirs,
    iter_image_files,
)


FACE_MASK_LABEL_ALIASES = {
    "withmask": 0,
    "mask": 0,
    "wearingmask": 0,
    "with_mask": 0,
    "withoutmask": 1,
    "nomask": 1,
    "no_mask": 1,
    "without_mask": 1,
    "unmasked": 1,
    "maskwearedincorrect": 1,
    "mask_weared_incorrect": 1,
    "incorrectmask": 1,
    "incorrect_mask": 1,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare datasets for the Face Mask experiment project.")
    parser.add_argument("--dataset", required=True, choices=DATASET_KEYS)
    parser.add_argument("--raw-dir", help="Existing raw Face Mask dataset directory. Defaults to datasets/raw/face_mask.")
    parser.add_argument(
        "--kaggle-dataset",
        default=DEFAULT_FACE_MASK_DATASET_SLUG,
        help="Kaggle dataset slug used when raw Face Mask data is not already present.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for train/val/test splitting.")
    parser.add_argument("--force", action="store_true", help="Rebuild the processed dataset even if it already exists.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_workspace_dirs()

    if args.dataset == "voc_smoke":
        prepare_voc_smoke()
        return 0

    raw_dir = Path(args.raw_dir).expanduser().resolve() if args.raw_dir else DATASETS_DIR / "raw" / "face_mask"
    prepare_face_mask(raw_dir=raw_dir, kaggle_dataset=args.kaggle_dataset, seed=args.seed, force=args.force)
    return 0


def prepare_voc_smoke() -> None:
    root = dataset_root("voc_smoke")
    image_train = root / "images" / "train"
    image_val = root / "images" / "val"
    label_train = root / "labels" / "train"
    label_val = root / "labels" / "val"

    required = [image_train, image_val, label_train, label_val]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"VOC smoke dataset is incomplete. Missing: {', '.join(missing)}")

    payload = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": VOC_SMOKE_NAMES,
    }
    dump_yaml(dataset_yaml_path("voc_smoke"), payload)

    print(
        "Prepared VOC smoke dataset config:",
        f"train_images={len(iter_image_files(image_train))}",
        f"val_images={len(iter_image_files(image_val))}",
        sep=" ",
    )


def prepare_face_mask(raw_dir: Path, kaggle_dataset: str, seed: int, force: bool) -> None:
    processed_root = dataset_root("face_mask")
    if force and processed_root.exists():
        shutil.rmtree(processed_root)

    if not raw_dir.exists():
        download_face_mask_dataset(kaggle_dataset, raw_dir)

    image_index = index_images(raw_dir)
    annotations = sorted(raw_dir.rglob("*.xml"))
    if not annotations:
        raise FileNotFoundError(
            f"No Pascal VOC XML annotations were found in {raw_dir}. "
            "Place the Kaggle dataset there or run with Kaggle credentials configured."
        )

    records = []
    for xml_path in annotations:
        image_path = image_index.get(xml_path.stem)
        if not image_path:
            continue
        boxes = parse_pascal_voc_annotation(xml_path)
        records.append((image_path, boxes))

    if not records:
        raise RuntimeError(f"No valid Face Mask records could be constructed from {raw_dir}.")

    random.Random(seed).shuffle(records)
    splits = split_records(records)

    for split, items in splits.items():
        image_out = processed_root / "images" / split
        label_out = processed_root / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        for image_path, boxes in items:
            target_image = image_out / image_path.name
            target_label = label_out / f"{image_path.stem}.txt"
            shutil.copy2(image_path, target_image)
            target_label.write_text("\n".join(format_yolo_box(box) for box in boxes), encoding="utf-8")

    payload = {
        "path": str(processed_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": FACE_MASK_NAMES,
    }
    dump_yaml(dataset_yaml_path("face_mask"), payload)

    print(
        "Prepared Face Mask dataset:",
        f"train={len(splits['train'])}",
        f"val={len(splits['val'])}",
        f"test={len(splits['test'])}",
        f"processed_root={processed_root}",
        sep=" ",
    )


def download_face_mask_dataset(kaggle_dataset: str, raw_dir: Path) -> None:
    kaggle_binary = shutil.which("kaggle")
    if not kaggle_binary:
        raise FileNotFoundError(
            "The Kaggle CLI was not found. Install it and configure credentials, "
            "or provide --raw-dir pointing to an already-downloaded dataset."
        )

    raw_dir.mkdir(parents=True, exist_ok=True)
    command = [kaggle_binary, "datasets", "download", "-d", kaggle_dataset, "-p", str(raw_dir), "--force"]
    subprocess.run(command, check=True)
    extract_archives(raw_dir)


def extract_archives(root: Path) -> None:
    for archive_path in root.glob("*.zip"):
        extract_dir = root / archive_path.stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(extract_dir)


def index_images(root: Path) -> dict[str, Path]:
    image_map = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            image_map.setdefault(path.stem, path)
    return image_map


def parse_pascal_voc_annotation(xml_path: Path) -> list[dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Annotation is missing <size>: {xml_path}")

    width = float(size.findtext("width", default="0"))
    height = float(size.findtext("height", default="0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Annotation has invalid image size: {xml_path}")

    boxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name", default="").strip()
        class_id = normalize_face_mask_label(name)
        if class_id is None:
            continue
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        xmin = float(bndbox.findtext("xmin", default="0"))
        ymin = float(bndbox.findtext("ymin", default="0"))
        xmax = float(bndbox.findtext("xmax", default="0"))
        ymax = float(bndbox.findtext("ymax", default="0"))
        x_center = ((xmin + xmax) / 2.0) / width
        y_center = ((ymin + ymax) / 2.0) / height
        box_width = max(xmax - xmin, 0.0) / width
        box_height = max(ymax - ymin, 0.0) / height
        boxes.append(
            {
                "class_id": class_id,
                "x_center": clamp01(x_center),
                "y_center": clamp01(y_center),
                "width": clamp01(box_width),
                "height": clamp01(box_height),
            }
        )
    return boxes


def normalize_face_mask_label(name: str) -> int | None:
    key = "".join(ch for ch in name.lower() if ch.isalnum() or ch == "_")
    return FACE_MASK_LABEL_ALIASES.get(key)


def split_records(records: list[tuple[Path, list[dict]]]) -> dict[str, list[tuple[Path, list[dict]]]]:
    total = len(records)
    train_end = max(1, int(total * 0.8))
    val_end = max(train_end + 1, int(total * 0.9))
    return {
        "train": records[:train_end],
        "val": records[train_end:val_end],
        "test": records[val_end:],
    }


def format_yolo_box(box: dict) -> str:
    return (
        f"{box['class_id']} "
        f"{box['x_center']:.6f} {box['y_center']:.6f} "
        f"{box['width']:.6f} {box['height']:.6f}"
    )


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


if __name__ == "__main__":
    raise SystemExit(main())
