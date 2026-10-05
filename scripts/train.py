"""
Preview dataset configuration before Colab training.
Shows class counts, split sizes, and validates data.yaml.

Usage:
    python scripts/train.py              # Show dataset stats
    python scripts/train.py --dry-run    # Same output
    python scripts/train.py --class-counts
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "datasets" / "data.yaml"
DATASET_DIR = ROOT / "datasets" / "coco" / "images_by_class"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_split_dirs() -> tuple[Path, Path]:
    if not DATA_YAML.exists():
        return ROOT / "datasets" / "train", ROOT / "datasets" / "val"
    cfg = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8")) or {}
    base_path = Path(cfg.get("path", ROOT / "datasets"))
    train_rel = cfg.get("train", "train/images")
    val_rel = cfg.get("val", "val/images")
    train_dir = (base_path / train_rel).parent if "images" in train_rel else (base_path / train_rel)
    val_dir = (base_path / val_rel).parent if "images" in val_rel else (base_path / val_rel)
    return train_dir, val_dir


def header(text: str) -> None:
    print(f"\n{'=' * 60}", flush=True)
    print(f"  {text}", flush=True)
    print(f"{'=' * 60}", flush=True)


def load_class_names() -> list[str]:
    if not DATA_YAML.exists():
        return []
    cfg = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8")) or {}
    names = cfg.get("names", [])
    if isinstance(names, dict):
        return [str(names.get(i, names.get(str(i)))) for i in range(len(names))]
    return [str(name) for name in names]


def count_files(folder: Path, exts: set[str]) -> int:
    if not folder.exists():
        return 0
    try:
        return sum(1 for f in os.listdir(folder) if Path(f).suffix.lower() in exts)
    except Exception:
        return 0


def count_boxes(label_dir: Path, class_names: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not label_dir.exists():
        return counts
    try:
        with os.scandir(label_dir) as it:
            for entry in it:
                if entry.name.endswith(".txt"):
                    with open(entry.path, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            parts = line.split()
                            if parts:
                                try:
                                    idx = int(float(parts[0]))
                                    if 0 <= idx < len(class_names):
                                        counts[class_names[idx]] += 1
                                except ValueError:
                                    pass
    except Exception:
        pass
    return counts


def main() -> None:
    header("SmartNav 33-Class Dataset Overview")
    show_class_counts = "--class-counts" in sys.argv
    class_names = load_class_names()
    TRAIN_DIR, VAL_DIR = get_split_dirs()

    # Source previews
    if DATASET_DIR.exists():
        dirs = sorted(d for d in DATASET_DIR.iterdir() if d.is_dir())
        print(f"  Model classes: {len(class_names)}", flush=True)
        print(f"  Kept source preview folders: {len(dirs)}", flush=True)
        for d in dirs[:5]:
            print(f"    {d.name:<30} {count_files(d, IMAGE_EXTENSIONS):>4} preview images", flush=True)
        if len(dirs) > 5:
            print(f"    ... and {len(dirs) - 5} more", flush=True)
        print("  Note: preview folders are for reference; train/val below are the active training split.\n", flush=True)

    # Active Split
    train_img_dir = TRAIN_DIR / "images"
    train_lbl_dir = TRAIN_DIR / "labels"
    val_img_dir = VAL_DIR / "images"
    val_lbl_dir = VAL_DIR / "labels"

    if train_img_dir.exists():
        train_imgs = count_files(train_img_dir, IMAGE_EXTENSIONS)
        val_imgs = count_files(val_img_dir, IMAGE_EXTENSIONS)
        train_lbls = count_files(train_lbl_dir, {".txt"})
        val_lbls = count_files(val_lbl_dir, {".txt"})
        print(f"  Active Dataset: {TRAIN_DIR.parent}", flush=True)
        print(f"  Train images: {train_imgs}", flush=True)
        print(f"  Val images:   {val_imgs}", flush=True)
        print(f"  Train labels: {train_lbls}", flush=True)
        print(f"  Val labels:   {val_lbls}", flush=True)
    else:
        print(f"  Train directory not found: {train_img_dir}", flush=True)

    # data.yaml
    if DATA_YAML.exists():
        nc = len(class_names)
        print(f"\n  data.yaml: {DATA_YAML} ({nc} classes)", flush=True)
        if show_class_counts and train_lbl_dir.exists():
            print("  Counting per-class boxes...", flush=True)
            train_boxes = count_boxes(train_lbl_dir, class_names)
            val_boxes = count_boxes(val_lbl_dir, class_names)
            print("\n  Per-class box counts:")
            print(f"    {'Class Name':<20} {'Train Boxes':>12} {'Val Boxes':>10} {'Total Boxes':>12}")
            print(f"    {'-'*20} {'-'*12} {'-'*10} {'-'*12}")
            for name in class_names:
                t_cnt = train_boxes[name]
                v_cnt = val_boxes[name]
                print(f"    {name:<20} {t_cnt:>12} {v_cnt:>10} {t_cnt + v_cnt:>12}")

        print(f"\n  Ready for Colab Edge Training (YOLO26n @ 416x416)!", flush=True)
        print(f"  -> Zip and upload datasets/ to Google Drive root as datasets_colab.zip", flush=True)
    else:
        print(f"\n  No data.yaml found.", flush=True)


if __name__ == "__main__":
    main()
