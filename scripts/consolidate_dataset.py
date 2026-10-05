"""
High-performance consolidation, cleaning, normalization, and restructuring of the SmartNav dataset to 33 classes.
Builds the final dataset at: datasets/smartnav_v1_33cls_20260920/
"""

import os
import sys
import shutil
import hashlib
import json
import random
from pathlib import Path
from collections import Counter
import zipfile
import yaml

ROOT = Path(__file__).resolve().parent.parent

# Output directory with today's date
DATE_STR = "20260920"
TARGET_DATASET_DIR = ROOT / "datasets" / f"smartnav_v1_33cls_{DATE_STR}"
TRAIN_IMG_DIR = TARGET_DATASET_DIR / "train" / "images"
TRAIN_LBL_DIR = TARGET_DATASET_DIR / "train" / "labels"
VAL_IMG_DIR = TARGET_DATASET_DIR / "val" / "images"
VAL_LBL_DIR = TARGET_DATASET_DIR / "val" / "labels"

CLASS_NAMES = [
    "person",         # 0
    "car",            # 1
    "bus",            # 2
    "truck",          # 3
    "motorcycle",     # 4
    "bicycle",        # 5
    "rickshaw",       # 6
    "van",            # 7
    "ambulance",      # 8
    "police_car",     # 9
    "pothole",        # 10
    "manhole",        # 11
    "speed_breaker",  # 12
    "cone",           # 13
    "stop_sign",      # 14
    "traffic_light",  # 15
    "crosswalk",      # 16
    "pole",           # 17
    "dog",            # 18
    "cow",            # 19
    "door",           # 20
    "stairs",         # 21
    "chair",          # 22
    "table",          # 23
    "bed",            # 24
    "couch",          # 25
    "white_cane",     # 26
    "wheelchair",     # 27
    "handrail",       # 28
    "backpack",       # 29
    "handbag",        # 30
    "shelf",          # 31
    "sink",           # 32
]

CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

OLD_59_TO_NEW_33 = {
    0: CLASS_TO_ID["person"],
    2: CLASS_TO_ID["car"],
    4: CLASS_TO_ID["bus"],
    5: CLASS_TO_ID["truck"],
    3: CLASS_TO_ID["motorcycle"],
    7: CLASS_TO_ID["bicycle"],
    8: CLASS_TO_ID["rickshaw"],
    6: CLASS_TO_ID["van"],
    10: CLASS_TO_ID["ambulance"],
    9: CLASS_TO_ID["police_car"],
    20: CLASS_TO_ID["pothole"],
    21: CLASS_TO_ID["manhole"],
    22: CLASS_TO_ID["speed_breaker"],
    18: CLASS_TO_ID["cone"],
    16: CLASS_TO_ID["stop_sign"],
    15: CLASS_TO_ID["traffic_light"],
    26: CLASS_TO_ID["crosswalk"],
    23: CLASS_TO_ID["pole"],
    11: CLASS_TO_ID["dog"],
    14: CLASS_TO_ID["cow"],
    27: CLASS_TO_ID["door"],
    24: CLASS_TO_ID["stairs"],
    29: CLASS_TO_ID["chair"],
    31: CLASS_TO_ID["table"],
    30: CLASS_TO_ID["bed"],
    57: CLASS_TO_ID["white_cane"],
    56: CLASS_TO_ID["wheelchair"],
    25: CLASS_TO_ID["handrail"],
    52: CLASS_TO_ID["backpack"],
    35: CLASS_TO_ID["shelf"],
    33: CLASS_TO_ID["sink"],
}

COCO_80_TO_NEW_33 = {
    0: CLASS_TO_ID["person"],
    1: CLASS_TO_ID["bicycle"],
    2: CLASS_TO_ID["car"],
    3: CLASS_TO_ID["motorcycle"],
    5: CLASS_TO_ID["bus"],
    7: CLASS_TO_ID["truck"],
    9: CLASS_TO_ID["traffic_light"],
    11: CLASS_TO_ID["stop_sign"],
    16: CLASS_TO_ID["dog"],
    19: CLASS_TO_ID["cow"],
    24: CLASS_TO_ID["backpack"],
    26: CLASS_TO_ID["handbag"],
    56: CLASS_TO_ID["chair"],
    57: CLASS_TO_ID["couch"],
    59: CLASS_TO_ID["bed"],
    60: CLASS_TO_ID["table"],
}

def fast_copy(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def main():
    print("=" * 70, flush=True)
    print("Starting SmartNav 33-Class Dataset Consolidation (Lightning Mode)", flush=True)
    print(f"Target Directory: {TARGET_DATASET_DIR}", flush=True)
    print("=" * 70, flush=True)

    for d in [TRAIN_IMG_DIR, TRAIN_LBL_DIR, VAL_IMG_DIR, VAL_LBL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    seen_keys = set()
    collected_samples = []

    # 1. Existing train/val from E:
    print("\n[1/4] Processing existing train and val splits from E:...", flush=True)
    existing_splits = [
        ("train", ROOT / "datasets" / "train" / "images", ROOT / "datasets" / "train" / "labels"),
        ("val", ROOT / "datasets" / "val" / "images", ROOT / "datasets" / "val" / "labels"),
    ]

    for split_name, img_dir, lbl_dir in existing_splits:
        if not img_dir.exists() or not lbl_dir.exists():
            continue
        print(f"  Indexing images in {split_name}...", flush=True)
        img_index = {}
        with os.scandir(img_dir) as it:
            for entry in it:
                if entry.is_file():
                    img_index[Path(entry.name).stem] = Path(entry.path)
        print(f"  Indexed {len(img_index)} images. Parsing label files...", flush=True)

        count_in_split = 0
        with os.scandir(lbl_dir) as it:
            for entry in it:
                if not entry.name.endswith(".txt"):
                    continue
                stem = entry.name[:-4]

                # Fast check: does image exist in RAM index?
                img_path = img_index.get(stem)
                if not img_path:
                    continue

                valid_boxes = []
                sample_classes = set()
                with open(entry.path, "r", encoding="utf-8", errors="ignore") as fp:
                    for line in fp:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                old_cid = int(float(parts[0]))
                                if old_cid in OLD_59_TO_NEW_33:
                                    new_cid = OLD_59_TO_NEW_33[old_cid]
                                    coords = [max(0.0, min(1.0, float(x))) for x in parts[1:5]]
                                    valid_boxes.append((new_cid, *coords))
                                    sample_classes.add(new_cid)
                            except (ValueError, IndexError):
                                pass

                if valid_boxes:
                    key = img_path.name
                    if key not in seen_keys:
                        seen_keys.add(key)
                        collected_samples.append({
                            "image_path": img_path,
                            "boxes": valid_boxes,
                            "source": f"existing_{split_name}",
                            "classes": sample_classes,
                        })
                        count_in_split += 1

        print(f"  -> Kept {count_in_split} valid images from {split_name}.", flush=True)

    print(f"Total from existing splits: {len(collected_samples)} images.", flush=True)

    # 2. COCO val2017 Ground Truth (couch, handbag, chair, backpack, table, cow)
    print("\n[2/4] Sourcing COCO val2017 images & labels for couch, handbag, chair, backpack...", flush=True)
    coco_labels_dir = ROOT / "cache" / "coco_val2017_labels"
    coco_source_folders = [
        ("couch_E", ROOT / "datasets" / "coco" / "images_by_class" / "couch"),
        ("handbag_E", ROOT / "datasets" / "coco" / "images_by_class" / "handbag"),
        ("chair_D", Path("D:/datasets/coco/images_by_class/chair")),
        ("backpack_D", Path("D:/datasets/coco/images_by_class/backpack")),
        ("couch_D", Path("D:/datasets/coco/images_by_class/couch")),
        ("handbag_D", Path("D:/datasets/coco/images_by_class/handbag")),
        ("table_D", Path("D:/datasets/coco/images_by_class/dining table")),
        ("cow_D", Path("D:/datasets/coco/images_by_class/cow")),
    ]

    # Pre-index COCO label contents in memory for ultra-fast lookup
    print("  Indexing COCO label files in RAM...", flush=True)
    coco_label_cache = {}
    if coco_labels_dir.exists():
        with os.scandir(coco_labels_dir) as it:
            for entry in it:
                if entry.name.endswith(".txt"):
                    stem = entry.name[:-4]
                    with open(entry.path, "r", encoding="utf-8", errors="ignore") as fp:
                        coco_label_cache[stem] = fp.read()

    coco_added = 0
    for source_tag, folder in coco_source_folders:
        if not folder.exists():
            continue
        folder_added = 0
        with os.scandir(folder) as it:
            for entry in it:
                if not entry.name.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue
                raw_stem = Path(entry.name).stem
                coco_stem = raw_stem[len("train_"):] if raw_stem.startswith("train_") else raw_stem
                lbl_content = coco_label_cache.get(coco_stem)
                if not lbl_content:
                    continue

                valid_boxes = []
                sample_classes = set()
                for line in lbl_content.splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            coco_cid = int(float(parts[0]))
                            if coco_cid in COCO_80_TO_NEW_33:
                                new_cid = COCO_80_TO_NEW_33[coco_cid]
                                coords = [max(0.0, min(1.0, float(x))) for x in parts[1:5]]
                                valid_boxes.append((new_cid, *coords))
                                sample_classes.add(new_cid)
                        except (ValueError, IndexError):
                            pass

                if valid_boxes:
                    key = f"{source_tag}_{entry.name}"
                    if key not in seen_keys:
                        seen_keys.add(key)
                        collected_samples.append({
                            "image_path": Path(entry.path),
                            "boxes": valid_boxes,
                            "source": f"coco_{source_tag}",
                            "classes": sample_classes,
                        })
                        coco_added += 1
                        folder_added += 1
        print(f"  -> Added {folder_added} images from {source_tag} ({folder}).", flush=True)

    print(f"Total COCO images added: {coco_added}", flush=True)

    # 3. Supplementary Roboflow Zips (manhole, sign1)
    print("\n[3/4] Checking supplementary Roboflow zips for manhole and rare classes...", flush=True)
    supplementary_zips = [
        ("roboflow_manhole", ROOT / "roboflow_zips" / "manhole.zip", {1: CLASS_TO_ID["manhole"]}),
        ("roboflow_sign1", ROOT / "roboflow_zips" / "sign1.zip", {
            12: CLASS_TO_ID["speed_breaker"],
            14: CLASS_TO_ID["stop_sign"],
            15: CLASS_TO_ID["traffic_light"],
        }),
    ]

    extract_tmp = ROOT / "cache" / "tmp_zip_extract"
    extract_tmp.mkdir(parents=True, exist_ok=True)
    zip_added = 0

    for source_tag, zpath, class_map in supplementary_zips:
        if not zpath.exists():
            continue
        try:
            with zipfile.ZipFile(zpath, "r") as z:
                all_files = set(z.namelist())
                img_files = [f for f in all_files if f.lower().endswith((".jpg", ".png", ".jpeg"))]
                for img_rel in img_files:
                    p = Path(img_rel)
                    stem = p.stem
                    label_candidates = [
                        (p.parent.parent / "labels" / f"{stem}.txt").as_posix(),
                        (p.parent / f"{stem}.txt").as_posix(),
                    ]
                    lbl_member = next((lc for lc in label_candidates if lc in all_files), None)
                    if not lbl_member:
                        continue

                    lbl_content = z.read(lbl_member).decode("utf-8", errors="ignore")
                    valid_boxes = []
                    sample_classes = set()
                    for line in lbl_content.splitlines():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                local_cid = int(float(parts[0]))
                                if local_cid in class_map:
                                    new_cid = class_map[local_cid]
                                    coords = [max(0.0, min(1.0, float(x))) for x in parts[1:5]]
                                    valid_boxes.append((new_cid, *coords))
                                    sample_classes.add(new_cid)
                            except (ValueError, IndexError):
                                pass

                    if valid_boxes:
                        key = f"{source_tag}_{p.name}"
                        if key not in seen_keys:
                            seen_keys.add(key)
                            extracted_img = extract_tmp / f"{source_tag}_{p.name}"
                            with open(extracted_img, "wb") as fp:
                                fp.write(z.read(img_rel))
                            collected_samples.append({
                                "image_path": extracted_img,
                                "boxes": valid_boxes,
                                "source": source_tag,
                                "classes": sample_classes,
                            })
                            zip_added += 1
        except Exception as e:
            print(f"  Warning: error reading {zpath.name}: {e}", flush=True)

    print(f"  -> Added {zip_added} supplementary images from Roboflow zips.", flush=True)
    print(f"\nTotal Consolidated Dataset Images: {len(collected_samples)}", flush=True)

    # 4. Stratified 85 / 15 Train/Val Split
    print("\n[4/4] Generating stratified 85/15 train/val split...", flush=True)
    random.seed(42)
    random.shuffle(collected_samples)

    total_class_boxes = Counter()
    total_class_images = Counter()
    for s in collected_samples:
        for cid in s["classes"]:
            total_class_images[cid] += 1
        for box in s["boxes"]:
            total_class_boxes[box[0]] += 1

    val_samples = []
    train_samples = []
    val_class_counts = Counter()
    train_class_counts = Counter()

    target_val_ratio = 0.15

    # Prioritize rare classes so they always get val representation
    collected_samples.sort(key=lambda s: min(total_class_images[cid] for cid in s["classes"]))

    for sample in collected_samples:
        needs_val = False
        for cid in sample["classes"]:
            current_val = val_class_counts[cid]
            total_imgs = total_class_images[cid]
            target_val = max(1, int(round(total_imgs * target_val_ratio)))
            if current_val < target_val:
                needs_val = True
                break

        if needs_val and len(val_samples) < len(collected_samples) * 0.18:
            val_samples.append(sample)
            for cid in sample["classes"]:
                val_class_counts[cid] += 1
        else:
            train_samples.append(sample)
            for cid in sample["classes"]:
                train_class_counts[cid] += 1

    print(f"  -> Train Images: {len(train_samples)}", flush=True)
    print(f"  -> Val Images:   {len(val_samples)} ({len(val_samples)/len(collected_samples)*100:.1f}%)", flush=True)

    source_manifest = []

    def write_split(samples, img_dest_dir, lbl_dest_dir, split_name):
        for idx, s in enumerate(samples):
            fname = f"{split_name}_{idx:06d}{s['image_path'].suffix.lower()}"
            lbl_name = f"{split_name}_{idx:06d}.txt"
            dst_img = img_dest_dir / fname
            dst_lbl = lbl_dest_dir / lbl_name

            fast_copy(s["image_path"], dst_img)

            with open(dst_lbl, "w", encoding="utf-8") as fp:
                for b in s["boxes"]:
                    fp.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

            source_manifest.append({
                "file": fname,
                "split": split_name,
                "original_path": str(s["image_path"]),
                "source": s["source"],
                "box_count": len(s["boxes"]),
                "classes": [CLASS_NAMES[cid] for cid in s["classes"]],
            })

    print("  Writing train files with fast-link...", flush=True)
    write_split(train_samples, TRAIN_IMG_DIR, TRAIN_LBL_DIR, "train")
    print("  Writing val files with fast-link...", flush=True)
    write_split(val_samples, VAL_IMG_DIR, VAL_LBL_DIR, "val")

    # 5. Metadata Files Generation
    print("  Writing data.yaml and classes.yaml...", flush=True)
    data_yaml_content = {
        "path": str(TARGET_DATASET_DIR).replace("\\", "/"),
        "train": "train/images",
        "val": "val/images",
        "nc": len(CLASS_NAMES),
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    with open(TARGET_DATASET_DIR / "data.yaml", "w", encoding="utf-8") as fp:
        yaml.dump(data_yaml_content, fp, sort_keys=False)

    with open(TARGET_DATASET_DIR / "classes.yaml", "w", encoding="utf-8") as fp:
        yaml.dump({i: name for i, name in enumerate(CLASS_NAMES)}, fp, sort_keys=False)

    # Sync project-level datasets/data.yaml and datasets/classes.yaml
    with open(ROOT / "datasets" / "data.yaml", "w", encoding="utf-8") as fp:
        yaml.dump(data_yaml_content, fp, sort_keys=False)

    with open(ROOT / "datasets" / "classes.yaml", "w", encoding="utf-8") as fp:
        yaml.dump({i: name for i, name in enumerate(CLASS_NAMES)}, fp, sort_keys=False)

    with open(TARGET_DATASET_DIR / "source_manifest.json", "w", encoding="utf-8") as fp:
        json.dump(source_manifest, fp, indent=2)

    print("\n" + "=" * 70, flush=True)
    print("CONSOLIDATION COMPLETED SUCCESSFULLY", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Class ID':8s} | {'Class Name':18s} | {'Train Imgs':10s} | {'Val Imgs':10s} | {'Total Boxes':12s}", flush=True)
    print("-" * 70, flush=True)
    for cid, cname in enumerate(CLASS_NAMES):
        t_img = train_class_counts[cid]
        v_img = val_class_counts[cid]
        b_cnt = total_class_boxes[cid]
        print(f"{cid:8d} | {cname:18s} | {t_img:10d} | {v_img:10d} | {b_cnt:12d}", flush=True)

    print(f"\nFinal Consolidated Dataset Location: {TARGET_DATASET_DIR}", flush=True)

if __name__ == "__main__":
    main()
