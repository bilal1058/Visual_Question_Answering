"""
All-in-one dataset setup for the Visual QA detector.

Downloads:
  - Retained COCO val2017 classes, precise labels
  - Retained Open Images expansion classes, precise labels
  - Optional Roboflow import/merge for additional classes

Usage:
    python scripts/setup.py                    # Retained COCO + OI + roboflow_zips merge if present
    python scripts/setup.py --coco-only       # Only retained COCO classes
    python scripts/setup.py --oi-only         # Only retained OI expansion
    python scripts/setup.py --labels          # Regenerate YOLO labels + splits from existing images
    python scripts/setup.py --fix-splits      # Rebalance existing train/val so every class gets val samples
    python scripts/setup.py --dry-run                      # Preview only
    python scripts/setup.py --roboflow ZIP                 # Import single Roboflow YOLOv8 export
    python scripts/setup.py --roboflow-dir DIR             # Merge all zips from a directory
    python scripts/setup.py --force-roboflow-merge         # Re-merge zips already recorded in the manifest
    python scripts/setup.py --rebuild-previews              # Rebuild images_by_class from train/val data
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import requests
from tqdm import tqdm
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.class_taxonomy import (
    FINAL_ASSISTIVE_CLASSES,
    CLASS_ALIASES,
    canonical_label,
    class_key as taxonomy_class_key,
)

COCO_ROOT = ROOT / "datasets" / "coco"
OI_OUTPUT_DIR = COCO_ROOT / "images_by_class"
LABELS_CACHE = ROOT / "cache" / "labels"
TRAIN_DIR = ROOT / "datasets" / "train"
VAL_DIR = ROOT / "datasets" / "val"
DATA_YAML = ROOT / "datasets" / "data.yaml"
ROBOFLOW_CACHE = ROOT / "cache" / "roboflow"
ROBOFLOW_MERGE_MANIFEST = ROBOFLOW_CACHE / "merged_zips.json"
DEFAULT_ROBOFLOW_DIR = ROOT / "roboflow_zips"
ROBOFLOW_BUNDLE_CANDIDATES = [
    ROOT / "datasets" / "roboflow_export.zip",
    ROOT / "roboflow_export.zip",
]
DATASET_SOURCE_CONFIGS = [
    ROOT / "config" / "dataset_sources.json",
    ROOT / "dataset_sources.json",
]
MAX_OI_PER_CLASS = 80
ROBOFLOW_SAMPLE_LIMIT_PER_CLASS: int | None = None
TRAIN_SPLIT = 0.9
VAL_TARGET_IMAGES_PER_CLASS = 25
MIN_TRAIN_IMAGES_PER_CLASS = 25
VAL_TARGET_BOXES_PER_CLASS = 25
MIN_TRAIN_BOXES_PER_CLASS = 25

RETAINED_CLASS_FOLDERS = list(FINAL_ASSISTIVE_CLASSES)
COCO_SOURCE_CLASS_ALIASES = ["cell phone", "dining table"]
ROBOFLOW_PREVIEW_PREFIX = "rf_"

# Maps Roboflow zip sub-class names to clean target class names.
# "__all__" maps every local class in that zip to one target.
# Explicit entries override specific local names.
# "None" skips that local class.
ROBOFLOW_CLASS_MAP: dict[str, dict[str, str | None]] = {
    "pothole_and_speed_breaker": {
        "pothole": "pothole",
        "speed breaker": "speed breaker",
        "speed bump": "speed breaker",
        "speedbreaker": "speed breaker"
    },
    "bike": {
        "bike-and-person": "motorcycle",
        "bike": "motorcycle",
        "motorcycle": "motorcycle",
        "motorbike": "motorcycle",
        "bicycle": "bicycle",
        "cycle": "bicycle"
    },
    "dogs": {
        "__all__": "dog"
    },
    "cow": {
        "__all__": "cow"
    },
    "human_detection": {
        "__all__": "person"
    },
    "pedestrian_detection": {
        "__all__": "person"
    },
    "person_detection": {
        "__all__": "person"
    },
    "people_detection": {
        "__all__": "person"
    },
    "fridge": {
        "fridge": "refrigerator",
        "refrigerator": "refrigerator"
    },
    "window1": {
        "window": "Window",
        "windows": "Window"
    },
    "door1": {
        "door": "Door",
        "doors": "Door"
    },
    "door2": {
        "door": "Door",
        "doors": "Door"
    },
    "bus": {
        "bus": "bus",
        "buses": "bus"
    },
    "wheelchair": {
        "__all__": "Wheelchair"
    },
    "bed": {
        "bed": "bed",
        "beds": "bed"
    },
    "table": {
        "table": "table",
        "tables": "table"
    },
    "truck": {
        "__all__": "truck"
    },
    "air_conditioner": {"__all__": "air conditioner"},
    "ambulance": {"__all__": "Ambulance"},
    "apple": {"__all__": "apple"},
    "backpack": {"__all__": "backpack"},
    "bench": {"__all__": None},
    "bicycle": {"__all__": "bicycle"},
    "book": {"__all__": "book"},
    "bottle": {"__all__": "bottle"},
    "bowl": {"__all__": None},
    "building": {"__all__": "building"},
    "car": {"__all__": "car"},
    "chair": {"__all__": "chair"},
    "charger": {"Charger": "charger", "__default__": None},
    "cell_phone": {"__all__": "mobile"},
    "clothes": {"__all__": None},
    "cone": {"__all__": "cone"},
    "construction_barrier": {"__all__": None},
    "crosswalk": {"__all__": "crosswalk"},
    "curtain": {"__all__": None},
    "door": {"__all__": "Door"},
    "dress": {"__all__": None},
    "fan": {"__all__": "fan"},
    "gas_stove": {"__all__": "Gas stove"},
    "glasses": {"__all__": "glasses"},
    "goat": {"__all__": "goat"},
    "handrail": {"__all__": "handrail"},
    "handrail_yolo": {"__all__": "handrail"},
    "knife": {"__all__": "knife"},
    "laptop": {"__all__": "laptop"},
    "license_plate": {"__all__": None},
    "manhole": {"Manhole cover detection - v1 2023-12-26 9-14pm": "manhole", "__default__": None},
    "medicine": {"__all__": "medicine"},
    "microwave": {"__all__": None},
    "mobile": {"__all__": "mobile"},
    "mobile_phone": {"__all__": "mobile"},
    "mug": {"__all__": "mug"},
    "pillow": {"__all__": "Pillow"},
    "pole": {"__all__": "pole"},
    "police_car": {"__all__": "police car"},
    "pothole": {"__all__": "pothole"},
    "rickshaw": {"__all__": "rickshaw"},
    "remote": {"__all__": "remote"},
    "remote_control": {"__all__": "remote"},
    "router": {"0": "router", "__default__": None},
    "scissors": {"__all__": None},
    "sheep": {"__all__": "sheep"},
    "shelf": {"__all__": "Shelf"},
    "shalwar_kameez": {"__all__": None},
    "shalwar_qameez": {"__all__": None},
    "shoes": {"__all__": "shoes"},
    "sign": {"__all__": "sign"},
    "road_signs": {"__all__": "sign"},
    "road_signs_vanga": {"__all__": "sign"},
    "sink": {"__all__": "sink"},
    "speed_breaker": {"__all__": "speed breaker"},
    "spoon": {"__all__": None},
    "stairs": {"__all__": "Stairs"},
    "suitcase": {"Suitcase": "suitcase", "__default__": None},
    "switch": {"__all__": "switch"},
    "table": {"__all__": "table"},
    "tie": {"__all__": "tie"},
    "toaster": {"__all__": "toaster"},
    "traffic_light": {"__all__": "traffic light"},
    "trash_can": {"__all__": "trash can"},
    "tree": {"__all__": "tree"},
    "umbrella": {"__all__": "umbrella"},
    "white_cane": {"__all__": "white cane"},
    "window": {"__all__": "Window"},
    "worker": {"__all__": "worker"},
    "yolov8_crosswalk_detection": {"__all__": "crosswalk"},
    "crosswalk_detection": {"__all__": "crosswalk"},
    # Backward-compatible keys for older, unrenamed local exports.
    "gasstove": {"__all__": "Gas stove"},
    "licenseplate": {"__all__": "license plate"},
    "policecar": {"__all__": "police car"},
    "speedbreaker": {"__all__": "speed breaker"},
    "trashcan": {"__all__": "trash can"},
}

# Safer maps for large, high-signal Roboflow datasets. These let you keep your
# final assistive setup while importing only useful overlapping classes.
STREET_RECOMMENDED_CLASS_MAP: dict[str, str | None] = {
    "__default__": None,
    "bike": "bicycle",
    "bicycle": "bicycle",
    "bus": "bus",
    "car": "car",
    "crosswalk": "crosswalk",
    "cycle": "bicycle",
    "dog": "dog",
    "fire hydrant": "fire hydrant",
    "fire_hydrant": "fire hydrant",
    "motor": "motorcycle",
    "motorbike": "motorcycle",
    "motorcycle": "motorcycle",
    "parking meter": "parking meter",
    "person": "person",
    "pedestrian": "person",
    "rider": "person",
    "road sign": "sign",
    "road barrier": None,
    "handrail": "handrail",
    "railing": "handrail",
    "sign": "sign",
    "stairs": "Stairs",
    "stop sign": "stop sign",
    "stop_sign": "stop sign",
    "tl_green": "traffic light",
    "tl_none": "traffic light",
    "tl_red": "traffic light",
    "tl_yellow": "traffic light",
    "traffic light": "traffic light",
    "traffic-light": "traffic light",
    "traffic signal": "traffic light",
    "traffic sign": "sign",
    "traffic-sign": "sign",
    "zebra crossing": "crosswalk",
    "train": "train",
    "truck": "truck",
    "van": "Van",
}

LOGISTICS_RECOMMENDED_CLASS_MAP: dict[str, str | None] = {
    "__default__": None,
    "car": "car",
    "truck": "truck",
    "van": "Van",
    "traffic light": "traffic light",
    "person": "person",
    "road sign": "sign",
    "license plate": "license plate",
    "traffic cone": "cone",
}

FOOD_RECOMMENDED_CLASS_MAP: dict[str, str | None] = {
    "__default__": None,
    "apple": "apple",
    "banana": "banana",
    "broccoli": "broccoli",
    "carrot": "carrot",
    "orange": "orange",
    "pizza": "pizza",
    "sandwich": "sandwich",
}

HOUSEHOLD_RECOMMENDED_CLASS_MAP: dict[str, str | None] = {
    "__default__": None,
    "air conditioner": "air conditioner",
    "air-conditioner": "air conditioner",
    "bed": "bed",
    "book": "book",
    "bottle": "bottle",
    "bowl": "bowl",
    "cell phone": "mobile",
    "cellphone": "mobile",
    "chair": "chair",
    "charger": "charger",
    "couch": "couch",
    "cup": "cup",
    "curtain": "Curtain",
    "dining table": "table",
    "door": "Door",
    "dress": None,
    "fan": "fan",
    "fridge": "refrigerator",
    "headphones": "Headphones",
    "kettle": "Kettle",
    "keyboard": "keyboard",
    "laptop": "laptop",
    "microwave": "microwave",
    "mirror": "Mirror",
    "mobile": "mobile",
    "mobile phone": "mobile",
    "mouse": "mouse",
    "oven": "oven",
    "phone": "mobile",
    "pillow": "Pillow",
    "plate": "Plate",
    "printer": "Printer",
    "refrigerator": "refrigerator",
    "remote": "remote",
    "remote control": "remote",
    "router": "router",
    "shelf": "Shelf",
    "shalwar kameez": None,
    "shalwar qameez": None,
    "shalwar-kameez": None,
    "shalwar-qameez": None,
    "sink": "sink",
    "sofa": "couch",
    "table": "table",
    "toaster": "toaster",
    "toilet": "toilet",
    "toothbrush": "toothbrush",
    "tv": "tv",
    "tvmonitor": "tv",
    "tv remote": "remote",
    "washing machine": "Washing machine",
    "woman dress": None,
    "women dress": None,
    "women's dress": None,
    "womens dress": None,
    "wheel chair": "Wheelchair",
    "window": "Window",
}

SPORT_RECOMMENDED_CLASS_MAP: dict[str, str | None] = {
    "__default__": None,
    "baseball bat": "baseball bat",
    "baseball-bat": "baseball bat",
    "baseball glove": "baseball glove",
    "baseball-glove": "baseball glove",
    "frisbee": "frisbee",
    "skateboard": "skateboard",
    "skis": "skis",
    "snowboard": "snowboard",
    "surfboard": "surfboard",
    "tennis racket": "tennis racket",
}

GENERAL_RECOMMENDED_CLASS_MAP: dict[str, str | None] = {
    "__default__": None,
    **STREET_RECOMMENDED_CLASS_MAP,
    **LOGISTICS_RECOMMENDED_CLASS_MAP,
    **FOOD_RECOMMENDED_CLASS_MAP,
    **HOUSEHOLD_RECOMMENDED_CLASS_MAP,
    **SPORT_RECOMMENDED_CLASS_MAP,
    "aeroplane": "airplane",
    "airplane": "airplane",
    "ambulance": "Ambulance",
    "backpack": "backpack",
    "bear": "bear",
    "bench": "bench",
    "boat": "boat",
    "clock": "clock",
    "cow": "cow",
    "elephant": "elephant",
    "giraffe": "giraffe",
    "handbag": "handbag",
    "horse": "horse",
    "sheep": "sheep",
    "suitcase": "suitcase",
    "teddy bear": "teddy bear",
    "tie": "tie",
    "truck": "truck",
    "umbrella": "umbrella",
    "wheel chair": "Wheelchair",
    "zebra": "zebra",
}

ROBOFLOW_CLASS_MAP.update({
    # Broad benchmark datasets: import only overlapping useful classes.
    "fsod": GENERAL_RECOMMENDED_CLASS_MAP,
    "fsod_few_shot_object_detection_dataset": GENERAL_RECOMMENDED_CLASS_MAP,
    "fsod_few_shot_object_detection_dataset_7tiwq": GENERAL_RECOMMENDED_CLASS_MAP,
    "objects_365": GENERAL_RECOMMENDED_CLASS_MAP,
    "objects365": GENERAL_RECOMMENDED_CLASS_MAP,

    # Street / driving datasets, useful for night scenes and navigation VQA.
    "bdd100k": STREET_RECOMMENDED_CLASS_MAP,
    "bdd100k_finetune": STREET_RECOMMENDED_CLASS_MAP,
    "flir": STREET_RECOMMENDED_CLASS_MAP,
    "flir_camera_objects": STREET_RECOMMENDED_CLASS_MAP,
    "flir_data_set": STREET_RECOMMENDED_CLASS_MAP,
    "idd": STREET_RECOMMENDED_CLASS_MAP,
    "idd_project": STREET_RECOMMENDED_CLASS_MAP,
    "safewalkbd": STREET_RECOMMENDED_CLASS_MAP,
    "safewalkbd_final": STREET_RECOMMENDED_CLASS_MAP,
    "sod": STREET_RECOMMENDED_CLASS_MAP,
    "obstacle_finder": STREET_RECOMMENDED_CLASS_MAP,
    "blind_assist_object_detection": GENERAL_RECOMMENDED_CLASS_MAP,

    # Logistics has strong mAP and clean labels for vehicle/sign classes.
    "logistics": LOGISTICS_RECOMMENDED_CLASS_MAP,
    "logistics_sz9jr": LOGISTICS_RECOMMENDED_CLASS_MAP,

    # Food/produce.
    "complete_food": FOOD_RECOMMENDED_CLASS_MAP,
    "detection_fruits": FOOD_RECOMMENDED_CLASS_MAP,
    "fruit_detection": FOOD_RECOMMENDED_CLASS_MAP,
    "fruit_detection_skagr": FOOD_RECOMMENDED_CLASS_MAP,
    "fruit_detection_by_yolov8": FOOD_RECOMMENDED_CLASS_MAP,
    "fruits_and_veg_2": FOOD_RECOMMENDED_CLASS_MAP,

    # Indoor/household.
    "assistance": HOUSEHOLD_RECOMMENDED_CLASS_MAP,
    "detected_household_appliances": HOUSEHOLD_RECOMMENDED_CLASS_MAP,
    "house_items": HOUSEHOLD_RECOMMENDED_CLASS_MAP,
    "household_appliances": HOUSEHOLD_RECOMMENDED_CLASS_MAP,
    "household_items": HOUSEHOLD_RECOMMENDED_CLASS_MAP,
    "households_objects": HOUSEHOLD_RECOMMENDED_CLASS_MAP,
    "merge_practice_run": HOUSEHOLD_RECOMMENDED_CLASS_MAP,
    "netra": HOUSEHOLD_RECOMMENDED_CLASS_MAP,
    "cell_phone": HOUSEHOLD_RECOMMENDED_CLASS_MAP,
    "mobile_phone": HOUSEHOLD_RECOMMENDED_CLASS_MAP,
    "phone_detect": HOUSEHOLD_RECOMMENDED_CLASS_MAP,
    "phone_detection": HOUSEHOLD_RECOMMENDED_CLASS_MAP,
    "using_phone": HOUSEHOLD_RECOMMENDED_CLASS_MAP,

    # Sports equipment weak classes.
    "gold_fg_sports_equipment_merged_b_release_test": SPORT_RECOMMENDED_CLASS_MAP,
    "sports_equipment": SPORT_RECOMMENDED_CLASS_MAP,
    "suas_2025_targets_no_polygons": SPORT_RECOMMENDED_CLASS_MAP,
})
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

PRIORITY_OI_CLASSES = [
    "Window", "Wheel", "Door", "Building", "House", "Camera", "Lamp",
    "Helmet", "Hat", "Sunglasses", "Glove", "Belt", "Boot", "Sandal",
    "Shirt", "Jacket", "Shorts", "Towel", "Pillow", "Curtain",
    "Mirror", "Plate", "Box", "Coin", "Key", "Pen", "Poster",
    "Flag", "Balloon", "Ladder", "Hammer", "Screwdriver", "Axe", "Saw",
    "Guitar", "Piano", "Drum", "Trumpet", "Violin", "Flute", "Saxophone",
    "Microphone", "Headphones", "Watch", "Necklace", "Ring", "Earrings",
    "Candle", "Envelope", "Printer", "Monitor", "Shelf", "Cabinet",
    "Shower", "Television", "Fridge", "Stove", "Jar", "Can", "Bucket",
    "Basket", "Bag", "Glasses", "Coat", "Sock", "Trousers", "Crown",
    "Bracelet", "Tire", "Helicopter", "Tractor", "Van", "Ambulance",
    "Police car", "Fire truck", "Wheelchair", "Bicycle helmet", "Luggage",
    "Tent", "Christmas tree", "Flower", "Tree", "Plant", "Mountain",
    "Bridge", "Castle", "Church", "Fountain", "Monument", "Statue",
    "Tower", "Windmill", "Stairs", "Escalator", "Elevator",
    "Flashlight", "Battery", "Game controller", "Speaker", "Projector",
    "Binoculars", "Microscope", "Stethoscope", "Hard hat", "Life jacket",
    "Vacuum cleaner", "Coffee maker", "Kettle", "Mug", "Bread", "Cheese",
    "Egg", "Milk", "Juice", "Coffee", "Tea",
]

TARGET_OI_CLASSES = [
    "Air conditioner", "Ambulance", "Door", "Gas stove",
    "Glasses", "Goat", "Kettle", "Mirror", "Mug", "Plate",
    "Police car", "Shelf", "Stairs", "Van", "Wheelchair", "Window",
    # Extra OI matches for former Bing classes (precise bbox > centered bbox)
    "Mechanical fan", "Light switch",
]

# Maps OI class names to normalized folder names.
OI_FOLDER_OVERRIDES = {
    "Mechanical fan": "fan",
    "Light switch": "switch",
    "Mobile phone": "mobile",
    "Table": "table",
}

COCO80_ID_TO_INDEX = {1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8,10:9,11:10,13:11,14:12,15:13,16:14,17:15,18:16,19:17,20:18,21:19,22:20,23:21,24:22,25:23,27:24,28:25,31:26,32:27,33:28,34:29,35:30,36:31,37:32,38:33,39:34,40:35,41:36,42:37,43:38,44:39,46:40,47:41,48:42,49:43,50:44,51:45,52:46,53:47,54:48,55:49,56:50,57:51,58:52,59:53,60:54,61:55,62:56,63:57,64:58,65:59,67:60,70:61,72:62,73:63,74:64,75:65,76:66,77:67,78:68,79:69,80:70,81:71,82:72,84:73,85:74,86:75,87:76,88:77,89:78,90:79}


def header(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


DRY_RUN = False


def _class_key(name: str) -> str:
    return taxonomy_class_key(name)


def _retained_class_keys() -> set[str]:
    retained = {_class_key(folder) for folder in RETAINED_CLASS_FOLDERS}
    retained.update(_class_key(name) for name in COCO_SOURCE_CLASS_ALIASES)
    retained.update(_class_key(alias) for alias, target in CLASS_ALIASES.items() if target in FINAL_ASSISTIVE_CLASSES)
    return retained


def _target_folder_for_class_name(name: str) -> str:
    return _safe_class_folder(canonical_label(name))


def _oi_folder_name(class_name: str) -> str:
    return OI_FOLDER_OVERRIDES.get(class_name, class_name.replace(" ", "_"))


def _retained_coco_classes() -> list[str]:
    retained = _retained_class_keys()
    return [name for name in COCO80 if _class_key(name) in retained]


def _retained_oi_classes() -> list[str]:
    retained = _retained_class_keys()
    return [name for name in TARGET_OI_CLASSES if _class_key(_oi_folder_name(name)) in retained]


def _class_dir_has_images(folder_name: str) -> bool:
    class_dir = OI_OUTPUT_DIR / folder_name
    return class_dir.is_dir() and any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in class_dir.iterdir()
    )


def _class_dirs_complete(folder_names: list[str]) -> bool:
    return bool(folder_names) and all(_class_dir_has_images(name) for name in folder_names)


def _source_image_files(directory: Path, include_roboflow_previews: bool = True) -> list[Path]:
    if not directory.exists():
        return []
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not include_roboflow_previews:
        files = [p for p in files if not p.stem.startswith(ROBOFLOW_PREVIEW_PREFIX)]
    return sorted(files)


def download_file(url: str, dest: Path, desc: str = "") -> None:
    if DRY_RUN:
        print(f"  [dry-run] Would download {desc or dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {desc or dest.name} ...")
    r = requests.get(url, stream=True, timeout=120, verify=False)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 / total
                    bar_len = 40
                    filled = int(bar_len * downloaded / total)
                    bar = "#" * filled + "-" * (bar_len - filled)
                    sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%")
                    sys.stdout.flush()
    mb = dest.stat().st_size / 1024 / 1024
    print(f"\n  -> {mb:.1f} MB")


# ═══════════════════════════════════════════════════════════════════════════════
# PART A — COCO download (80 classes)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_yolo_weights() -> None:
    header("YOLO Weights")
    weights = ROOT / "yolov8n.pt"
    if weights.exists():
        print("  yolov8n.pt already exists.")
        return
    print("  Downloading yolov8n.pt via ultralytics ...")
    from ultralytics import YOLO
    YOLO("yolov8n.pt")


def _dl_extract(name: str, url: str, extract_to: Path) -> None:
    zips_dir = LABELS_CACHE.parent / "zips"
    zips_dir.mkdir(parents=True, exist_ok=True)
    zip_path = zips_dir / f"{name}.zip"
    download_file(url, zip_path, desc=name)
    if not zip_path.exists():
        return
    print(f"  Extracting ...")
    shutil.unpack_archive(str(zip_path), str(extract_to))
    zip_path.unlink(missing_ok=True)
    print("  Done.")


def _download_coco() -> None:
    header("COCO val2017")
    retained_coco = _retained_coco_classes()
    retained_folders = [_target_folder_for_class_name(name) for name in retained_coco]
    if _class_dirs_complete(retained_folders):
        total = sum(len(_source_image_files(OI_OUTPUT_DIR / name)) for name in retained_folders)
        print(f"  Retained COCO classes already organized ({total} images across {len(retained_coco)} class dirs). Skipping download.")
        return

    images_dir = COCO_ROOT / "images"
    ann_dir = COCO_ROOT / "annotations"
    val_dir = images_dir / "val2017"
    ann_file = ann_dir / "instances_val2017.json"

    if val_dir.exists() and any(val_dir.iterdir()):
        print(f"  Images exist ({len(list(val_dir.glob('*.jpg')))}). Skipping.")
    else:
        _dl_extract("val2017", "http://images.cocodataset.org/zips/val2017.zip", images_dir)

    if ann_file.exists():
        print("  Annotations exist. Skipping.")
    else:
        _dl_extract("annotations", "http://images.cocodataset.org/annotations/annotations_trainval2017.zip", COCO_ROOT)


def _coco_labels() -> None:
    header("COCO YOLO labels")
    ann_file = COCO_ROOT / "annotations" / "instances_val2017.json"
    labels_dir = COCO_ROOT / "labels" / "val2017"

    if labels_dir.exists() and len(list(labels_dir.glob("*.txt"))) >= 4000:
        print(f"  Labels exist. Skipping.")
        return

    if not ann_file.exists():
        print(f"  Annotations not found — skipping labels.")
        return

    labels_dir.mkdir(parents=True, exist_ok=True)
    with open(ann_file, encoding="utf-8") as f:
        data = json.load(f)
    image_info = {img["id"]: img for img in data["images"]}
    by_img = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0): continue
        cid = ann["category_id"]
        if cid not in COCO80_ID_TO_INDEX: continue
        img = image_info[ann["image_id"]]
        w, h = img["width"], img["height"]
        x, y, bw, bh = ann["bbox"]
        cls = COCO80_ID_TO_INDEX[cid]
        by_img[ann["image_id"]].append(f"{cls} {(x+bw/2)/w:.6f} {(y+bh/2)/h:.6f} {bw/w:.6f} {bh/h:.6f}")

    for img_id, img in image_info.items():
        (labels_dir / f"{Path(img['file_name']).stem}.txt").write_text("\n".join(by_img.get(img_id, [])))

    print(f"  {len(image_info)} label files written.")


def _organize_coco() -> None:
    header("Organize COCO by class")
    ann_file = COCO_ROOT / "annotations" / "instances_val2017.json"
    images_dir = COCO_ROOT / "images" / "val2017"
    output = COCO_ROOT / "images_by_class"
    retained_coco = set(_retained_coco_classes())
    retained_folders = sorted({_target_folder_for_class_name(name) for name in retained_coco})

    if _class_dirs_complete(retained_folders):
        total = sum(len(_source_image_files(output / name)) for name in retained_folders)
        print(f"  Retained COCO classes already exist ({total} images). Skipping.")
        return

    with open(ann_file, encoding="utf-8") as f:
        data = json.load(f)
    cat_map = {c["id"]: c["name"] for c in data["categories"]}
    id_to_fn = {img["id"]: img["file_name"] for img in data["images"]}
    img_cls = defaultdict(set)
    for ann in data["annotations"]:
        img_cls[ann["image_id"]].add(cat_map[ann["category_id"]])

    for cat in data["categories"]:
        if cat["name"] in retained_coco:
            (output / _target_folder_for_class_name(cat["name"])).mkdir(parents=True, exist_ok=True)

    total = 0
    for img_id, classes in img_cls.items():
        fn = id_to_fn.get(img_id)
        if not fn: continue
        src = images_dir / fn
        if not src.exists(): continue
        for cls in classes:
            if cls not in retained_coco:
                continue
            dst = output / _target_folder_for_class_name(cls) / fn
            if not dst.exists():
                shutil.copy2(src, dst)
                total += 1

    print(f"  Copied {total} images into class folders.")
    # Clean up originals
    for folder in [images_dir, COCO_ROOT / "annotations", COCO_ROOT / "labels", LABELS_CACHE.parent / "zips"]:
        if folder.exists(): shutil.rmtree(folder); print(f"  Cleaned: {folder}")


def run_coco() -> None:
    print(f"\n{'=' * 55}")
    print("  COCO DATASET SETUP")
    print(f"  val2017 | 5000 images | {len(_retained_coco_classes())} retained classes")
    print(f"{'=' * 55}")
    if DRY_RUN:
        print("  [dry-run] Would ensure YOLO weights, download COCO val2017,")
        print("  [dry-run] generate precise COCO labels, and organize class folders.")
        return
    COCO_ROOT.mkdir(parents=True, exist_ok=True)
    _get_yolo_weights()
    _download_coco()
    _coco_labels()
    _organize_coco()
    print(f"\n{'=' * 55}\n  COCO DONE!\n{'=' * 55}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# PART B — Open Images expansion (17 targeted classes)
# ═══════════════════════════════════════════════════════════════════════════════

def _oi_fetch_classes() -> dict[str, str]:
    header("OI — Fetching class list")
    csv_path = LABELS_CACHE / "oi_classes.csv"
    if DRY_RUN:
        print("  [dry-run] Would download OI class list")
        return {"_placeholder_": "_"}
    download_file(
        "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv",
        csv_path, "Class descriptions",
    )
    classes = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2: classes[row[1].strip()] = row[0].strip()
    print(f"  OI classes: {len(classes)}")
    return classes


def _oi_select_classes(all_classes: dict[str, str]) -> list[str]:
    coco_lower = {c.lower() for c in COCO80}
    targets = _retained_oi_classes()
    selected = [c for c in targets if c in all_classes and c.lower() not in coco_lower]
    print(f"  Selected {len(selected)} OI classes (filtered COCO overlap):")
    for n in selected:
        print(f"    - {n}")
    return selected


def _oi_download_annotations(selected: list[str], all_classes: dict[str, str]) -> dict[str, list[dict]]:
    header("OI — Filtering annotations")
    ann_path = LABELS_CACHE / "oi_annotations.csv"
    download_file(
        "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
        ann_path, "Validation annotations",
    )

    selected_ids = {all_classes[n]: n for n in selected}
    oi_id_to_name = selected_ids

    class_images = defaultdict(list)
    with open(ann_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Filtering", unit="rows"):
            cid = (row.get("LabelName") or "").strip()
            if not cid or cid not in oi_id_to_name: continue
            try:
                xmin, xmax, ymin, ymax = map(float, [row["XMin"], row["XMax"], row["YMin"], row["YMax"]])
            except (ValueError, KeyError): continue
            if xmin >= xmax or ymin >= ymax: continue
            cls_name = oi_id_to_name[cid]
            class_images[cls_name].append({"image_id": row["ImageID"].strip(), "class_name": cls_name,
                                           "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})

    for n, a in sorted(class_images.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"    {n:<30} {len(a):>6} annotations")
    return class_images


def _oi_download_images(class_images: dict[str, list[dict]]) -> None:
    header(f"OI — Downloading (max {MAX_OI_PER_CLASS}/class)")
    sampled: dict[str, list[dict]] = {}
    for name, anns in class_images.items():
        if len(anns) < 5:
            print(f"  Skipping {name}: only {len(anns)} annotations"); continue
        by_img = {}
        for ann in anns: by_img.setdefault(ann["image_id"], ann)
        items = list(by_img.values())[:MAX_OI_PER_CLASS]
        items.sort(key=lambda x: x["image_id"])
        sampled[name] = items

    total = sum(len(v) for v in sampled.values())
    print(f"  Classes: {len(sampled)}, Images: {total}")
    if not sampled: print("  No classes."); return

    img_anns = defaultdict(list)
    for name, anns in sampled.items():
        for ann in anns: img_anns[ann["image_id"]].append(ann)

    img_dir = LABELS_CACHE / "oi_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(img_dir.glob("*.jpg")))
    remaining = [iid for iid in img_anns if not (img_dir / f"{iid}.jpg").exists()]
    print(f"  Already: {existing}, To download: {len(remaining)}")

    import concurrent.futures
    def dl(iid):
        dest = img_dir / f"{iid}.jpg"
        if dest.exists(): return True
        try:
            r = requests.get(
                f"https://open-images-dataset.s3.amazonaws.com/validation/{iid}.jpg",
                timeout=30, headers={"User-Agent": "Mozilla/5.0"}, verify=False,
            )
            if r.status_code == 200 and len(r.content) > 1000:
                with open(dest, "wb") as f: f.write(r.content)
                return True
            return False
        except Exception: return False

    success, failed = existing, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(dl, iid): iid for iid in remaining}
        with tqdm(total=len(remaining), desc="Downloading", unit="img") as pbar:
            for f in concurrent.futures.as_completed(futures):
                if f.result(): success += 1
                else: failed += 1
                pbar.update(1)
    print(f"  Success: {success}, Failed: {failed}")

    # Copy to class folders
    img_to_cls = defaultdict(set)
    for name, anns in sampled.items():
        for ann in anns: img_to_cls[ann["image_id"]].add(name)

    total_copies = 0
    for iid, names in tqdm(img_to_cls.items(), desc="Organizing", unit="img"):
        src = img_dir / f"{iid}.jpg"
        if not src.exists(): continue
        for cls_name in names:
            cls_dir = OI_OUTPUT_DIR / _oi_folder_name(cls_name)
            cls_dir.mkdir(parents=True, exist_ok=True)
            dst = cls_dir / f"{iid}.jpg"
            if not dst.exists(): shutil.copy2(src, dst); total_copies += 1

    print(f"  Copied {total_copies} images.")
    # Clean OI cache
    shutil.rmtree(img_dir)
    print("  OI cache cleaned.")


def run_openimages() -> None:
    targets = _retained_oi_classes()
    print(f"\n{'=' * 60}")
    print("  OPEN IMAGES TARGETED DOWNLOAD")
    print(f"  {len(targets)} retained classes (max {MAX_OI_PER_CLASS} images each)")
    print(f"{'=' * 60}")
    if DRY_RUN:
        print("  [dry-run] Would download Open Images metadata, filter validation")
        print("  [dry-run] annotations, and fetch precise labeled images for:")
        for name in targets:
            folder = _oi_folder_name(name)
            print(f"    - {name} -> {folder}")
        return

    # Skip if already fully organized in class folders
    expected = [_oi_folder_name(name) for name in targets]
    if OI_OUTPUT_DIR.exists() and _class_dirs_complete(expected):
        total = sum(len(_source_image_files(OI_OUTPUT_DIR / d)) for d in expected)
        print(f"  Already complete ({total} images across {len(expected)} class dirs). Skipping.")
        return

    OI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_CACHE.mkdir(parents=True, exist_ok=True)
    all_classes = _oi_fetch_classes()
    selected = _oi_select_classes(all_classes)
    if not selected:
        print("  No OI classes to download.")
        return
    class_images = _oi_download_annotations(selected, all_classes)
    _oi_download_images(class_images)
    # Clean cache
    ann_path = LABELS_CACHE / "oi_annotations.csv"
    cls_path = LABELS_CACHE / "oi_classes.csv"
    if ann_path.exists(): ann_path.unlink()
    if cls_path.exists(): cls_path.unlink()
    print(f"\n{'=' * 60}\n  OI DONE!\n{'=' * 60}\n")




# ═══════════════════════════════════════════════════════════════════════════════
# PART D — Roboflow Universe import (YOLOv8 export)
# ═══════════════════════════════════════════════════════════════════════════════

def _arg_value(*names: str) -> str | None:
    for name in names:
        if name in sys.argv:
            idx = sys.argv.index(name)
            if idx + 1 >= len(sys.argv) or sys.argv[idx + 1].startswith("--"):
                print(f"  Missing value for {name}")
                sys.exit(2)
            return sys.argv[idx + 1]
    return None


def _read_dotenv_values() -> dict[str, str]:
    values: dict[str, str] = {}
    for env_path in [ROOT / ".env", ROOT / ".env.local"]:
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")
    return values


def _roboflow_from_config() -> str | None:
    for config_path in DATASET_SOURCE_CONFIGS:
        if not config_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"  Invalid dataset source config: {config_path} ({exc})")
            sys.exit(1)

        direct = config.get("roboflow_export_url") or config.get("roboflow_export_zip")
        if direct:
            return str(direct)
        roboflow = config.get("roboflow")
        if isinstance(roboflow, dict):
            nested = roboflow.get("export_url") or roboflow.get("zip")
            if nested:
                return str(nested)
    return None


def _configured_roboflow_source() -> tuple[str | None, str]:
    for candidate in ROBOFLOW_BUNDLE_CANDIDATES:
        if candidate.exists():
            return str(candidate), f"local zip ({candidate})"

    dotenv = _read_dotenv_values()
    for key in ["ROBOFLOW_EXPORT_URL", "ROBOFLOW_EXPORT_ZIP", "ROBOFLOW_DATASET_URL"]:
        value = os.environ.get(key) or dotenv.get(key)
        if value:
            return value, key

    config_source = _roboflow_from_config()
    if config_source:
        return config_source, "config/dataset_sources.json"

    return None, ""


def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _display_source(source: str) -> str:
    if _is_url(source) and "?" in source:
        return f"{source.split('?', 1)[0]}?..."
    return source


def _image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _yaml_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def _safe_class_folder(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in " ._-" else "_" for ch in name.strip())
    folder = cleaned.replace(" ", "_").strip("._")
    return folder or "class"


def _link_or_copy_image(src: Path, dst: Path) -> bool:
    if dst.exists():
        return False
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return True


def _class_folder_for_name(name: str) -> str:
    key = _class_key(name)
    if OI_OUTPUT_DIR.exists():
        for folder in sorted(d.name for d in OI_OUTPUT_DIR.iterdir() if d.is_dir()):
            if _class_key(folder) == key:
                return folder
    for folder in RETAINED_CLASS_FOLDERS:
        if _class_key(folder) == key:
            return folder
    return _safe_class_folder(name)


def _roboflow_zip_key(zip_path: Path) -> str:
    stem = zip_path.stem.lower()
    stem = re.sub(r"\.v\d[^.]*\.yolov8$", "", stem)
    stem = re.sub(r"\.v\d.*$", "", stem)
    return re.sub(r"[^a-z0-9]+", "_", stem).strip("_")


def _roboflow_class_map_for_zip(zip_path: Path) -> dict[str, str | None]:
    key = _roboflow_zip_key(zip_path)
    candidates = [
        zip_path.name,
        zip_path.stem,
        zip_path.name.lower(),
        zip_path.stem.lower(),
        key,
        re.sub(r"\d+$", "", key),
    ]
    for candidate in candidates:
        if candidate in ROBOFLOW_CLASS_MAP:
            return ROBOFLOW_CLASS_MAP[candidate]
    return {}


def _map_roboflow_class_name(zip_map: dict[str, str | None], name: str) -> str | None:
    def normalize_target(target: str | None) -> str | None:
        if target is None:
            return None
        canonical = canonical_label(target)
        if canonical not in FINAL_ASSISTIVE_CLASSES:
            return None
        return canonical

    if "__all__" in zip_map:
        return normalize_target(zip_map["__all__"])
    if name in zip_map:
        return normalize_target(zip_map[name])

    name_key = _class_key(name)
    for source, target in zip_map.items():
        if source.startswith("__"):
            continue
        if _class_key(source) == name_key:
            return normalize_target(target)
    return normalize_target(zip_map.get("__default__", name))


def _zip_fingerprint(zip_path: Path) -> dict[str, str | int]:
    digest = hashlib.sha256()
    with zip_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"size": zip_path.stat().st_size, "sha256": digest.hexdigest()}


def _directory_fingerprint(directory: Path) -> dict[str, str | int]:
    total_size = 0
    total_files = 0
    max_mtime_ns = 0
    yaml_digest = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        stat = path.stat()
        total_size += stat.st_size
        total_files += 1
        max_mtime_ns = max(max_mtime_ns, stat.st_mtime_ns)
        if path.name.lower() in {"data.yaml", "data.yml"}:
            yaml_digest.update(path.read_bytes())
    return {
        "size": total_size,
        "files": total_files,
        "max_mtime_ns": max_mtime_ns,
        "data_yaml_sha256": yaml_digest.hexdigest(),
    }


def _source_fingerprint(source_path: Path) -> dict[str, str | int]:
    if source_path.is_dir():
        return _directory_fingerprint(source_path)
    return _zip_fingerprint(source_path)


def _read_roboflow_merge_manifest() -> dict[str, dict[str, str | int]]:
    if not ROBOFLOW_MERGE_MANIFEST.exists():
        return {}
    try:
        data = json.loads(ROBOFLOW_MERGE_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(name): value for name, value in data.items() if isinstance(value, dict)}


def _write_roboflow_merge_manifest(manifest: dict[str, dict[str, str | int]]) -> None:
    ROBOFLOW_MERGE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    ROBOFLOW_MERGE_MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _has_roboflow_split_outputs() -> bool:
    for split_dir in [TRAIN_DIR, VAL_DIR]:
        image_dir = split_dir / "images"
        if image_dir.exists() and any(image_dir.glob(f"{ROBOFLOW_PREVIEW_PREFIX}*")):
            return True
    return False


def _is_roboflow_export_dir(directory: Path) -> bool:
    return directory.is_dir() and (
        (directory / "data.yaml").exists()
        or (directory / "data.yml").exists()
        or any(directory.glob("*/data.yaml"))
        or any(directory.glob("*/data.yml"))
    )


def _roboflow_merge_sources(source_dir: Path) -> list[Path]:
    if _is_roboflow_export_dir(source_dir):
        return [source_dir]

    sources = list(source_dir.glob("*.zip"))
    sources.extend(
        path for path in source_dir.iterdir()
        if path.is_dir() and _is_roboflow_export_dir(path)
    )
    return sorted(sources, key=lambda path: path.name.lower())


def _normalize_yolo_names(raw_names) -> list[str]:
    if isinstance(raw_names, list):
        return [str(name) for name in raw_names]
    if isinstance(raw_names, dict):
        def key_order(item) -> tuple[int, str]:
            key = item[0]
            try:
                return int(key), str(key)
            except (TypeError, ValueError):
                return 999999, str(key)

        return [str(value) for _, value in sorted(raw_names.items(), key=key_order)]
    raise ValueError("Roboflow data.yaml is missing a YOLO names list.")


def _resolve_roboflow_source(source: str) -> Path:
    if _is_url(source):
        ROBOFLOW_CACHE.mkdir(parents=True, exist_ok=True)
        dest = ROBOFLOW_CACHE / "roboflow_export.zip"
        download_file(source, dest, "Roboflow YOLOv8 export")
        return dest

    source_path = Path(source).expanduser()
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    if not source_path.exists():
        print(f"  Roboflow export not found: {source_path}")
        sys.exit(1)
    return source_path


def _find_roboflow_yaml(extract_dir: Path) -> tuple[Path, dict]:
    import yaml

    candidates = sorted(extract_dir.rglob("data.yaml")) + sorted(extract_dir.rglob("data.yml"))
    for yaml_path in candidates:
        with open(yaml_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if "names" in cfg and ("train" in cfg or (yaml_path.parent / "train" / "images").exists()):
            return yaml_path, cfg
    print("  No Roboflow YOLO data.yaml found in export.")
    sys.exit(1)


def _safe_extract_zip(zip_path: Path, extract_dir: Path) -> None:
    extract_root = extract_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (extract_dir / member.filename).resolve()
            try:
                target.relative_to(extract_root)
            except ValueError:
                raise ValueError(f"Unsafe zip path: {member.filename}") from None
        zf.extractall(extract_dir)


def _resolve_split_dir(yaml_path: Path, cfg: dict, split_key: str, fallback: str) -> Path:
    base = yaml_path.parent
    dataset_root = Path(cfg["path"]) if cfg.get("path") else base
    if not dataset_root.is_absolute():
        dataset_root = (base / dataset_root).resolve()

    raw_value = cfg.get(split_key)
    if raw_value is None and split_key == "val":
        raw_value = cfg.get("valid")
    split_path = Path(raw_value) if raw_value else Path(fallback)
    if not split_path.is_absolute():
        split_path = (dataset_root / split_path).resolve()

    # If resolved path escapes the extract dir (e.g. ../train/images), remap
    # to look inside the extract tree instead.
    try:
        split_path.relative_to(base.resolve())
    except ValueError:
        # Path went outside extract dir — search for the images dir inside
        split_name = split_path.parent.name  # e.g. "train", "valid"
        for candidate_name in [split_name, split_key]:
            candidate = base / candidate_name / "images"
            if candidate.exists() and any(
                p.suffix.lower() in IMAGE_EXTENSIONS for p in candidate.iterdir() if p.is_file()
            ):
                return candidate
        # Last resort: return the original path but resolved inside base
        return base / Path(raw_value or fallback).name
    return split_path


def _label_dir_for_images(image_dir: Path) -> Path:
    if image_dir.name.lower() == "images":
        return image_dir.parent / "labels"
    return image_dir / "labels"


def _copy_roboflow_split(split_name: str, image_dir: Path, target_dir: Path) -> int:
    images = _image_files(image_dir)
    label_dir = _label_dir_for_images(image_dir)
    print(f"  {split_name:<5} source: {image_dir}")
    print(f"  {split_name:<5} images: {len(images)}")
    if not images:
        return 0

    if target_dir.exists():
        shutil.rmtree(target_dir)
    (target_dir / "images").mkdir(parents=True, exist_ok=True)
    (target_dir / "labels").mkdir(parents=True, exist_ok=True)

    for idx, image_path in enumerate(images):
        stem = f"rf_{split_name}_{idx:06d}_{image_path.stem}"
        dst_image = target_dir / "images" / f"{stem}{image_path.suffix.lower()}"
        dst_label = target_dir / "labels" / f"{stem}.txt"
        shutil.copy2(image_path, dst_image)

        src_label = label_dir / f"{image_path.stem}.txt"
        if src_label.exists():
            shutil.copy2(src_label, dst_label)
        else:
            dst_label.write_text("", encoding="utf-8")
    return len(images)


def _write_project_data_yaml(names: list[str]) -> None:
    yaml_lines = [
        f"train: {_yaml_path(TRAIN_DIR / 'images')}\n",
        f"val: {_yaml_path(VAL_DIR / 'images')}\n\n",
        f"nc: {len(names)}\n",
        "names:\n",
    ]
    for idx, name in enumerate(names):
        yaml_lines.append(f"  {idx}: {name!r}\n")
    DATA_YAML.parent.mkdir(parents=True, exist_ok=True)
    DATA_YAML.write_text("".join(yaml_lines), encoding="utf-8")
    print(f"  data.yaml written ({len(names)} classes)")


def _populate_speed_images_from_yolo(names: list[str]) -> None:
    if OI_OUTPUT_DIR.exists():
        shutil.rmtree(OI_OUTPUT_DIR)
    OI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    counts: dict[int, int] = defaultdict(int)
    copied = 0
    for split_dir in [TRAIN_DIR, VAL_DIR]:
        image_dir = split_dir / "images"
        label_dir = split_dir / "labels"
        for image_path in _image_files(image_dir):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            class_ids = set()
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    class_id = int(float(parts[0]))
                except ValueError:
                    continue
                if 0 <= class_id < len(names):
                    class_ids.add(class_id)

            for class_id in sorted(class_ids):
                if ROBOFLOW_SAMPLE_LIMIT_PER_CLASS is not None and counts[class_id] >= ROBOFLOW_SAMPLE_LIMIT_PER_CLASS:
                    continue
                class_dir = OI_OUTPUT_DIR / _safe_class_folder(names[class_id])
                class_dir.mkdir(parents=True, exist_ok=True)
                dst = class_dir / image_path.name
                if _link_or_copy_image(image_path, dst):
                    copied += 1
                counts[class_id] += 1

    print(f"  Benchmark preview images: {copied} across {len(counts)} labeled classes")


def run_roboflow_import(source: str) -> None:
    header("Roboflow Universe YOLOv8 import")
    print("  This replaces project train/val splits with the Roboflow export.")
    print("  Roboflow labels are preserved; no centered fallback boxes are generated.")
    if DRY_RUN:
        print(f"  [dry-run] Would import: {_display_source(source)}")
        print("  [dry-run] Expected format: Roboflow YOLOv8 zip with data.yaml, train/images, train/labels.")
        return

    zip_path = _resolve_roboflow_source(source)
    extract_dir = ROBOFLOW_CACHE / "extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Extracting: {zip_path}")
    try:
        _safe_extract_zip(zip_path, extract_dir)
    except zipfile.BadZipFile:
        print("  Roboflow source is not a valid zip file.")
        sys.exit(1)
    except ValueError as exc:
        print(f"  {exc}")
        sys.exit(1)

    yaml_path, cfg = _find_roboflow_yaml(extract_dir)
    names = _normalize_yolo_names(cfg.get("names"))
    train_images = _resolve_split_dir(yaml_path, cfg, "train", "train/images")
    val_images = _resolve_split_dir(yaml_path, cfg, "val", "valid/images")

    train_count = _copy_roboflow_split("train", train_images, TRAIN_DIR)
    val_count = _copy_roboflow_split("val", val_images, VAL_DIR)
    if train_count == 0 or val_count == 0:
        print("  Roboflow export must include non-empty train and val/valid image splits.")
        sys.exit(1)

    _write_project_data_yaml(names)
    _populate_speed_images_from_yolo(names)
    shutil.rmtree(extract_dir, ignore_errors=True)
    print("\n  Roboflow import complete. Ready for training and benchmark.")


def _read_class_list() -> list[str]:
    """Read class names from existing data.yaml, or scan OI_OUTPUT_DIR."""
    if DATA_YAML.exists():
        import yaml
        with open(DATA_YAML, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        names = cfg.get("names", {})
        if isinstance(names, dict):
            return [
                canonical_label(names.get(i, names.get(str(i))))
                for i in sorted(int(k) for k in names.keys())
            ]
        return [canonical_label(name) for name in list(names)]
    if OI_OUTPUT_DIR.exists():
        dirs = sorted(d for d in OI_OUTPUT_DIR.iterdir() if d.is_dir())
        available = {canonical_label(d.name.replace("_", " ")) for d in dirs}
        return [name for name in FINAL_ASSISTIVE_CLASSES if name in available]
    return []


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _label_coords_to_bbox(coords: list[float]) -> tuple[float, float, float, float] | None:
    if len(coords) == 4:
        x, y, w, h = (_clamp01(value) for value in coords)
        if w <= 0 or h <= 0:
            return None
        return x, y, w, h

    if len(coords) >= 6 and len(coords) % 2 == 0:
        xs = [_clamp01(value) for value in coords[0::2]]
        ys = [_clamp01(value) for value in coords[1::2]]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return None
        return x1 + w / 2, y1 + h / 2, w, h

    return None


def _remap_label_line_to_box(line: str, local_to_global: dict[int, int]) -> str | None:
    parts = line.strip().split()
    if not parts:
        return None
    try:
        local = int(float(parts[0]))
        coords = [float(value) for value in parts[1:]]
    except ValueError:
        return None

    global_idx = local_to_global.get(local)
    bbox = _label_coords_to_bbox(coords)
    if global_idx is None or bbox is None:
        return None

    x, y, w, h = bbox
    return f"{global_idx} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"


def _copy_roboflow_preview_image(image_path: Path, label_lines: list[str],
                                 class_names: list[str], counts: dict[int, int]) -> None:
    class_ids = set()
    for line in label_lines:
        parts = line.split()
        if not parts:
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            continue
        if 0 <= class_id < len(class_names):
            class_ids.add(class_id)

    for class_id in sorted(class_ids):
        if ROBOFLOW_SAMPLE_LIMIT_PER_CLASS is not None and counts[class_id] >= ROBOFLOW_SAMPLE_LIMIT_PER_CLASS:
            continue
        class_dir = OI_OUTPUT_DIR / _class_folder_for_name(class_names[class_id])
        class_dir.mkdir(parents=True, exist_ok=True)
        dst = class_dir / image_path.name
        _link_or_copy_image(image_path, dst)
        counts[class_id] += 1


def _merge_roboflow_split(split_name: str, image_dir: Path, target_dir: Path,
                          local_to_global: dict[int, int], start_idx: int,
                          class_names: list[str],
                          preview_counts: dict[int, int]) -> int:
    """Copy a Roboflow split with class-index remapping, appending to target_dir."""
    images = _image_files(image_dir)
    label_dir = _label_dir_for_images(image_dir)
    (target_dir / "images").mkdir(parents=True, exist_ok=True)
    (target_dir / "labels").mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped_no_label = 0
    for i, img_path in enumerate(images):
        safe_source_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", img_path.stem)[:80].strip("._-")
        if not safe_source_stem:
            safe_source_stem = hashlib.sha1(img_path.stem.encode("utf-8", errors="ignore")).hexdigest()[:12]
        stem = f"rf_{split_name}_{start_idx + i:06d}_{safe_source_stem}"
        dst_img = target_dir / "images" / f"{stem}{img_path.suffix.lower()}"
        dst_lbl = target_dir / "labels" / f"{stem}.txt"
        src_lbl = label_dir / f"{img_path.stem}.txt"
        lines = []
        if src_lbl.exists():
            for line in src_lbl.read_text(encoding="utf-8").splitlines():
                remapped = _remap_label_line_to_box(line, local_to_global)
                if remapped is not None:
                    lines.append(remapped)
        lines = list(dict.fromkeys(lines))
        if not lines:
            skipped_no_label += 1
            continue

        shutil.copy2(img_path, dst_img)
        dst_lbl.write_text("\n".join(lines), encoding="utf-8")
        _copy_roboflow_preview_image(dst_img, lines, class_names, preview_counts)
        copied += 1
    if skipped_no_label:
        print(f"    ({skipped_no_label} images skipped - no remapped labels)")
    return copied


def rebuild_preview_images(class_names: list[str] | None = None) -> None:
    """Rebuild images_by_class from existing train/val splits.

    This ensures the benchmark/preview folder always reflects the actual
    training data, even when Roboflow zips were already merged in a prior run.
    """
    header("Rebuilding images_by_class previews")
    if class_names is None:
        class_names = _read_class_list()
    if not class_names:
        print("  No class list available. Skipping.")
        return
    if OI_OUTPUT_DIR.exists():
        removed = 0
        for class_dir in OI_OUTPUT_DIR.iterdir():
            if not class_dir.is_dir():
                continue
            for image_path in class_dir.iterdir():
                if image_path.is_file() and image_path.name.startswith(ROBOFLOW_PREVIEW_PREFIX):
                    image_path.unlink()
                    removed += 1
        if removed:
            print(f"  Removed {removed} old Roboflow preview files")

    counts: dict[int, int] = defaultdict(int)
    # Count existing preview images
    if OI_OUTPUT_DIR.exists():
        for d in OI_OUTPUT_DIR.iterdir():
            if not d.is_dir():
                continue
            key = _class_key(d.name.replace("_", " "))
            for idx, name in enumerate(class_names):
                if _class_key(name) == key:
                    existing_count = len(_source_image_files(d))
                    counts[idx] = existing_count
                    break

    copied = 0
    for split_dir in [TRAIN_DIR, VAL_DIR]:
        image_dir = split_dir / "images"
        label_dir = split_dir / "labels"
        if not image_dir.exists() or not label_dir.exists():
            continue
        for label_path in sorted(label_dir.glob("*.txt")):
            class_ids: set[int] = set()
            for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    cid = int(float(parts[0]))
                except ValueError:
                    continue
                if 0 <= cid < len(class_names):
                    class_ids.add(cid)

            for cid in sorted(class_ids):
                if ROBOFLOW_SAMPLE_LIMIT_PER_CLASS is not None and counts[cid] >= ROBOFLOW_SAMPLE_LIMIT_PER_CLASS:
                    continue
                # Find matching image
                img_path = None
                for ext in IMAGE_EXTENSIONS:
                    candidate = image_dir / f"{label_path.stem}{ext}"
                    if candidate.exists():
                        img_path = candidate
                        break
                if img_path is None:
                    continue
                class_dir = OI_OUTPUT_DIR / _class_folder_for_name(class_names[cid])
                class_dir.mkdir(parents=True, exist_ok=True)
                dst = class_dir / img_path.name
                if _link_or_copy_image(img_path, dst):
                    copied += 1
                counts[cid] += 1

    total_classes = sum(1 for c in counts.values() if c > 0)
    print(f"  Preview images added: {copied} across {total_classes} classes")
    low = [(class_names[cid], c) for cid, c in sorted(counts.items()) if 0 < c < 10]
    if low:
        print(f"  Low-preview classes: {', '.join(f'{n}({c})' for n, c in low[:15])}")


def run_roboflow_merge(zip_dir: Path, force: bool = False) -> None:
    """Merge all Roboflow zip files from a directory into existing COCO+OI dataset."""
    sources = _roboflow_merge_sources(zip_dir)
    if not sources:
        print(f"  No Roboflow zip files or extracted exports found in {zip_dir}")
        return

    header(f"Merging {len(sources)} Roboflow source{'s' if len(sources) != 1 else ''} from {zip_dir}")
    if DRY_RUN:
        for source_path in sources:
            print(f"  [dry-run] Would merge {source_path.name} as key '{_roboflow_zip_key(source_path)}'")
        return

    # Read existing classes from data.yaml (from COCO+OI run_labels)
    existing = _read_class_list()
    class_names = list(existing)  # class_names[idx] = name
    preview_counts: dict[int, int] = defaultdict(int)
    merged_zips = 0
    print(f"  Existing classes: {len(class_names)}")

    # On force-merge, remove old Roboflow files to prevent duplicates
    if force:
        for split_dir in [TRAIN_DIR, VAL_DIR]:
            for subdir in ["images", "labels"]:
                d = split_dir / subdir
                if not d.exists():
                    continue
                removed = 0
                for f in d.iterdir():
                    if f.name.startswith("rf_"):
                        f.unlink(missing_ok=True)
                        removed += 1
                if removed:
                    print(f"  Cleaned {removed} old rf_ files from {split_dir.name}/{subdir}")

    manifest = {} if force else _read_roboflow_merge_manifest()
    if not force and not manifest and _has_roboflow_split_outputs():
        manifest = {source_path.name: _source_fingerprint(source_path) for source_path in sources}
        _write_roboflow_merge_manifest(manifest)
        print("  Existing Roboflow outputs found; recorded current zips as already merged.")
        print("  Add new zip files, then run setup.py again to merge only the new additions.")
        return

    for source_path in sources:
        fingerprint = _source_fingerprint(source_path)
        if not force and manifest.get(source_path.name) == fingerprint:
            print(f"\n  [{source_path.name}]")
            print("    Already merged. Skipping.")
            continue

        print(f"\n  [{source_path.name}]")
        extract_dir = ROBOFLOW_CACHE / "merge_extract"
        cleanup_extract = False
        if source_path.is_dir():
            extract_dir = source_path
        else:
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            extract_dir.mkdir(parents=True)
            cleanup_extract = True
            try:
                _safe_extract_zip(source_path, extract_dir)
            except (zipfile.BadZipFile, ValueError) as e:
                print(f"    SKIP: {e}")
                continue

        yaml_path, cfg = _find_roboflow_yaml(extract_dir)
        local_names = _normalize_yolo_names(cfg.get("names"))
        print(f"    Raw classes: {local_names}")

        # Map local classes to target names using ROBOFLOW_CLASS_MAP
        zip_map = _roboflow_class_map_for_zip(source_path)
        mapped_names: list[str | None] = []
        for name in local_names:
            mapped_names.append(_map_roboflow_class_name(zip_map, name))

        # Build local -> global index map (skip None-mapped and non-taxonomy classes)
        allowed_lower = {_class_key(n) for n in FINAL_ASSISTIVE_CLASSES}
        existing_lower = {_class_key(n): n for n in class_names}
        local_to_global = {}
        for li, name in enumerate(mapped_names):
            if name is None:
                print(f"    Skip class: {local_names[li]!r}")
                continue
            key = _class_key(name)
            if key not in allowed_lower:
                print(f"    Skip class: {local_names[li]!r} -> {name!r} (not in taxonomy)")
                continue
            if key in existing_lower:
                canonical = existing_lower[key]
            else:
                canonical = name
                class_names.append(canonical)
                existing_lower[key] = canonical
            local_to_global[li] = class_names.index(canonical)

        kept = len(local_to_global)
        active = {class_names[local_to_global[li]] for li in local_to_global}
        print(f"    Mapped to {kept} class{'es' if kept != 1 else ''}: {sorted(active)}")
        if not local_to_global:
            print("    SKIP: no mapped classes left after filtering.")
            shutil.rmtree(extract_dir, ignore_errors=True)
            continue

        # Count existing images in target dirs for unique filenames
        train_count = len(list((TRAIN_DIR / "images").glob("*"))) if TRAIN_DIR.exists() else 0
        val_count = len(list((VAL_DIR / "images").glob("*"))) if VAL_DIR.exists() else 0

        extract_root = yaml_path.parent
        train_img_dir = _resolve_split_dir(yaml_path, cfg, "train", "train/images")
        if not _image_files(train_img_dir):
            # Scan extract tree for actual image directories
            for candidate in ["train/images", "images/train", "images"]:
                d = extract_root / candidate
                if _image_files(d):
                    train_img_dir = d
                    break

        # Try possible val dir names
        val_img_dir = _resolve_split_dir(yaml_path, cfg, "val", "valid/images")
        if not _image_files(val_img_dir):
            val_img_dir = None
        for candidate in ["valid/images", "val/images", "validation/images"]:
            if val_img_dir is not None:
                break
            d = extract_root / candidate
            if _image_files(d):
                val_img_dir = d
                break

        test_img_dir = _resolve_split_dir(yaml_path, cfg, "test", "test/images")
        if not _image_files(test_img_dir):
            test_img_dir = None
        for candidate in ["test/images", "tests/images"]:
            if test_img_dir is not None:
                break
            d = extract_root / candidate
            if _image_files(d):
                test_img_dir = d
                break

        t = _merge_roboflow_split(
            "train", train_img_dir, TRAIN_DIR, local_to_global,
            train_count, class_names, preview_counts,
        )
        test_as_val = 0
        if val_img_dir:
            v = _merge_roboflow_split(
                "val", val_img_dir, VAL_DIR, local_to_global,
                val_count, class_names, preview_counts,
            )
        else:
            v = 0
        if test_img_dir and (not val_img_dir or test_img_dir.resolve() != val_img_dir.resolve()):
            test_as_val = _merge_roboflow_split(
                "test", test_img_dir, VAL_DIR, local_to_global,
                val_count + v, class_names, preview_counts,
            )
        print(f"    Added: {t} train, {v + test_as_val} val ({test_as_val} from test)")
        merged_zips += 1
        manifest[source_path.name] = fingerprint
        if cleanup_extract:
            shutil.rmtree(extract_dir, ignore_errors=True)

    # Write final data.yaml with all classes
    _write_project_data_yaml(class_names)
    if not force:
        _write_roboflow_merge_manifest(manifest)
    rebalance_existing_splits(class_names, dry_run=False)
    rebuild_preview_images(class_names)

    print(f"\n{'=' * 60}")
    print(f"  Merged {merged_zips} zips | Total classes: {len(class_names)}")
    print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════════════════════════
# PART E — YOLO label generation + train/val splits
# ═══════════════════════════════════════════════════════════════════════════════

def _scan_classes() -> dict[int, str]:
    if not OI_OUTPUT_DIR.exists():
        print(f"  No class folders. Run setup.py first.")
        return {}
    dirs = sorted(d for d in OI_OUTPUT_DIR.iterdir() if d.is_dir())
    available = {
        canonical_label(d.name.replace("_", " "))
        for d in dirs
    }
    ordered = [name for name in FINAL_ASSISTIVE_CLASSES if name in available]
    return {idx: name for idx, name in enumerate(ordered)}


def _load_coco_annotations() -> tuple[dict, dict]:
    header("Loading COCO annotations for labels")
    coco_zip = LABELS_CACHE / "coco_annotations.zip"
    coco_dir = LABELS_CACHE / "coco_ann"
    coco_json = coco_dir / "annotations" / "instances_val2017.json"

    if not coco_json.exists():
        download_file(
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            coco_zip, "COCO annotations",
        )
        shutil.unpack_archive(str(coco_zip), str(coco_dir))
        coco_zip.unlink()

    with open(coco_json, encoding="utf-8") as f:
        data = json.load(f)

    coco_by_file = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0): continue
        img = next((i for i in data["images"] if i["id"] == ann["image_id"]), None)
        if not img: continue
        cat = next((c["name"] for c in data["categories"] if c["id"] == ann["category_id"]), "")
        w, h = img["width"], img["height"]
        x, y, bw, bh = ann["bbox"]
        coco_by_file[img["file_name"]].append((cat, (x+bw/2)/w, (y+bh/2)/h, bw/w, bh/h))

    print(f"  {len(data['images'])} images, {len(data['annotations'])} annotations")
    return coco_by_file, {c["name"] for c in data["categories"]}


def _load_oi_annotations(classes: dict[int, str]) -> dict:
    header("Loading OI annotations for labels")
    cls_csv = LABELS_CACHE / "oi_classes.csv"
    ann_csv = LABELS_CACHE / "oi_annotations.csv"

    if not cls_csv.exists():
        download_file(
            "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv",
            cls_csv, "OI classes",
        )

    if not ann_csv.exists():
        download_file(
            "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
            ann_csv, "OI annotations",
        )

    oi_name_to_id = {}
    with open(cls_csv, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2: oi_name_to_id[row[1].strip()] = row[0].strip()

    oi_id_to_idx = {}
    our_oids = set()
    override_lookup = {_class_key(folder): oi_name for oi_name, folder in OI_FOLDER_OVERRIDES.items()}
    for idx, name in classes.items():
        oi_name = override_lookup.get(_class_key(name), name)
        oid = oi_name_to_id.get(oi_name)
        if oid: oi_id_to_idx[oid] = idx; our_oids.add(oid)

    oi_by_img = defaultdict(list)
    with open(ann_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = (row.get("LabelName") or "").strip()
            if not cid or cid not in our_oids: continue
            try: x1, x2, y1, y2 = map(float, [row["XMin"], row["XMax"], row["YMin"], row["YMax"]])
            except (ValueError, KeyError): continue
            if x1 >= x2 or y1 >= y2: continue
            oi_by_img[row["ImageID"].strip()].append({
                "idx": oi_id_to_idx[cid], "xc": (x1+x2)/2, "yc": (y1+y2)/2,
                "w": x2-x1, "h": y2-y1,
            })

    print(f"  {len(oi_by_img)} images with annotations")
    return oi_by_img


def generate_labels(classes: dict[int, str], dry_run: bool = False) -> dict[str, str]:
    header("Generating YOLO labels")
    name_to_idx = {name: idx for idx, name in classes.items()}
    for source, target in CLASS_ALIASES.items():
        if target in name_to_idx:
            name_to_idx[source] = name_to_idx[target]
    for name, idx in list(name_to_idx.items()):
        name_to_idx[_class_key(name)] = idx
    coco_by_file, coco_names = _load_coco_annotations() if not dry_run else ({}, set())
    oi_by_img = _load_oi_annotations(classes) if not dry_run else {}

    label_dir = LABELS_CACHE / "yolo_labels"
    if not dry_run: label_dir.mkdir(parents=True, exist_ok=True)

    seen = {}
    allowed_classes = {canonical_label(name) for name in classes.values()}
    for d in OI_OUTPUT_DIR.iterdir():
        if not d.is_dir(): continue
        if canonical_label(d.name.replace("_", " ")) not in allowed_classes:
            continue
        for f in _source_image_files(d, include_roboflow_previews=False):
            seen.setdefault(f.stem, f)

    labeled = 0
    fallback = 0
    label_map = {}

    for stem, path in seen.items():
        lines = []
        if stem.isdigit():
            fname = f"{stem.zfill(12)}.jpg"
            for cat, xc, yc, bw, bh in coco_by_file.get(fname, []):
                idx = name_to_idx.get(cat)
                if idx is None:
                    idx = name_to_idx.get(canonical_label(cat))
                if idx is None:
                    idx = name_to_idx.get(_class_key(cat))
                if idx is not None: lines.append(f"{idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        else:
            for ann in oi_by_img.get(stem, []):
                lines.append(f"{ann['idx']} {ann['xc']:.6f} {ann['yc']:.6f} {ann['w']:.6f} {ann['h']:.6f}")

        if lines: labeled += 1
        else:
            # Fallback: centered bbox with correct class from parent folder
            parent_name = path.parent.name
            cls_name = canonical_label(parent_name.replace("_", " "))
            fallback_idx = name_to_idx.get(cls_name, 0)
            fallback += 1
            lines.append(f"{fallback_idx} 0.500000 0.500000 0.900000 0.900000")

        label_map[stem] = "\n".join(lines)
        if not dry_run:
            (label_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"  Total: {len(seen)} | With bboxes: {labeled} | Fallback: {fallback}")
    return label_map


def _classes_from_label_text(text: str) -> set[int]:
    class_ids: set[int] = set()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].isdigit():
            class_ids.add(int(parts[0]))
    return class_ids


def _target_val_images(total_images: int) -> int:
    if total_images < 2:
        return 0
    if total_images <= MIN_TRAIN_IMAGES_PER_CLASS:
        return max(1, total_images // 5)
    available_for_val = total_images - MIN_TRAIN_IMAGES_PER_CLASS
    target = max(math.ceil(total_images * (1.0 - TRAIN_SPLIT)), VAL_TARGET_IMAGES_PER_CLASS)
    return max(1, min(available_for_val, target))


def _min_train_images(total_images: int) -> int:
    target_val = _target_val_images(total_images)
    if target_val <= 0:
        return total_images
    return max(1, total_images - target_val)


def _target_val_boxes(total_boxes: int) -> int:
    if total_boxes < 2:
        return 0
    if total_boxes <= MIN_TRAIN_BOXES_PER_CLASS:
        return max(1, total_boxes // 5)
    available_for_val = total_boxes - MIN_TRAIN_BOXES_PER_CLASS
    return max(1, min(VAL_TARGET_BOXES_PER_CLASS, available_for_val))


def _min_train_boxes(total_boxes: int) -> int:
    target_val = _target_val_boxes(total_boxes)
    if target_val <= 0:
        return total_boxes
    return max(1, total_boxes - target_val)


def _rebalance_item_split(
    train_items: list[tuple[str, Path]],
    val_items: list[tuple[str, Path]],
    label_map: dict[str, str],
) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]], int]:
    """Move source items from train to val until every class has a useful val sample."""
    train = list(train_items)
    val = list(val_items)
    item_classes = {
        stem: _classes_from_label_text(label_map.get(stem, ""))
        for stem, _ in train + val
    }
    total_counts = Counter()
    val_counts = Counter()
    train_counts = Counter()
    for stem, _ in train:
        train_counts.update(item_classes.get(stem, set()))
        total_counts.update(item_classes.get(stem, set()))
    for stem, _ in val:
        val_counts.update(item_classes.get(stem, set()))
        total_counts.update(item_classes.get(stem, set()))

    moved = 0
    for class_id, total in sorted(total_counts.items(), key=lambda item: item[0]):
        target = _target_val_images(total)
        while val_counts[class_id] < target:
            needed = {
                cid
                for cid, cid_total in total_counts.items()
                if val_counts[cid] < _target_val_images(cid_total)
            }
            candidates = [
                (idx, stem, src, item_classes.get(stem, set()))
                for idx, (stem, src) in enumerate(train)
                if class_id in item_classes.get(stem, set())
            ]
            candidates = [
                c for c in candidates
                if all(train_counts[cid] > 1 for cid in c[3])
            ]
            if not candidates:
                break

            candidates.sort(
                key=lambda item: (
                    -len(item[3] & needed),
                    len(item[3]),
                    item[1],
                )
            )
            idx, stem, src, cls_ids = candidates[0]
            train.pop(idx)
            val.append((stem, src))
            train_counts.subtract(cls_ids)
            val_counts.update(cls_ids)
            moved += 1
    return train, val, moved


def _label_class_ids(label_path: Path) -> set[int]:
    return set(_label_class_counts(label_path).keys())


def _label_class_counts(label_path: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    if not label_path.exists():
        return counts
    with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5 and parts[0].isdigit():
                counts[int(parts[0])] += 1
    return counts


def _image_for_label(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    matches = [p for p in image_dir.glob(f"{stem}.*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    return matches[0] if matches else None


def _split_records(split_dir: Path) -> list[dict]:
    image_dir = split_dir / "images"
    label_dir = split_dir / "labels"
    records = []
    if not label_dir.exists():
        return records
    image_by_stem = {path.stem: path for path in _source_image_files(image_dir)}
    for label_path in sorted(label_dir.glob("*.txt")):
        image_path = image_by_stem.get(label_path.stem)
        class_counts = _label_class_counts(label_path)
        class_ids = set(class_counts)
        if image_path is None or not class_ids:
            continue
        records.append({
            "stem": label_path.stem,
            "image": image_path,
            "label": label_path,
            "classes": class_ids,
            "class_counts": class_counts,
        })
    return records


def _unique_destination_pair(target_dir: Path, stem: str, image_suffix: str) -> tuple[Path, Path]:
    image_dir = target_dir / "images"
    label_dir = target_dir / "labels"
    candidate_stem = stem
    counter = 1
    while True:
        image_path = image_dir / f"{candidate_stem}{image_suffix.lower()}"
        label_path = label_dir / f"{candidate_stem}.txt"
        if not image_path.exists() and not label_path.exists():
            return image_path, label_path
        candidate_stem = f"{stem}_rebalance{counter}"
        counter += 1


def _clear_label_caches() -> None:
    for cache_path in (ROOT / "datasets").rglob("*.cache"):
        cache_path.unlink(missing_ok=True)


def rebalance_existing_splits(class_names: list[str] | None = None, dry_run: bool = False) -> None:
    """Repair train/val folders so classes from merged zips are represented in val."""
    header("Class-aware validation rebalance")
    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        print("  train/val folders not found. Skipping.")
        return

    train_records = _split_records(TRAIN_DIR)
    val_records = _split_records(VAL_DIR)
    train_counts = Counter()
    val_counts = Counter()
    total_counts = Counter()
    train_box_counts = Counter()
    val_box_counts = Counter()
    total_box_counts = Counter()
    for record in train_records:
        train_counts.update(record["classes"])
        total_counts.update(record["classes"])
        train_box_counts.update(record["class_counts"])
        total_box_counts.update(record["class_counts"])
    for record in val_records:
        val_counts.update(record["classes"])
        total_counts.update(record["classes"])
        val_box_counts.update(record["class_counts"])
        total_box_counts.update(record["class_counts"])

    needs = {
        cid: (_target_val_images(total), _target_val_boxes(total_box_counts[cid]))
        for cid, total in total_counts.items()
        if (
            val_counts[cid] < _target_val_images(total)
            or val_box_counts[cid] < _target_val_boxes(total_box_counts[cid])
        )
    }
    if not needs:
        print("  Validation split already has class coverage.")
        return

    class_to_train_records: dict[int, list[dict]] = defaultdict(list)
    for record in train_records:
        for cid in record["classes"]:
            class_to_train_records[cid].append(record)
    for records in class_to_train_records.values():
        records.sort(key=lambda record: (len(record["classes"]), record["stem"]))

    moved_record_ids: set[int] = set()
    moved = 0
    for class_id, (target_images, target_boxes) in sorted(needs.items()):
        while val_counts[class_id] < target_images or val_box_counts[class_id] < target_boxes:
            still_needed = {
                cid
                for cid, total in total_counts.items()
                if (
                    val_counts[cid] < _target_val_images(total)
                    or val_box_counts[cid] < _target_val_boxes(total_box_counts[cid])
                )
            }
            candidate = None
            best_key = None
            for record in class_to_train_records.get(class_id, []):
                if id(record) in moved_record_ids:
                    continue
                blocked = False
                for cid in record["classes"]:
                    cid_needs_val = cid in still_needed
                    if not cid_needs_val:
                        continue
                    if train_counts[cid] <= _min_train_images(total_counts[cid]):
                        blocked = True
                        break
                    if train_box_counts[cid] - record["class_counts"][cid] < _min_train_boxes(total_box_counts[cid]):
                        blocked = True
                        break
                if blocked:
                    continue
                key = (-len(record["classes"] & still_needed), len(record["classes"]), record["stem"])
                if best_key is None or key < best_key:
                    candidate = record
                    best_key = key

            if candidate is None:
                label = class_names[class_id] if class_names and class_id < len(class_names) else str(class_id)
                print(
                    f"  WARN: could not create val sample for {label!r} "
                    f"(images {val_counts[class_id]}/{target_images}, boxes {val_box_counts[class_id]}/{target_boxes})."
                )
                break
            record = candidate
            if not dry_run:
                dst_img, dst_lbl = _unique_destination_pair(
                    VAL_DIR, record["stem"], record["image"].suffix
                )
                shutil.move(str(record["image"]), str(dst_img))
                shutil.move(str(record["label"]), str(dst_lbl))
                record["image"] = dst_img
                record["label"] = dst_lbl
            moved_record_ids.add(id(record))
            val_records.append(record)
            train_counts.subtract(record["classes"])
            val_counts.update(record["classes"])
            train_box_counts.subtract(record["class_counts"])
            val_box_counts.update(record["class_counts"])
            moved += 1

    if moved:
        if not dry_run:
            _clear_label_caches()
        print(f"  Moved {moved} image/label pair{'s' if moved != 1 else ''} from train to val.")
    else:
        print("  No pairs moved.")

    missing = []
    for cid, total in sorted(total_counts.items()):
        target_images = _target_val_images(total)
        target_boxes = _target_val_boxes(total_box_counts[cid])
        if (
            (target_images and val_counts[cid] < target_images)
            or (target_boxes and val_box_counts[cid] < target_boxes)
        ):
            label = class_names[cid] if class_names and cid < len(class_names) else str(cid)
            missing.append(
                f"{label} img {val_counts[cid]}/{target_images}, box {val_box_counts[cid]}/{target_boxes}"
            )
    if missing:
        print("  Remaining low-val classes: " + ", ".join(missing[:20]))
    else:
        print("  All classes that can meet the target now have validation image/box coverage.")


def create_splits(classes: dict[int, str], label_map: dict[str, str], dry_run: bool = False) -> None:
    header("Creating train/val splits")
    seen = {}
    allowed_classes = {canonical_label(name) for name in classes.values()}
    for d in OI_OUTPUT_DIR.iterdir():
        if not d.is_dir(): continue
        if canonical_label(d.name.replace("_", " ")) not in allowed_classes:
            continue
        for f in _source_image_files(d, include_roboflow_previews=False):
            seen.setdefault(f.stem, f)

    items = list(seen.items())
    random.shuffle(items)
    split = int(len(items) * TRAIN_SPLIT)
    train_items, val_items = items[:split], items[split:]
    train_items, val_items, moved_for_balance = _rebalance_item_split(train_items, val_items, label_map)
    print(f"  Unique images: {len(items)} | Train: {len(train_items)} | Val: {len(val_items)}")
    if moved_for_balance:
        print(f"  Class-aware rebalance moved {moved_for_balance} source images into val.")

    if dry_run: return

    for d in [TRAIN_DIR, VAL_DIR]:
        if d.exists(): shutil.rmtree(d)
        (d / "images").mkdir(parents=True)
        (d / "labels").mkdir(parents=True)

    def copy(items_sub, target, prefix):
        for i, (stem, src) in enumerate(items_sub):
            shutil.copy2(src, target / "images" / f"{prefix}_{stem}{src.suffix}")
            lbl = target / "labels" / f"{prefix}_{stem}.txt"
            lbl.write_text(label_map.get(stem, "0 0.500000 0.500000 0.900000 0.900000\n"), encoding="utf-8")
            if (i+1) % 2000 == 0: print(f"    {prefix}: {i+1}/{len(items_sub)}")

    print("  Copying training set...")
    copy(train_items, TRAIN_DIR, "train")
    print("  Copying validation set...")
    copy(val_items, VAL_DIR, "val")

    # Write data.yaml
    yaml_lines = [
        f"train: {TRAIN_DIR.resolve()}\n",
        f"val: {VAL_DIR.resolve()}\n\n",
        f"nc: {len(classes)}\n",
        "names:\n",
    ]
    for idx, name in classes.items():
        yaml_lines.append(f"  {idx}: {name!r}\n")
    DATA_YAML.parent.mkdir(parents=True, exist_ok=True)
    DATA_YAML.write_text("".join(yaml_lines), encoding="utf-8")
    print(f"  data.yaml written ({len(classes)} classes)")
    rebalance_existing_splits([classes[idx] for idx in sorted(classes)], dry_run=False)
    print(f"\n  Ready for Colab! Zip datasets/ and upload.")


def run_labels() -> None:
    print(f"\n{'=' * 60}")
    print("  LABEL GENERATION + SPLITS")
    print(f"{'=' * 60}")
    classes = _scan_classes()
    if not classes: sys.exit(1)
    label_map = generate_labels(classes)
    create_splits(classes, label_map)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    global DRY_RUN
    dry_run = "--dry-run" in sys.argv
    DRY_RUN = dry_run
    coco_only = "--coco-only" in sys.argv
    oi_only = "--oi-only" in sys.argv
    labels_only = "--labels" in sys.argv
    fix_splits = "--fix-splits" in sys.argv
    no_roboflow = "--no-roboflow" in sys.argv
    force_roboflow_merge = "--force-roboflow-merge" in sys.argv
    roboflow_source = _arg_value("--roboflow", "--roboflow-url", "--roboflow-zip")
    roboflow_dir_raw = _arg_value("--roboflow-dir")
    roboflow_dir = Path(roboflow_dir_raw).resolve() if roboflow_dir_raw else None
    explicit_roboflow_dir = roboflow_dir is not None
    if roboflow_dir and not roboflow_dir.is_dir():
        print(f"  Roboflow directory not found: {roboflow_dir}")
        sys.exit(1)

    if dry_run:
        print("\n  DRY RUN - no downloads or writes\n")

    if roboflow_source:
        run_roboflow_import(roboflow_source)
        return

    if fix_splits:
        class_names = _read_class_list() if DATA_YAML.exists() else []
        rebalance_existing_splits(class_names, dry_run=dry_run)
        return

    rebuild_previews = "--rebuild-previews" in sys.argv
    if rebuild_previews:
        class_names = _read_class_list() if DATA_YAML.exists() else []
        rebuild_preview_images(class_names)
        return

    default_pipeline = not any([coco_only, oi_only, labels_only, explicit_roboflow_dir])
    if default_pipeline and not no_roboflow:
        auto_source, source_label = _configured_roboflow_source()
        if auto_source:
            print(f"  Auto dataset source: {source_label}")
            run_roboflow_import(auto_source)
            return
        if DEFAULT_ROBOFLOW_DIR.is_dir() and any(DEFAULT_ROBOFLOW_DIR.glob("*.zip")):
            roboflow_dir = DEFAULT_ROBOFLOW_DIR.resolve()
            print(f"  Auto Roboflow merge dir: {roboflow_dir}")

    if labels_only:
        classes = _scan_classes()
        if not classes: sys.exit(1)
        label_map = generate_labels(classes, dry_run=dry_run)
        create_splits(classes, label_map, dry_run=dry_run)
        return

    if roboflow_dir and DATA_YAML.exists() and TRAIN_DIR.exists():
        run_roboflow_merge(roboflow_dir, force=force_roboflow_merge)
        return

    if oi_only:
        run_openimages()
    elif coco_only:
        run_coco()
    else:
        run_coco()
        run_openimages()

    if dry_run:
        if roboflow_dir:
            run_roboflow_merge(roboflow_dir, force=force_roboflow_merge)
        return

    run_labels()

    if roboflow_dir:
        run_roboflow_merge(roboflow_dir, force=force_roboflow_merge)


if __name__ == "__main__":
    main()
