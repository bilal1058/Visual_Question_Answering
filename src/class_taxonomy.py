"""Assistive class policy shared by dataset setup, model training, and VQA reasoning.

SmartNav 33-Class Assistive Vision Taxonomy for Visually Impaired Navigation:
- TIER 1: Outdoor safety-critical (20 classes)
- TIER 2: Indoor physical hazards (6 classes)
- TIER 3: Personal object locator / context (7 classes)
"""

from __future__ import annotations

FINAL_ASSISTIVE_CLASSES = [
    # Tier 1 — Outdoor safety-critical (20 classes)
    "person",
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "rickshaw",
    "van",
    "ambulance",
    "police_car",
    "pothole",
    "manhole",
    "speed_breaker",
    "cone",
    "stop_sign",
    "traffic_light",
    "crosswalk",
    "pole",
    "dog",
    "cow",
    # Tier 2 — Indoor physical hazards (6 classes)
    "door",
    "stairs",
    "chair",
    "table",
    "bed",
    "couch",
    # Tier 3 — Personal object locator / context (7 classes)
    "white_cane",
    "wheelchair",
    "handrail",
    "backpack",
    "handbag",
    "shelf",
    "sink",
]

TIER_1_OUTDOOR_SAFETY = [
    "person", "car", "bus", "truck", "motorcycle", "bicycle", "rickshaw",
    "van", "ambulance", "police_car", "pothole", "manhole", "speed_breaker",
    "cone", "stop_sign", "traffic_light", "crosswalk", "pole", "dog", "cow"
]

TIER_2_INDOOR_HAZARDS = [
    "door", "stairs", "chair", "table", "bed", "couch"
]

TIER_3_PERSONAL_LOCATORS = [
    "white_cane", "wheelchair", "handrail", "backpack", "handbag", "shelf", "sink"
]

PEOPLE = ["person"]

VEHICLES = [
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "rickshaw",
    "van",
    "ambulance",
    "police_car",
]

HAZARDS = [
    "pothole",
    "manhole",
    "speed_breaker",
    "cone",
    "stop_sign",
    "traffic_light",
    "crosswalk",
    "pole",
    "stairs",
]

INDOOR = [
    "door",
    "stairs",
    "chair",
    "table",
    "bed",
    "couch",
    "shelf",
    "sink",
]

ANIMALS = ["dog", "cow"]

LOCATORS = [
    "white_cane",
    "wheelchair",
    "handrail",
    "backpack",
    "handbag",
    "shelf",
    "sink",
]

CATEGORY_GROUPS = {
    "hazards": HAZARDS,
    "indoor": INDOOR,
    "vehicles": VEHICLES,
    "people": PEOPLE,
    "animals": ANIMALS,
    "locators": LOCATORS,
}


def class_key(name: str) -> str:
    normalized = str(name).replace("_", " ").replace("-", " ").lower()
    return " ".join(normalized.split())


_CANONICAL_BY_KEY = {class_key(name): name for name in FINAL_ASSISTIVE_CLASSES}

CLASS_ALIASES = {
    "dining table": "table",
    "dining-table": "table",
    "traffic_light": "traffic_light",
    "traffic-light": "traffic_light",
    "traffic signal": "traffic_light",
    "traffic light": "traffic_light",
    "stop_sign": "stop_sign",
    "stop sign": "stop_sign",
    "speed_breaker": "speed_breaker",
    "speed-breaker": "speed_breaker",
    "speed breaker": "speed_breaker",
    "speed bump": "speed_breaker",
    "speedbreaker": "speed_breaker",
    "police_car": "police_car",
    "police car": "police_car",
    "white_cane": "white_cane",
    "white cane": "white_cane",
    "wheel chair": "wheelchair",
    "wheelchair": "wheelchair",
    "stairs": "stairs",
    "stair": "stairs",
    "hand rail": "handrail",
    "hand-rail": "handrail",
    "stair railing": "handrail",
    "stairs railing": "handrail",
    "railing": "handrail",
    "zebra crossing": "crosswalk",
    "zebra_crossing": "crosswalk",
    "pedestrian crossing": "crosswalk",
    "sofa": "couch",
}


def canonical_label(name: str) -> str:
    key = class_key(name)
    alias = CLASS_ALIASES.get(key)
    if alias:
        return alias
    return _CANONICAL_BY_KEY.get(key, str(name).replace(" ", "_").strip().lower())


def is_final_class(name: str) -> bool:
    return canonical_label(name) in FINAL_ASSISTIVE_CLASSES


def category_for(label: str) -> str:
    canonical = canonical_label(label)
    for category, labels in CATEGORY_GROUPS.items():
        if canonical in labels:
            return category
    return "objects"


def labels_for_category(category: str, labels: set[str] | list[str]) -> list[str]:
    wanted = set(CATEGORY_GROUPS.get(category, []))
    present = {canonical_label(label) for label in labels}
    return [label for label in CATEGORY_GROUPS.get(category, []) if label in wanted and label in present]


def assistive_alert(label: str, count: int = 1) -> str | None:
    canonical = canonical_label(label)
    if canonical not in HAZARDS and canonical not in {"door", "white_cane"}:
        return None
    if canonical in {"traffic_light", "stop_sign", "crosswalk"}:
        prefix = "Navigation cue ahead"
    elif canonical in {"pothole", "manhole", "speed_breaker", "pole", "stairs"}:
        prefix = "Ground / Path hazard detected"
    else:
        prefix = "Attention"
    suffix = f" ({count})" if count > 1 else ""
    return f"{prefix}: {canonical.replace('_', ' ')}{suffix}."
