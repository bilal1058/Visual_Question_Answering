from __future__ import annotations

import argparse
import base64
import os
import re
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from ultralytics import YOLO

from src.class_taxonomy import (
    CATEGORY_GROUPS,
    HAZARDS,
    VEHICLES,
    ANIMALS,
    assistive_alert,
    canonical_label,
    category_for,
)

load_dotenv()

# Theme colors for annotated output. OpenCV uses BGR.
_BOX_OUTER = (8, 12, 18)
_BOX_INNER = (255, 207, 0)          # cyan-lime
_LABEL_BORDER = (0, 207, 255)       # electric cyan
_LABEL_FILL = (8, 20, 40)
_TEXT_BRIGHT = (232, 245, 255)

# ─── Color Detection ────────────────────────────────────────────────────────
# Named HSV ranges for chromatic colors.
# Each entry: list of ([h_lo, s_lo, v_lo], [h_hi, s_hi, v_hi]) bands.
_COLOR_RANGES: dict[str, list[tuple[list[int], list[int]]]] = {
    "red":    [([0,   100, 60],  [10,  255, 255]), ([165, 100, 60],  [180, 255, 255])],
    "orange": [([10,  100, 60],  [22,  255, 255])],
    "yellow": [([22,   80, 60],  [35,  255, 255])],
    "green":  [([35,   55, 40],  [90,  255, 255])],
    "blue":   [([90,   65, 40],  [130, 255, 255])],
    "purple": [([130,  50, 45],  [165, 255, 255])],
    "pink":   [([155,  40, 140], [175, 220, 255]), ([0,   30, 170], [12,  130, 255])],
    "brown":  [([8,    80, 20],  [24,  200, 160])],
    # Warm tan / beige / cream — critical for animal fur detection
    "tan":    [([15,   25, 130], [32,  120, 230])],
    "cream":  [([18,   10, 180], [35,   65, 255])],
    "tawny":  [([10,   60, 100], [22,  180, 210])],
}


def get_groq_client() -> Groq | None:
    api_key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=api_key) if api_key else None


def build_scene_description(counts: Counter[str], max_classes: int = 16) -> str:
    if not counts:
        return "no objects detected"
    normalized = Counter()
    for label, total in counts.items():
        normalized[canonical_label(label)] += total
    parts = []
    sorted_counts = sorted(normalized.items(), key=lambda item: (-item[1], item[0]))
    for label, total in sorted_counts[:max_classes]:
        suffix = "s" if total != 1 else ""
        parts.append(f"{total} {label}{suffix}")
    hidden = len(sorted_counts) - max_classes
    if hidden > 0:
        parts.append(f"{hidden} other classes")
    return ", ".join(parts)


# ── Improved color extraction ──────────────────────────────────────────────
# Strategy:
#   1. Use k-means (k=4) to find the dominant pixel clusters in the crop.
#   2. For each cluster center convert BGR→HSV and look up the color name.
#   3. Weight by cluster size; skip near-neutral clusters unless they dominate.
#   This is far more robust than scanning all pixels against HSV ranges because
#   it is not misled by sparse noise or mixed-texture backgrounds.

def _bgr_to_color_name(bgr: np.ndarray) -> str:
    """Map a single BGR pixel (shape (3,) float32) to a color name string."""
    pixel = bgr.reshape(1, 1, 3).astype(np.uint8)
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    # Neutral tones — decided by saturation / value alone
    if v < 45:
        return "black"
    if v > 210 and s < 50:
        return "white"
    if s < 38:
        if v < 85:
            return "dark gray"
        if v > 185:
            return "light gray"
        return "gray"

    # Chromatic — scan named ranges
    best_name = "unknown"
    for color_name, bands in _COLOR_RANGES.items():
        for lo, hi in bands:
            if lo[0] <= h <= hi[0] and lo[1] <= s <= hi[1] and lo[2] <= v <= hi[2]:
                return color_name
    return best_name


def extract_dominant_color(crop_bgr: np.ndarray) -> str:
    """Return a human-readable dominant color for a BGR crop using k-means."""
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown"
    h, w = crop_bgr.shape[:2]
    if h < 8 or w < 8:
        return "unknown"

    # Focus on inner 70% to reduce background bleed
    y1, y2 = int(h * 0.15), int(h * 0.85)
    x1, x2 = int(w * 0.15), int(w * 0.85)
    focused = crop_bgr[y1:y2, x1:x2]

    # Resize to small fixed canvas for speed
    small = cv2.resize(focused, (48, 48), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3).astype(np.float32)

    # k-means with k=4 clusters — handles multi-color animals well
    k = min(4, pixels.shape[0])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
    try:
        _, labels, centers = cv2.kmeans(
            pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
    except cv2.error:
        return "unknown"

    label_array = labels.flatten()
    total = len(label_array)
    cluster_sizes = [int(np.count_nonzero(label_array == i)) for i in range(k)]

    # Score each cluster: weight = size fraction, ignore near-black clusters
    scored: list[tuple[float, str]] = []
    for idx, (size, center) in enumerate(zip(cluster_sizes, centers)):
        fraction = size / total
        if fraction < 0.06:
            continue
        name = _bgr_to_color_name(center)
        # Down-weight very dark clusters unless they truly dominate
        if name in ("black", "dark gray") and fraction < 0.55:
            fraction *= 0.5
        scored.append((fraction, name))

    if not scored:
        return "unknown"

    scored.sort(reverse=True)
    top_name = scored[0][1]

    # Build a two-color answer when two distinct colors are both prominent
    if len(scored) >= 2:
        second_name = scored[1][1]
        second_frac = scored[1][0]
        top_frac = scored[0][0]
        if (
            second_name != top_name
            and second_frac >= 0.22
            and top_frac - second_frac < 0.35
            and second_name not in ("unknown",)
        ):
            return f"{top_name} and {second_name}"

    return top_name if top_name != "unknown" else "unknown"


# ─── Position / size helpers ────────────────────────────────────────────────

def _position_label(cx: float, cy: float, width: int, height: int) -> str:
    parts = []
    if cy < height * 0.33:
        parts.append("top")
    elif cy > height * 0.67:
        parts.append("bottom")
    if cx < width * 0.33:
        parts.append("left")
    elif cx > width * 0.67:
        parts.append("right")
    return "-".join(parts) if parts else "center"


def _size_category(box_area: float, image_area: float) -> str:
    ratio = box_area / image_area if image_area > 0 else 0
    if ratio > 0.25:
        return "large"
    if ratio > 0.06:
        return "medium"
    return "small"


# ─── Deduplication ──────────────────────────────────────────────────────────

def _iou(a: list[int], b: list[int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center_distance(a: list[int], b: list[int], diagonal: float) -> float:
    ax = (a[0] + a[2]) / 2.0
    ay = (a[1] + a[3]) / 2.0
    bx = (b[0] + b[2]) / 2.0
    by = (b[1] + b[3]) / 2.0
    dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    return dist / diagonal if diagonal > 0 else 1.0


def _raw_detections(result, class_names: dict[int, str]) -> list[dict]:
    detections: list[dict] = []
    if result.boxes is None:
        return detections
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        detections.append(
            {
                "label": canonical_label(class_names[int(box.cls.item())]),
                "conf": float(box.conf.item()),
                "xyxy": [x1, y1, x2, y2],
            }
        )
    return detections


def _dedupe_detections(
    detections: list[dict], image_shape: tuple[int, int, int]
) -> list[dict]:
    """
    Remove redundant boxes for the same class.

    Thresholds tightened vs. the original:
      - IoU ≥ 0.40  (was 0.55) → aggressively merges heavily overlapping boxes
      - IoU ≥ 0.22 AND centre distance < 0.06 (was 0.32 / 0.05) → catches
        near-duplicate predictions with slight offset
    This prevents the same object from appearing multiple times in the output.
    """
    if not detections:
        return []

    height, width = image_shape[:2]
    diagonal = float((height ** 2 + width ** 2) ** 0.5)
    kept: list[dict] = []

    for det in sorted(detections, key=lambda item: item["conf"], reverse=True):
        x1, y1, x2, y2 = det["xyxy"]
        if x2 <= x1 or y2 <= y1:
            continue

        duplicate = False
        for existing in kept:
            if existing["label"] != det["label"]:
                continue
            overlap = _iou(det["xyxy"], existing["xyxy"])
            center_gap = _center_distance(det["xyxy"], existing["xyxy"], diagonal)
            # More aggressive dedup: catch same-object multi-detection
            if overlap >= 0.35 or (overlap >= 0.18 and center_gap < 0.10):
                duplicate = True
                break

        if not duplicate:
            kept.append(det)

    return kept


def _collect_detections(
    image: np.ndarray,
    result,
    class_names: dict[int, str],
    pad: int = 8,
) -> list[dict]:
    raw = _raw_detections(result, class_names)
    clean = _dedupe_detections(raw, image.shape)

    height, width = image.shape[:2]
    for det in clean:
        x1, y1, x2, y2 = det["xyxy"]
        crop = image[
            max(0, y1 - pad): min(height, y2 + pad),
            max(0, x1 - pad): min(width, x2 + pad),
        ].copy()
        det["crop_bgr"] = crop
        # FIX: use improved k-means color detection (not YOLO labels)
        det["color"] = extract_dominant_color(crop)

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        det["position"] = _position_label(cx, cy, width, height)

        box_area = max(0, x2 - x1) * max(0, y2 - y1)
        det["size"] = _size_category(box_area, width * height)

    return clean


# ─── Attribute enrichment ───────────────────────────────────────────────────

def enrich_crops_with_attributes(
    crops: list[dict], image_shape: tuple[int, int, int]
) -> list[dict]:
    height, width = image_shape[:2]
    image_area = width * height
    for crop in crops:
        if "crop_bgr" in crop:
            crop["color"] = extract_dominant_color(crop["crop_bgr"])
        x1, y1, x2, y2 = crop["xyxy"]
        crop["position"] = _position_label(
            (x1 + x2) / 2.0, (y1 + y2) / 2.0, width, height
        )
        crop["size"] = _size_category(
            max(0, x2 - x1) * max(0, y2 - y1), image_area
        )
    return crops


# ─── Spatial reasoning ──────────────────────────────────────────────────────

def compute_spatial_relationships(
    crops: list[dict], image_shape: tuple[int, int, int]
) -> list[str]:
    if len(crops) < 2:
        return []

    height, width = image_shape[:2]
    diagonal = float((height ** 2 + width ** 2) ** 0.5)
    relationships: list[str] = []

    for index, left in enumerate(crops[:8]):
        for right in crops[index + 1: 8]:
            overlap = _iou(left["xyxy"], right["xyxy"])
            distance = _center_distance(left["xyxy"], right["xyxy"], diagonal)

            if overlap > 0.28:
                relationships.append(f"{left['label']} overlaps {right['label']}")
            elif distance < 0.14:
                relationships.append(f"{left['label']} is near {right['label']}")

            if left["xyxy"][3] < right["xyxy"][1] and distance < 0.22:
                relationships.append(f"{left['label']} is above {right['label']}")

    return relationships[:10]


def build_rich_scene_description(
    crops: list[dict],
    relationships: list[str],
    max_classes: int = 12,
    max_examples_per_class: int = 2,
) -> str:
    if not crops:
        return "No objects detected in this frame."

    grouped: dict[str, list[dict]] = {}
    for crop in crops:
        crop["label"] = canonical_label(crop["label"])
        grouped.setdefault(crop["label"], []).append(crop)

    parts = []
    category_parts = []
    for category in ["hazards", "vehicles", "indoor", "personal items", "food", "accessibility"]:
        labels = [
            label for label in CATEGORY_GROUPS.get(category, [])
            if label in grouped
        ]
        if labels:
            category_parts.append(
                f"{category}: " + ", ".join(f"{len(grouped[label])} {label}" for label in labels)
            )

    sorted_groups = sorted(
        grouped.items(), key=lambda item: (-len(item[1]), item[0])
    )
    for label, items in sorted_groups[:max_classes]:
        object_bits = []
        for item in items[:max_examples_per_class]:
            attrs = []
            if item.get("color") and item["color"] != "unknown":
                attrs.append(item["color"])
            if item.get("size"):
                attrs.append(item["size"])
            if item.get("position"):
                attrs.append(item["position"])
            if attrs:
                object_bits.append(f"{label} ({', '.join(attrs)})")
            else:
                object_bits.append(label)

        total = len(items)
        name = f"{label}s" if total != 1 else label
        parts.append(f"{total} {name}: " + "; ".join(object_bits))

    summary = ""
    if category_parts:
        summary += "ASSISTIVE CATEGORIES: " + " | ".join(category_parts) + "\n"
    summary += "DETECTED OBJECTS: " + " | ".join(parts)
    hidden_groups = sorted_groups[max_classes:]
    if hidden_groups:
        other_text = ", ".join(f"{len(items)} {label}" for label, items in hidden_groups[:10])
        summary += f" | Other lower-priority classes: {other_text}"
    if relationships:
        summary += "\nSPATIAL RELATIONSHIPS: " + ", ".join(relationships)
    return summary


def generate_alerts(crops: list[dict]) -> list[str]:
    counts: Counter[str] = Counter(canonical_label(crop["label"]) for crop in crops)
    alerts: list[str] = []

    for label in HAZARDS:
        if counts.get(label, 0):
            alert = assistive_alert(label, counts[label])
            if alert:
                alerts.append(alert)

    person_count = counts.get("person", 0)
    if person_count > 5:
        alerts.append(f"Crowd detected: {person_count} people in frame.")
    elif person_count > 3:
        alerts.append(f"Group detected: {person_count} people in frame.")

    animals = sum(counts.get(name, 0) for name in ANIMALS)
    vehicles = sum(counts.get(name, 0) for name in VEHICLES)
    if animals > 0 and vehicles > 0:
        alerts.append("Animals and vehicles appear in the same frame.")

    if counts.get("bicycle", 0) or counts.get("motorcycle", 0):
        if person_count:
            alerts.append("Person detected near a rideable vehicle.")

    return alerts


# ─── LLM layer ──────────────────────────────────────────────────────────────

def _history_to_groq_messages(chat_history: list[dict]) -> list[dict]:
    messages = []
    for message in chat_history[-6:]:
        role = "user" if message["role"] == "user" else "assistant"
        messages.append({"role": role, "content": message["content"]})
    return messages


def _image_to_groq_data_url(image_bgr: np.ndarray) -> str:
    max_side = max(320, int(os.getenv("GROQ_VISION_MAX_SIDE", "768")))
    quality = max(45, min(92, int(os.getenv("GROQ_VISION_JPEG_QUALITY", "72"))))
    image = image_bgr
    height, width = image.shape[:2]
    largest_side = max(height, width)
    if largest_side > max_side:
        scale = max_side / largest_side
        image = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def answer_with_groq(
    question: str,
    counts: Counter[str],
    client: Groq,
    chat_history: list[dict] | None = None,
    rich_scene_desc: str | None = None,
    alerts: list[str] | None = None,
    image_bgr: np.ndarray | None = None,
) -> str:
    model_name = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
    is_vision = "llama-4" in model_name.lower() or "vision" in model_name.lower()

    scene_info = rich_scene_desc or f"Detected objects: {build_scene_description(counts)}"
    alert_text = "\n".join(alerts) if alerts else "No active alerts."

    system_content = (
        "You are an assistive computer vision assistant for navigation, safety, "
        "and daily living. Use the image provided to answer accurately.\n\n"
        f"YOLO detected: {scene_info}\n\n"
        f"Alerts: {alert_text}\n\n"
        "Rules:\n"
        "1. Look at the image itself — don't rely only on YOLO detections.\n"
        "2. If the question asks about something YOLO missed, use the image.\n"
        "3. Keep answers concise (1-3 sentences).\n"
    )

    messages: list[dict] = [{"role": "system", "content": system_content}]
    if chat_history:
        messages.extend(_history_to_groq_messages(chat_history))

    if is_vision and image_bgr is not None:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": _image_to_groq_data_url(image_bgr)}},
            ],
        })
    else:
        messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2,
        max_completion_tokens=300,
    )
    return response.choices[0].message.content.strip()


def answer_with_crop_vision(
    question: str,
    crop_bgr: np.ndarray,
    label: str,
    conf: float,
    client: Groq,
    chat_history: list[dict] | None = None,
) -> str:
    messages: list[dict] = []
    if chat_history:
        messages.extend(_history_to_groq_messages(chat_history))
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": _image_to_groq_data_url(crop_bgr)},
                },
                {
                    "type": "text",
                    "text": (
                        f"This crop was detected as {label} with {conf:.0%} confidence. "
                        f"Answer only about this crop: {question}"
                    ),
                },
            ],
        }
    )

    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
        messages=messages,
        temperature=0.2,
        max_completion_tokens=300,
    )
    return response.choices[0].message.content.strip()


def _singularize(word: str) -> str:
    word = word.lower().strip()
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 2:
        return word[:-2]
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word


def _find_target_label(
    question: str, counts: Counter[str], crops: list[dict] | None = None
) -> str | None:
    tokens = re.findall(r"[a-zA-Z]+", question.lower())
    available = set(counts.keys())

    for token in tokens:
        if token in available:
            return token
        singular = _singularize(token)
        if singular in available:
            return singular

    if crops and len(crops) == 1:
        return crops[0]["label"]
    return None


def answer_with_rules(
    question: str, counts: Counter[str], crops: list[dict] | None = None
) -> str:
    q = question.lower().strip().rstrip("?")
    present = [label for label, total in counts.items() if total > 0]
    target = _find_target_label(q, counts, crops)

    if q.startswith("how many "):
        noun = q.replace("how many ", "", 1).replace(" are there", "").strip()
        noun = _singularize(noun)
        total = counts.get(noun, 0)
        suffix = "s" if total != 1 else ""
        return f"There are {total} {noun}{suffix}."

    if q.startswith("is there any "):
        noun = _singularize(q.replace("is there any ", "", 1).strip())
        return "Yes." if counts.get(noun, 0) > 0 else "No."

    if "what color" in q or "which color" in q or "colour" in q:
        if not target:
            return "Please ask about a detected object so I can estimate its color from the crop."
        if not crops:
            return f"{target.capitalize()} was detected, but crop details are not available."
        matching = [crop for crop in crops if crop["label"] == target]
        if not matching:
            return f"No {target} detected in this frame."
        best = max(matching, key=lambda item: item["conf"])
        color = best.get("color", "unknown")
        if color == "unknown":
            return f"I cannot estimate the color of the {target} confidently from this frame."
        return f"The {target} appears {color}."

    if "which objects" in q or "what objects" in q or q == "objects":
        return "Objects present: " + (", ".join(sorted(present)) if present else "none")

    if "total" in q:
        return f"Total detected objects: {sum(counts.values())}"

    if "describe" in q or "summary" in q:
        if not crops:
            return "No objects detected in this frame."
        descriptions = []
        for crop in crops[:4]:
            pieces = [crop["label"]]
            color = crop.get("color")
            if color and color != "unknown":
                pieces.append(color)
            pieces.append(crop.get("position", "center"))
            descriptions.append(" ".join(pieces))
        return "Scene summary: " + ", ".join(descriptions) + "."

    return "Supported prompts: count objects, check presence, ask object colors, list objects, or request a short summary."


def _can_answer_locally(question: str) -> bool:
    q = question.lower().strip().rstrip("?")
    simple_starts = ("how many ", "is there any ")
    if q.startswith(simple_starts):
        return True
    simple_phrases = (
        "what objects",
        "which objects",
        "objects present",
        "detected objects",
        "total",
        "what color",
        "which color",
        "colour",
    )
    return q == "objects" or any(phrase in q for phrase in simple_phrases)


def answer_question(
    question: str,
    counts: Counter[str],
    chat_history: list[dict] | None = None,
    rich_scene_desc: str | None = None,
    alerts: list[str] | None = None,
    crops: list[dict] | None = None,
    image_bgr: np.ndarray | None = None,
) -> str:
    if _can_answer_locally(question):
        return answer_with_rules(question, counts, crops)

    client = get_groq_client()
    if client:
        try:
            return answer_with_groq(
                question, counts, client, chat_history, rich_scene_desc, alerts, image_bgr
            )
        except Exception as exc:
            print(f"[Scene Reasoning] Groq failed: {exc}")
    return answer_with_rules(question, counts, crops)


def answer_focused_crop(
    question: str, crop_info: dict, chat_history: list[dict] | None = None
) -> str:
    client = get_groq_client()
    if client:
        try:
            return answer_with_crop_vision(
                question,
                crop_info["crop_bgr"],
                crop_info["label"],
                crop_info["conf"],
                client,
                chat_history,
            )
        except Exception as exc:
            print(f"[Scene Reasoning] Groq vision failed: {exc}")

    label = crop_info["label"]
    color = crop_info.get("color", "unknown")
    if "color" in question.lower() and color != "unknown":
        return f"The {label} appears {color}."
    return answer_with_rules(question, Counter({label: 1}), [crop_info])


def answer_followup(
    question: str,
    counts: Counter[str] | dict[str, int],
    chat_history: list[dict] | None = None,
    rich_scene_desc: str | None = None,
    alerts: list[str] | None = None,
    crops: list[dict] | None = None,
    image_bgr: np.ndarray | None = None,
) -> str:
    if not isinstance(counts, Counter):
        counts = Counter(counts)
    return answer_question(question, counts, chat_history, rich_scene_desc, alerts, crops, image_bgr)


# ─── Drawing ────────────────────────────────────────────────────────────────

def draw_detections(
    image: np.ndarray, result, class_names: dict[int, str]
) -> Counter[str]:
    counts: Counter[str] = Counter()
    detections = _collect_detections(image, result, class_names)
    if not detections:
        return counts

    height, width = image.shape[:2]
    scale = max(width, height) / 1000.0
    font_scale = max(0.68, scale * 0.90)
    outer_thickness = max(3, int(scale * 3.2))
    inner_thickness = max(2, int(scale * 2.0))
    text_thickness = max(2, int(scale * 1.7))
    pad = max(8, int(scale * 8))

    for det in detections:
        label = det["label"]
        conf = det["conf"]
        x1, y1, x2, y2 = det["xyxy"]
        counts[label] += 1
        text = f"{label}  {conf:.0%}"

        cv2.rectangle(image, (x1, y1), (x2, y2), _BOX_OUTER, outer_thickness)
        cv2.rectangle(image, (x1, y1), (x2, y2), _BOX_INNER, inner_thickness)

        (txt_w, txt_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )

        label_top = max(0, y1 - txt_h - pad * 2)
        label_bottom = min(height - 1, label_top + txt_h + baseline + pad * 2)
        label_right = min(width - 1, x1 + txt_w + pad * 2)

        cv2.rectangle(
            image, (x1, label_top), (label_right, label_bottom), _LABEL_BORDER, cv2.FILLED
        )
        cv2.rectangle(
            image,
            (x1 + 2, label_top + 2),
            (max(x1 + 3, label_right - 2), max(label_top + 3, label_bottom - 2)),
            _LABEL_FILL,
            cv2.FILLED,
        )

        text_y = min(height - baseline - 2, label_top + txt_h + pad)
        cv2.putText(
            image, text, (x1 + pad, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, _TEXT_BRIGHT, text_thickness, cv2.LINE_AA,
        )

    return counts


def extract_crops(
    image: np.ndarray, result, class_names: dict[int, str], pad: int = 8
) -> list[dict]:
    detections = _collect_detections(image, result, class_names, pad)
    detections.sort(key=lambda item: item["conf"], reverse=True)
    return detections


# ─── I/O helpers ────────────────────────────────────────────────────────────

def read_image(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def decode_uploaded_image(file_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode the uploaded image.")
    return image


def run_detection(
    model_or_path: YOLO | str | Path,
    image: np.ndarray,
    question: str,
    conf: float = 0.25,
    chat_history: list[dict] | None = None,
) -> tuple[np.ndarray, str, Counter[str], list[dict], dict[str, int]]:
    total_start = time.perf_counter()

    model = model_or_path if isinstance(model_or_path, YOLO) else YOLO(str(model_or_path))

    yolo_start = time.perf_counter()
    # FIX: lower NMS iou from 0.45 → 0.35 to suppress overlapping duplicate boxes
    results = model.predict(source=image, conf=conf, iou=0.35, imgsz=640, verbose=False)
    yolo_ms = int((time.perf_counter() - yolo_start) * 1000)

    result = results[0]
    rendered = image.copy()
    counts = draw_detections(rendered, result, result.names)
    crops = extract_crops(image, result, result.names)
    crops = enrich_crops_with_attributes(crops, image.shape)
    relationships = compute_spatial_relationships(crops, image.shape)
    rich_desc = build_rich_scene_description(crops, relationships)
    alerts = generate_alerts(crops)

    llm_start = time.perf_counter()
    answer = answer_question(question, counts, chat_history, rich_desc, alerts, crops)
    llm_ms = int((time.perf_counter() - llm_start) * 1000)

    total_ms = int((time.perf_counter() - total_start) * 1000)
    timing = {"yolo_ms": yolo_ms, "llm_ms": llm_ms, "total_ms": total_ms}
    return rendered, answer, counts, crops, timing


def save_result(image: np.ndarray, output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), image)
    return target.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO + Groq scene question answering.")
    parser.add_argument("--model", required=True, help="YOLO weights (.pt).")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--question", required=True, help="Question about the image.")
    parser.add_argument("--output", default="artifacts/prediction.jpg", help="Output image path.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = read_image(args.image)
    rendered, answer, counts, crops, timing = run_detection(
        args.model, image, args.question, args.conf
    )
    saved = save_result(rendered, args.output)
    print("Question :", args.question)
    print("Answer   :", answer)
    print("Counts   :", dict(counts))
    print("Timing   :", timing)
    print(
        "Crops    :",
        [(crop["label"], f"{crop['conf']:.0%}", crop.get("color", "unknown")) for crop in crops],
    )
    print("Saved    :", saved)


if __name__ == "__main__":
    main()
