"""Benchmark runner — callable from CLI script or FastAPI endpoint."""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
SPEED_DIR = ROOT / "datasets" / "coco" / "images_by_class"
TRAIN_DIR = ROOT / "datasets" / "train"
VAL_DIR = ROOT / "datasets" / "val"
VAL_YAML = ROOT / "datasets" / "data.yaml"
MODEL_PATH = ROOT / "yolov8n.pt"
FINETUNED_PATH = ROOT / "best.pt"

MAX_IMAGES_PER_CLASS = 6
MAX_SPEED_IMAGES = 2000
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


class BenchmarkRunner:
    def __init__(self, model_path: str | Path = MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = YOLO(str(self.model_path))
        self.results: dict = {}
        self.progress: dict = {"status": "idle", "current": 0, "total": 0, "message": ""}

    # ── Speed benchmark (images_by_class) ──────────────────────────────

    def _read_class_names(self) -> list[str]:
        if not VAL_YAML.exists():
            return []
        import yaml
        with open(VAL_YAML) as f:
            cfg = yaml.safe_load(f)
        names = cfg.get("names", [])
        if isinstance(names, dict):
            return [names[i] for i in sorted(int(k) for k in names.keys())]
        return list(names)

    def _collect_images(self, all_classes: bool = False) -> list[tuple[Path, str]]:
        items: list[tuple[Path, str]] = []
        limit = MAX_SPEED_IMAGES if all_classes else None

        # Collect from images_by_class (COCO + OI classes)
        if SPEED_DIR.exists():
            for d in sorted(SPEED_DIR.iterdir()):
                if not d.is_dir(): continue
                imgs = sorted(d.glob("*.jpg")) + sorted(d.glob("*.jpeg"))
                for p in imgs[:MAX_IMAGES_PER_CLASS]:
                    items.append((p, d.name.replace("_", " ")))

        # Collect from val set (Roboflow classes not in images_by_class)
        existing = {d.name for d in SPEED_DIR.iterdir() if d.is_dir()} if SPEED_DIR.exists() else set()
        class_names = self._read_class_names()
        label_dir = VAL_DIR / "labels"
        image_dir = VAL_DIR / "images"
        if label_dir.exists() and image_dir.exists():
            per_class: dict[str, list[Path]] = {}
            for lbl_path in sorted(label_dir.glob("*.txt")):
                img_stem = lbl_path.stem
                for ext in [".jpg", ".jpeg", ".png"]:
                    img = image_dir / f"{img_stem}{ext}"
                    if img.exists(): break
                else:
                    continue
                for line in lbl_path.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split()
                    if not parts: continue
                    try: cid = int(float(parts[0]))
                    except ValueError: continue
                    if 0 <= cid < len(class_names):
                        name = class_names[cid]
                        folder = name.replace(" ", "_")
                        if folder in existing: continue
                        per_class.setdefault(folder, []).append(img)
            for folder in sorted(per_class):
                for p in per_class[folder][:MAX_IMAGES_PER_CLASS]:
                    items.append((p, folder.replace("_", " ")))

        if limit and len(items) > limit:
            import random
            random.shuffle(items)
            items = items[:limit]

        return items

    def _speed_benchmark(self, all_classes: bool = False, on_progress: Callable | None = None) -> dict:
        images = self._collect_images(all_classes=all_classes)
        if not images:
            return {"error": "No images found for speed benchmark"}

        latencies: list[float] = []
        fps_samples: list[float] = []
        total_detections = 0
        images_with_objects = 0
        per_class: dict[str, int] = defaultdict(int)
        class_names = self.model.names if hasattr(self.model, "names") else {}

        total_start = time.perf_counter()
        for idx, (img_path, _) in enumerate(images):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            start_t = time.perf_counter()
            results = self.model.predict(
                source=img, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                imgsz=640, verbose=False,
            )
            elapsed_ms = (time.perf_counter() - start_t) * 1000
            latencies.append(elapsed_ms)
            fps_samples.append(1000 / elapsed_ms if elapsed_ms > 0 else 0)

            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_id = int(box.cls.item())
                    label = class_names.get(cls_id, "unknown")
                    per_class[label] += 1
                total_detections += len(results[0].boxes)
                if len(results[0].boxes):
                    images_with_objects += 1

            if (idx + 1) % 50 == 0 or idx == len(images) - 1:
                msg = f"Speed: {idx+1}/{len(images)}"
                self.progress = {"status": "running", "current": idx + 1, "total": len(images), "message": msg}
                if on_progress:
                    on_progress(idx + 1, len(images), msg)

        total_elapsed = time.perf_counter() - total_start

        return {
            "num_images": len(images),
            "images_with_objects": images_with_objects,
            "total_detections": total_detections,
            "avg_latency_ms": round(float(np.mean(latencies)), 1) if latencies else 0,
            "p50_latency_ms": round(float(np.percentile(latencies, 50)), 1) if latencies else 0,
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 1) if latencies else 0,
            "p99_latency_ms": round(float(np.percentile(latencies, 99)), 1) if latencies else 0,
            "avg_fps": round(float(np.mean(fps_samples)), 1) if fps_samples else 0,
            "overall_fps": round(len(images) / total_elapsed, 1) if total_elapsed > 0 else 0,
            "total_time_s": round(total_elapsed, 1),
        }

    # ── Accuracy evaluation (val set, proper mAP) ──────────────────────

    def _accuracy_eval(self, on_progress: Callable | None = None) -> dict:
        import shutil, yaml

        model_nc = len(self.model.names) if hasattr(self.model, "names") else 0

        if not VAL_YAML.exists():
            return {"error": "data.yaml not found"}

        with open(VAL_YAML) as f:
            cfg = yaml.safe_load(f)
        ds_nc = int(cfg.get("nc", 0))

        if model_nc not in (80, ds_nc):
            return {
                "note": f"Model ({model_nc} classes) unexpected — expected 80 or {ds_nc}",
                "model_classes": model_nc,
                "dataset_classes": ds_nc,
                "eval_dataset": f"project validation ({ds_nc} classes)",
                "mAP50": None, "mAP50_95": None,
                "precision": None, "recall": None,
            }

        if on_progress:
            on_progress(0, 0, "Running mAP evaluation...")

        try:
            # For baseline (80 cls): filter to COCO80 classes only
            if model_nc == 80 and ds_nc > 80:
                tmp_dir = VAL_YAML.parent / ".benchmark_tmp"
                tmp_dir.mkdir(parents=True, exist_ok=True)

                tmp_yaml = tmp_dir / "data_coco80.yaml"
                tmp_labels = tmp_dir / "labels"
                tmp_labels.mkdir(parents=True, exist_ok=True)

                # Write filtered data.yaml with only 80 COCO classes
                names = cfg.get("names", [])
                if isinstance(names, dict):
                    names = [names[i] for i in sorted(int(k) for k in names.keys())]
                coco80_names = names[:80]
                yaml_lines = [
                    f"train: {cfg.get('train', '')}\n",
                    f"val: {tmp_dir.resolve()}/images\n\n",
                    "nc: 80\n",
                    "names:\n",
                ]
                for idx, name in enumerate(coco80_names):
                    yaml_lines.append(f"  {idx}: {name!r}\n")
                tmp_yaml.write_text("".join(yaml_lines), encoding="utf-8")

                # Symlink or copy images
                val_img_dir = VAL_DIR / "images"
                tmp_img = tmp_dir / "images"
                if not tmp_img.exists():
                    try:
                        import os
                        os.symlink(str(val_img_dir.resolve()), str(tmp_img.resolve()))
                    except (OSError, AttributeError):
                        shutil.copytree(val_img_dir, tmp_img, dirs_exist_ok=True)

                # Filter labels: keep only class IDs 0-79
                src_label_dir = VAL_DIR / "labels"
                for lbl_path in src_label_dir.glob("*.txt"):
                    lines = []
                    for line in lbl_path.read_text(encoding="utf-8").splitlines():
                        parts = line.strip().split()
                        if not parts: continue
                        try: cid = int(float(parts[0]))
                        except ValueError: continue
                        if cid >= 80: continue
                        lines.append(line)
                    if lines:
                        (tmp_labels / lbl_path.name).write_text("\n".join(lines), encoding="utf-8")

                val_yaml = str(tmp_yaml)
                eval_dataset = "COCO80 subset of project validation"
            else:
                val_yaml = str(VAL_YAML)
                eval_dataset = f"project validation ({ds_nc} classes)"

            val_results = self.model.val(data=val_yaml, imgsz=640, verbose=False)
            box = val_results.box
            return {
                "model_classes": model_nc,
                "dataset_classes": ds_nc,
                "eval_dataset": eval_dataset,
                "mAP50": round(float(box.map50), 4),
                "mAP50_95": round(float(box.map), 4),
                "precision": round(float(box.mp), 4),
                "recall": round(float(box.mr), 4),
            }
        except Exception as e:
            return {"error": f"mAP evaluation failed: {e}"}
        finally:
            tmp_dir = VAL_YAML.parent / ".benchmark_tmp"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Full run ───────────────────────────────────────────────────────

    def run(self, all_classes: bool = False, on_progress: Callable | None = None) -> dict:
        self.progress = {"status": "running", "current": 0, "total": 0, "message": "Starting..."}

        model_nc = len(self.model.names) if hasattr(self.model, "names") else 0
        speed = self._speed_benchmark(all_classes=all_classes, on_progress=on_progress)
        self.progress["message"] = "Evaluating accuracy..."
        if on_progress:
            on_progress(0, 0, "Evaluating accuracy...")
        accuracy = self._accuracy_eval(on_progress)

        result = {
            "status": "completed",
            "model": self.model_path.name,
            "num_classes": model_nc,
            "model_task": "COCO pretrained baseline" if model_nc == 80 else "expanded custom detector",
            **speed,
            "accuracy": accuracy,
        }
        self.results = result
        self.progress = {"status": "completed", "current": 100, "total": 100, "message": "Done"}
        return result

    def get_progress(self) -> dict:
        return self.progress

    def get_latest_results(self) -> dict:
        return self.results


_runners: dict[str, BenchmarkRunner] = {}
CACHE_FILE = ROOT / ".benchmark_cache.json"


def _result_class_count(result: dict, default: int | None = None) -> int | None:
    if not isinstance(result, dict):
        return default
    accuracy = result.get("accuracy") if isinstance(result.get("accuracy"), dict) else {}
    value = result.get("num_classes") or accuracy.get("model_classes") or default
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return default


def _project_dataset_class_count() -> int | None:
    if not VAL_YAML.exists():
        return None
    try:
        for line in VAL_YAML.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("nc:"):
                return int(line.split(":", 1)[1].strip())
    except Exception:
        return None
    return None


def build_comparison_meta(results: dict) -> dict:
    baseline = results.get("baseline", {}) if isinstance(results, dict) else {}
    finetuned = results.get("finetuned", {}) if isinstance(results, dict) else {}
    baseline_default = 80 if baseline.get("model") == MODEL_PATH.name else None
    finetuned_default = _project_dataset_class_count() if finetuned.get("model") == FINETUNED_PATH.name else None
    baseline_classes = _result_class_count(baseline, baseline_default)
    finetuned_classes = _result_class_count(finetuned, finetuned_default)
    same_dataset = baseline.get("accuracy", {}).get("eval_dataset") == finetuned.get("accuracy", {}).get("eval_dataset")
    different_scopes = bool(baseline_classes and finetuned_classes and baseline_classes != finetuned_classes)

    return {
        "baseline_classes": baseline_classes,
        "finetuned_classes": finetuned_classes,
        "baseline_label": f"Base ({baseline_classes} cls)" if baseline_classes else "Base",
        "finetuned_label": f"Tuned ({finetuned_classes} cls)" if finetuned_classes else "Tuned",
        "comparison_scope": "same-class-set" if same_dataset else "different-class-sets",
        "warning": (
            "Baseline knows only 80 COCO classes; mAP computed only on those 80 within the 122-class val set."
            if same_dataset and different_scopes else
            ("Same validation dataset used for both." if same_dataset else
             "Models evaluated on different datasets — mAP not directly comparable.")
        ),
        "recommended_scope": (
            "For fair comparison, compare only the 80 shared COCO classes. "
            "Tuned model's additional 42 classes only appear in its own metrics."
        ),
    }


def attach_comparison_meta(output: dict) -> dict:
    if not isinstance(output, dict):
        return output
    comparison = output.get("comparison")
    if isinstance(comparison, dict):
        output["meta"] = build_comparison_meta(comparison)
    return output


def get_runner(model_path: str | Path = MODEL_PATH, force_new: bool = False) -> BenchmarkRunner:
    key = str(Path(model_path).resolve())
    if force_new or key not in _runners:
        _runners[key] = BenchmarkRunner(model_path)
    return _runners[key]


def compare_models(on_progress: Callable | None = None, force: bool = False) -> dict:
    """Run benchmark on both baseline and fine-tuned, return comparison.
    
    Results are cached to disk — subsequent calls return cached data.
    Pass force=True to re-run.
    """
    if not force and CACHE_FILE.exists():
        import json
        try:
            cached = json.loads(CACHE_FILE.read_text())
            return attach_comparison_meta(cached)
        except Exception:
            pass

    results = {}
    for label, path in [("baseline", MODEL_PATH), ("finetuned", FINETUNED_PATH)]:
        if not Path(path).exists():
            results[label] = {"error": f"Not found: {path.name}"}
            continue
        runner = get_runner(path, force_new=True)

        def cb(label=label):
            def wrapper(current, total, msg):
                if on_progress:
                    on_progress(current, total, f"{label}: {msg}")
            return wrapper

        r = runner.run(on_progress=cb())
        results[label] = r

    output = attach_comparison_meta({"comparison": results})

    try:
        import json
        CACHE_FILE.write_text(json.dumps(output, indent=2))
    except Exception:
        pass

    return output
