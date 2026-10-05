"""
DirectML inference engine for AMD GPUs on Windows.
Uses ONNX Runtime with DirectML to accelerate YOLO on RX 590 etc.

Usage:
    from src.dml_engine import DMLInferenceEngine
    engine = DMLInferenceEngine("yolov8n.onnx")
    results = engine.predict(image)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "yolov8n.pt"
ONNX_PATH = ROOT / "yolov8n.onnx"

# Optional dependency — don't crash if not installed
_HAVE_DML = False
try:
    import onnxruntime  # noqa: F401
    _HAVE_DML = True
except ImportError:
    _HAVE_DML = False

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "mobile", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def export_to_onnx(pt_path: Path = MODEL_PATH, onnx_path: Path = ONNX_PATH) -> Path:
    """Export YOLOv8 PyTorch weights to ONNX format (one-time)."""
    if onnx_path.exists():
        print(f"  ONNX model already exists: {onnx_path}")
        return onnx_path

    from ultralytics import YOLO
    model = YOLO(str(pt_path))
    model.export(format="onnx", imgsz=640, half=False, simplify=True)
    # Ultralytics saves onnx next to the pt file
    default_onnx = pt_path.with_suffix(".onnx")
    if default_onnx.exists() and default_onnx != onnx_path:
        import shutil
        shutil.copy2(default_onnx, onnx_path)
    print(f"  ONNX model exported: {onnx_path}")
    return onnx_path


class DMLInferenceEngine:
    def __init__(self, onnx_path: str | Path = ONNX_PATH):
        if not _HAVE_DML:
            raise RuntimeError("onnxruntime not installed. Run: pip install onnxruntime-directml")

        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            export_to_onnx(MODEL_PATH, self.onnx_path)

        import onnxruntime as ort
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        )
        self.provider = self.session.get_providers()[0]
        print(f"  DML Engine: {self.provider}")
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
    ) -> list[dict]:
        orig_h, orig_w = image.shape[:2]

        # Preprocess: resize + pad to 640x640
        input_size = 640
        scale = min(input_size / orig_w, input_size / orig_h)
        nw, nh = int(orig_w * scale), int(orig_h * scale)
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # Create padded 640x640 image
        padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        pad_x = (input_size - nw) // 2
        pad_y = (input_size - nh) // 2
        padded[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

        # Normalize and add batch dim
        blob = padded.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]

        # Run inference
        outputs = self.session.run(None, {self.input_name: blob})[0][0]

        # Postprocess: filter by confidence, NMS, scale back
        detections = []
        for det in outputs:
            scores = det[4:]
            cls_id = int(np.argmax(scores))
            conf = float(scores[cls_id])
            if conf < conf_threshold:
                continue

            xc, yc, bw, bh = det[:4]
            # Adjust for padding
            xc = (xc * input_size - pad_x) / scale
            yc = (yc * input_size - pad_y) / scale
            bw = bw * input_size / scale
            bh = bh * input_size / scale

            x1 = int(max(0, xc - bw / 2))
            y1 = int(max(0, yc - bh / 2))
            x2 = int(min(orig_w, xc + bw / 2))
            y2 = int(min(orig_h, yc + bh / 2))

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append({
                "label": COCO_NAMES[cls_id] if cls_id < len(COCO_NAMES) else f"class_{cls_id}",
                "confidence": conf,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "class_id": cls_id,
            })

        # NMS
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        keep = []
        for det in detections:
            overlapping = False
            for kept in keep:
                iou = self._iou(
                    [det["x1"], det["y1"], det["x2"], det["y2"]],
                    [kept["x1"], kept["y1"], kept["x2"], kept["y2"]],
                )
                if iou > iou_threshold and det["label"] == kept["label"]:
                    overlapping = True
                    break
            if not overlapping:
                keep.append(det)
        return keep

    @staticmethod
    def _iou(box_a: list[int], box_b: list[int]) -> float:
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
        area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0


def get_engine() -> DMLInferenceEngine | None:
    if not _HAVE_DML:
        return None
    try:
        return DMLInferenceEngine()
    except Exception:
        return None


if __name__ == "__main__":
    export_to_onnx()
