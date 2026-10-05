"""
Video stream handler with async frame processing, object tracking, and sliding window.

Includes lightweight IoU-based object tracker (formerly src/tracker.py).
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from src.database import insert_frame, insert_detections, insert_latency
from src.class_taxonomy import canonical_label


# ─── Object Tracker ─────────────────────────────────────────────────────────
# Lightweight IoU-based tracker (approximates ByteTrack behavior).
# Tracks objects across video frames using IoU matching + velocity prediction.


@dataclass
class Track:
    track_id: int
    label: str
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    age: int = 0
    hit_streak: int = 1
    is_confirmed: bool = False
    vel_x: float = 0.0
    vel_y: float = 0.0


class ObjectTracker:
    def __init__(self, max_lost: int = 15, iou_threshold: float = 0.35):
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
        self.next_id = 0
        self.tracks: dict[int, Track] = {}
        self.lost: dict[int, int] = {}

    def _iou(self, a: list[int], b: list[int]) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections: list[dict]) -> list[Track]:
        matched_det = set()
        matched_trk = set()

        # Predict next position for existing tracks
        for tid, track in list(self.tracks.items()):
            w = track.x2 - track.x1
            h = track.y2 - track.y1
            track.x1 += int(track.vel_x)
            track.y1 += int(track.vel_y)
            track.x2 = track.x1 + w
            track.y2 = track.y1 + h
            track.age += 1

        # Match detections to tracks (greedy IoU)
        matches: list[tuple[int, int, float]] = []
        for di, det in enumerate(detections):
            dbox = [det["x1"], det["y1"], det["x2"], det["y2"]]
            best_iou = self.iou_threshold
            best_tid = -1
            for tid, track in self.tracks.items():
                if tid in matched_trk:
                    continue
                if track.label != det["label"]:
                    continue
                tbox = [track.x1, track.y1, track.x2, track.y2]
                iou = self._iou(dbox, tbox)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            if best_tid >= 0:
                matches.append((di, best_tid, best_iou))
                matched_det.add(di)
                matched_trk.add(best_tid)

        # Update matched tracks
        for di, tid, _ in matches:
            det = detections[di]
            track = self.tracks[tid]
            track.x1, track.y1, track.x2, track.y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            track.confidence = det["confidence"]
            track.vel_x = (track.x2 - track.x1) * 0.1
            track.vel_y = (track.y2 - track.y1) * 0.1
            track.hit_streak += 1
            track.age = 0
            track.is_confirmed = track.hit_streak >= 3
            self.lost.pop(tid, None)

        # Create new tracks for unmatched detections
        for di, det in enumerate(detections):
            if di in matched_det:
                continue
            self.tracks[self.next_id] = Track(
                track_id=self.next_id,
                label=det["label"],
                x1=det["x1"], y1=det["y1"],
                x2=det["x2"], y2=det["y2"],
                confidence=det["confidence"],
            )
            self.next_id += 1

        # Mark unmatched tracks as lost
        for tid in list(self.tracks.keys()):
            if tid not in matched_trk:
                self.lost[tid] = self.lost.get(tid, 0) + 1
                if self.lost[tid] > self.max_lost:
                    del self.tracks[tid]
                    del self.lost[tid]

        return list(self.tracks.values())

    def reset(self) -> None:
        self.tracks.clear()
        self.lost.clear()
        self.next_id = 0


# ─── Stream Processor ───────────────────────────────────────────────────────


class StreamProcessor:
    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.35,
        iou: float = 0.45,
    ):
        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.iou = iou
        self.tracker = ObjectTracker()
        self.running = False
        self.frame_count = 0
        self.fps = 0.0
        self._last_time = 0.0

    async def process_frame(
        self, frame: np.ndarray
    ) -> dict:
        start = time.perf_counter()

        preprocess_start = time.perf_counter()
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000

        yolo_start = time.perf_counter()
        results = self.model.predict(
            source=frame, conf=self.confidence, iou=self.iou,
            imgsz=640, verbose=False,
        )
        yolo_ms = (time.perf_counter() - yolo_start) * 1000

        postprocess_start = time.perf_counter()
        result = results[0]
        raw_detections = []
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                raw_detections.append({
                    "label": canonical_label(result.names[int(box.cls.item())]),
                    "confidence": float(box.conf.item()),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                })

        # Track
        tracked = self.tracker.update(raw_detections)
        postprocess_ms = (time.perf_counter() - postprocess_start) * 1000

        total_ms = (time.perf_counter() - start) * 1000

        # FPS
        now = time.time()
        if self._last_time > 0:
            self.fps = self.fps * 0.9 + (1.0 / (now - self._last_time)) * 0.1
        self._last_time = now
        self.frame_count += 1

        # Draw tracking info
        annotated = frame.copy()
        for t in tracked:
            color = (37, 99, 235) if t.is_confirmed else (100, 100, 100)
            cv2.rectangle(annotated, (t.x1, t.y1), (t.x2, t.y2), color, 2)
            label = f"{t.label}#{t.track_id} {t.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (t.x1, t.y1 - th - 6), (t.x1 + tw + 6, t.y1), color, -1)
            cv2.putText(annotated, label, (t.x1 + 3, t.y1 - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # FPS overlay
        cv2.putText(annotated, f"{self.fps:.1f} FPS", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (37, 99, 235), 2, cv2.LINE_AA)

        return {
            "annotated": annotated,
            "detections": raw_detections,
            "tracked": [
                {"track_id": t.track_id, "label": t.label,
                 "x1": t.x1, "y1": t.y1, "x2": t.x2, "y2": t.y2,
                 "confidence": t.confidence, "confirmed": t.is_confirmed}
                for t in tracked
            ],
            "timing": {
                "preprocess_ms": preprocess_ms,
                "yolo_ms": yolo_ms,
                "postprocess_ms": postprocess_ms,
                "total_ms": total_ms,
            },
            "fps": self.fps,
        }

    async def process_video_file(
        self, video_path: str | Path, frame_skip: int = 2
    ) -> AsyncGenerator[dict, None]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.running = True
        skip_counter = 0
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                skip_counter += 1
                if skip_counter % frame_skip != 0:
                    continue
                result = await self.process_frame(frame)
                yield result
                await asyncio.sleep(0.001)
        finally:
            cap.release()
            self.running = False

    async def process_camera_stream(
        self, camera_id: int = 0, max_fps: float = 15.0
    ) -> AsyncGenerator[dict, None]:
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera: {camera_id}")

        self.running = True
        min_interval = 1.0 / max_fps
        try:
            while self.running:
                loop_start = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    await asyncio.sleep(0.01)
                    continue
                result = await self.process_frame(frame)
                yield result
                elapsed = time.perf_counter() - loop_start
                sleep_time = min_interval - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        finally:
            cap.release()
            self.running = False

    def stop(self) -> None:
        self.running = False
