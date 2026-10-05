"""
Data layer: SQLite storage, latency logging, and Pydantic schemas.

Consolidates former standalone modules:
  - src/database.py       — SQLite frame/detection/query/latency tables
  - src/latency_logger.py — In-memory latency recording + CSV export
  - src/schemas.py        — Pydantic request/response models
"""

from __future__ import annotations

import csv
import sqlite3
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

DB_PATH = Path("data") / "vqa.db"


# ═══════════════════════════════════════════════════════════════════════════════
# SQLite Storage
# ═══════════════════════════════════════════════════════════════════════════════


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                source TEXT DEFAULT 'upload',
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id INTEGER NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                x1 INTEGER NOT NULL,
                y1 INTEGER NOT NULL,
                x2 INTEGER NOT NULL,
                y2 INTEGER NOT NULL,
                color TEXT DEFAULT 'unknown',
                position TEXT DEFAULT 'center',
                size TEXT DEFAULT 'medium',
                track_id INTEGER,
                FOREIGN KEY (frame_id) REFERENCES frames(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id INTEGER NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (frame_id) REFERENCES frames(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS latency (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id INTEGER NOT NULL,
                yolo_ms REAL NOT NULL,
                llm_ms REAL NOT NULL,
                preprocess_ms REAL DEFAULT 0,
                postprocess_ms REAL DEFAULT 0,
                total_ms REAL NOT NULL,
                num_objects INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                FOREIGN KEY (frame_id) REFERENCES frames(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_detections_frame ON detections(frame_id);
            CREATE INDEX IF NOT EXISTS idx_queries_frame ON queries(frame_id);
            CREATE INDEX IF NOT EXISTS idx_latency_frame ON latency(frame_id);
            CREATE INDEX IF NOT EXISTS idx_latency_created ON latency(created_at);
        """)


# ─── Frame operations ─────────────────────────────────────────────────────────

def insert_frame(image_id: str, filename: str, width: int, height: int, source: str = "upload") -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO frames (image_id, filename, width, height, source, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (image_id, filename, width, height, source, time.time()),
        )
        if cur.lastrowid:
            return cur.lastrowid
        row = conn.execute("SELECT id FROM frames WHERE image_id = ?", (image_id,)).fetchone()
        return row["id"] if row else 0


def get_frame(frame_id: int) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM frames WHERE id = ?", (frame_id,)).fetchone()
        return dict(row) if row else None


def get_frame_by_image_id(image_id: str) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM frames WHERE image_id = ?", (image_id,)).fetchone()
        return dict(row) if row else None


# ─── Detection operations ─────────────────────────────────────────────────────

def insert_detections(frame_id: int, detections: list[dict]) -> None:
    with get_connection() as conn:
        conn.executemany(
            """INSERT INTO detections
               (frame_id, label, confidence, x1, y1, x2, y2, color, position, size)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (frame_id, d["label"], d["confidence"], d["x1"], d["y1"],
                 d["x2"], d["y2"], d.get("color", "unknown"),
                 d.get("position", "center"), d.get("size", "medium"))
                for d in detections
            ],
        )


def get_detections(frame_id: int) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM detections WHERE frame_id = ? ORDER BY confidence DESC",
            (frame_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# ─── Query operations ─────────────────────────────────────────────────────────

def insert_query(frame_id: int, question: str, answer: str) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO queries (frame_id, question, answer, created_at) VALUES (?, ?, ?, ?)",
            (frame_id, question, answer, time.time()),
        )
        return cur.lastrowid


def get_queries(frame_id: int, limit: int = 20) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM queries WHERE frame_id = ? ORDER BY created_at DESC LIMIT ?",
            (frame_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


# ─── Latency operations ───────────────────────────────────────────────────────

def insert_latency(frame_id: int, timing: dict) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO latency
               (frame_id, yolo_ms, llm_ms, preprocess_ms, postprocess_ms, total_ms, num_objects, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                frame_id,
                timing.get("yolo_ms", 0),
                timing.get("llm_ms", 0),
                timing.get("preprocess_ms", 0),
                timing.get("postprocess_ms", 0),
                timing.get("total_ms", 0),
                timing.get("num_objects", 0),
                time.time(),
            ),
        )
        return cur.lastrowid


def get_latency_stats(limit: int = 50) -> dict:
    with get_connection() as conn:
        recent = conn.execute(
            "SELECT * FROM latency ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()

        agg = conn.execute(
            """SELECT
                COUNT(*) as total,
                AVG(yolo_ms) as avg_yolo,
                AVG(llm_ms) as avg_llm,
                AVG(total_ms) as avg_total,
                MAX(total_ms) as max_total,
                MIN(total_ms) as min_total,
                AVG(CASE WHEN total_ms > 0 THEN 1000.0/total_ms ELSE 0 END) as avg_fps
               FROM latency
               WHERE created_at > ?""",
            (time.time() - 3600,),
        ).fetchone()

        stats = dict(agg) if agg else {}
        stats["recent"] = [dict(r) for r in recent[:10]]
        return stats


def get_latency_history(minutes: int = 60) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT created_at, yolo_ms, llm_ms, total_ms, num_objects
               FROM latency WHERE created_at > ?
               ORDER BY created_at ASC""",
            (time.time() - minutes * 60,),
        ).fetchall()
        return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════════════════════
# Latency Logger (in-memory + CSV export)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class LatencyRecord:
    timestamp: float
    yolo_ms: float
    llm_ms: float
    preprocess_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_ms: float = 0.0
    num_objects: int = 0
    num_crops: int = 0
    image_size: tuple[int, int] = (0, 0)
    source: Literal["upload", "camera", "video"] = "upload"


class LatencyLogger:
    def __init__(self, max_records: int = 500, log_dir: str | Path | None = None) -> None:
        self.records: deque[LatencyRecord] = deque(maxlen=max_records)
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def record(self, **kwargs) -> LatencyRecord:
        record = LatencyRecord(timestamp=time.time(), **kwargs)
        self.records.append(record)
        self._write_csv(record)
        return record

    def _write_csv(self, record: LatencyRecord) -> None:
        if not self.log_dir:
            return
        csv_path = self.log_dir / "latency_log.csv"
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "timestamp", "yolo_ms", "llm_ms", "preprocess_ms",
                    "postprocess_ms", "total_ms", "num_objects",
                    "num_crops", "image_w", "image_h", "source",
                ])
            writer.writerow([
                record.timestamp, record.yolo_ms, record.llm_ms,
                record.preprocess_ms, record.postprocess_ms, record.total_ms,
                record.num_objects, record.num_crops,
                record.image_size[0], record.image_size[1], record.source,
            ])

    def stats(self) -> dict[str, float]:
        if not self.records:
            return {}
        yolo = [r.yolo_ms for r in self.records if r.yolo_ms > 0]
        llm = [r.llm_ms for r in self.records if r.llm_ms > 0]
        total = [r.total_ms for r in self.records if r.total_ms > 0]
        return {
            "yolo_avg_ms": sum(yolo) / len(yolo) if yolo else 0,
            "yolo_max_ms": max(yolo) if yolo else 0,
            "yolo_min_ms": min(yolo) if yolo else 0,
            "llm_avg_ms": sum(llm) / len(llm) if llm else 0,
            "total_avg_ms": sum(total) / len(total) if total else 0,
            "total_max_ms": max(total) if total else 0,
            "total_records": len(self.records),
            "avg_fps": 1000 / (sum(total) / len(total)) if total else 0,
        }

    def summary_html(self, timing: dict[str, int]) -> str:
        s = self.stats()
        if not s:
            return '<div class="vq-timing-bar"><div class="vq-timing-item"><span>No data yet</span></div></div>'

        avg_yolo = s.get("yolo_avg_ms", 0)
        avg_total = s.get("total_avg_ms", 0)
        avg_fps = s.get("avg_fps", 0)
        records = s.get("total_records", 0)

        recent = list(self.records)[-5:]
        recent_yolo = sum(r.yolo_ms for r in recent) / len(recent) if recent else 0
        recent_total = sum(r.total_ms for r in recent) / len(recent) if recent else 0

        return f"""
<div class="vq-timing-bar">
  <div class="vq-timing-item">
    <span>Detector</span>
    <strong>{timing.get("yolo_ms", int(recent_yolo))}ms</strong>
    <small style="display:block;color:var(--muted);font-size:0.65rem;">avg {avg_yolo:.0f}ms</small>
  </div>
  <div class="vq-timing-item">
    <span>Reasoner</span>
    <strong>{timing.get("llm_ms", 0)}ms</strong>
  </div>
  <div class="vq-timing-item">
    <span>Total</span>
    <strong>{timing.get("total_ms", int(recent_total))}ms</strong>
    <small style="display:block;color:var(--muted);font-size:0.65rem;">{avg_fps:.1f} FPS avg</small>
  </div>
  <div class="vq-timing-item">
    <span>Runs</span>
    <strong>{records}</strong>
  </div>
</div>
"""


_logger: LatencyLogger | None = None


def get_logger() -> LatencyLogger:
    global _logger
    if _logger is None:
        _logger = LatencyLogger(log_dir=Path("logs"))
    return _logger


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic Schemas
# ═══════════════════════════════════════════════════════════════════════════════


class DetectionResult(BaseModel):
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    color: str = "unknown"
    position: str = "center"
    size: str = "medium"


class DetectRequest(BaseModel):
    confidence: float = 0.35
    iou: float = 0.45


class DetectResponse(BaseModel):
    task_id: str
    status: str = "processing"


class VQARequest(BaseModel):
    image_id: str
    question: str
    history: list[dict] = []


class VQACropRequest(BaseModel):
    image_id: str
    question: str
    crop_index: int = 0
    history: list[dict] = []


class VQAResponse(BaseModel):
    answer: str
    timing_ms: dict[str, int]


class MetricsResponse(BaseModel):
    total_runs: int
    avg_yolo_ms: float
    avg_llm_ms: float
    avg_total_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    avg_fps: float
    recent_runs: list[dict]


class FrameRecord(BaseModel):
    id: Optional[int] = None
    image_id: str
    filename: str
    width: int
    height: int
    timestamp: datetime = None
    source: str = "upload"

    class Config:
        from_attributes = True


class DetectionRecord(BaseModel):
    id: Optional[int] = None
    frame_id: int
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    color: str
    position: str
    size: str

    class Config:
        from_attributes = True


class QueryRecord(BaseModel):
    id: Optional[int] = None
    frame_id: int
    question: str
    answer: str
    timestamp: datetime = None

    class Config:
        from_attributes = True


class LatencyRecordSchema(BaseModel):
    id: Optional[int] = None
    frame_id: int
    yolo_ms: float
    llm_ms: float
    preprocess_ms: float = 0
    postprocess_ms: float = 0
    total_ms: float
    num_objects: int = 0
    timestamp: datetime = None

    class Config:
        from_attributes = True


# ─── Init on import ───────────────────────────────────────────────────────────

init_db()
