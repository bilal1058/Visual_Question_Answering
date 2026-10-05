"""
FastAPI backend for Visual Question Answering.
Provides async REST endpoints for detection, VQA, metrics, and video streaming.

Run:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

Or:
    python app.py
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

if os.name == "nt":
    os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2
import numpy as np
import requests
from fastapi import FastAPI, File, Form, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

from src.database import (
    get_detections, get_frame_by_image_id, get_latency_history,
    get_latency_stats, get_queries, insert_detections, insert_frame,
    insert_latency, insert_query,
)
from src.detect_and_answer import (
    answer_followup, build_rich_scene_description, compute_spatial_relationships,
    decode_uploaded_image, enrich_crops_with_attributes,
    extract_crops, generate_alerts, get_groq_client,
)
from collections import Counter
from src.class_taxonomy import canonical_label
from src.database import LatencyLogger
from src.database import DetectResponse, MetricsResponse, VQACropRequest, VQARequest, VQAResponse
from src.stream_handler import StreamProcessor
from src.benchmark_runner import get_runner

ROOT = Path(__file__).resolve().parent
BASELINE_PATH = ROOT / "yolov8n.pt"
FINETUNED_PATH = ROOT / "best.pt"
MODEL_PATH = FINETUNED_PATH if FINETUNED_PATH.exists() else BASELINE_PATH
STATIC_DIR = ROOT / "static"

latency_logger = LatencyLogger(max_records=1000, log_dir=ROOT / "logs")
_model: YOLO | None = None
_stream_processor: StreamProcessor | None = None


def camera_backend_candidates() -> tuple[int, ...]:
    if os.name == "nt":
        return (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY)
    return (cv2.CAP_ANY,)


def open_camera_capture(camera_id: int = 0) -> cv2.VideoCapture | None:
    for backend in camera_backend_candidates():
        try:
            cap = cv2.VideoCapture(camera_id, backend)
        except Exception:
            continue
        if cap is not None and cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 854)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size:
                    return cap
                time.sleep(0.04)
        if cap is not None:
            cap.release()
    return None


def find_camera_capture(camera_id: int = 0) -> tuple[int, cv2.VideoCapture | None]:
    camera_ids = [camera_id] + [idx for idx in range(5) if idx != camera_id]
    for candidate_id in camera_ids:
        cap = open_camera_capture(candidate_id)
        if cap is not None:
            return candidate_id, cap
    return camera_id, None


def get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(str(MODEL_PATH))
    return _model


def is_low_light(image: np.ndarray) -> bool:
    # Use the average brightness from HSV value channel.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(hsv[..., 2].mean()) < 92.0


def enhance_low_light_image(image: np.ndarray) -> np.ndarray:
    # Brighten dark scenes while preserving local contrast for detection.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge((l_channel, a_channel, b_channel))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    gamma = 1.22
    table = np.array([((value / 255.0) ** (1.0 / gamma)) * 255 for value in range(256)]).astype("uint8")
    return cv2.LUT(enhanced, table)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    yield
    global _stream_processor
    if _stream_processor:
        _stream_processor.stop()


app = FastAPI(title="VQA API", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─── Background task ──────────────────────────────────────────────────────────

async def run_detection_task(
    image_bytes: bytes,
    task_id: str,
    confidence: float,
    question: str = "What objects are present in the image?",
):
    loop = asyncio.get_event_loop()
    raw_image = await loop.run_in_executor(None, decode_uploaded_image, image_bytes)
    detect_image = raw_image
    low_light = is_low_light(raw_image)
    confidence = max(0.10, min(1.0, confidence))
    effective_conf = confidence
    if low_light:
        detect_image = enhance_low_light_image(raw_image)
        effective_conf = max(0.10, confidence - 0.07)
    h, w = raw_image.shape[:2]

    # Generate a unique image_id
    image_id = task_id
    frame_id = insert_frame(image_id, f"{task_id}.jpg", w, h)

    model = get_model()
    yolo_start = time.perf_counter()
    results = await loop.run_in_executor(
        None,
        lambda: model.predict(source=detect_image, conf=effective_conf, iou=0.45, imgsz=640, verbose=False),
    )
    result = results[0]

    # In dark scenes, give the model one rescue pass for small or partially
    # hidden safety/context objects. Use class names because custom datasets
    # do not keep COCO's "person == class 0" ordering.
    if low_light:
        rescue_labels = {"person", "dog", "traffic light"}

        def count_rescue_labels(yolo_result) -> int:
            boxes = yolo_result.boxes
            if boxes is None:
                return 0
            names = yolo_result.names
            total = 0
            for box in boxes:
                cls_id = int(box.cls.item())
                raw_label = str(names.get(cls_id, cls_id)) if hasattr(names, "get") else str(names[cls_id])
                label = canonical_label(raw_label).lower()
                if label in rescue_labels:
                    total += 1
            return total

        base_rescue_count = count_rescue_labels(result)
        if base_rescue_count < 2:
            rescue_results = await loop.run_in_executor(
                None,
                lambda: model.predict(source=detect_image, conf=max(0.10, effective_conf - 0.05), iou=0.40, imgsz=768, verbose=False),
            )
            rescue_result = rescue_results[0]
            rescue_count = count_rescue_labels(rescue_result)
            if rescue_count > base_rescue_count:
                result = rescue_result

    yolo_ms = int((time.perf_counter() - yolo_start) * 1000)
    crops = extract_crops(raw_image, result, result.names)
    crops = enrich_crops_with_attributes(crops, raw_image.shape)
    # Compute counts from deduped crops
    counts = {}
    for c in crops:
        counts[c["label"]] = counts.get(c["label"], 0) + 1
    relationships = compute_spatial_relationships(crops, raw_image.shape)
    rich_desc = build_rich_scene_description(crops, relationships)
    alerts = generate_alerts(crops)

    # Save detections to DB
    detections_data = []
    for crop in crops:
        x1, y1, x2, y2 = crop["xyxy"]
        crop_b64 = ""
        if crop.get("crop_bgr") is not None and crop["crop_bgr"].size > 0:
            ok, crop_buf = cv2.imencode(".jpg", crop["crop_bgr"], [cv2.IMWRITE_JPEG_QUALITY, 88])
            if ok:
                crop_b64 = base64.b64encode(crop_buf).decode("utf-8")
        detections_data.append({
            "label": crop["label"],
            "confidence": crop["conf"],
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "color": crop.get("color", "unknown"),
            "position": crop.get("position", "center"),
            "size": crop.get("size", "medium"),
            "crop_b64": crop_b64,
        })
    insert_detections(frame_id, detections_data)

    # LLM answer (with vision if GROQ_MODEL is a vision model)
    llm_start = time.perf_counter()
    answer = answer_followup(question, counts, [], rich_desc, alerts, crops, raw_image)
    llm_ms = int((time.perf_counter() - llm_start) * 1000)

    total_ms = yolo_ms + llm_ms
    timing = {"yolo_ms": yolo_ms, "llm_ms": llm_ms, "total_ms": total_ms, "num_objects": sum(counts.values())}

    insert_latency(frame_id, timing)
    insert_query(frame_id, question, answer)
    latency_logger.record(**timing, image_size=(w, h), source="upload")

    # Store results (send CLEAN image - no OpenCV bboxes, frontend draws them)
    clean_image = raw_image.copy()
    _, buffer = cv2.imencode(".jpg", clean_image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    result_b64 = base64.b64encode(buffer).decode("utf-8")

    app.state._results[task_id] = {
        "status": "completed",
        "annotated": result_b64,
        "detections": detections_data,
        "counts": dict(counts),
        "answer": answer,
        "timing": timing,
    }


# ─── REST Endpoints ───────────────────────────────────────────────────────────

@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    confidence: float = Form(0.25),
    question: str = Form("What objects are present in the image?"),
) -> dict:
    if not hasattr(app.state, "_results"):
        app.state._results = {}

    image_bytes = await file.read()
    task_id = str(uuid.uuid4())[:8]
    app.state._results[task_id] = {"status": "processing"}

    asyncio.create_task(run_detection_task(image_bytes, task_id, confidence, question))

    return {"task_id": task_id, "status": "processing"}


@app.get("/detect/{task_id}")
async def get_detection_result(task_id: str) -> dict:
    if not hasattr(app.state, "_results"):
        return {"status": "not_found"}
    result = app.state._results.get(task_id, {"status": "not_found"})
    return result


@app.post("/vqa")
async def vqa(request: VQARequest) -> VQAResponse:
    frame = get_frame_by_image_id(request.image_id)
    if not frame:
        return VQAResponse(answer="No image found with that ID.", timing_ms={})

    detections = get_detections(frame["id"])
    counts = {}
    for d in detections:
        counts[d["label"]] = counts.get(d["label"], 0) + 1

    llm_start = time.perf_counter()
    answer = answer_followup(request.question, counts, request.history)
    llm_ms = int((time.perf_counter() - llm_start) * 1000)

    insert_query(frame["id"], request.question, answer)

    return VQAResponse(answer=answer, timing_ms={"llm_ms": llm_ms})


@app.post("/vqa/crop")
async def vqa_crop(request: VQACropRequest) -> VQAResponse:
    frame = get_frame_by_image_id(request.image_id)
    if not frame:
        return VQAResponse(answer="No image found with that ID.", timing_ms={})

    detections = get_detections(frame["id"])
    if request.crop_index < 0 or request.crop_index >= len(detections):
        return VQAResponse(answer="Invalid crop selection.", timing_ms={})

    crop = detections[request.crop_index]
    counts = Counter({crop["label"]: 1})

    llm_start = time.perf_counter()
    focus_prompt = f"Focus on the detected {crop['label']} ({crop.get('color', 'unknown')}, {crop.get('position', 'center')}, {crop.get('size', 'medium')}). {request.question}"
    answer = answer_followup(focus_prompt, counts, request.history)
    llm_ms = int((time.perf_counter() - llm_start) * 1000)

    insert_query(frame["id"], request.question, answer)

    return VQAResponse(answer=answer, timing_ms={"llm_ms": llm_ms})


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)) -> dict:
    audio_bytes = await audio.read()
    if not audio_bytes:
        return JSONResponse(status_code=400, content={"text": "", "error": "No audio received."})

    filename = audio.filename or "voice.webm"
    content_type = audio.content_type or "audio/webm"
    provider = os.getenv("TRANSCRIBE_PROVIDER", "auto").strip().lower()
    assembly_key = os.getenv("ASSEMBLYAI_API_KEY") or os.getenv("ASS_API")
    groq_key = os.getenv("GROQ_API_KEY")
    loop = asyncio.get_event_loop()

    def groq_transcribe_models() -> list[str]:
        models = os.getenv("GROQ_TRANSCRIBE_MODELS") or os.getenv("GROQ_TRANSCRIBE_MODEL")
        names = [name.strip() for name in (models or "whisper-large-v3-turbo,whisper-large-v3").split(",")]
        names = [name for name in names if name]
        if "whisper-large-v3-turbo" in names and "whisper-large-v3" not in names:
            names.append("whisper-large-v3")
        return names

    def transcribe_with_groq() -> str:
        client = get_groq_client()
        if client is None:
            raise RuntimeError("GROQ_API_KEY is not configured.")
        errors = []
        for model_name in groq_transcribe_models():
            try:
                result = client.audio.transcriptions.create(
                    file=(filename, io.BytesIO(audio_bytes), content_type),
                    model=model_name,
                    response_format="json",
                    temperature=0,
                )
                text = getattr(result, "text", "")
                if not text and isinstance(result, dict):
                    text = result.get("text", "")
                return str(text).strip()
            except Exception as exc:
                errors.append(f"{model_name}: {exc}")
        raise RuntimeError("Groq transcription failed for all models. " + " | ".join(errors))

    def transcribe_with_assemblyai() -> str:
        if not assembly_key:
            raise RuntimeError("ASSEMBLYAI_API_KEY or ASS_API is not configured.")
        base_url = os.getenv("ASSEMBLYAI_BASE_URL", "https://api.assemblyai.com").rstrip("/")
        headers = {"authorization": assembly_key}
        upload = requests.post(
            f"{base_url}/v2/upload",
            headers=headers,
            data=audio_bytes,
            timeout=60,
        )
        upload.raise_for_status()
        audio_url = upload.json()["upload_url"]
        payload = {
            "audio_url": audio_url,
            "language_detection": True,
        }
        models = os.getenv("ASSEMBLYAI_SPEECH_MODELS", "universal-3-pro")
        speech_models = [name.strip() for name in models.split(",") if name.strip()]
        if speech_models:
            payload["speech_models"] = speech_models

        start = requests.post(f"{base_url}/v2/transcript", json=payload, headers=headers, timeout=30)
        start.raise_for_status()
        transcript_id = start.json()["id"]
        deadline = time.monotonic() + float(os.getenv("TRANSCRIBE_TIMEOUT_SECONDS", "90"))

        while time.monotonic() < deadline:
            poll = requests.get(f"{base_url}/v2/transcript/{transcript_id}", headers=headers, timeout=30)
            poll.raise_for_status()
            result = poll.json()
            status = result.get("status")
            if status == "completed":
                return (result.get("text") or "").strip()
            if status == "error":
                raise RuntimeError(result.get("error") or "AssemblyAI transcription failed.")
            time.sleep(1.2)
        raise TimeoutError("AssemblyAI transcription timed out.")

    def run_transcription() -> str:
        if provider == "assemblyai":
            return transcribe_with_assemblyai()
        if provider == "groq":
            return transcribe_with_groq()
        if provider in {"auto", "fast", "fastest"}:
            if groq_key:
                try:
                    return transcribe_with_groq()
                except Exception as groq_error:
                    if not assembly_key:
                        raise groq_error
            if assembly_key:
                return transcribe_with_assemblyai()
            raise RuntimeError("No transcription provider key configured.")
        return transcribe_with_groq()

    try:
        text = await loop.run_in_executor(None, run_transcription)
    except Exception as exc:
        return JSONResponse(status_code=502, content={"text": "", "error": str(exc)})
    return {"text": text}


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> dict:
    stats = get_latency_stats(limit=50)
    s = latency_logger.stats()
    totals = [r.total_ms for r in latency_logger.records if r.total_ms > 0]
    p50 = float(np.percentile(totals, 50)) if totals else 0
    p95 = float(np.percentile(totals, 95)) if totals else 0
    recent = [
        {"yolo_ms": r.yolo_ms, "llm_ms": r.llm_ms, "total_ms": r.total_ms,
         "num_objects": r.num_objects, "timestamp": r.timestamp}
        for r in list(latency_logger.records)[-10:]
    ]
    return {
        "total_runs": s.get("total_records", stats.get("total", 0)),
        "avg_yolo_ms": s.get("yolo_avg_ms", stats.get("avg_yolo", 0)),
        "avg_llm_ms": s.get("llm_avg_ms", stats.get("avg_llm", 0)),
        "avg_total_ms": s.get("total_avg_ms", stats.get("avg_total", 0)),
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "avg_fps": s.get("avg_fps", stats.get("avg_fps", 0)),
        "recent_runs": recent,
    }


@app.get("/metrics/history")
async def metrics_history(minutes: int = Query(60)) -> list[dict]:
    return get_latency_history(minutes)


# ─── Benchmark ────────────────────────────────────────────────────────────────

@app.post("/benchmark/start")
async def benchmark_start(force: bool = Form(False)):
    """Start a benchmark comparison in background."""
    from src.benchmark_runner import CACHE_FILE, attach_comparison_meta, compare_models

    import threading

    if not force and CACHE_FILE.exists():
        import json
        try:
            app.state._benchmark_result = attach_comparison_meta(json.loads(CACHE_FILE.read_text()))
            return {"status": "cached"}
        except Exception:
            pass

    app.state._benchmark_result = None

    app.state._benchmark_progress = {"status": "running", "current": 0, "total": 0, "message": "Starting..."}

    def on_progress(current, total, msg):
        app.state._benchmark_progress = {"status": "running", "current": current, "total": total, "message": msg}

    def run():
        app.state._benchmark_result = compare_models(force=force, on_progress=on_progress)
        app.state._benchmark_progress = {"status": "completed", "current": 100, "total": 100, "message": "Done"}

    threading.Thread(target=run, daemon=True).start()
    return {"status": "started"}


@app.get("/benchmark/status")
async def benchmark_status() -> dict:
    progress = getattr(app.state, "_benchmark_progress", {"status": "running", "current": 0, "total": 0, "message": "Starting..."})
    if not hasattr(app.state, "_benchmark_result"):
        return {"status": "running", "progress": progress}
    if app.state._benchmark_result is None:
        return {"status": "running", "progress": progress}
    return {"status": "completed", "results": app.state._benchmark_result, "progress": progress}


# ─── Video streaming (MJPEG) ──────────────────────────────────────────────────

async def generate_mjpeg_stream(source_type: str, source_id: int = 0):
    """Async MJPEG generator for /stream endpoint."""
    global _stream_processor
    if _stream_processor is None:
        _stream_processor = StreamProcessor(str(MODEL_PATH), confidence=0.35)
    _stream_processor.running = True

    cap: cv2.VideoCapture | None = None
    try:
        if source_type == "camera":
            resolved_id, cap = find_camera_capture(source_id)
            if cap is None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: text/plain\r\n\r\n"
                    b"Camera not found or blocked.\r\n"
                )
                return
            source_id = resolved_id
        else:
            return

        while _stream_processor.running:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue
            frame = cv2.resize(frame, (854, 480))
            result = await _stream_processor.process_frame(frame)
            _, buffer = cv2.imencode(".jpg", result["annotated"], [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
            await asyncio.sleep(0.02)
    finally:
        if cap:
            cap.release()
        _stream_processor.running = False


@app.get("/stream")
async def video_stream(source: str = Query("camera"), camera_id: int = Query(0)):
    return StreamingResponse(
        generate_mjpeg_stream(source, camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/stream/status")
async def stream_status(camera_id: int = Query(0)) -> dict:
    resolved_id, cap = find_camera_capture(camera_id)
    if cap is None:
        return {
            "ok": False,
            "camera_id": camera_id,
            "message": "OpenCV cannot access a camera right now. Browser capture may still work because it uses separate browser camera access.",
        }
    cap.release()
    return {"ok": True, "camera_id": resolved_id, "message": f"Camera {resolved_id} ready."}


@app.post("/stream/stop")
async def stop_stream():
    global _stream_processor
    if _stream_processor:
        _stream_processor.stop()
    return {"status": "stopped"}


# ─── WebSocket for real-time detections ───────────────────────────────────────

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    global _stream_processor
    if _stream_processor is None:
        _stream_processor = StreamProcessor(str(MODEL_PATH))

    resolved_id, cap = find_camera_capture(0)
    if cap is None:
        await websocket.send_json({"error": "Camera not found or blocked."})
        await websocket.close()
        return
    _stream_processor.running = True
    try:
        while _stream_processor.running:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue
            result = await _stream_processor.process_frame(frame)
            _, buffer = cv2.imencode(".jpg", result["annotated"], [cv2.IMWRITE_JPEG_QUALITY, 80])
            payload = {
                "image": base64.b64encode(buffer).decode("utf-8"),
                "fps": result["fps"],
                "detections": result["detections"],
                "tracked": result["tracked"],
                "timing": result["timing"],
            }
            await websocket.send_json(payload)
            await asyncio.sleep(0.03)
    except WebSocketDisconnect:
        pass
    finally:
        cap.release()
        _stream_processor.running = False


# ─── Frontend ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>VQA API</h1><p>Frontend not found. Run with static/index.html</p>")


# ─── CLI runner ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
