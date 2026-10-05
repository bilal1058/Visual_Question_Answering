Project instructions for Codex

This is a Visual Q&A computer vision app.

Do not rewrite the app from scratch.
Preserve all object detection, VQA, upload, camera, threshold, thumbnail, stream, benchmark, and timing logic.
Prefer UI and CSS improvements over backend changes.
Keep the interface readable and accessible.
Use a premium dark AI dashboard style.
Use reusable components where possible.
Do not add unnecessary dependencies.
Do not remove existing features.
Run available checks after changes.
Explain changed files at the end.

# Project setup guide

## Quick start
```powershell
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```
Open http://localhost:8000

## Dataset
```powershell
python scripts/setup.py                      # Auto Roboflow if configured; else COCO80 + Open Images
python scripts/setup.py --roboflow ZIP_OR_URL # Import Roboflow Universe YOLOv8 export
python scripts/setup.py --no-roboflow        # Force COCO80 + Open Images fallback
python scripts/setup.py --coco-only          # Only COCO80
python scripts/setup.py --oi-only            # Only Open Images expansion
python scripts/setup.py --labels             # Regenerate labels + splits from existing images
```

## Training (on Colab T4)
```powershell
python scripts/train.py --dry-run            # Preview dataset stats
```
Then zip datasets/ and upload to Drive, run `colab_train.ipynb` in Colab.

## Benchmark
Runs in the UI at /metrics, or CLI:
```powershell
python scripts/benchmark.py
python scripts/benchmark.py --all-classes
```

## Common commands
```powershell
uvicorn app:app --reload --port 8000    # Start server
pip install -r requirements.txt         # Install all deps
```

## File structure
```
├── app.py                 # FastAPI server (main entry point)
├── colab_train.ipynb      # Colab training notebook
├── scripts/
│   ├── setup.py           # Dataset download + expansion + labels + splits
│   ├── train.py           # Dataset preview (actual training on Colab)
│   └── benchmark.py       # CLI benchmark
├── src/
│   ├── detect_and_answer.py  # Core detection + VQA logic
│   ├── database.py           # SQLite storage
│   ├── latency_logger.py     # Inference timing logger
│   ├── dml_engine.py         # AMD GPU inference (DirectML)
│   ├── benchmark_runner.py   # Shared benchmark logic
│   ├── tracker.py            # Object tracking (ByteTrack-style)
│   ├── stream_handler.py     # Video stream processor
│   └── schemas.py            # Pydantic models
├── static/
│   ├── index.html           # FastAPI frontend
│   ├── style.css            # Frontend styles
│   └── app.js               # Frontend logic
└── datasets/
    └── coco/
        └── images_by_class/   # Class-organized images
```
