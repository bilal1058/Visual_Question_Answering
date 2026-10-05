"""
Benchmark YOLOv8 on the expanded dataset.
Results shown on the UI at /metrics. This CLI version is for terminal use.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --all-classes
    python scripts/benchmark.py --model yolov8n.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.benchmark_runner import BenchmarkRunner


def main() -> None:
    all_classes = "--all-classes" in sys.argv
    model_flag = None
    for i, arg in enumerate(sys.argv):
        if arg == "--model" and i + 1 < len(sys.argv):
            model_flag = sys.argv[i + 1]

    model_path = Path(model_flag) if model_flag else ROOT / "yolov8n.pt"
    runner = BenchmarkRunner(model_path)

    def progress(current, total, msg):
        print(f"\r  [{current}/{total}] {msg}", end="", flush=True)

    results = runner.run(all_classes=all_classes, on_progress=progress)
    print()

    if "error" in results:
        print(f"  ERROR: {results['error']}")
        return

    print(f"\n  {'-'*50}")
    print(f"  {'Metric':<35} {'Value':>12}")
    print(f"  {'-'*50}")
    for k in ["num_images", "images_with_objects", "total_detections",
              "avg_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
              "avg_fps", "overall_fps", "total_time_s"]:
        if k in results:
            unit = " ms" if "ms" in k else " s" if k == "total_time_s" else ""
            print(f"  {k:<35} {results[k]:>10}{unit}")
    if "mAP50_P" in results:
        print(f"  {'mAP@50 Precision':<35} {results['mAP50_P']:>12.3f}")
        print(f"  {'mAP@50 Recall':<35} {results['mAP50_R']:>12.3f}")
    print(f"  {'-'*50}")

    if results.get("per_class"):
        print(f"\n  {'Class':<30} {'Detections':>10}")
        print(f"  {'-'*40}")
        for cls, cnt in list(results["per_class"].items())[:20]:
            print(f"  {cls:<30} {cnt:>10}")


if __name__ == "__main__":
    main()
