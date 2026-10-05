"""Run YOLO validation with per-class metrics — like Colab end-of-training."""
from ultralytics import YOLO
from pathlib import Path
import sys

MODEL = Path("best.pt")
DATA = Path("datasets/data.yaml")

if not MODEL.exists():
    print(f"Model not found: {MODEL}")
    sys.exit(1)
if not DATA.exists():
    print(f"data.yaml not found: {DATA}")
    sys.exit(1)

print(f"Model: {MODEL}")
print(f"Data:  {DATA}")
print("=" * 90)
print("Running validation on val set... (this may take a few minutes)\n")

model = YOLO(str(MODEL))

# Run validation — this produces per-class metrics
results = model.val(
    data=str(DATA),
    imgsz=640,
    batch=8,
    conf=0.001,       # low conf for proper mAP calculation
    iou=0.6,          # standard COCO IoU for mAP
    verbose=True,
    plots=True,        # generates confusion matrix, PR curves etc.
    save_dir="val_results",
)

# Print summary
print("\n" + "=" * 90)
print("OVERALL METRICS")
print("=" * 90)
print(f"  mAP50:      {results.box.map50:.4f}")
print(f"  mAP50-95:   {results.box.map:.4f}")
print(f"  Precision:  {results.box.mp:.4f}")
print(f"  Recall:     {results.box.mr:.4f}")

# Per-class metrics
print("\n" + "=" * 90)
print(f"{'Class':<22} {'Images':>7} {'Instances':>10} {'P':>8} {'R':>8} {'mAP50':>8} {'mAP50-95':>10}")
print("-" * 90)

names = model.names
ap50 = results.box.ap50           # per-class AP50
ap = results.box.ap               # per-class AP50-95
p = results.box.p                 # per-class precision  
r = results.box.r                 # per-class recall

# Get per-class instance counts from results
nc = len(names)
for i in range(nc):
    cls_name = names[i]
    cls_p = p[i] if i < len(p) else 0
    cls_r = r[i] if i < len(r) else 0
    cls_ap50 = ap50[i] if i < len(ap50) else 0
    cls_ap = ap[i] if i < len(ap) else 0
    
    # Color-code: flag weak classes
    flag = ""
    if cls_ap50 < 0.3:
        flag = " << WEAK"
    elif cls_ap50 < 0.5:
        flag = " < needs work"
    
    print(f"  {cls_name:<20} {'':>7} {'':>10} {cls_p:>8.3f} {cls_r:>8.3f} {cls_ap50:>8.3f} {cls_ap:>10.3f}{flag}")

print("-" * 90)
print(f"  {'ALL':<20} {'':>7} {'':>10} {results.box.mp:>8.3f} {results.box.mr:>8.3f} {results.box.map50:>8.3f} {results.box.map:>10.3f}")
print("=" * 90)

# Flag problem classes
weak = [(names[i], ap50[i]) for i in range(nc) if i < len(ap50) and ap50[i] < 0.3]
if weak:
    print(f"\nWEAK CLASSES (mAP50 < 0.30) — {len(weak)} classes:")
    for name, score in sorted(weak, key=lambda x: x[1]):
        print(f"  {name:<20} mAP50={score:.3f}")

good = [(names[i], ap50[i]) for i in range(nc) if i < len(ap50) and ap50[i] >= 0.7]
if good:
    print(f"\nSTRONG CLASSES (mAP50 >= 0.70) — {len(good)} classes:")
    for name, score in sorted(good, key=lambda x: -x[1]):
        print(f"  {name:<20} mAP50={score:.3f}")

print(f"\nPlots saved to: val_results/")
print("  - confusion_matrix.png")
print("  - PR_curve.png") 
print("  - F1_curve.png")
print("  - P_curve.png")
print("  - R_curve.png")
