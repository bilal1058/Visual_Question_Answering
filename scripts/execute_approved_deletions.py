import os
import shutil
from pathlib import Path

ROOT = Path("E:/CV-MID")

# 1. 36 folders to delete in images_by_class
base_e = ROOT / "datasets" / "coco" / "images_by_class"
classes_to_delete = {
    "air_conditioner", "apple", "banana", "book", "bottle", "cat", "charger", "clock",
    "cup", "fan", "gas_stove", "glasses", "goat", "kettle", "keyboard", "knife",
    "laptop", "medicine", "mirror", "mobile", "mug", "orange", "oven", "plate",
    "refrigerator", "remote", "router", "shoes", "sign", "switch", "toilet",
    "trash_can", "tv", "umbrella", "window", "worker"
}

deleted_folders = []
deleted_folder_bytes = 0
deleted_folder_images = 0

if base_e.exists():
    for entry in os.listdir(base_e):
        entry_lower = entry.lower().replace(" ", "_")
        if entry_lower in classes_to_delete or entry.lower() in classes_to_delete:
            p = base_e / entry
            if p.is_dir():
                count = sum(1 for _, _, files in os.walk(p) for _ in files)
                size = sum(os.path.getsize(os.path.join(r, f)) for r, _, files in os.walk(p) for f in files)
                shutil.rmtree(p)
                deleted_folders.append((entry, count, size))
                deleted_folder_bytes += size
                deleted_folder_images += count

print("=== DELETED EXCLUDED CLASS FOLDERS ===")
print(f"Total Folders Deleted: {len(deleted_folders)}")
print(f"Total Images Deleted:  {deleted_folder_images}")
print(f"Total Space Freed:     {deleted_folder_bytes / (1024*1024):.2f} MB ({deleted_folder_bytes / (1024*1024*1024):.2f} GB)")
for name, cnt, sz in sorted(deleted_folders, key=lambda x: x[1], reverse=True):
    print(f"  Deleted {name:20s}: {cnt:5d} images ({sz/(1024*1024):6.2f} MB)")

# 2. 12 Orphaned & duplicate .pt weights
orphaned_pts = [
    "besst.pt", "lasst.pt", "bestt.pt", "last.pt",
    "best (1).pt", "last (1).pt", "best8n.pt",
    "best (2).pt", "last (2).pt",
    "best (3).pt", "last (3).pt",
    "bestold.pt"
]

deleted_pts = []
deleted_pt_bytes = 0

print("\n=== DELETED ORPHANED & DUPLICATE WEIGHTS ===")
for pt_name in orphaned_pts:
    pt_path = ROOT / pt_name
    if pt_path.exists():
        sz = pt_path.stat().st_size
        pt_path.unlink()
        deleted_pts.append((pt_name, sz))
        deleted_pt_bytes += sz
        print(f"  Deleted {pt_name:15s}: {sz/(1024*1024):6.2f} MB")
    else:
        print(f"  {pt_name} does not exist (already removed)")

print(f"Total Weights Deleted: {len(deleted_pts)} files ({deleted_pt_bytes/(1024*1024):.2f} MB)")
print("Preserved weights: best.pt (active baseline) and yolov8n.pt (clean baseline)")
