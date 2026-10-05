import json
from pathlib import Path

nb_path = Path("E:/CV-MID/colab_train.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update markdown heading cell 0
nb["cells"][0]["source"] = [
    "# SmartNav YOLO26n Edge Training on T4 GPU (33 Classes - 416x416 Mobile Optimized)\n",
    "\n",
    "## Visual Obstacle Detection & Indoor Navigation for Wearable Assistive Glasses\n",
    "\n",
    "### Workflow\n",
    "1. Dataset `datasets/smartnav_v1_33cls_20260920` is already configured in `datasets/data.yaml`.\n",
    "2. Zip `datasets/smartnav_v1_33cls_20260920/train`, `val`, and `data.yaml` as `datasets_colab.zip`:\n",
    "   ```powershell\n",
    "   Compress-Archive -Path datasets\\smartnav_v1_33cls_20260920\\train,datasets\\smartnav_v1_33cls_20260920\\val,datasets\\smartnav_v1_33cls_20260920\\data.yaml -DestinationPath datasets_colab.zip -Force\n",
    "   ```\n",
    "3. Upload `datasets_colab.zip` to Google Drive root (`MyDrive/datasets_colab.zip`).\n",
    "4. Run cells below:\n",
    "   - **Mount & Extract**\n",
    "   - **Rare-Class Oversampling** (boosts manhole, rickshaw, van, stop_sign, cow)\n",
    "   - **Train YOLO26n** (imgsz=416, epochs=120, patience=20)\n",
    "   - **Export INT8 TFLite** for on-device Android phone deployment\n"
]

# Find train cell and update
train_cell = None
resume_cell = None
for c in nb["cells"]:
    if c.get("metadata", {}).get("id") == "train":
        train_cell = c
    elif c.get("metadata", {}).get("id") == "resume":
        resume_cell = c

# Oversampling cell to insert before train cell
oversample_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"id": "oversample"},
    "outputs": [],
    "source": [
        "# Class-weighted oversampling for rare classes before training\n",
        "# Duplicates rare class image & label entries so they appear more frequently per epoch\n",
        "import os, shutil\n",
        "from pathlib import Path\n",
        "from collections import Counter\n",
        "\n",
        "DATASET_ROOT = Path('/content/datasets')\n",
        "train_lbl_dir = DATASET_ROOT / 'train' / 'labels'\n",
        "train_img_dir = DATASET_ROOT / 'train' / 'images'\n",
        "\n",
        "# Rare class IDs to oversample: manhole(11), rickshaw(6), van(7), stop_sign(14), cow(19)\n",
        "OVERSAMPLE_CONFIG = {\n",
        "    11: 8,  # manhole: multiply by 8\n",
        "    6:  3,  # rickshaw: multiply by 3\n",
        "    7:  4,  # van: multiply by 4\n",
        "    14: 3,  # stop_sign: multiply by 3\n",
        "    19: 2,  # cow: multiply by 2\n",
        "}\n",
        "\n",
        "added_count = 0\n",
        "for lbl_file in list(train_lbl_dir.glob('*.txt')):\n",
        "    stem = lbl_file.stem\n",
        "    if '_os_' in stem:\n",
        "        continue\n",
        "    cids = set()\n",
        "    with open(lbl_file, 'r', errors='ignore') as fp:\n",
        "        for line in fp:\n",
        "            p = line.split()\n",
        "            if p:\n",
        "                try:\n",
        "                    cids.add(int(p[0]))\n",
        "                except ValueError:\n",
        "                    pass\n",
        "    max_mult = max([OVERSAMPLE_CONFIG.get(cid, 1) for cid in cids], default=1)\n",
        "    if max_mult > 1:\n",
        "        # Find source image\n",
        "        img_path = None\n",
        "        for ext in ['.jpg', '.png', '.jpeg']:\n",
        "            cand = train_img_dir / f'{stem}{ext}'\n",
        "            if cand.exists():\n",
        "                img_path = cand\n",
        "                break\n",
        "        if img_path:\n",
        "            for m in range(1, max_mult):\n",
        "                os_stem = f'{stem}_os_{m}'\n",
        "                shutil.copy2(lbl_file, train_lbl_dir / f'{os_stem}.txt')\n",
        "                shutil.copy2(img_path, train_img_dir / f'{os_stem}{img_path.suffix}')\n",
        "                added_count += 1\n",
        "\n",
        "print(f'Oversampling complete: added {added_count} rare-class training samples.')\n",
        "print(f'Total train images after oversampling: {len(os.listdir(train_img_dir))}')\n"
    ]
}

if train_cell:
    train_cell["source"] = [
        "import shutil, os\n",
        "from ultralytics import YOLO\n",
        "\n",
        "CHECKPOINT_DIR = '/content/drive/MyDrive/vqa_training'\n",
        "LAST_CKPT = f'{CHECKPOINT_DIR}/last.pt'\n",
        "BEST_CKPT  = f'{CHECKPOINT_DIR}/best.pt'\n",
        "RUNS_DIR   = '/content/runs/smartnav_yolo26n'\n",
        "os.makedirs(CHECKPOINT_DIR, exist_ok=True)\n",
        "\n",
        "def save_checkpoint(trainer):\n",
        "    src_last = f'{trainer.save_dir}/weights/last.pt'\n",
        "    src_best = f'{trainer.save_dir}/weights/best.pt'\n",
        "    if os.path.exists(src_last):\n",
        "        shutil.copy(src_last, LAST_CKPT)\n",
        "    if os.path.exists(src_best):\n",
        "        shutil.copy(src_best, BEST_CKPT)\n",
        "    print(f'  -> Checkpoints saved to Drive ({CHECKPOINT_DIR})')\n",
        "\n",
        "# Use YOLO26n specifically (NMS-free end-to-end head for mobile edge deployment)\n",
        "model = YOLO('yolo26n.pt')\n",
        "model.add_callback('on_train_epoch_end', save_checkpoint)\n",
        "\n",
        "# Training config: imgsz=416, epochs=120, patience=20, mobile augmentation\n",
        "results = model.train(\n",
        "    data='/content/datasets/data.yaml',\n",
        "    epochs=120,\n",
        "    patience=20,\n",
        "    imgsz=416,\n",
        "    batch=32,\n",
        "    optimizer='AdamW',\n",
        "    lr0=0.001,\n",
        "    device=0,\n",
        "    workers=4,\n",
        "    seed=42,\n",
        "    cos_lr=True,\n",
        "    warmup_epochs=3,\n",
        "    amp=True,\n",
        "    mosaic=1.0,\n",
        "    mixup=0.1,\n",
        "    degrees=10.0,\n",
        "    fliplr=0.5,\n",
        "    project='/content/runs',\n",
        "    name='smartnav_yolo26n',\n",
        "    exist_ok=True\n",
        ")\n"
    ]

if resume_cell:
    resume_cell["source"] = [
        "import shutil, os\n",
        "from ultralytics import YOLO\n",
        "\n",
        "CHECKPOINT_DIR = '/content/drive/MyDrive/vqa_training'\n",
        "LAST_CKPT = f'{CHECKPOINT_DIR}/last.pt'\n",
        "RUNS_DIR  = '/content/runs/smartnav_yolo26n'\n",
        "\n",
        "if not os.path.exists(LAST_CKPT):\n",
        "    raise FileNotFoundError(f'{LAST_CKPT} not found. Run Train cell first.')\n",
        "\n",
        "os.makedirs(f'{RUNS_DIR}/weights', exist_ok=True)\n",
        "shutil.copy(LAST_CKPT, f'{RUNS_DIR}/weights/last.pt')\n",
        "\n",
        "def save_checkpoint(trainer):\n",
        "    src_last = f'{trainer.save_dir}/weights/last.pt'\n",
        "    src_best = f'{trainer.save_dir}/weights/best.pt'\n",
        "    if os.path.exists(src_last):\n",
        "        shutil.copy(src_last, LAST_CKPT)\n",
        "    if os.path.exists(src_best):\n",
        "        shutil.copy(src_best, f'{CHECKPOINT_DIR}/best.pt')\n",
        "    print(f'  -> Checkpoints saved to Drive')\n",
        "\n",
        "model = YOLO(f'{RUNS_DIR}/weights/last.pt')\n",
        "model.add_callback('on_train_epoch_end', save_checkpoint)\n",
        "model.train(resume=True)\n"
    ]

# Add TFLite INT8 Export cell at end
export_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"id": "tflite_export"},
    "outputs": [],
    "source": [
        "# Export to TFLite INT8 for On-Device Android Phone Deployment\n",
        "from ultralytics import YOLO\n",
        "import shutil, os\n",
        "\n",
        "CHECKPOINT_DIR = '/content/drive/MyDrive/vqa_training'\n",
        "BEST_CKPT = f'{CHECKPOINT_DIR}/best.pt'\n",
        "\n",
        "print('Loading best trained YOLO26n model...')\n",
        "model = YOLO(BEST_CKPT)\n",
        "\n",
        "print('Exporting to TFLite with INT8 quantization at imgsz=416...')\n",
        "tflite_path = model.export(\n",
        "    format='tflite',\n",
        "    int8=True,\n",
        "    data='/content/datasets/data.yaml',\n",
        "    imgsz=416\n",
        ")\n",
        "\n",
        "print(f'TFLite export complete: {tflite_path}')\n",
        "dst_drive = '/content/drive/MyDrive/smartnav_yolo26n_int8.tflite'\n",
        "shutil.copy(tflite_path, dst_drive)\n",
        "print(f'Saved INT8 model to Drive: {dst_drive} ({os.path.getsize(dst_drive)/1024/1024:.2f} MB)')\n"
    ]
}

# Insert oversample_cell before train_cell
train_idx = None
for i, c in enumerate(nb["cells"]):
    if c.get("metadata", {}).get("id") == "train":
        train_idx = i
        break

if train_idx is not None:
    nb["cells"].insert(train_idx, oversample_cell)

# Append export cell
nb["cells"].append(export_cell)

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated colab_train.ipynb successfully.")
