# Visual QA

Premium dark AI dashboard for object detection, Visual Q&A, typed or voice questions, camera capture, live stream inference, benchmarks, and latency metrics.

## Quick Start

```powershell
python -m pip install -r requirements.txt
python scripts/setup.py
uvicorn app:app --reload --port 8000
```

Open `http://localhost:8000`.

## Portable Dataset Setup

`python scripts/setup.py` is the normal setup command for new users.

It chooses the dataset source in this order:

1. `datasets/roboflow_export.zip`
2. `roboflow_export.zip`
3. `ROBOFLOW_EXPORT_URL` / `ROBOFLOW_EXPORT_ZIP` in `.env` or the shell
4. `config/dataset_sources.json`
5. COCO + Open Images fallback, then local `roboflow_zips/` merge if present

Roboflow exports should be downloaded as **YOLOv8** format.

For a private Roboflow URL:

```powershell
Copy-Item .env.example .env
# then set ROBOFLOW_EXPORT_URL=... in .env
python scripts/setup.py
```

For a fully portable release bundle, place the exported zip at:

```text
datasets/roboflow_export.zip
```

Then new users only run `python scripts/setup.py`.

For multiple manually downloaded Roboflow exports, place them in:

```text
roboflow_zips/
```

Use clear class-style names such as `backpack.zip`, `traffic_light.zip`, and `stairs.zip`. If two zips are for the same class, use `stairs.zip`, `stairs1.zip`, etc. The setup script merges these into the existing train/val dataset instead of replacing COCO/Open Images classes. If a Roboflow class conflicts with an existing `images_by_class` class, it is merged into the existing class name. Already merged zips are recorded in `cache/roboflow/merged_zips.json`, so rerunning setup will not duplicate the same Roboflow images. Use `--force-roboflow-merge` only when you intentionally want to re-import recorded zips.

For broad Roboflow datasets, rename downloads to one of these supported names so extra classes are filtered into the current class list instead of creating random new classes:

```text
logistics.zip
flir_camera_objects.zip
fsod.zip
objects_365.zip
bdd100k.zip
idd.zip
sod.zip
fruit_detection.zip
household_items.zip
sports_equipment.zip
```

Extracted Roboflow exports also work with `--roboflow-dir`:

```powershell
python scripts/setup.py --roboflow-dir Logistics
```

After label generation or Roboflow merge, setup also rebalances train/val so classes from zips without a validation split still get validation samples. The rebalance tries to give each class up to 25 validation images and 25 validation boxes while keeping at least 25 training images/boxes when possible.

## Removing Classes

To remove weak or unwanted classes, remove them from every source that can recreate them:

1. Delete the class folder from `datasets/coco/images_by_class/`.
2. Remove the class name from `RETAINED_CLASS_FOLDERS` in `scripts/setup.py` if it is a retained COCO/Open Images class.
3. Delete or move the matching zip from `roboflow_zips/` if the class comes from Roboflow.
4. Rebuild train/val/data.yaml:

```powershell
Remove-Item -Recurse -Force datasets\train,datasets\val,datasets\data.yaml
python scripts/setup.py
python scripts/train.py --dry-run
```

If you only removed class folders and do not want to merge Roboflow again, run:

```powershell
python scripts/setup.py --labels
```

Do not remove classes only from `data.yaml`; labels and class indexes must be regenerated together.

## Final Assistive Dataset

The final mobile-focused dataset is defined in `src/class_taxonomy.py`. It renames old labels such as `cell phone` to `mobile` and `dining table` to `table`, and groups classes into assistive categories such as hazards, indoor, personal items, vehicles, and food.

The current local dataset has already been pruned to the final assistive class list. To add the missing `handrail` and `crosswalk` zips, place the YOLOv8 Roboflow zip files in `roboflow_zips/`, then run:

```powershell
python scripts/setup.py --roboflow-dir roboflow_zips
python scripts/train.py --dry-run
```

Discarded data from earlier pruning runs is kept in `cache/final_dataset_prune/` instead of being permanently deleted.

## Dataset Commands

```powershell
python scripts/setup.py --roboflow path_or_url     # Import a Roboflow YOLOv8 export directly
python scripts/setup.py --roboflow-dir roboflow_zips # Merge all Roboflow zips from a folder
python scripts/setup.py --no-roboflow              # Force COCO + Open Images fallback
python scripts/setup.py --coco-only                # Only COCO80
python scripts/setup.py --oi-only                  # Only Open Images expansion
python scripts/setup.py --labels                   # Regenerate labels/splits from class folders
python scripts/setup.py --fix-splits               # Repair existing train/val class coverage
python scripts/setup.py --rebuild-previews         # Refresh images_by_class from train/val labels
python scripts/setup.py --dry-run                  # Preview without downloads/writes
python scripts/setup.py --force-roboflow-merge     # Intentionally re-import recorded Roboflow zips
```

Run `python scripts/setup.py --fix-splits` if diagnostics show `NO_VAL` or very low-validation classes such as `traffic light` with train boxes but too few validation boxes.

The old web-search expansion is disabled by default because it can download unrelated images. Use `--web-scrape` only for manual experiments after reviewing images.

## Training Preview

```powershell
python scripts/train.py
```

For Colab training, run setup, preview the dataset, then zip only the training artifacts:

```powershell
python scripts/setup.py
python scripts/train.py --dry-run
python scripts/train.py --class-counts
Compress-Archive -Path datasets\train,datasets\val,datasets\data.yaml -DestinationPath datasets_colab.zip -Force
```

Upload `datasets_colab.zip` to Colab and unzip so the files are under `/content/datasets/`. If you are replacing an older upload in the same Colab runtime, remove old label caches before training:

```python
!rm -f /content/datasets/train/labels.cache /content/datasets/val/labels.cache
```

Then run `colab_train.ipynb`.

## Training Warnings

If Ultralytics prints this warning:

```text
Box and segment counts should be equal ... only boxes will be used
```

the dataset has mixed detect and segmentation labels. The setup script now converts Roboflow polygon rows into detect-only YOLO boxes during merge. If you see this after re-uploading to Colab, re-run local setup or re-zip the cleaned `datasets/train`, `datasets/val`, and `datasets/data.yaml`, then delete the Colab `labels.cache` files.

If Ultralytics prints:

```text
duplicate labels removed
```

the label file had repeated rows. The setup script removes exact duplicate rows during Roboflow merge. If this appears from an old zip, regenerate or re-zip after running the current setup script.

## Voice Questions

After uploading or capturing an image and running Analyze, users can ask VQA questions by typing or by clicking `Mic`.

Voice input first tries the browser speech-recognition API in Chrome/Edge on `localhost` or HTTPS. If the browser speech service is blocked or unavailable, the app falls back to recording audio in the browser and sending it to the FastAPI `/transcribe` endpoint.

Server transcription supports Groq or AssemblyAI. Choose with:

```env
TRANSCRIBE_PROVIDER=auto
```

`auto` uses the fastest fallback chain for short recorded mic clips: Groq `whisper-large-v3-turbo`, then Groq `whisper-large-v3`, then AssemblyAI `universal-3-pro`.

Groq transcription:

```env
GROQ_API_KEY=your_key_here
GROQ_TRANSCRIBE_MODELS=whisper-large-v3-turbo,whisper-large-v3
```

AssemblyAI transcription:

```env
ASSEMBLYAI_API_KEY=your_key_here
# ASS_API=your_key_here also works
ASSEMBLYAI_SPEECH_MODELS=universal-3-pro
```

If the mic does not start:

1. Use Chrome or Edge.
2. Open the app at `http://127.0.0.1:8000` or `http://localhost:8000`.
3. Allow microphone access from the browser address-bar permission popup.
4. Hard refresh after frontend updates with `Ctrl + F5`.
5. If browser speech recognition says the service is unavailable, click `Mic`, speak, then click `Rec` again to use server transcription.

## Benchmark

```powershell
python scripts/benchmark.py
python scripts/benchmark.py --all-classes
```

## Dataset Diagnostics

Older one-off diagnosis and Open Images helper scripts are archived in `cache/archived_scripts/`. Use `python scripts/train.py --dry-run` for the quick dataset readiness check, and use `python scripts/benchmark.py` after training to compare model speed and detections.
