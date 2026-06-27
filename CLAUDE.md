# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## What this is

**Construction Safety Monitor** — a CSCI435 computer-vision project. A FastAPI app
where you upload one construction-site image or video and it runs every safety
capability at once, returning a single combined verdict plus one annotated output.
A React dashboard is served at `/`.

## Architecture

```
main.py                  FastAPI app: /detect/image, /detect/video; serves the frontend
services/
  config.py              Model paths, thresholds, inference-time switches (VIDEO_STRIDE, *_CONF, USE_TTA_IMAGE)
  enhancement.py         CLAHE image enhancement (low-light/dusty/smoky robustness)
  ppe_detector.py        PPE detection (YOLOv8) + ByteTrack tracking
  firesmoke_detector.py  Fire/smoke detection (YOLOv8); class names MUST be {0:'fire',1:'smoke'}
  segmentation.py        Fire/smoke pixel masking (HSV + morphology) -> severity = % of frame
  pose_detector.py       YOLOv8-pose keypoints + fall heuristic (yolov8s-pose.pt auto-downloads)
  tracking.py            Per-worker violation aggregation across video frames
  alerts.py              Combined alert + severity/level logic
  pipeline.py            Orchestrates everything for image & video
frontend/                React + Vite dashboard (see "Frontend" below)
models/                  ppe_model.pt, firesmoke_model.pt (committed; ~21MB / ~50MB)
training/                Retraining notebooks + scripts (Kaggle/Colab)
docs/                    Confusion matrices / training curves
uploads/ outputs/        Sample inputs / annotated results
```

The pipeline is the entry point for all CV work — read `services/pipeline.py` first.

## Capabilities (the CSCI435 requirements)

- **Object detection + recognition** — two YOLOv8 models: PPE (14 classes incl. `NO-` violations) and fire/smoke
- **Object tracking** — ByteTrack assigns persistent worker IDs across video frames; violations deduped per worker
- **Keypoint/pose detection** — YOLOv8-pose + posture heuristic for fall detection
- **Image segmentation + severity** — classical HSV + morphology masks fire/smoke pixels -> spread %
- **Binary morphological operations** — opening/closing to clean masks
- **Image enhancement** — CLAHE preprocessing

## Running the app

Backend (serves the built frontend at `/`):
```bash
.venv/bin/uvicorn main:app --reload --port 8000
```
Open http://localhost:8000/ . The first run downloads `yolov8s-pose.pt` (~22 MB).

`main.py` serves `frontend/dist` if it exists, else `frontend/`. **`frontend/dist` is
committed** so the app runs without a Node build. Background servers launched by tooling
get reaped — run uvicorn in your own terminal for a persistent server.

## Frontend (React + Vite)

Source in `frontend/src/` (`App.jsx`, `index.css`, `main.jsx`); built output in
`frontend/dist/`. **After ANY change to `frontend/src/`, you MUST rebuild** or the running
app won't reflect it:
```bash
cd frontend && npm run build      # regenerates frontend/dist (which main.py serves)
```
- Design system: custom CSS in `src/index.css` — CSS vars (`--yellow #F2C200`, `--red`,
  `--green`, steel grays), fonts Oswald/Inter/Roboto Mono, numbered `.panel`/`.phead` sections.
- **Model Performance panel** (section "07"): `src/ModelPerformance.jsx` renders an old-vs-new
  metric comparison (mAP50 / mAP50-95 / Precision / Recall, per model). Its data is
  `src/model_stats.json` (bundled at build). Update that JSON when retrains land.

## Models & training

| Model | File | Classes | Status |
|---|---|---|---|
| PPE | `models/ppe_model.pt` | 14 (Hardhat, Vest, Mask, Gloves, Goggles, +`NO-` variants, Person, Ladder, Cone, Fall) | **yolov8m, mAP50 0.774 -> 0.785** (768px, 30ep; same Roboflow v4 val set) |
| Fire/smoke | `models/firesmoke_model.pt` | fire, smoke | yolov8m, **smoke mAP50 0.667 -> 0.801** after merged-data retrain |

Backend loads whatever weights are in `models/` — drop a new `best.pt` in (named
`ppe_model.pt` / `firesmoke_model.pt`) and restart uvicorn. No code change needed, **as long
as fire/smoke class names are `fire`/`smoke`** (the cubeai dataset ships Chinese `火`/`烟` —
rename in data.yaml before training).

Training notebooks (run on Kaggle, GPU):
- `training/kaggle_train_firesmoke_v2.ipynb` — fire/smoke: merges multiple datasets, remaps
  all classes to `fire=0/smoke=1`, oversamples smoke to fix imbalance, trains @ 768px.
- `training/kaggle_train_ppe_v2.ipynb` — PPE: yolov8m @ 768px on Roboflow v4 (gitignored:
  contains a hardcoded Roboflow key).
- `training/kaggle_train.ipynb`, `colab_train.ipynb` — older combined notebooks (gitignored).

### Hard-won Kaggle/training gotchas
- **P100 needs a torch downgrade.** P100 is Pascal (sm_60); the default ultralytics install
  pulls a CUDA-12.8 torch built only for sm_70+, which crashes with
  `CUDA error: no kernel image is available`. Fix: `pip install torch==2.6.0
  torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124` (cu124 still ships
  sm_60). **T4 needs no downgrade** (sm_75 works with stock torch) and is faster for AMP.
- **Restarting a Kaggle session WIPES pip installs.** Install must be the first cell, then
  run everything in the same session — never restart after installing. `import torch` returns
  the already-loaded module, so the install cell must run before any torch import.
- The `google-adk`/`dask-cuda`/`cuml` pip "dependency conflict" errors are harmless noise.
- Old-vs-new mAP is only comparable on the **same** val set (PPE uses the same Roboflow v4 set,
  so its comparison is direct; fire/smoke baseline was on a different set).

## Conventions & gotchas

- Use the project venv: `.venv/bin/python`, `.venv/bin/uvicorn`.
- Tune accuracy without retraining via `services/config.py` (`*_CONF`, `USE_TTA_IMAGE`,
  `VIDEO_STRIDE`, `VIDEO_MAX_WIDTH`). Lower `FIRESMOKE_CONF` to raise smoke recall.
- Video is processed every Nth frame and downscaled (CPU inference) — see config.
- `.gitignore` excludes `.venv/`, `frontend/node_modules/`, `runs/`, `yolov8s-pose.pt`, and
  the key-bearing notebooks. `frontend/dist/` is intentionally committed.

## Common commands

```bash
.venv/bin/uvicorn main:app --reload --port 8000     # run the app
cd frontend && npm run build                         # rebuild the dashboard after src/ edits
.venv/bin/python -c "from ultralytics import YOLO; print(YOLO('models/firesmoke_model.pt').names)"
```
