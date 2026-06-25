# Construction Safety Monitor — CSCI435 Project

A FastAPI application that scans construction-site images and video for safety hazards and
returns a single combined safety verdict. One upload runs every capability and produces one
annotated output plus a unified alerts/severity summary, viewed through a built-in web UI.

## Capabilities

| CV capability | What it does here | Hazard pillar |
|---|---|---|
| **Object detection + recognition** | YOLOv8 detects & classifies PPE (hardhat, vest, mask, gloves, goggles + `NO-` violations) and fire/smoke | PPE, fire |
| **Object tracking** | ByteTrack assigns persistent worker IDs across video frames; alerts are deduped and violation durations reported per worker | PPE / monitoring |
| **Keypoint / pose detection** | YOLOv8-pose finds body keypoints; a posture heuristic flags falls (OR'd with the `Fall-Detected` class) | Falls |
| **Image segmentation + severity** | Classical HSV + morphology masks the actual fire/smoke pixels inside detected regions → spread **% of frame** as a severity signal | Fire & smoke |
| **Image enhancement** | CLAHE preprocessing improves detection in low-light / dusty / smoky frames | (robustness) |
| **Binary morphological operations** | Opening/closing cleans the fire/smoke masks | (segmentation) |

## Structure

```
main.py                       FastAPI app: unified routes + serves the frontend
services/
  config.py                   Model paths, thresholds, inference-time switches
  enhancement.py              CLAHE image enhancement
  ppe_detector.py             PPE detection + ByteTrack tracking
  firesmoke_detector.py       Fire/smoke detection
  segmentation.py             Fire/smoke masking + morphology + severity
  pose_detector.py            Pose keypoints + fall heuristic
  tracking.py                 Per-worker violation aggregation
  alerts.py                   Combined alert + severity logic
  pipeline.py                 Orchestrates everything (image & video)
frontend/                     React (Vite) dashboard — src/ source, dist/ built bundle (served at /)
models/                       ppe_model.pt, firesmoke_model.pt (yolov8s-pose.pt auto-downloads)
training/                     Retraining + evaluation scripts (Colab)
docs/                         Confusion matrices / training curves
uploads/ outputs/             Sample inputs / annotated results
```

## How to run

**Prerequisites:** Python 3.10+ (tested on 3.11) and `git`. Node.js is **only** needed if
you want to rebuild the UI — the React dashboard ships pre-built in `frontend/dist/`.

1. **Get the code**
   ```bash
   git clone <repo-url>
   cd Construction-Site-Safety-PPE-Fire-Detection
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Start the server**
   ```bash
   uvicorn main:app --reload
   ```

4. **Open the app** at <http://localhost:8000/>, drag in an image or video, and click
   **Run analysis**.

**Notes**
- The first run downloads the pretrained pose model (`yolov8s-pose.pt`, ~22 MB) — needs
  internet once.
- Try the bundled samples in [uploads/](uploads/) — e.g.
  `Construction_no_safety.jpg` (image) or `Construction_video_test.mp4` (video).
- No GPU is required; a GPU only speeds up video processing.

### Rebuilding the frontend (optional — only to change the UI)

The React app is pre-built into `frontend/dist/`, which FastAPI serves directly, so
running the demo needs **no Node**. To edit the dashboard:

```bash
cd frontend
npm install
npm run build        # rebuilds frontend/dist (served by FastAPI at /)
# or: npm run dev     # Vite dev server on :5173, proxies API calls to uvicorn on :8000
```

## Endpoints

- `POST /detect/image` — runs all capabilities on an image; returns detections, alerts,
  fire/smoke severity, fall status, and an annotated image URL.
- `POST /detect/video` — runs all capabilities frame-by-frame with tracking; returns a
  per-worker violation summary, peak severity, annotated video + key-frame URLs.

Video is processed every Nth frame and downscaled (see `VIDEO_STRIDE` / `VIDEO_MAX_WIDTH`
in [services/config.py](services/config.py)) to keep CPU inference and the output file
practical — 4K every-frame on CPU is too slow for an interactive demo.

## Accuracy

Two tracks, both supported:

- **Inference-time (no retraining):** test-time augmentation (`USE_TTA_IMAGE`), CLAHE
  enhancement, and tunable per-model `conf`/`iou` thresholds — all in
  [services/config.py](services/config.py).
- **Retraining (Colab GPU):** upgraded configs in
  [training/ppe/train_ppe.py](training/ppe/train_ppe.py) and
  [training/firesmoke/train_firesmoke.py](training/firesmoke/train_firesmoke.py)
  (yolov8m base, more epochs, stronger augmentation; smoke class targeted specifically).
  Measure with [training/evaluate.py](training/evaluate.py) and drop new `best.pt` files
  into `models/` — the backend loads whatever weights are there, no code change needed.

| Model | Classes | Baseline mAP50 |
|---|---|---|
| `ppe_model.pt` | Hardhat, Vest, Mask, Gloves, Goggles, Fall-Detected (+ NO- variants) | 0.75+ |
| `firesmoke_model.pt` | fire, smoke | 0.763 (smoke 0.667 — the retraining target) |

Dataset sources and full metrics are in `training/ppe/README.md` and
`training/firesmoke/README.md`.
