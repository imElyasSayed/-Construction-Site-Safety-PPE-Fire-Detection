# Construction Safety Monitor — CSCI435 Project

A FastAPI backend that detects PPE violations and fire/smoke hazards on construction
sites from uploaded images and video.

## Structure

```
Safety-Monitor-Project/
  main.py                     FastAPI app and API routes
  services/
    ppe_detector.py           PPE violation detection (YOLOv8, fine-tuned)
    firesmoke_detector.py     Fire/smoke detection (YOLOv8, fine-tuned)
  models/
    ppe_model.pt              Fine-tuned PPE weights
    firesmoke_model.pt        Fine-tuned fire/smoke weights
  uploads/                    Sample input files
  outputs/                    Sample annotated outputs
  training/
    ppe/                      PPE training notebook + module README
    firesmoke/                Fire/smoke training script, data.yaml, module README
  docs/                       Confusion matrix / training curve images for the report
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload
```

## Endpoints

- `POST /detect/image` — upload an image, returns PPE detections, alerts, and an
  annotated output image path.
- `POST /detect/video` — upload a video, returns frame-by-frame PPE detections and an
  annotated output video path.

Fire/smoke detection (`services/firesmoke_detector.py`) is implemented and ready to use
but not yet wired into a `main.py` route — see `training/firesmoke/README.md` for its
`detect_from_path` / `detect_from_bytes` API.

## Models

| Model | Classes | mAP50 |
|---|---|---|
| `ppe_model.pt` | Hardhat, Vest, Mask, Gloves, Goggles, Fall-Detected (+ NO- variants) | 0.75+ |
| `firesmoke_model.pt` | fire, smoke | 0.763 |

Training details, dataset sources, and metrics are documented in `training/ppe/README.md`
and `training/firesmoke/README.md`.
