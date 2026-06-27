# PPE Detection Module — YOLOv8 Fine-Tuning
## CSCI435 Computer Vision Project — Construction Site Safety Monitor

---

## Overview

This module provides a fine-tuned YOLOv8 model for real-time Personal Protective Equipment (PPE) detection on construction sites. The model detects 14 classes of PPE and safety violations including helmets, safety vests, masks, gloves, goggles, and their missing counterparts. The **deployed** model (`models/ppe_model.pt`) is the YOLOv8m retrain; the original YOLOv8s baseline is kept for comparison.

---

## Model Details

| Property | Baseline | Deployed (fine-tuned) |
|---|---|---|
| Base model | YOLOv8s (COCO) | **YOLOv8m** (COCO) |
| Dataset | Roboflow PPE Combined Model v4 (31k train / 8.8k val, 14 classes) | same |
| Epochs | 20 | 30 |
| Image size | 640×640 | 768×768 |

*(Deployed config read from the checkpoint: `model=yolov8m.pt, imgsz=768, epochs=30`.)*

---

## Classes Detected

| ID | Class | Type |
|---|---|---|
| 0 | Fall-Detected | Hazard |
| 1 | Gloves | PPE Present |
| 2 | Goggles | PPE Present |
| 3 | Hardhat | PPE Present |
| 4 | Ladder | Equipment |
| 5 | Mask | PPE Present |
| 6 | NO-Gloves | PPE Violation |
| 7 | NO-Goggles | PPE Violation |
| 8 | NO-Hardhat | PPE Violation |
| 9 | NO-Mask | PPE Violation |
| 10 | NO-Safety Vest | PPE Violation |
| 11 | Person | Person |
| 12 | Safety Cone | Equipment |
| 13 | Safety Vest | PPE Present |

---

## Performance Metrics

Overall, on the same Roboflow v4 validation set (8,814 images) — a direct, fair comparison:

| Metric | Baseline (yolov8s) | Deployed (yolov8m) |
|---|---|---|
| Precision | 0.714 | 0.712 |
| Recall | 0.815 | **0.837** |
| mAP50 | 0.774 | **0.785** |
| mAP50-95 | 0.501 | **0.518** |

The yolov8m retrain at 768px improves mAP50 (0.774 → 0.785) and mAP50-95 (0.501 → 0.518), driven
mainly by higher recall (0.815 → 0.837). Numbers are the deployed checkpoint's recorded
`train_metrics`; the `docs/ppe_results.png` / `ppe_confusion_matrix.png` charts are from the
yolov8s baseline run.

---

## Files

```
ppe_detection/
  best.pt          — fine-tuned model weights
  standalone_inference.py     — inference script (image, video, webcam)
  data.yaml        — dataset configuration
  results.png      — training curves
  confusion_matrix.png — per-class confusion matrix
  README.md        — this file
```

---

## Usage

### Install dependencies
```bash
pip install ultralytics opencv-python
```

### Run on image
```bash
python standalone_inference.py --mode image --input path/to/image.jpg --output output.jpg
```

### Run on video
```bash
python standalone_inference.py --mode video --input path/to/video.mp4 --output output.mp4
```

### Run on webcam
```bash
python standalone_inference.py --mode webcam
```

---

## Dataset

Dataset: Roboflow PPE Combined Model v4
URL: https://universe.roboflow.com/roboflow-universe-projects/personal-protective-equipment-combined-model/dataset/4
License: CC BY 4.0

---

## Integration

The backend module loads the model as follows:

```python
from ultralytics import YOLO
model = YOLO('best.pt')
results = model('image.jpg')
```

The `standalone_inference.py` script exposes three functions for direct import:
- `run_on_image(image_path, output_path)`
- `run_on_video(video_path, output_path)`
- `run_on_webcam(camera_index)`
