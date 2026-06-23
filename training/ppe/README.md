# PPE Detection Module — YOLOv8 Fine-Tuning
## CSCI435 Computer Vision Project — Construction Site Safety Monitor

---

## Overview

This module provides a fine-tuned YOLOv8s model for real-time Personal Protective Equipment (PPE) detection on construction sites. The model detects 14 classes of PPE and safety violations including helmets, safety vests, masks, gloves, goggles, and their missing counterparts.

---

## Model Details

| Property | Value |
|---|---|
| Base Model | YOLOv8s (pretrained on COCO) |
| Dataset | Roboflow PPE Combined Model v4 |
| Training Images | 31,000+ |
| Validation Images | 8,800+ |
| Classes | 14 |
| Epochs | 20 |
| Image Size | 640x640 |

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

| Metric | Score |
|---|---|
| mAP50 | 0.75+ |
| Precision | 0.80+ |
| Recall | 0.75+ |

---

## Files

```
ppe_detection/
  best.pt          — fine-tuned model weights
  inference.py     — inference script (image, video, webcam)
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
python inference.py --mode image --input path/to/image.jpg --output output.jpg
```

### Run on video
```bash
python inference.py --mode video --input path/to/video.mp4 --output output.mp4
```

### Run on webcam
```bash
python inference.py --mode webcam
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

The `inference.py` script exposes three functions for direct import:
- `run_on_image(image_path, output_path)`
- `run_on_video(video_path, output_path)`
- `run_on_webcam(camera_index)`
