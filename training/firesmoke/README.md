# Fire & Smoke Detection Module

A YOLOv8-based fire and smoke detection model fine-tuned on a large real-world dataset. This module is designed to plug directly into a FastAPI backend for real-time detection from camera feeds or uploaded images.

---

## Model Overview

| Property | Details |
|---|---|
| Base Model | YOLOv8s (pretrained on COCO) |
| Classes | `fire`, `smoke` |
| Dataset | Cubeai Fire & Smoke Detection Dataset (Kaggle) |
| Training Images | ~7,000 |
| Validation Images | ~2,000 |
| Epochs | 50 |
| Image Size | 640×640 |
| Training Platform | Google Colab (Tesla T4 GPU) |
| Training Time | ~2.15 hours |

---

## Results

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| all | 0.794 | 0.710 | 0.763 | 0.457 |
| fire | 0.865 | 0.798 | 0.859 | 0.549 |
| smoke | 0.731 | 0.620 | 0.667 | 0.365 |

---

## Files

```
firesmoke/
├── best.pt          # Trained model weights (hand this to the backend)
├── inference.py     # Detection script (hand this to the backend)
├── data.yaml        # Dataset config (classes + paths)
└── train_firesmoke.py  # Training script (for reference)
```

---

## How to Use

### Install dependencies
```bash
pip install ultralytics opencv-python
```

### Run on a single image
```bash
python inference.py path/to/image.jpg
```

This will print detections and save an annotated image with bounding boxes.

### Use in FastAPI backend

Import the two key functions from `inference.py`:

```python
from inference import detect_from_bytes, detect_from_path

# For uploaded image files (FastAPI UploadFile)
detections, annotated_image = detect_from_bytes(image_bytes)

# For local image paths
detections, annotated_image = detect_from_path("image.jpg")
```

### Detection output format
```python
[
    {
        "class": "fire",         # or "smoke"
        "confidence": 0.87,      # 0 to 1
        "box": [x1, y1, x2, y2] # bounding box in pixels
    }
]
```

---

## Notes for Backend Integration

- Default confidence threshold is `0.25` — lower it to `0.1` for more sensitive detection
- The model runs on CPU with ~200ms inference time per image
- Input can be any image format (JPG, PNG, etc.)
- `annotated_image` returned is a NumPy array (BGR) — convert with `cv2.imencode` to send over API

---

## Training Details

The model was fine-tuned from `yolov8s.pt` using the Ultralytics library. An initial attempt was made with a 100-image Pascal VOC dataset which yielded a low mAP50 of 0.315. The dataset was replaced with a larger 7,000-image YOLO-format dataset which significantly improved performance to a final mAP50 of 0.763.
