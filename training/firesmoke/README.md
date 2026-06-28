# Fire & Smoke Detection Module

A YOLOv8-based fire and smoke detection model fine-tuned on a large real-world dataset. This module is designed to plug directly into a FastAPI backend for real-time detection from camera feeds or uploaded images.

---

## Model Overview

The **deployed** model (`models/firesmoke_model.pt`) is the fine-tuned **YOLOv8m**, retrained on a
merged, smoke-balanced dataset to fix the weak smoke class. The original YOLOv8s baseline is kept
for comparison.

| Property | Baseline | Deployed (fine-tuned) |
|---|---|---|
| Base model | YOLOv8s (COCO) | **YOLOv8m** (COCO) |
| Classes | `fire`, `smoke` | `fire`, `smoke` |
| Dataset | Cubeai Fire & Smoke (Kaggle), ~7k train / ~2k val | **Merged**: Cubeai + sayedgamal99, smoke oversampled toward 1:1 |
| Image size | 640×640 | 768×768 (higher res for thin/distant smoke) |
| Epochs | 50 | 40 |
| Augmentation | HSV, scale, flip, mosaic | + wider HSV/scale, mixup, rotation |
| Training platform | Kaggle/Colab T4 | Kaggle T4 (P100 compatible) |

---

## Results

**Baseline (YOLOv8s):**

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| all | 0.794 | 0.710 | 0.763 | 0.457 |
| fire | 0.865 | 0.798 | 0.859 | 0.549 |
| smoke | 0.731 | 0.620 | 0.667 | 0.365 |

**Fine-tuned (YOLOv8m, merged data) — the deployed model:**

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| all | 0.845 | 0.769 | 0.820 | 0.499 |
| fire | 0.863 | 0.775 | 0.839 | 0.510 |
| smoke | 0.830 | 0.761 | **0.801** | 0.488 |

**Headline:** smoke mAP50 jumped **0.667 → 0.801** — the weak class was the whole point of the
retrain. Fire dipped slightly (0.859 → 0.839), an acceptable trade since fire was already strong.

> ⚠️ Provisional comparison: the baseline was measured on the original Cubeai val set and the
> fine-tuned model on the merged val set — different yardsticks. For an airtight same-val-set
> number, run the eval cell in `training/kaggle_train_firesmoke_v2.ipynb`.

---

## Files

```
training/firesmoke/
├── data.yaml            # baseline dataset config (classes + paths)
├── train_firesmoke.py   # standalone retrain script (yolov8m, reference)
├── training_args.yaml   # exact args from the baseline run
└── README.md            # this file
```

Deployed weights: `models/firesmoke_model.pt` (loaded by `services/firesmoke_detector.py`).
Full merge + retrain pipeline: `training/kaggle_train_firesmoke_v2.ipynb`.

---

## How to Use

Inference is **not** standalone here — the deployed model is loaded and run by the backend module
`services/firesmoke_detector.py`, which the unified pipeline calls automatically. Just run the app
(see the main README) and upload an image/video.

To call the detector directly from Python:

```python
from services import firesmoke_detector
import cv2

frame = cv2.imread("uploads/fire_on_construction_sites.jpg")
detections = firesmoke_detector.detect(frame)   # uses models/firesmoke_model.pt
```

### Detection output format
```python
[
    {
        "class": "fire",         # or "smoke"
        "confidence": 0.87,      # 0 to 1
        "box": [x1, y1, x2, y2]  # bounding box in pixels
    }
]
```

### Retraining
```bash
# standalone script (yolov8m, early-stopped)
python train_firesmoke.py --data path/to/merged/data.yaml
# or the full merge + oversample + train pipeline:
#   training/kaggle_train_firesmoke_v2.ipynb  (run on Kaggle GPU)
```
Drop the resulting `best.pt` into `models/firesmoke_model.pt` and restart the server — no code
change needed, as long as the class names stay `{0:'fire', 1:'smoke'}`.

---

## Notes for Backend Integration

- Default confidence threshold is `0.25` — lower it to `0.1` for more sensitive detection
- Fire/smoke inference is ~200 ms/image on CPU; the full app pipeline (3 models) reaches ~13 FPS
  on a Tesla T4 GPU — see the main README's Performance section
- Input can be any image format (JPG, PNG, etc.)
- `annotated_image` returned is a NumPy array (BGR) — convert with `cv2.imencode` to send over API

---

## Training Details

The baseline was fine-tuned from `yolov8s.pt`. An initial 100-image Pascal VOC attempt yielded a
low mAP50 of 0.315; replacing it with the ~7,000-image Cubeai YOLO dataset lifted that to 0.763.

**Smoke-focused retrain (deployed model).** Smoke was the weak class (recall ~0.62, capped by a
~5:1 fire:smoke imbalance), so the v2 retrain:

1. **merges** several fire/smoke datasets (Cubeai + sayedgamal99) and remaps every label to a
   unified `fire=0 / smoke=1`. This also normalizes the Cubeai dataset's Chinese class names
   (`火`/`烟`) — they **must** be remapped or the backend's `{0:'fire', 1:'smoke'}` contract breaks;
2. **oversamples** smoke-containing images toward a 1:1 balance;
3. trains **YOLOv8m at 768px for 40 epochs** with stronger colour/scale augmentation + mixup.

Full pipeline is in `training/kaggle_train_firesmoke_v2.ipynb`; `train_firesmoke.py` is the
standalone reference script. (Exact deployed config read from the checkpoint:
`model=yolov8m.pt, imgsz=768, epochs=40`.)
