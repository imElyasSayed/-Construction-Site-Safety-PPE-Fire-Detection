# Retraining on Google Colab (free GPU)

You do **not** need to create or annotate a dataset — both models use existing public
datasets you just download. Run these cells in a Colab notebook with a GPU runtime
(`Runtime → Change runtime type → T4 GPU`).

The only inputs you must provide are your own API keys (free accounts):
- **Roboflow API key** — for the PPE dataset (account → Settings → API key)
- **Kaggle API token** — `kaggle.json` from kaggle.com → Account → Create New API Token

---

## 0. Setup

```python
!pip install -q ultralytics roboflow kaggle
```

---

## 1. PPE dataset (Roboflow) + train

Dataset: <https://universe.roboflow.com/roboflow-universe-projects/personal-protective-equipment-combined-model/dataset/4>

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")        # <-- paste your key
project = rf.workspace("roboflow-universe-projects") \
            .project("personal-protective-equipment-combined-model")
dataset = project.version(4).download("yolov8")        # downloads + writes data.yaml
print("PPE data.yaml:", dataset.location + "/data.yaml")
```

```python
# Train (uses the upgraded config: yolov8m, 80 epochs, stronger augmentation)
!python train_ppe.py --data {dataset.location}/data.yaml
```

---

## 2. Fire/smoke dataset (Kaggle) + train

Kaggle's new API tokens are a single string like `KGAT_...` (not a `kaggle.json` file).
Set it as an environment variable — don't write it into the notebook:

```python
import os
from getpass import getpass
os.environ["KAGGLE_API_TOKEN"] = getpass("Kaggle API token (KGAT_...): ")

# Cubeai Fire & Smoke Detection (YOLOv8 format) — the dataset the original model used.
!kaggle datasets download -d cubeai/fire-and-smoke-detection-for-yolov8 -p /content/firesmoke --unzip
```

Make sure the dataset's `data.yaml` has correct `train:`/`val:` paths and
`names: ['fire','smoke']` (the repo's [firesmoke/data.yaml](firesmoke/data.yaml) is the
template — its `path:` currently points at a Windows folder, so use the downloaded one).

```python
!python train_firesmoke.py --data /content/firesmoke/data.yaml
```

---

## 3. Evaluate (before vs. after)

```python
# New weights
!python evaluate.py --weights runs/detect/ppe_v4/weights/best.pt        --data {dataset.location}/data.yaml
!python evaluate.py --weights runs/detect/fire_smoke_v3/weights/best.pt --data /content/firesmoke/data.yaml
```

Compare `mAP50` / per-class numbers against the baselines (PPE 0.75, fire 0.859,
smoke 0.667). The confusion matrices / PR curves land in `runs/detect/<name>/` — copy the
good ones into `docs/` for the report.

---

## 4. Ship the new weights

Download the two `best.pt` files from Colab and drop them into the backend:

```
runs/detect/ppe_v4/weights/best.pt        ->  models/ppe_model.pt
runs/detect/fire_smoke_v3/weights/best.pt ->  models/firesmoke_model.pt
```

The app loads whatever weights are in `models/` — no code change needed. Restart
`uvicorn` and the higher-accuracy models are live.
