"""Fire/smoke retraining — accuracy-focused config (run on Colab GPU).

Reproduces the shipped model (mAP50 0.820, smoke 0.667 -> 0.801) vs the 0.763 baseline,
aimed at the weak smoke class:
  - Base model yolov8s -> yolov8m  (more capacity)
  - Epochs 50 -> 40, image size 640 -> 768
  - Stronger augmentation          (mosaic + mixup + copy_paste + wider HSV/scale)
    Smoke is diffuse and greyish, so heavier colour/scale jitter + copy_paste helps
    the model generalise instead of memorising a few smoke shapes.

Usage (Colab / local GPU):
    pip install ultralytics
    python train_firesmoke.py --data /path/to/data.yaml
The new weights land at runs/fire_smoke_v3/weights/best.pt — copy that to
    models/firesmoke_model.pt
in the backend (the app loads whatever weights are there; no code change needed).
"""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.yaml", help="dataset config")
    parser.add_argument("--model", default="yolov8m.pt", help="base weights")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--name", default="fire_smoke_v3")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        patience=20,
        save=True,
        val=True,
        # --- stronger augmentation for the weak smoke class ---
        hsv_h=0.02, hsv_s=0.7, hsv_v=0.5,
        scale=0.6, translate=0.1, fliplr=0.5,
        mosaic=1.0, mixup=0.15, copy_paste=0.2,
        close_mosaic=10,
        cos_lr=True,
    )

    metrics = model.val()
    print("\nTraining complete. Validation metrics:")
    print(metrics.results_dict)
    print(f"Best weights: runs/detect/{args.name}/weights/best.pt")
    print("Copy best.pt -> models/firesmoke_model.pt in the backend.")


if __name__ == "__main__":
    main()
