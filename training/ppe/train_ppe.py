"""PPE retraining — accuracy-focused config (run on Colab GPU).

The shipped PPE model (mAP50 0.75+) was trained for only 20 epochs on yolov8s. This
script raises the ceiling:
  - Base model yolov8s -> yolov8m
  - Epochs 20 -> 80 (patience early-stop)
  - Stronger augmentation (mosaic + mixup + copy_paste + HSV/scale jitter)

Dataset: Roboflow "PPE Combined Model v4" (14 classes), same as the original.
Export it in YOLOv8 format and point --data at its data.yaml.

Usage (Colab / local GPU):
    pip install ultralytics roboflow
    python train_ppe.py --data /path/to/PPE-Combined-4/data.yaml
New weights -> runs/detect/ppe_v4/weights/best.pt; copy to models/ppe_model.pt.
"""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to Roboflow PPE data.yaml")
    parser.add_argument("--model", default="yolov8m.pt", help="base weights")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", default="ppe_v4")
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
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        scale=0.5, translate=0.1, fliplr=0.5,
        mosaic=1.0, mixup=0.1, copy_paste=0.1,
        close_mosaic=10,
        cos_lr=True,
    )

    metrics = model.val()
    print("\nTraining complete. Validation metrics:")
    print(metrics.results_dict)
    print(f"Best weights: runs/detect/{args.name}/weights/best.pt")
    print("Copy best.pt -> models/ppe_model.pt in the backend.")


if __name__ == "__main__":
    main()
