"""Evaluate a trained YOLOv8 model on its validation set.

Run before and after retraining to quantify the accuracy gain, and to regenerate the
confusion matrix / PR curves used in the report. Plots are written by ultralytics into
the run directory; copy the relevant ones into docs/.

Usage:
    python evaluate.py --weights ../../models/ppe_model.pt --data /path/to/ppe/data.yaml
    python evaluate.py --weights ../../models/firesmoke_model.pt --data firesmoke/data.yaml
"""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="path to .pt weights")
    parser.add_argument("--data", required=True, help="dataset data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", default="eval")
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz, name=args.name, plots=True)

    print("\n=== Overall ===")
    print(f"mAP50    : {metrics.box.map50:.3f}")
    print(f"mAP50-95 : {metrics.box.map:.3f}")
    print(f"precision: {metrics.box.mp:.3f}")
    print(f"recall   : {metrics.box.mr:.3f}")

    print("\n=== Per class (mAP50) ===")
    for i, name in model.names.items():
        try:
            print(f"{name:18s}: {metrics.box.maps[i]:.3f}")
        except (IndexError, TypeError):
            pass

    print(f"\nPlots (confusion_matrix.png, *_curve.png) saved under runs/detect/{args.name}/")


if __name__ == "__main__":
    main()
