import cv2
import torch
from ultralytics import YOLO
from pathlib import Path

WEIGHTS_PATH = "best.pt"
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45

CLASS_COLORS = {
    "Hardhat":        (0, 255, 0),
    "NO-Hardhat":     (0, 0, 255),
    "Safety Vest":    (0, 255, 128),
    "NO-Safety Vest": (0, 0, 200),
    "Mask":           (255, 165, 0),
    "NO-Mask":        (0, 100, 255),
    "Gloves":         (255, 200, 0),
    "NO-Gloves":      (0, 80, 255),
    "Goggles":        (200, 100, 255),
    "NO-Goggles":     (100, 0, 200),
    "Person":         (200, 200, 200),
    "Safety Cone":    (0, 165, 255),
    "Ladder":         (150, 150, 0),
    "Fall-Detected":  (0, 0, 255),
}
DEFAULT_COLOR = (255, 255, 0)

model = YOLO(WEIGHTS_PATH)
model.conf = CONF_THRESHOLD
model.iou  = IOU_THRESHOLD
CLASS_NAMES = model.names


def draw_results(frame, results):
    detected = []
    for r in results:
        for box in r.boxes:
            cls_id   = int(box.cls[0])
            cls_name = CLASS_NAMES[cls_id]
            conf     = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
            label = f"{cls_name} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            detected.append(cls_name)
    non_compliant = [d for d in detected if d.startswith("NO-") or d == "Fall-Detected"]
    if non_compliant:
        status = f"NON-COMPLIANT: {', '.join(set(non_compliant))}"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 200), -1)
        cv2.putText(frame, status, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    elif detected:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 180, 0), -1)
        cv2.putText(frame, "COMPLIANT", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return frame


def run_on_image(image_path, output_path=None):
    frame = cv2.imread(str(image_path))
    results = model(frame)
    frame = draw_results(frame, results)
    if output_path:
        cv2.imwrite(str(output_path), frame)
    return frame


def run_on_video(video_path, output_path=None):
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if output_path:
        writer = cv2.VideoWriter(str(output_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)
        frame   = draw_results(frame, results)
        if writer:
            writer.write(frame)
        cv2.imshow("PPE Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def run_on_webcam(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    print("Webcam running - press Q to quit")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)
        frame   = draw_results(frame, results)
        cv2.imshow("PPE Detection - Live", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["image", "video", "webcam"], default="webcam")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    if args.mode == "image":
        run_on_image(args.input, args.output)
    elif args.mode == "video":
        run_on_video(args.input, args.output)
    else:
        run_on_webcam()