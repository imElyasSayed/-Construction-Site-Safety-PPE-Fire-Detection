from ultralytics import YOLO
import cv2
import os

MODEL_PATH = "models/ppe_model.pt"
OUTPUT_FOLDER = "outputs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = YOLO(MODEL_PATH)


def detect_ppe(image_path):
    results = model(image_path, conf=0.35, iou=0.45)

    detections = []

    image = cv2.imread(image_path)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "label": label,
                "confidence": round(confidence, 2),
                "box": [x1, y1, x2, y2]
            })

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{label} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    filename = os.path.basename(image_path)
    output_path = os.path.join(OUTPUT_FOLDER, f"detected_{filename}")

    cv2.imwrite(output_path, image)

    return detections, output_path

def detect_ppe_video(video_path):
    output_path = os.path.join(
        OUTPUT_FOLDER,
        f"detected_{os.path.basename(video_path)}"
    )

    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_count = 0
    all_detections = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        results = model(frame, conf=0.35, iou=0.45)

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                all_detections.append({
                    "frame": frame_count,
                    "label": label,
                    "confidence": round(confidence, 2),
                    "box": [x1, y1, x2, y2]
                })

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        writer.write(frame)

    cap.release()
    writer.release()

    return all_detections, output_path