from ultralytics import YOLO
import cv2
import numpy as np

MODEL_PATH = "models/firesmoke_model.pt"
model = YOLO(MODEL_PATH)

CLASSES = ['fire', 'smoke']
COLORS = {'fire': (0, 0, 255), 'smoke': (128, 128, 128)}  # BGR


def detect(image, conf_threshold=0.25):
    """
    Run fire/smoke detection on an image.

    Args:
        image: numpy array (BGR, as returned by cv2.imread)
        conf_threshold: minimum confidence to include a detection (0-1)

    Returns:
        detections: list of dicts with keys:
                    'class', 'confidence', 'box' [x1, y1, x2, y2]
        annotated:  image with bounding boxes drawn on it
    """
    results = model(image, conf=conf_threshold)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_name = CLASSES[cls_id]

        detections.append({
            'class': class_name,
            'confidence': round(confidence, 3),
            'box': [x1, y1, x2, y2]
        })

    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det['box']
        color = COLORS.get(det['class'], (0, 255, 0))
        label = f"{det['class']} {det['confidence']:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return detections, annotated


def detect_from_path(image_path, conf_threshold=0.25):
    """Run detection on an image file path."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return detect(image, conf_threshold)


def detect_from_bytes(image_bytes, conf_threshold=0.25):
    """
    Run detection on raw image bytes (e.g. from FastAPI UploadFile).
    Returns detections list and annotated image as numpy array.
    """
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image bytes")
    return detect(image, conf_threshold)
