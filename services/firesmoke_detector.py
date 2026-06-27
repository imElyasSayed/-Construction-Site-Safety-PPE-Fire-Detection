"""Fire / smoke detection (YOLOv8, fine-tuned).

Returns region detections only; pixel-level masking and severity are handled by
``services.segmentation`` so detection stays a single responsibility.
"""

from ultralytics import YOLO

from services.config import (
    FIRESMOKE_MODEL_PATH,
    FIRESMOKE_CONF,
    USE_TTA_IMAGE,
    LIVE_CONF_BOOST,
    LIVE_FIRESMOKE_IMGSZ,
)

model = YOLO(FIRESMOKE_MODEL_PATH)
NAMES = model.names  # {0: 'fire', 1: 'smoke'}


def detect(frame, conf=FIRESMOKE_CONF, augment=USE_TTA_IMAGE):
    """Run fire/smoke detection on a single BGR frame."""
    results = model.predict(frame, conf=conf, augment=augment, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            detections.append({
                "class": NAMES[class_id],
                "confidence": round(float(box.conf[0]), 3),
                "box": list(map(int, box.xyxy[0])),
            })
    return detections


def detect_live(frame):
    """Run low-latency fire/smoke detection for webcam frames."""
    results = model.predict(
        frame,
        conf=min(0.95, FIRESMOKE_CONF + LIVE_CONF_BOOST),
        imgsz=LIVE_FIRESMOKE_IMGSZ,
        augment=False,
        verbose=False,
    )

    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            detections.append({
                "class": NAMES[class_id],
                "confidence": round(float(box.conf[0]), 3),
                "box": list(map(int, box.xyxy[0])),
            })
    return detections
