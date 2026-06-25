"""PPE detection + tracking (YOLOv8, fine-tuned).

Exposes a frame-based interface so the unified pipeline can compose it with the other
capabilities:
    detect(frame) -> list of detections          (single image path)
    track(frame)  -> list of detections + track_id (video path, ByteTrack)

The module is model-agnostic: it loads whatever weights live at PPE_MODEL_PATH, so
retraining to a larger variant requires no code change.
"""

from ultralytics import YOLO

from services.config import (
    PPE_MODEL_PATH,
    PPE_CONF,
    PPE_IOU,
    USE_TTA_IMAGE,
)

model = YOLO(PPE_MODEL_PATH)
NAMES = model.names


def _parse(results, with_id=False):
    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            detection = {
                "label": NAMES[class_id],
                "confidence": round(float(box.conf[0]), 2),
                "box": list(map(int, box.xyxy[0])),
            }
            if with_id:
                detection["track_id"] = (
                    int(box.id[0]) if box.id is not None else None
                )
            detections.append(detection)
    return detections


def detect(frame, conf=PPE_CONF, iou=PPE_IOU, augment=USE_TTA_IMAGE):
    """Run PPE detection on a single BGR frame."""
    results = model.predict(frame, conf=conf, iou=iou, augment=augment, verbose=False)
    return _parse(results)


def track(frame, conf=PPE_CONF, iou=PPE_IOU):
    """Run PPE detection with persistent ByteTrack IDs (video frames)."""
    results = model.track(
        frame,
        conf=conf,
        iou=iou,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )
    return _parse(results, with_id=True)
