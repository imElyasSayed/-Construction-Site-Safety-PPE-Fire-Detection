"""Human pose / keypoint detection for robust fall detection.

A bounding-box "Fall-Detected" class is fragile. Body keypoints let us reason about
*posture*: a fallen worker's torso is roughly horizontal and their person-box is wider
than it is tall. We OR this with the model's Fall-Detected class downstream for higher
recall.

Uses the pretrained COCO pose model (auto-downloaded by ultralytics). If the model can't
load (e.g. no network on first run), pose degrades gracefully to "unavailable" rather
than breaking the pipeline.
"""

from services.config import POSE_MODEL_PATH, POSE_CONF

# COCO keypoint indices
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12

try:
    from ultralytics import YOLO

    _model = YOLO(POSE_MODEL_PATH)
    AVAILABLE = True
except Exception as exc:  # pragma: no cover - depends on runtime/network
    _model = None
    AVAILABLE = False
    _LOAD_ERROR = str(exc)


def _midpoint(kps, i, j, min_conf=0.3):
    """Average of two keypoints if both are confident, else None."""
    if kps is None or len(kps) <= max(i, j):
        return None
    a, b = kps[i], kps[j]
    if a[2] < min_conf or b[2] < min_conf:
        return None
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _is_fallen(box, kps):
    """Heuristic fall test from posture + box aspect ratio."""
    x1, y1, x2, y2 = box
    width, height = max(1, x2 - x1), max(1, y2 - y1)
    wide_box = width > height * 1.2

    shoulders = _midpoint(kps, L_SHOULDER, R_SHOULDER)
    hips = _midpoint(kps, L_HIP, R_HIP)
    torso_horizontal = False
    if shoulders and hips:
        dx = abs(shoulders[0] - hips[0])
        dy = abs(shoulders[1] - hips[1])
        torso_horizontal = dx > dy  # torso lies flatter than it stands

    return bool(wide_box or torso_horizontal)


def detect(frame, conf=POSE_CONF):
    """Detect people + keypoints and flag fallen postures.

    Returns (persons, fall_detected):
        persons: list of {box, keypoints, fall}
        fall_detected: True if any person looks fallen
    """
    if not AVAILABLE:
        return [], False

    results = _model.predict(frame, conf=conf, verbose=False)
    persons = []
    fall_detected = False

    for result in results:
        if result.keypoints is None:
            continue
        kp_data = result.keypoints.data.cpu().numpy()  # (n, 17, 3)
        for box, kps in zip(result.boxes, kp_data):
            xyxy = list(map(int, box.xyxy[0]))
            fallen = _is_fallen(xyxy, kps)
            fall_detected = fall_detected or fallen
            persons.append({
                "box": xyxy,
                "keypoints": kps.tolist(),
                "fall": fallen,
            })

    return persons, fall_detected
