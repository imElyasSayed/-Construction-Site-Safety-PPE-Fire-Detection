"""Fire / smoke region segmentation + severity.

YOLO gives us *where* fire/smoke is (a box). To report *how bad* it is, we segment the
actual fire/smoke pixels inside each detected box using classical colour thresholding in
HSV, cleaned with binary morphological opening/closing. Severity is the share of the
frame covered by each hazard — a far more actionable signal than a yes/no flag.

This deliberately avoids a trained segmentation model: our dataset has bounding-box
labels, not masks, and this approach needs no new annotated data.
"""

import cv2
import numpy as np

_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def _clip_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    return x1, y1, x2, y2


def _fire_mask(roi_hsv):
    """Bright orange/red/yellow, high-value pixels = fire."""
    lower_warm = np.array([0, 80, 150], dtype=np.uint8)
    upper_warm = np.array([35, 255, 255], dtype=np.uint8)
    lower_red = np.array([160, 80, 150], dtype=np.uint8)
    upper_red = np.array([180, 255, 255], dtype=np.uint8)
    return cv2.bitwise_or(
        cv2.inRange(roi_hsv, lower_warm, upper_warm),
        cv2.inRange(roi_hsv, lower_red, upper_red),
    )


def _smoke_mask(roi_hsv):
    """Low-saturation, mid/high-value greyish pixels = smoke."""
    lower = np.array([0, 0, 80], dtype=np.uint8)
    upper = np.array([180, 65, 225], dtype=np.uint8)
    return cv2.inRange(roi_hsv, lower, upper)


def _clean(mask):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL)
    return mask


def segment(frame, detections):
    """Build fire/smoke masks from detections and compute severity percentages.

    Returns a dict:
        fire_mask, smoke_mask : full-frame uint8 masks (0/255)
        fire_severity, smoke_severity : % of frame area covered (0-100)
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    fire_mask = np.zeros((h, w), dtype=np.uint8)
    smoke_mask = np.zeros((h, w), dtype=np.uint8)

    for det in detections:
        x1, y1, x2, y2 = _clip_box(det["box"], w, h)
        if x2 <= x1 or y2 <= y1:
            continue
        roi_hsv = hsv[y1:y2, x1:x2]
        if det["class"] == "fire":
            fire_mask[y1:y2, x1:x2] = cv2.bitwise_or(
                fire_mask[y1:y2, x1:x2], _fire_mask(roi_hsv)
            )
        else:
            smoke_mask[y1:y2, x1:x2] = cv2.bitwise_or(
                smoke_mask[y1:y2, x1:x2], _smoke_mask(roi_hsv)
            )

    fire_mask = _clean(fire_mask)
    smoke_mask = _clean(smoke_mask)

    total = float(h * w)
    return {
        "fire_mask": fire_mask,
        "smoke_mask": smoke_mask,
        "fire_severity": round(float(np.count_nonzero(fire_mask)) / total * 100, 1),
        "smoke_severity": round(float(np.count_nonzero(smoke_mask)) / total * 100, 1),
    }


def overlay(frame, seg, fire_color=(0, 0, 255), smoke_color=(160, 160, 160), alpha=0.45):
    """Blend the fire/smoke masks onto a copy of the frame for visualization."""
    out = frame.copy()
    for mask, color in ((seg["fire_mask"], fire_color), (seg["smoke_mask"], smoke_color)):
        if np.count_nonzero(mask) == 0:
            continue
        colored = np.zeros_like(frame)
        colored[mask > 0] = color
        out = cv2.addWeighted(out, 1.0, colored, alpha, 0)
    return out
