"""Image enhancement (CLAHE) preprocessing.

Construction footage is often low-light, dusty, or smoke-obscured. Contrast Limited
Adaptive Histogram Equalization on the luminance channel improves local contrast so
the detectors see PPE / fire / people more reliably, without distorting colour.
"""

import cv2

from services.config import USE_ENHANCEMENT

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def enhance(frame, enabled=USE_ENHANCEMENT):
    """Return a contrast-enhanced copy of a BGR frame (or the original if disabled)."""
    if not enabled:
        return frame

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
