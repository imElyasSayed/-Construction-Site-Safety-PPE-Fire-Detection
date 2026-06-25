"""Central configuration for the Construction Safety Monitor.

Tuning detection thresholds, model paths, and inference-time accuracy switches in
one place keeps the capability modules clean and lets us calibrate accuracy without
touching detection logic.
"""

import os

# --- Model paths ---------------------------------------------------------------
MODELS_DIR = "models"
PPE_MODEL_PATH = os.path.join(MODELS_DIR, "ppe_model.pt")
FIRESMOKE_MODEL_PATH = os.path.join(MODELS_DIR, "firesmoke_model.pt")
# Pretrained COCO pose model — auto-downloaded by ultralytics on first use.
POSE_MODEL_PATH = "yolov8s-pose.pt"

# --- Detection thresholds (tuned on the validation sets) -----------------------
PPE_CONF = 0.35
PPE_IOU = 0.45
FIRESMOKE_CONF = 0.25
FIRESMOKE_IOU = 0.45
POSE_CONF = 0.40

# --- Inference-time accuracy switches ------------------------------------------
# Test-time augmentation improves accuracy but ~3x slows inference, so we enable
# it for single images only and keep video real-time-ish.
USE_TTA_IMAGE = True
USE_TTA_VIDEO = True

# CLAHE preprocessing helps in low-light / dusty / smoky frames.
USE_ENHANCEMENT = True

# Video is processed every Nth frame and downscaled to keep CPU inference and the
# output file web-friendly (4K every-frame on CPU is impractically slow).
VIDEO_STRIDE = 3
VIDEO_MAX_WIDTH = 1280

# --- Folders -------------------------------------------------------------------
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

# --- Labels --------------------------------------------------------------------
# PPE violation labels that should raise an alert.
PPE_VIOLATION_LABELS = {
    "NO-Hardhat",
    "NO-Safety Vest",
    "NO-Mask",
    "NO-Gloves",
    "NO-Goggles",
}
FALL_LABEL = "Fall-Detected"
