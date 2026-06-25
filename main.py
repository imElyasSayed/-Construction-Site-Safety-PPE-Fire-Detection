"""Construction Safety Monitor — FastAPI app.

A single upload runs every capability (PPE detection + tracking, fire/smoke detection +
segmentation/severity, pose-based fall detection) and returns one combined annotated
output plus a unified safety summary. The vanilla web frontend is served from /.
"""

import os
import shutil
from urllib.parse import quote

import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles

from services import pipeline
from services.config import UPLOAD_FOLDER, OUTPUT_FOLDER

app = FastAPI(title="Construction Safety Monitor")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def _output_url(path):
    """Map a local outputs/ path to its served URL."""
    return f"/outputs/{quote(os.path.basename(path))}"


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    frame = cv2.imread(file_path)
    if frame is None:
        return {"error": "Could not read image file", "filename": file.filename}

    result, annotated = pipeline.analyze_image(frame)
    output_path = os.path.join(OUTPUT_FOLDER, f"detected_{file.filename}")
    cv2.imwrite(output_path, annotated)

    result.update({
        "type": "image",
        "filename": file.filename,
        "output_image": _output_url(output_path),
    })
    return result


@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result, output_path, keyframe_path = pipeline.analyze_video(file_path)

    result.update({
        "type": "video",
        "filename": file.filename,
        "output_video": _output_url(output_path),
        "keyframe": _output_url(keyframe_path) if keyframe_path else None,
    })
    return result


# Serve generated results and uploads, then the built React frontend at the root.
# The "/" mount is registered last so the API routes above take precedence.
# Run `npm install && npm run build` in frontend/ to (re)generate frontend/dist.
FRONTEND_DIR = "frontend/dist" if os.path.isdir("frontend/dist") else "frontend"
app.mount("/outputs", StaticFiles(directory=OUTPUT_FOLDER), name="outputs")
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
