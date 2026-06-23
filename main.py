from fastapi import FastAPI, UploadFile, File
import shutil
import os

from services.ppe_detector import detect_ppe, detect_ppe_video

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def generate_alert(detections):
    alert_messages = []

    for detection in detections:
        label = detection["label"]

        if label == "NO-Hardhat":
            alert_messages.append("Worker without hardhat detected")

        elif label == "NO-Safety Vest":
            alert_messages.append("Worker without safety vest detected")

        elif label == "NO-Mask":
            alert_messages.append("Worker without mask detected")

        elif label == "NO-Gloves":
            alert_messages.append("Worker without gloves detected")

        elif label == "NO-Goggles":
            alert_messages.append("Worker without goggles detected")

        elif label == "Fall-Detected":
            alert_messages.append("Fall hazard detected")

    if alert_messages:
        return list(set(alert_messages))

    return ["No safety issue detected"]


@app.get("/")
def home():
    return {"message": "Construction Safety Monitor Backend"}


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ppe_results, output_image_path = detect_ppe(file_path)
    alerts = generate_alert(ppe_results)

    return {
        "filename": file.filename,
        "saved_path": file_path,
        "output_image": output_image_path,
        "ppe_detections": ppe_results,
        "alerts": alerts
    }


@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    video_detections, output_video_path = detect_ppe_video(file_path)
    alerts = generate_alert(video_detections)

    return {
        "filename": file.filename,
        "saved_path": file_path,
        "output_video": output_video_path,
        "total_detections": len(video_detections),
        "ppe_detections": video_detections[:50],
        "alerts": alerts,
        "note": "Showing first 50 detections only"
    }