"""Unified analysis pipeline.

One upload runs every capability and produces a single combined annotated output plus a
unified safety summary. This is the only orchestration point — the API routes just call
``analyze_image`` / ``analyze_video``.

Image:  enhance -> PPE + fire/smoke(+segmentation) + pose/fall -> compose -> summary
Video:  same per frame, but PPE runs through ByteTrack; per-worker violations and peak
        fire/smoke severity are aggregated across frames.
"""

import os

import cv2

from services import (
    enhancement,
    ppe_detector,
    firesmoke_detector,
    segmentation,
    pose_detector,
    alerts,
)
from services.tracking import TrackAggregator
from services.config import (
    PPE_VIOLATION_LABELS,
    FALL_LABEL,
    OUTPUT_FOLDER,
    VIDEO_STRIDE,
    VIDEO_MAX_WIDTH,
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# COCO skeleton (pairs of keypoint indices)
_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11),
    (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6),
]

_GREEN = (0, 200, 0)
_RED = (0, 0, 255)
_ORANGE = (0, 140, 255)
_GREY = (160, 160, 160)
_LEVEL_COLORS = {"ok": (0, 160, 0), "warning": (0, 170, 255), "critical": (0, 0, 220)}


# ----------------------------------------------------------------------------- drawing
def _draw_ppe(img, detections, show_ids=False):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        if label == FALL_LABEL:
            color = _ORANGE
        elif label in PPE_VIOLATION_LABELS or label.startswith("NO-"):
            color = _RED
        else:
            color = _GREEN
        text = label
        if show_ids and det.get("track_id") is not None:
            text = f"#{det['track_id']} {label}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{text} {det['confidence']:.2f}", (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def _draw_firesmoke(img, detections):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = _RED if det["class"] == "fire" else _GREY
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{det['class']} {det['confidence']:.2f}", (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def _draw_pose(img, persons):
    for person in persons:
        kps = person["keypoints"]
        color = _RED if person["fall"] else (0, 220, 220)
        for i, j in _SKELETON:
            if i < len(kps) and j < len(kps) and kps[i][2] > 0.3 and kps[j][2] > 0.3:
                cv2.line(img, (int(kps[i][0]), int(kps[i][1])),
                         (int(kps[j][0]), int(kps[j][1])), color, 2)
        for x, y, c in kps:
            if c > 0.3:
                cv2.circle(img, (int(x), int(y)), 3, color, -1)
        if person["fall"]:
            x1, y1, x2, y2 = person["box"]
            cv2.rectangle(img, (x1, y1), (x2, y2), _RED, 2)
            cv2.putText(img, "FALL", (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, _RED, 2)


def _draw_banner(img, level, text):
    color = _LEVEL_COLORS.get(level, _GREEN)
    cv2.rectangle(img, (0, 0), (img.shape[1], 36), color, -1)
    cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def _compose(frame, ppe, firesmoke, seg, persons, alert_info, show_ids=False):
    out = segmentation.overlay(frame, seg)
    _draw_ppe(out, ppe, show_ids=show_ids)
    _draw_firesmoke(out, firesmoke)
    _draw_pose(out, persons)
    _draw_banner(out, alert_info["level"], alert_info["alerts"][0])
    return out


# ----------------------------------------------------------------------------- image
def analyze_image(frame):
    enhanced = enhancement.enhance(frame)
    ppe = ppe_detector.detect(enhanced)
    firesmoke = firesmoke_detector.detect(enhanced)
    seg = segmentation.segment(frame, firesmoke)
    persons, pose_fall = pose_detector.detect(enhanced)

    fall = pose_fall or any(d["label"] == FALL_LABEL for d in ppe)
    alert_info = alerts.generate_alerts(ppe, firesmoke, fall, seg)
    annotated = _compose(frame, ppe, firesmoke, seg, persons, alert_info)

    result = {
        "ppe_detections": ppe,
        "ppe_summary": alerts.ppe_compliance(d["label"] for d in ppe),
        "firesmoke_detections": firesmoke,
        "persons_detected": len(persons),
        "fall_detected": fall,
        "pose_available": pose_detector.AVAILABLE,
        "severity": {
            "fire": seg["fire_severity"],
            "smoke": seg["smoke_severity"],
        },
        "alerts": alert_info["alerts"],
        "level": alert_info["level"],
    }
    return result, annotated


def analyze_live_frame(frame):
    ppe = ppe_detector.detect_live(frame)
    firesmoke = firesmoke_detector.detect_live(frame)

    h, w = frame.shape[:2]
    total = max(1.0, float(h * w))
    fire_area = 0.0
    smoke_area = 0.0
    for det in firesmoke:
        x1, y1, x2, y2 = det["box"]
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if det["class"] == "fire":
            fire_area += area
        elif det["class"] == "smoke":
            smoke_area += area
    seg = {
        "fire_severity": round(min(100.0, fire_area / total * 100), 1),
        "smoke_severity": round(min(100.0, smoke_area / total * 100), 1),
    }

    fall = any(d["label"] == FALL_LABEL for d in ppe)
    alert_info = alerts.generate_alerts(ppe, firesmoke, fall, seg)

    return {
        "ppe_detections": ppe,
        "ppe_summary": alerts.ppe_compliance(d["label"] for d in ppe),
        "firesmoke_detections": firesmoke,
        "persons_detected": 0,
        "fall_detected": fall,
        "pose_available": False,
        "severity": {
            "fire": seg["fire_severity"],
            "smoke": seg["smoke_severity"],
        },
        "alerts": alert_info["alerts"],
        "level": alert_info["level"],
    }


# ----------------------------------------------------------------------------- video
def _open_writer(path, fps, size):
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
    return None


def _downscale(frame):
    """Shrink a frame to VIDEO_MAX_WIDTH for faster inference / web-friendly output."""
    h, w = frame.shape[:2]
    if w <= VIDEO_MAX_WIDTH:
        return frame
    scale = VIDEO_MAX_WIDTH / float(w)
    return cv2.resize(frame, (VIDEO_MAX_WIDTH, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out_fps = max(1.0, src_fps / VIDEO_STRIDE)

    base = os.path.basename(video_path)
    output_path = os.path.join(OUTPUT_FOLDER, f"detected_{base}")
    keyframe_path = os.path.join(OUTPUT_FOLDER, f"detected_{os.path.splitext(base)[0]}_key.jpg")

    aggregator = TrackAggregator(out_fps)
    all_alerts = set()
    seen_labels = set()
    peak_fire = peak_smoke = 0.0
    fall_ever = False
    best_score = -1
    best_frame = None
    writer = None
    frame_index = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1
        if (frame_index - 1) % VIDEO_STRIDE != 0:
            continue
        processed += 1

        frame = _downscale(frame)
        if writer is None:
            h, w = frame.shape[:2]
            writer = _open_writer(output_path, out_fps, (w, h))

        enhanced = enhancement.enhance(frame)
        tracked = ppe_detector.track(enhanced)
        firesmoke = firesmoke_detector.detect(enhanced)
        seg = segmentation.segment(frame, firesmoke)
        persons, pose_fall = pose_detector.detect(enhanced)

        aggregator.update(tracked, frame_index)
        seen_labels.update(d["label"] for d in tracked)
        fall = pose_fall or any(d["label"] == FALL_LABEL for d in tracked)
        fall_ever = fall_ever or fall
        peak_fire = max(peak_fire, seg["fire_severity"])
        peak_smoke = max(peak_smoke, seg["smoke_severity"])

        alert_info = alerts.generate_alerts(tracked, firesmoke, fall, seg)
        for msg in alert_info["alerts"]:
            if msg != "No safety issue detected":
                all_alerts.add(msg)

        annotated = _compose(frame, tracked, firesmoke, seg, persons, alert_info, show_ids=True)
        if writer is not None:
            writer.write(annotated)

        # keep the most hazardous frame as a representative snapshot
        score = {"ok": 0, "warning": 1, "critical": 2}[alert_info["level"]] * 100 + len(alert_info["alerts"])
        if score > best_score:
            best_score = score
            best_frame = annotated

    cap.release()
    if writer is not None:
        writer.release()
    if best_frame is not None:
        cv2.imwrite(keyframe_path, best_frame)

    level = "critical" if (peak_fire > 0 or fall_ever) else ("warning" if all_alerts else "ok")
    summary = aggregator.summary()

    result = {
        "frames_total": frame_index,
        "frames_processed": processed,
        "frame_stride": VIDEO_STRIDE,
        "ppe_summary": alerts.ppe_compliance(seen_labels),
        "tracking": summary,
        "fall_detected": fall_ever,
        "pose_available": pose_detector.AVAILABLE,
        "peak_severity": {"fire": round(peak_fire, 1), "smoke": round(peak_smoke, 1)},
        "alerts": sorted(all_alerts) or ["No safety issue detected"],
        "level": level,
        "output_video": output_path,
        "keyframe": keyframe_path if best_frame is not None else None,
        "video_playable": writer is not None,
    }
    return result, output_path, (keyframe_path if best_frame is not None else None)
