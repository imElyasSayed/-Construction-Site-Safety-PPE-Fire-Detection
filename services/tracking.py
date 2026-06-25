"""Per-worker violation aggregation across a video.

ByteTrack (run inside ``ppe_detector.track``) gives each worker a stable ID across
frames. This aggregator turns the raw per-frame detections into a per-worker summary so
the system reports *sustained* violations ("Worker 3 had no hardhat for 12.4s") instead
of re-alerting on every frame.
"""

from collections import defaultdict

from services.config import PPE_VIOLATION_LABELS, FALL_LABEL


class TrackAggregator:
    def __init__(self, fps):
        self.fps = fps if fps and fps > 0 else 25.0
        # track_id -> {label -> frame count}
        self._violations = defaultdict(lambda: defaultdict(int))
        self._first_seen = {}
        self._last_seen = {}

    def update(self, detections, frame_index):
        """Feed one frame's tracked detections."""
        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue
            label = det["label"]
            self._first_seen.setdefault(track_id, frame_index)
            self._last_seen[track_id] = frame_index
            if label in PPE_VIOLATION_LABELS or label == FALL_LABEL:
                self._violations[track_id][label] += 1

    def summary(self):
        """Per-worker violation summary with durations in seconds."""
        workers = []
        for track_id in sorted(self._violations):
            violations = {
                label: {
                    "frames": count,
                    "duration_sec": round(count / self.fps, 1),
                }
                for label, count in self._violations[track_id].items()
            }
            workers.append({
                "worker_id": track_id,
                "violations": violations,
            })
        return {
            "tracked_workers": len(self._last_seen),
            "workers_with_violations": workers,
        }
