"""Combined safety alert logic.

Turns raw detections from every capability into human-readable alerts and a single
severity-ranked summary. Centralised here so both the image and video paths share one
source of truth for what counts as a hazard.
"""

from services.config import PPE_VIOLATION_LABELS, FALL_LABEL

_PPE_MESSAGES = {
    "NO-Hardhat": "Worker without hardhat detected",
    "NO-Safety Vest": "Worker without safety vest detected",
    "NO-Mask": "Worker without mask detected",
    "NO-Gloves": "Worker without gloves detected",
    "NO-Goggles": "Worker without goggles detected",
}


def generate_alerts(ppe_detections, firesmoke_detections, fall_detected, seg):
    """Build the alert list + severity level from all capability outputs.

    Returns {alerts: [..], level: 'ok'|'warning'|'critical', fire_severity, smoke_severity}.
    """
    alerts = []

    # PPE violations
    ppe_labels = {d["label"] for d in ppe_detections}
    for label in PPE_VIOLATION_LABELS:
        if label in ppe_labels:
            alerts.append(_PPE_MESSAGES[label])

    # Fall (pose heuristic OR the model's Fall-Detected class)
    if fall_detected or FALL_LABEL in ppe_labels:
        alerts.append("Fall hazard detected")

    # Fire / smoke — severity is reported separately via the gauges, so keep the
    # message text stable (embedding the % would break per-frame video dedup).
    fire_sev = seg.get("fire_severity", 0.0)
    smoke_sev = seg.get("smoke_severity", 0.0)
    classes = {d["class"] for d in firesmoke_detections}
    if "fire" in classes:
        alerts.append("Fire detected")
    if "smoke" in classes:
        alerts.append("Smoke detected")

    # Severity level
    if "fire" in classes or (fall_detected or FALL_LABEL in ppe_labels):
        level = "critical"
    elif alerts:
        level = "warning"
    else:
        level = "ok"

    if not alerts:
        alerts = ["No safety issue detected"]

    return {
        "alerts": alerts,
        "level": level,
        "fire_severity": fire_sev,
        "smoke_severity": smoke_sev,
    }


# PPE item -> body zone used by the dashboard's body diagram.
PPE_TYPES = [
    {"key": "hardhat", "label": "Hardhat",     "present": "Hardhat",     "violation": "NO-Hardhat",     "zone": "head"},
    {"key": "goggles", "label": "Goggles",     "present": "Goggles",     "violation": "NO-Goggles",     "zone": "eyes"},
    {"key": "mask",    "label": "Mask",        "present": "Mask",        "violation": "NO-Mask",        "zone": "face"},
    {"key": "vest",    "label": "Safety Vest", "present": "Safety Vest", "violation": "NO-Safety Vest", "zone": "torso"},
    {"key": "gloves",  "label": "Gloves",      "present": "Gloves",      "violation": "NO-Gloves",      "zone": "hands"},
]


def ppe_compliance(labels):
    """Reduce the set of detected labels to a per-item compliance summary.

    For each PPE item, status is:
        'violation' if any NO-<item> was seen (takes priority — it's the hazard),
        'present'   if the item was seen and no violation,
        'absent'    if the item was never observed.
    """
    labels = set(labels)
    summary = []
    for t in PPE_TYPES:
        if t["violation"] in labels:
            status = "violation"
        elif t["present"] in labels:
            status = "present"
        else:
            status = "absent"
        summary.append({
            "key": t["key"],
            "label": t["label"],
            "zone": t["zone"],
            "status": status,
        })
    return summary
