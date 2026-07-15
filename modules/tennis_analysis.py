from core.angle_engine import calculate_angle


def run(keypoints):
    issues = []

    for frame in keypoints:
        try:
            shoulder = frame.get("shoulder")
            elbow = frame.get("elbow")
            wrist = frame.get("wrist")

            if not shoulder or not elbow or not wrist:
                continue

            elbow_angle = calculate_angle(
                shoulder[:2], elbow[:2], wrist[:2]
            )

            if elbow_angle > 170:
                issues.append({
                    "joint": "elbow",
                    "issue": "overextension",
                    "value": round(elbow_angle, 2),
                    "target": 150
                })

        except Exception:
            continue

    return {
        "mode": "tennis",
        "issues": issues,
        "feedback": generate_feedback(issues)
    }


def generate_feedback(issues):
    if not issues:
        return "Safe tennis form"

    return "Reduce elbow extension to avoid injury"