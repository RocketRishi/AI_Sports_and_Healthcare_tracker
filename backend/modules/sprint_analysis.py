from core.angle_engine import calculate_angle


def run(keypoints):
    issues = []

    for frame in keypoints:
        try:
            hip = frame.get("hip")
            knee = frame.get("knee")
            ankle = frame.get("ankle")

            if not hip or not knee or not ankle:
                continue

            knee_angle = calculate_angle(hip[:2], knee[:2], ankle[:2])

            if knee_angle < 70:
                issues.append({
                    "joint": "knee",
                    "issue": "low knee drive",
                    "value": round(knee_angle, 2),
                    "target": 75
                })

        except Exception:
            continue

    return {
        "mode": "sprint",
        "issues": issues,
        "feedback": generate_feedback(issues)
    }


def generate_feedback(issues):
    if not issues:
        return "Good sprint form"

    return "Increase knee drive and improve running posture"