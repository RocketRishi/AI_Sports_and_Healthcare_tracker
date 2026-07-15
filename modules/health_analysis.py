from core.angle_engine import calculate_angle


def run(keypoints):
    issues = []

    for frame in keypoints:
        try:
            shoulder = frame.get("shoulder")
            hip = frame.get("hip")
            knee = frame.get("knee")

            if not shoulder or not hip or not knee:
                continue

            spine_angle = calculate_angle(
                shoulder[:2], hip[:2], knee[:2]
            )

            if spine_angle < 150:
                issues.append({
                    "joint": "spine",
                    "issue": "forward lean",
                    "value": round(spine_angle, 2),
                    "target": 170
                })

        except Exception:
            continue

    return {
        "mode": "health",
        "issues": issues,
        "feedback": generate_feedback(issues)
    }


def generate_feedback(issues):
    if not issues:
        return "Stable posture"

    return "Maintain upright posture and improve balance"