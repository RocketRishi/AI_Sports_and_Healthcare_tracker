from core.angle_engine import calculate_angle


def run(keypoints):
    issues = []

    for frame in keypoints:
        try:
            hip = frame.get("hip")
            knee = frame.get("knee")
            ankle = frame.get("ankle")
            shoulder = frame.get("shoulder")

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

            if shoulder:
                hip_angle = calculate_angle(shoulder[:2], hip[:2], knee[:2])

                if hip_angle < 160:
                    issues.append({
                        "joint": "hip",
                        "issue": "poor hip extension",
                        "value": round(hip_angle, 2),
                        "target": 170
                    })

            if shoulder and hip:
                torso_angle = calculate_angle(
                    (hip[0], hip[1] - 50), 
                    hip[:2],
                    shoulder[:2]
                )

                if torso_angle > 30:
                    issues.append({
                        "joint": "torso",
                        "issue": "excessive forward lean",
                        "value": round(torso_angle, 2),
                        "target": 20
                    })

            if knee and ankle:
                foot = (ankle[0], ankle[1] + 20)

                ankle_angle = calculate_angle(knee[:2], ankle[:2], foot)

                if ankle_angle < 140:
                    issues.append({
                        "joint": "ankle",
                        "issue": "weak push-off",
                        "value": round(ankle_angle, 2),
                        "target": 150
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
        return "Strong sprint mechanics. Maintain posture and consistency."

    feedback = []

    for issue in issues:
        if issue["joint"] == "knee":
            feedback.append("Increase knee drive for better stride power")

        elif issue["joint"] == "hip":
            feedback.append("Extend hips fully to improve acceleration")

        elif issue["joint"] == "torso":
            feedback.append("Keep torso stable and avoid excessive forward lean")

        elif issue["joint"] == "ankle":
            feedback.append("Improve ankle extension for stronger push-off")

    feedback = list(set(feedback))

    return ". ".join(feedback)
