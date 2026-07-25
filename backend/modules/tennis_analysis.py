import math
from typing import Dict, List, Optional

from core.angle_engine import calculate_flexion_angle, calculate_lean_angle
from data import tennis_benchmarks as bench

VISIBILITY_MIN = 0.3

# How far into the swing (as a fraction of usable frames) we still
# consider "loading" when searching for the trophy position, so a
# late-clip follow-through knee bend doesn't get mistaken for it.
_TROPHY_SEARCH_WINDOW = 0.7


def run(keypoints: List[Dict[str, list]]) -> Dict:
    side = _select_serving_arm(keypoints)
    metrics = _build_metric_sequence(keypoints, side)

    if len(metrics) < 5:
        issues = _general_safety_checks(metrics)

        return {
            "mode": "tennis",
            "dominant_arm": side,
            "phases_detected": {},
            "issues": issues,
            "feedback": (
                "Not enough clearly-tracked frames of the serving arm "
                "to evaluate swing phases (trophy / max external "
                "rotation / impact). Film from the side with the "
                "whole body and racket arm visible for a full "
                "biomechanical breakdown."
            ),
            "reference": _reference_block(),
        }

    phases = _detect_phases(metrics)
    issues = _general_safety_checks(metrics)
    issues += _phase_issues(metrics, phases)

    return {
        "mode": "tennis",
        "dominant_arm": side,
        "phases_detected": {
            name: (metrics[idx]["frame_index"] if idx is not None else None)
            for name, idx in phases.items()
        },
        "phase_metrics": {
            name: (_public_metrics(metrics[idx]) if idx is not None else None)
            for name, idx in phases.items()
        },
        "issues": issues,
        "feedback": generate_feedback(issues),
        "reference": _reference_block(),
    }


def _select_serving_arm(keypoints: List[Dict[str, list]]) -> str:
    left_score = 0.0
    right_score = 0.0

    for frame in keypoints:
        for joint in ("shoulder", "elbow", "wrist"):
            left = frame.get(f"left_{joint}")
            right = frame.get(f"right_{joint}")

            if left:
                left_score += left[2]
            if right:
                right_score += right[2]

    if left_score == 0.0 and right_score == 0.0:
        return "left"

    return "left" if left_score >= right_score else "right"


def _usable(point: Optional[list]) -> bool:
    return bool(point) and len(point) >= 3 and point[2] >= VISIBILITY_MIN


def _build_metric_sequence(
    keypoints: List[Dict[str, list]],
    side: str,
) -> List[Dict]:
    sequence = []
    previous_wrist = None

    for index, frame in enumerate(keypoints):
        shoulder = frame.get(f"{side}_shoulder") or frame.get("shoulder")
        elbow = frame.get(f"{side}_elbow") or frame.get("elbow")
        wrist = frame.get(f"{side}_wrist") or frame.get("wrist")
        hip = frame.get(f"{side}_hip") or frame.get("hip")
        knee = frame.get(f"{side}_knee") or frame.get("knee")
        ankle = frame.get(f"{side}_ankle") or frame.get("ankle")

        if not (_usable(shoulder) and _usable(elbow) and _usable(wrist) and _usable(hip)):
            previous_wrist = wrist if _usable(wrist) else previous_wrist
            continue

        elbow_flexion = calculate_flexion_angle(shoulder[:2], elbow[:2], wrist[:2])
        shoulder_abduction = calculate_flexion_angle(hip[:2], shoulder[:2], elbow[:2])
        trunk_lean = calculate_lean_angle(hip[:2], shoulder[:2])

        knee_flexion = None
        if _usable(knee) and _usable(ankle):
            knee_flexion = calculate_flexion_angle(hip[:2], knee[:2], ankle[:2])

        wrist_speed = None
        if _usable(wrist) and previous_wrist is not None:
            wrist_speed = math.hypot(
                wrist[0] - previous_wrist[0],
                wrist[1] - previous_wrist[1],
            )

        sequence.append({
            "frame_index": index,
            "elbow_flexion_deg": elbow_flexion,
            "shoulder_abduction_deg": shoulder_abduction,
            "trunk_lean_deg": trunk_lean,
            "knee_flexion_deg": knee_flexion,
            "wrist_speed": wrist_speed,
        })

        previous_wrist = wrist

    return sequence


def _public_metrics(entry: Dict) -> Dict:
    return {k: v for k, v in entry.items() if k != "wrist_speed"}


def _detect_phases(metrics: List[Dict]) -> Dict[str, Optional[int]]:
    n = len(metrics)
    trophy_cutoff = max(int(n * _TROPHY_SEARCH_WINDOW), 1)

    trophy_pos = _argmax(
        metrics[:trophy_cutoff], "knee_flexion_deg"
    )

    impact_search_start = trophy_pos if trophy_pos is not None else 0
    impact_pos = _argmax(
        metrics[impact_search_start:], "wrist_speed",
        offset=impact_search_start,
    )

    mer_search_end = impact_pos if impact_pos is not None else n
    mer_search_start = trophy_pos if trophy_pos is not None else 0
    mer_pos = _argmax(
        metrics[mer_search_start:mer_search_end], "elbow_flexion_deg",
        offset=mer_search_start,
    )

    return {
        "trophy": trophy_pos,
        "max_external_rotation": mer_pos,
        "impact": impact_pos,
    }


def _argmax(entries: List[Dict], key: str, offset: int = 0) -> Optional[int]:
    best_index = None
    best_value = None

    for i, entry in enumerate(entries):
        value = entry.get(key)

        if value is None:
            continue

        if best_value is None or value > best_value:
            best_value = value
            best_index = i + offset

    return best_index


def _within(value: float, benchmark: Dict) -> bool:
    if "range" in benchmark:
        low, high = benchmark["range"]
        return low <= value <= high

    if "mean" in benchmark and "sd" in benchmark:
        low = benchmark["mean"] - 2 * benchmark["sd"]
        high = benchmark["mean"] + 2 * benchmark["sd"]
        return low <= value <= high

    return True


def _target_range(benchmark: Dict):
    if "range" in benchmark:
        return list(benchmark["range"])

    if "mean" in benchmark and "sd" in benchmark:
        return [
            round(benchmark["mean"] - 2 * benchmark["sd"], 1),
            round(benchmark["mean"] + 2 * benchmark["sd"], 1),
        ]

    return None


def _make_issue(phase: str, joint: str, code: str, value: float, benchmark: Dict) -> Dict:
    return {
        "phase": phase,
        "joint": joint,
        "issue": code,
        "value": round(value, 2),
        "target": _target_range(benchmark),
        "source": benchmark.get("source"),
        "injury_risk": bench.INJURY_RISK_NOTES.get(code),
    }


def _general_safety_checks(metrics: List[Dict]) -> List[Dict]:
    issues = []

    for entry in metrics:
        if entry["elbow_flexion_deg"] < 5:
            issues.append(_make_issue(
                phase="any",
                joint="elbow",
                code="elbow_hyperextension",
                value=entry["elbow_flexion_deg"],
                benchmark={"range": (5, 180)},
            ))
            break  # one flag is enough; avoid spamming per-frame duplicates

    return issues


def _phase_issues(metrics: List[Dict], phases: Dict[str, Optional[int]]) -> List[Dict]:
    issues = []

    trophy_idx = phases.get("trophy")
    if trophy_idx is not None:
        trophy = metrics[trophy_idx]

        knee_bench = bench.TROPHY_POSITION["front_knee_flexion_deg"]
        if trophy["knee_flexion_deg"] is not None and not _within(trophy["knee_flexion_deg"], knee_bench):
            if trophy["knee_flexion_deg"] < knee_bench["range"][0]:
                issues.append(_make_issue(
                    "trophy", "knee", "low_trophy_knee_flexion",
                    trophy["knee_flexion_deg"], knee_bench,
                ))

        trunk_bench = bench.TROPHY_POSITION["trunk_lean_deg"]
        if not _within(trophy["trunk_lean_deg"], trunk_bench):
            code = (
                "excessive_trophy_trunk_lean"
                if trophy["trunk_lean_deg"] > trunk_bench["mean"]
                else "insufficient_trophy_trunk_lean"
            )
            issues.append(_make_issue(
                "trophy", "trunk", code,
                trophy["trunk_lean_deg"], trunk_bench,
            ))

    mer_idx = phases.get("max_external_rotation")
    if mer_idx is not None:
        mer = metrics[mer_idx]

        elbow_bench = bench.MAX_EXTERNAL_ROTATION["elbow_flexion_deg"]
        if not _within(mer["elbow_flexion_deg"], elbow_bench):
            if mer["elbow_flexion_deg"] < elbow_bench["range"][0]:
                issues.append(_make_issue(
                    "max_external_rotation", "elbow", "shallow_mer_elbow_flexion",
                    mer["elbow_flexion_deg"], elbow_bench,
                ))

        abduction_bench = bench.MAX_EXTERNAL_ROTATION["shoulder_abduction_deg"]
        if not _within(mer["shoulder_abduction_deg"], abduction_bench):
            code = (
                "high_shoulder_abduction"
                if mer["shoulder_abduction_deg"] > abduction_bench["mean"]
                else "low_shoulder_abduction"
            )
            issues.append(_make_issue(
                "max_external_rotation", "shoulder", code,
                mer["shoulder_abduction_deg"], abduction_bench,
            ))

    impact_idx = phases.get("impact")
    if impact_idx is not None:
        impact = metrics[impact_idx]

        elbow_bench = bench.IMPACT_POSITION["elbow_flexion_deg"]
        if not _within(impact["elbow_flexion_deg"], elbow_bench):
            if impact["elbow_flexion_deg"] < elbow_bench["range"][0]:
                issues.append(_make_issue(
                    "impact", "elbow", "early_elbow_extension",
                    impact["elbow_flexion_deg"], elbow_bench,
                ))

        knee_bench = bench.IMPACT_POSITION["knee_flexion_deg"]
        if impact["knee_flexion_deg"] is not None and not _within(impact["knee_flexion_deg"], knee_bench):
            if impact["knee_flexion_deg"] < knee_bench["range"][0]:
                issues.append(_make_issue(
                    "impact", "knee", "low_impact_knee_flexion",
                    impact["knee_flexion_deg"], knee_bench,
                ))

    return issues


def _reference_block() -> Dict:
    return {
        "peak_angular_velocity_sequence": bench.PEAK_ANGULAR_VELOCITY_SEQUENCE,
        "peak_joint_torque": bench.PEAK_JOINT_TORQUE,
        "max_external_rotation_literature": bench.MAX_EXTERNAL_ROTATION,
        "note": (
            "Angular velocity and torque figures are elite-player "
            "reference context; they cannot be measured from 2D "
            "video and are not computed for this specific clip."
        ),
    }


def generate_feedback(issues: List[Dict]) -> str:
    if not issues:
        return (
            "Serve mechanics are within elite benchmark ranges at "
            "the phases we could detect (trophy, max external "
            "rotation, impact)."
        )

    sentences = []

    for issue in issues:
        note = issue.get("injury_risk")
        if note:
            sentences.append(note)

    if not sentences:
        sentences.append("Review the flagged joint angles against the target ranges.")

    return " ".join(dict.fromkeys(sentences))
