from typing import Dict, List


def is_low_confidence(
    keypoints: List[Dict[str, list]],
    min_average_visibility: float = 0.55,
    max_empty_frame_ratio: float = 0.35,
    max_low_visibility_ratio: float = 0.30,
) -> bool:
    """
    Return True when pose landmarks are too unreliable
    for rule-based movement analysis.
    """
    if not keypoints:
        return True

    visibility_values = []
    empty_frames = 0
    low_visibility_points = 0

    for frame in keypoints:
        if not frame:
            empty_frames += 1
            continue

        for point in frame.values():
            if len(point) < 3:
                low_visibility_points += 1
                continue

            visibility = float(point[2])
            visibility_values.append(visibility)

            if visibility < 0.20:
                low_visibility_points += 1

    if not visibility_values:
        return True

    average_visibility = (
        sum(visibility_values) / len(visibility_values)
    )

    empty_frame_ratio = (
        empty_frames / len(keypoints)
    )

    total_points = (
        len(visibility_values) + low_visibility_points
    )

    low_visibility_ratio = (
        low_visibility_points / max(total_points, 1)
    )

    return (
        average_visibility < min_average_visibility
        or empty_frame_ratio > max_empty_frame_ratio
        or low_visibility_ratio > max_low_visibility_ratio
    )