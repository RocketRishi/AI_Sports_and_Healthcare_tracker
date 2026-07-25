from typing import Dict, List

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

# Bilateral landmark map. Earlier versions only captured the LEFT side,
# which meant any clip where the player's right side was the visible /
# unoccluded one (e.g. a right-handed serve filmed from the player's
# right) produced mostly low-visibility points and tripped the
# low-confidence fallback. Capturing both sides lets callers pick
# whichever side MediaPipe actually saw clearly.
LANDMARK_MAP = {
    "nose": mp_pose.PoseLandmark.NOSE,
    "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "left_hip": mp_pose.PoseLandmark.LEFT_HIP,
    "right_hip": mp_pose.PoseLandmark.RIGHT_HIP,
    "left_knee": mp_pose.PoseLandmark.LEFT_KNEE,
    "right_knee": mp_pose.PoseLandmark.RIGHT_KNEE,
    "left_ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
    "right_ankle": mp_pose.PoseLandmark.RIGHT_ANKLE,
    "left_elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
    "right_elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
    "left_wrist": mp_pose.PoseLandmark.LEFT_WRIST,
    "right_wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
    "left_foot_index": mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
    "right_foot_index": mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
}

# Joints used to alias a "dominant side" onto the legacy flat keys
# (shoulder/hip/knee/ankle/elbow/wrist) that sprint_analysis and
# health_analysis already read. This keeps those modules working
# unchanged while no longer silently defaulting to the left side.
_SIDE_JOINTS = ("shoulder", "hip", "knee", "ankle", "elbow", "wrist", "foot_index")

# Below this, a frame is treated as a detection failure worth retrying
# with relaxed settings rather than accepted as-is.
_RETRY_VISIBILITY_THRESHOLD = 0.35
_CORE_JOINTS_FOR_RETRY_CHECK = (
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
)


def _landmarks_to_frame_data(landmarks) -> Dict[str, list]:
    frame_data: Dict[str, list] = {}

    for name, landmark_index in LANDMARK_MAP.items():
        landmark = landmarks[landmark_index]
        frame_data[name] = [
            float(landmark.x),
            float(landmark.y),
            float(landmark.visibility),
        ]

    return frame_data


def _core_visibility(landmarks) -> float:
    values = [landmarks[idx].visibility for idx in _CORE_JOINTS_FOR_RETRY_CHECK]
    return sum(values) / len(values)


def _enhance_contrast(frame_bgr: np.ndarray) -> np.ndarray:
    """
    CLAHE contrast boost on the luminance channel only, so color
    balance is preserved. Helps MediaPipe find a person against
    backlit or overexposed outdoor courts/fields and dim indoor gyms,
    which are the most common reasons a frame yields no landmarks.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    enhanced = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def _detect(pose, frame_bgr: np.ndarray):
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    return pose.process(rgb_frame)


def _detect_with_fallbacks(
    primary_pose,
    frame_bgr: np.ndarray,
    fallback_pose_cache: Dict[str, object],
):
    """
    Cascade of detection attempts for a single frame:
      1. primary tracking-mode pass (fast, uses temporal smoothing)
      2. static-image pass with relaxed thresholds on the raw frame
         (recovers frames where the tracker lost lock, e.g. after fast
         motion/motion blur, without waiting for re-acquisition)
      3. static-image pass with relaxed thresholds on a contrast-
         enhanced frame (recovers frames lost to poor lighting)

    Fallback Pose instances are created lazily and cached so the
    expensive path only runs for frames that actually need it.
    """
    results = _detect(primary_pose, frame_bgr)

    if results.pose_landmarks and _core_visibility(results.pose_landmarks.landmark) >= _RETRY_VISIBILITY_THRESHOLD:
        return results

    if "relaxed" not in fallback_pose_cache:
        fallback_pose_cache["relaxed"] = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.25,
            min_tracking_confidence=0.25,
        )

    relaxed_results = _detect(fallback_pose_cache["relaxed"], frame_bgr)

    if relaxed_results.pose_landmarks and _core_visibility(relaxed_results.pose_landmarks.landmark) >= _RETRY_VISIBILITY_THRESHOLD:
        return relaxed_results

    enhanced_frame = _enhance_contrast(frame_bgr)
    enhanced_results = _detect(fallback_pose_cache["relaxed"], enhanced_frame)

    if enhanced_results.pose_landmarks:
        # Prefer whichever attempt saw more of the person.
        candidates = [
            r for r in (results, relaxed_results, enhanced_results)
            if r.pose_landmarks
        ]
        return max(candidates, key=lambda r: _core_visibility(r.pose_landmarks.landmark))

    # Nothing worked; return whichever attempt got furthest (may still
    # be landmark-free, which is a legitimate "no person here" result).
    for candidate in (results, relaxed_results):
        if candidate.pose_landmarks:
            return candidate

    return results


def _pick_dominant_side(all_keypoints: List[Dict[str, list]]) -> str:
    left_visibility = 0.0
    right_visibility = 0.0
    samples = 0

    for frame in all_keypoints:
        for joint in _SIDE_JOINTS:
            left = frame.get(f"left_{joint}")
            right = frame.get(f"right_{joint}")

            if left:
                left_visibility += left[2]
            if right:
                right_visibility += right[2]

            samples += 1

    if samples == 0:
        return "left"

    return "left" if left_visibility >= right_visibility else "right"


def _apply_legacy_aliases(all_keypoints: List[Dict[str, list]]) -> None:
    """
    Populate the flat shoulder/hip/knee/ankle/elbow/wrist/foot_index
    keys that older analyzer modules expect, aliased per-frame to
    whichever side had better visibility across the whole clip (not
    per-frame, so a swing analysis doesn't jump arms mid-motion).
    """
    dominant_side = _pick_dominant_side(all_keypoints)

    for frame in all_keypoints:
        for joint in _SIDE_JOINTS:
            sided_point = frame.get(f"{dominant_side}_{joint}")

            if sided_point is not None:
                frame[joint] = sided_point


def extract_keypoints(
    frames: List[np.ndarray],
) -> List[Dict[str, list]]:
    """
    Extract normalized x/y coordinates and visibility from BGR frames.

    Each frame's dict carries both explicit left_*/right_* landmarks
    and legacy flat aliases (shoulder/hip/knee/ankle/elbow/wrist)
    pointed at the more visible side for this clip.
    """
    all_keypoints: List[Dict[str, list]] = []
    fallback_pose_cache: Dict[str, object] = {}

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        for frame in frames:
            if frame is None or frame.size == 0:
                all_keypoints.append({})
                continue

            results = _detect_with_fallbacks(pose, frame, fallback_pose_cache)

            frame_data: Dict[str, list] = {}

            if results.pose_landmarks:
                frame_data = _landmarks_to_frame_data(results.pose_landmarks.landmark)

            all_keypoints.append(frame_data)

    if "relaxed" in fallback_pose_cache:
        fallback_pose_cache["relaxed"].close()

    _apply_legacy_aliases(all_keypoints)

    return all_keypoints
