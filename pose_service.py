from typing import Dict, List

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

LANDMARK_MAP = {
    "shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "hip": mp_pose.PoseLandmark.LEFT_HIP,
    "knee": mp_pose.PoseLandmark.LEFT_KNEE,
    "ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
    "elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
    "wrist": mp_pose.PoseLandmark.LEFT_WRIST,
}


def extract_keypoints(
    frames: List[np.ndarray],
) -> List[Dict[str, list]]:
    """
    Extract normalized x/y coordinates and visibility from BGR frames.
    """
    all_keypoints: List[Dict[str, list]] = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        for frame in frames:
            if frame is None or frame.size == 0:
                all_keypoints.append({})
                continue

            # OpenCV uses BGR.
            # MediaPipe expects RGB.
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            rgb_frame.flags.writeable = False

            results = pose.process(rgb_frame)

            frame_data: Dict[str, list] = {}

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                for name, landmark_index in LANDMARK_MAP.items():
                    landmark = landmarks[landmark_index]

                    frame_data[name] = [
                        float(landmark.x),
                        float(landmark.y),
                        float(landmark.visibility),
                    ]

            all_keypoints.append(frame_data)

    return all_keypoints