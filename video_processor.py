import os
import tempfile
from typing import List

import cv2
import numpy as np


def _resize_preserving_aspect_ratio(
    frame: np.ndarray,
    target_height: int = 512,
) -> np.ndarray:
    """
    Resize a frame without stretching the person's body proportions.
    """
    height, width = frame.shape[:2]

    if height <= 0 or width <= 0:
        raise ValueError("Invalid video frame dimensions")

    scale = target_height / float(height)
    target_width = max(int(round(width * scale)), 1)

    return cv2.resize(
        frame,
        (target_width, target_height),
    )


def process_video(
    video_bytes: bytes,
    target_fps: int = 10,
    target_height: int = 512,
    max_frames: int = 300,
) -> List[np.ndarray]:
    """
    Decode video bytes and return sampled BGR frames.
    """
    if not video_bytes:
        return []

    temp_path = None
    frames: List[np.ndarray] = []

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4",
        ) as temp:
            temp.write(video_bytes)
            temp_path = temp.name

        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)

        if not fps or fps <= 0:
            fps = 30.0

        safe_target_fps = max(target_fps, 1)

        frame_skip = max(
            int(round(fps / safe_target_fps)),
            1,
        )

        frame_index = 0

        while cap.isOpened() and len(frames) < max_frames:
            success, frame = cap.read()

            if not success:
                break

            if frame_index % frame_skip == 0:
                resized_frame = _resize_preserving_aspect_ratio(
                    frame,
                    target_height,
                )

                frames.append(resized_frame)

            frame_index += 1

        cap.release()

        return frames

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)