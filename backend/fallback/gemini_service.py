from typing import Any, Dict, List

import numpy as np


def analyze_with_gemini(
    frames: List[np.ndarray],
    mode: str,
) -> Dict[str, Any]:
    """
    Safe fallback response.

    This does not pretend that Gemini analyzed the video.
    Replace this function with a real Gemini vision request
    when the Gemini integration is configured.
    """
    frame_count = len(frames)

    return {
        "mode": mode,
        "used_fallback": True,
        "issues": [],
        "feedback": (
            "The pose could not be detected reliably in enough "
            "video frames. Please use a full-body video with the "
            "camera steady, good lighting, and the entire movement "
            "visible from the side."
        ),
        "diagnostics": {
            "frames_available": frame_count,
            "fallback_provider": "local_message",
        },
    }
