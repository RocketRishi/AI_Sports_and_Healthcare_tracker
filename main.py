import json
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from groq import Groq

from core.confidence import is_low_confidence
from core.pose_service import extract_keypoints
from core.video_processor import process_video
from fallback.gemini_service import analyze_with_gemini
from modules import health_analysis, sprint_analysis, tennis_analysis

app = FastAPI(title="Movement Analysis API")

FRONTEND_DIRECTORY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../frontend")
)

if os.path.isdir(FRONTEND_DIRECTORY):
    app.mount(
        "/static",
        StaticFiles(directory=FRONTEND_DIRECTORY),
        name="static",
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_groq_client() -> Optional[Groq]:
    api_key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=api_key) if api_key else None


@app.get("/")
def serve_frontend():
    index_path = os.path.join(FRONTEND_DIRECTORY, "index.html")

    if not os.path.isfile(index_path):
        return {
            "status": "ok",
            "message": "Movement Analysis API is running",
        }

    return FileResponse(index_path)


def translate_to_natural_language(data: Dict[str, Any], mode: str) -> str:
    """Convert structured output into plain-language coaching feedback."""
    fallback_feedback = str(
        data.get(
            "feedback",
            "Analysis completed, but no feedback was returned.",
        )
    )

    client = get_groq_client()

    if client is None:
        return fallback_feedback

    prompt = f"""
You are a sports and health AI coach.

Convert this JSON into clear, actionable advice for a user.

Mode: {mode}
JSON: {json.dumps(data)}

Requirements:
- Give a short explanation.
- Give clear improvement tips.
- Avoid technical jargon.
- Do not invent findings that are absent from the JSON.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.3,
        )

        content = response.choices[0].message.content
        return content or fallback_feedback

    except Exception:
        # Return the original feedback if Groq is unavailable.
        return fallback_feedback


def make_response(
    result: Dict[str, Any],
    mode: str,
    used_fallback: bool,
) -> Dict[str, Any]:
    """
    Always return the same top-level response structure.

    This prevents the frontend from receiving undefined values.
    """
    result["used_fallback"] = used_fallback

    return {
        "used_fallback": used_fallback,
        "json": result,
        "natural_language": translate_to_natural_language(result, mode),
    }


@app.post("/analyze")
async def analyze(
    mode: str,
    video: UploadFile = File(...),
):
    normalized_mode = mode.strip().lower()

    analyzers = {
        "sprint": sprint_analysis.run,
        "tennis": tennis_analysis.run,
        "health": health_analysis.run,
    }

    if normalized_mode not in analyzers:
        raise HTTPException(
            status_code=400,
            detail="Invalid mode. Use sprint, tennis, or health.",
        )

    filename = (video.filename or "").lower()

    valid_extension = filename.endswith(
        (
            ".mp4",
            ".mov",
            ".mkv",
            ".avi",
        )
    )

    valid_content_type = (
        video.content_type is None
        or video.content_type.startswith("video/")
    )

    if not valid_extension and not valid_content_type:
        raise HTTPException(
            status_code=400,
            detail="Please upload a video file.",
        )

    try:
        contents = await video.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="The uploaded video is empty.",
            )

        frames = process_video(contents)

        if not frames:
            raise HTTPException(
                status_code=400,
                detail="The video could not be decoded into frames.",
            )

        keypoints = extract_keypoints(frames)

        no_pose_detected = (
            not keypoints
            or all(not frame for frame in keypoints)
        )

        if no_pose_detected:
            fallback_result = analyze_with_gemini(
                frames,
                normalized_mode,
            )

            return make_response(
                fallback_result,
                normalized_mode,
                used_fallback=True,
            )

        if is_low_confidence(keypoints):
            fallback_result = analyze_with_gemini(
                frames,
                normalized_mode,
            )

            return make_response(
                fallback_result,
                normalized_mode,
                used_fallback=True,
            )

        analysis_function = analyzers[normalized_mode]
        result = analysis_function(keypoints)

        return make_response(
            result,
            normalized_mode,
            used_fallback=False,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {exc}",
        ) from exc