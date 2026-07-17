from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

from modules import sprint_analysis, tennis_analysis, health_analysis
from core.video_processor import process_video
from core.pose_service import extract_keypoints
from core.confidence import is_low_confidence
from fallback.gemini_service import analyze_with_gemini

app = FastAPI()

# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

# CORS (REQUIRED for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_frontend():
    return FileResponse("../frontend/index.html")


@app.post("/analyze")
async def analyze(mode: str, video: UploadFile = File(...)):
    try:
        # Read video bytes
        contents = await video.read()

        # Process video
        frames = process_video(contents)
        if not frames:
            raise HTTPException(status_code=400, detail="Video processing failed")

        # Pose detection
        keypoints = extract_keypoints(frames)
        if not keypoints:
            raise HTTPException(status_code=400, detail="Pose detection failed")

        # Confidence check
        if is_low_confidence(keypoints):
            return analyze_with_gemini(frames, mode)

        # Route to module
        if mode == "sprint":
            result = sprint_analysis.run(keypoints)

        elif mode == "tennis":
            result = tennis_analysis.run(keypoints)

        elif mode == "health":
            result = health_analysis.run(keypoints)

        else:
            raise HTTPException(status_code=400, detail="Invalid mode")

        return {
            "used_fallback": False,
            "data": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
