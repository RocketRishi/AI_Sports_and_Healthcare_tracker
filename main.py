from fastapi import FastAPI, UploadFile, File, HTTPException
from modules import sprint_analysis, tennis_analysis, health_analysis
from core.video_processor import process_video
from core.pose_service import extract_keypoints
from core.confidence import is_low_confidence
from fallback.gemini_service import analyze_with_gemini

app = FastAPI()


@app.post("/analyze")
async def analyze(mode: str, video: UploadFile = File(...)):
    try:
        # Step 1: Read video file properly
        contents = await video.read()

        # Step 2: Process video into frames
        frames = process_video(contents)

        if not frames:
            raise HTTPException(status_code=400, detail="Video processing failed")

        # Step 3: Pose detection
        keypoints = extract_keypoints(frames)

        if not keypoints:
            raise HTTPException(status_code=400, detail="Pose detection failed")

        # Step 4: Confidence check
        if is_low_confidence(keypoints):
            return analyze_with_gemini(frames, mode)

        # Step 5: Route to correct module
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
