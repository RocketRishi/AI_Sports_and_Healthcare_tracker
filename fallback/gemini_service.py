import requests
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
def analyze_with_gemini(frames, mode):
    prompt = f"Analyze {mode} posture and give improvement feedback."
    try:
        response_text = "AI feedback based on visual analysis"
    except Exception:
        response_text = "Fallback analysis failed"
    
    return {
        "mode":mode,
        "used_fallback":True,
        'issues':[],
        "feedback":response_text
    }