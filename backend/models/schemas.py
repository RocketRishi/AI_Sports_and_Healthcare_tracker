from pydantic import BaseModel
from typing import List, Optional

class Issue(BaseModel):
    joint: str
    issue: str
    value: float
    target: float

class AnalysisResponse(BaseModel):
    mode: str
    used_fallback: bool
    issues: List[Issue]
    feedback: str

