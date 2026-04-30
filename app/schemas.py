from pydantic import BaseModel, Field

class AnalyzeRequest(BaseModel):
    resume_text: str = Field(..., min_length=20)
    job_description: str = Field(..., min_length=20)

class AnalyzeResponse(BaseModel):
    match_score: float
    match_level: str
    matched_skills: list[str]
    missing_skills: list[str]
