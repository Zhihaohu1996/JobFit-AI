from fastapi import FastAPI
from app.schemas import AnalyzeRequest, AnalyzeResponse
from app.inference import analyze_resume_job
from app.database import init_db
from app.crud import save_analysis

app = FastAPI(title="JobFit AI")
init_db()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest):
    result = analyze_resume_job(payload.resume_text, payload.job_description)
    save_analysis(payload.resume_text, payload.job_description, result)
    return result
