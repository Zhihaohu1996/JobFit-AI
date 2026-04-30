import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.inference import analyze_resume_job

resume = "I have Python, FastAPI, SQL, and PyTorch project experience."
job = "We need an AI intern with Python, PyTorch, FastAPI, SQL, and REST API experience."
print(analyze_resume_job(resume, job))
