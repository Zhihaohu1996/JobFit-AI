import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import torch
torch.set_num_threads(1)
from model.jobfit_model import JobFitNet
from scripts.preprocess import build_features, extract_skills

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "jobfit_model.pt"
LEVELS = {0: "Low", 1: "Medium", 2: "High"}

def analyze_resume_job(resume_text: str, job_description: str) -> dict:
    """中文: 分析简历与岗位描述匹配度。English: Analyze resume-job match."""
    features = torch.tensor([build_features(resume_text, job_description)], dtype=torch.float32)
    model = JobFitNet(input_size=features.shape[1])
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        logits = model(features)
        probs = torch.softmax(logits, dim=1)[0]
        label = int(torch.argmax(probs).item())
    r_skills = extract_skills(resume_text)
    j_skills = extract_skills(job_description)
    return {
        "match_score": round(float(probs[label]) * 100, 2),
        "match_level": LEVELS[label],
        "matched_skills": sorted(r_skills & j_skills),
        "missing_skills": sorted(j_skills - r_skills),
    }
