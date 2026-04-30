import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILLS = json.loads((ROOT / "data" / "skill_keywords.json").read_text(encoding="utf-8"))
ALL_SKILLS = sorted({s for group in SKILLS.values() for s in group})

def normalize_text(text: str) -> str:
    """中文: 清洗文本。English: Normalize text for consistent matching."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#./ -]", " ", text)
    return " ".join(text.split())

def extract_skills(text: str) -> set[str]:
    """中文: 从文本中提取技能关键词。English: Extract skill keywords from text."""
    norm = normalize_text(text)
    return {skill for skill in ALL_SKILLS if skill in norm}

def build_features(resume_text: str, job_description: str) -> list[float]:
    """中文: 构造模型输入特征。English: Build numeric model features."""
    r_skills = extract_skills(resume_text)
    j_skills = extract_skills(job_description)
    overlap = len(r_skills & j_skills) / max(1, len(j_skills))
    missing = len(j_skills - r_skills)
    return [len(r_skills), len(j_skills), len(r_skills & j_skills), missing, overlap]
