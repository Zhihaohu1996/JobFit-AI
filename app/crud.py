import json
import sqlite3
from app.database import DB_PATH

def save_analysis(resume_text, job_description, result):
    """中文: 保存分析结果。English: Save analysis result."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO analyses (resume_text, job_description, match_score, match_level, missing_skills) VALUES (?, ?, ?, ?, ?)",
            (resume_text, job_description, result["match_score"], result["match_level"], json.dumps(result["missing_skills"]))
        )
