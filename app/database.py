import sqlite3
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "jobfit.db"

def init_db():
    """中文: 初始化数据库表。English: Initialize database table."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_text TEXT,
            job_description TEXT,
            match_score REAL,
            match_level TEXT,
            missing_skills TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """)
