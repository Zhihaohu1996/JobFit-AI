"""
中文：Day 3 文本预处理脚本。
English: Day 3 text preprocessing script.

中文：这个脚本会读取 Day 2 创建的 jobfit_training_data.csv，
提取简历和岗位描述中的技能关键词，并生成一个预处理预览文件。

English: This script reads the jobfit_training_data.csv file created on Day 2,
extracts skill keywords from resume and job description text,
and generates a preprocessing preview file.
"""

import json
import re
from pathlib import Path

import pandas as pd


# 中文：获取项目根目录。
# English: Get the project root directory.
ROOT = Path(__file__).resolve().parents[1]

# 中文：Day 2 创建的数据集路径。
# English: Dataset path created on Day 2.
DATA_PATH = ROOT / "data" / "jobfit_training_data.csv"

# 中文：Day 3 创建的技能关键词文件路径。
# English: Skill keyword file path created on Day 3.
SKILL_PATH = ROOT / "data" / "skill_keywords.json"

# 中文：预处理结果输出路径。
# English: Output path for the preprocessing preview.
OUTPUT_PATH = ROOT / "data" / "jobfit_preprocessed_preview.csv"


def normalize_text(text: str) -> str:
    """
    中文：标准化文本，包括转小写、去掉特殊符号、压缩多余空格。
    English: Normalize text by lowercasing, removing special symbols,
    and compressing extra spaces.
    """

    if not isinstance(text, str):
        return ""

    # 中文：转成小写。
    # English: Convert to lowercase.
    text = text.lower()

    # 中文：把非字母、数字、空格的字符替换为空格。
    # English: Replace non-letter, non-number, and non-space characters with spaces.
    text = re.sub(r"[^a-z0-9\s+#.-]", " ", text)

    # 中文：压缩连续空格。
    # English: Compress multiple spaces into one.
    text = " ".join(text.split())

    return text


def load_skill_keywords() -> dict:
    """
    中文：读取 skill_keywords.json。
    English: Load skill_keywords.json.
    """

    with open(SKILL_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_skills(text: str, skill_keywords: dict) -> list:
    """
    中文：从文本中提取技能。
    English: Extract skills from text.
    """

    normalized = normalize_text(text)
    found_skills = []

    for skill_name, keywords in skill_keywords.items():
        for keyword in keywords:
            keyword_normalized = normalize_text(keyword)

            if keyword_normalized in normalized:
                found_skills.append(skill_name)
                break

    return sorted(set(found_skills))


def calculate_overlap_score(resume_skills: list, job_skills: list) -> float:
    """
    中文：计算简历技能和岗位技能的重合比例。
    English: Calculate the overlap ratio between resume skills and job skills.
    """

    if not job_skills:
        return 0.0

    resume_set = set(resume_skills)
    job_set = set(job_skills)

    overlap = resume_set.intersection(job_set)

    return round(len(overlap) / len(job_set), 3)


def main():
    """
    中文：主流程：读取数据 -> 提取技能 -> 计算重合度 -> 保存预览结果。
    English: Main workflow: read data -> extract skills -> calculate overlap -> save preview.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset file: {DATA_PATH}")

    if not SKILL_PATH.exists():
        raise FileNotFoundError(f"Missing skill keyword file: {SKILL_PATH}")

    df = pd.read_csv(DATA_PATH)
    skill_keywords = load_skill_keywords()

    resume_skill_list = []
    job_skill_list = []
    overlap_scores = []

    for _, row in df.iterrows():
        resume_text = row["resume_text"]
        job_description = row["job_description"]

        resume_skills = extract_skills(resume_text, skill_keywords)
        job_skills = extract_skills(job_description, skill_keywords)
        overlap_score = calculate_overlap_score(resume_skills, job_skills)

        resume_skill_list.append(", ".join(resume_skills))
        job_skill_list.append(", ".join(job_skills))
        overlap_scores.append(overlap_score)

    df["resume_skills"] = resume_skill_list
    df["job_skills"] = job_skill_list
    df["skill_overlap_score"] = overlap_scores

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("Preprocessing completed successfully.")
    print(f"Output saved to: {OUTPUT_PATH}")
    print("\nPreview:")
    print(df[["resume_text", "job_description", "resume_skills", "job_skills", "skill_overlap_score"]].head())


if __name__ == "__main__":
    main()