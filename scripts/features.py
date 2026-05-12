"""
中文：Day 5 数字特征构建脚本。
English: Day 5 numeric feature engineering script.

中文：这个脚本会读取 Day 3 生成的 jobfit_preprocessed_preview.csv，
把 resume_text、job_description、resume_skills、job_skills 等信息转换成数字特征。

English: This script reads the jobfit_preprocessed_preview.csv file generated on Day 3
and converts resume text, job description, resume skills, and job skills into numeric features.
"""

from pathlib import Path

import pandas as pd


# 中文：获取项目根目录。
# English: Get the project root directory.
ROOT = Path(__file__).resolve().parents[1]

# 中文：Day 3 生成的预处理结果文件。
# English: Preprocessed file generated on Day 3.
INPUT_PATH = ROOT / "data" / "jobfit_preprocessed_preview.csv"

# 中文：Day 5 输出的数字特征文件。
# English: Numeric feature file generated on Day 5.
OUTPUT_PATH = ROOT / "data" / "jobfit_features.csv"


def count_words(text: str) -> int:
    """
    中文：统计文本中的单词数量。
    English: Count the number of words in a text.

    中文：这里用最简单的空格分词，适合当前英文 resume/JD 示例。
    English: We use simple whitespace splitting, which works for the current English resume/JD examples.
    """

    if not isinstance(text, str):
        return 0

    return len(text.split())


def parse_skill_list(skill_text) -> list:
    """
    中文：把 CSV 里的技能字符串转换成 Python list。
    English: Convert the skill string in the CSV into a Python list.

    示例 / Example:
    "python, fastapi, sql" -> ["python", "fastapi", "sql"]
    """

    if not isinstance(skill_text, str) or not skill_text.strip():
        return []

    return [skill.strip() for skill in skill_text.split(",") if skill.strip()]


def safe_ratio(numerator: float, denominator: float) -> float:
    """
    中文：安全计算比例，避免 denominator 为 0。
    English: Safely calculate a ratio and avoid division by zero.
    """

    if denominator == 0:
        return 0.0

    return round(numerator / denominator, 3)


def build_feature_row(row) -> dict:
    """
    中文：把一行原始预处理数据转换成一行数字特征。
    English: Convert one row of preprocessed data into one row of numeric features.
    """

    resume_text = str(row.get("resume_text", ""))
    job_description = str(row.get("job_description", ""))

    resume_skills = parse_skill_list(row.get("resume_skills", ""))
    job_skills = parse_skill_list(row.get("job_skills", ""))

    resume_skill_set = set(resume_skills)
    job_skill_set = set(job_skills)

    common_skills = resume_skill_set.intersection(job_skill_set)
    missing_skills = job_skill_set.difference(resume_skill_set)

    resume_word_count = count_words(resume_text)
    job_word_count = count_words(job_description)

    resume_skill_count = len(resume_skill_set)
    job_skill_count = len(job_skill_set)
    common_skill_count = len(common_skills)
    missing_skill_count = len(missing_skills)

    skill_overlap_score = float(row.get("skill_overlap_score", 0.0))
    length_ratio = safe_ratio(resume_word_count, job_word_count)

    return {
        "resume_word_count": resume_word_count,
        "job_word_count": job_word_count,
        "resume_skill_count": resume_skill_count,
        "job_skill_count": job_skill_count,
        "common_skill_count": common_skill_count,
        "missing_skill_count": missing_skill_count,
        "skill_overlap_score": skill_overlap_score,
        "length_ratio": length_ratio,
        "label": int(row["label"]),
        "role_type": row.get("role_type", "Unknown"),
    }


def main():
    """
    中文：主流程：读取预处理结果 -> 构建数字特征 -> 保存 feature CSV。
    English: Main workflow: read preprocessed data -> build numeric features -> save feature CSV.
    """

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_PATH}. Please run Day 3 preprocess.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "resume_text",
        "job_description",
        "resume_skills",
        "job_skills",
        "skill_overlap_score",
        "label",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    feature_rows = [build_feature_row(row) for _, row in df.iterrows()]
    feature_df = pd.DataFrame(feature_rows)

    feature_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("Feature engineering completed successfully.")
    print(f"Output saved to: {OUTPUT_PATH}")

    print("\nFeature preview:")
    print(feature_df.head())

    print("\nFeature columns:")
    print(feature_df.columns.tolist())


if __name__ == "__main__":
    main()