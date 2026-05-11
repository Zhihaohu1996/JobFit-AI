"""
中文：Day 4 传统机器学习 baseline 训练脚本。
English: Day 4 traditional machine learning baseline training script.

中文：这个脚本使用 TF-IDF + Logistic Regression 来建立 JobFit AI 的第一个基线模型。
English: This script uses TF-IDF + Logistic Regression to build the first baseline model for JobFit AI.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split


# 中文：获取项目根目录。
# English: Get the project root directory.
ROOT = Path(__file__).resolve().parents[1]

# 中文：Day 2 创建的数据集路径。
# English: Dataset path created on Day 2.
DATA_PATH = ROOT / "data" / "jobfit_training_data.csv"

# 中文：保存 baseline 模型和向量器的路径。
# English: Paths for saving the baseline model and vectorizer.
MODEL_DIR = ROOT / "model"
VECTORIZER_PATH = MODEL_DIR / "baseline_tfidf_vectorizer.joblib"
MODEL_PATH = MODEL_DIR / "baseline_logistic_model.joblib"


def build_input_text(row) -> str:
    """
    中文：把 resume_text 和 job_description 合并成一个模型输入文本。
    English: Combine resume_text and job_description into one model input text.

    中文：这里先用最简单的方法，把简历和岗位描述拼接起来。
    English: Here we use a simple approach: concatenate resume and job description.
    """

    resume_text = str(row["resume_text"])
    job_description = str(row["job_description"])

    return f"Resume: {resume_text} Job: {job_description}"


def main():
    """
    中文：主流程：读取数据 -> 构造文本 -> TF-IDF 向量化 -> 训练 Logistic Regression -> 保存模型。
    English: Main workflow: load data -> build text -> TF-IDF vectorization -> train Logistic Regression -> save model.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset file: {DATA_PATH}. Please finish Day 2 first."
        )

    MODEL_DIR.mkdir(exist_ok=True)

    # 中文：读取 Day 2 的训练数据。
    # English: Load the Day 2 training dataset.
    df = pd.read_csv(DATA_PATH)

    required_columns = {"resume_text", "job_description", "label"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # 中文：构造模型输入文本。
    # English: Build model input text.
    df["combined_text"] = df.apply(build_input_text, axis=1)

    X = df["combined_text"]
    y = df["label"]

    # 中文：划分训练集和测试集。
    # English: Split data into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    # 中文：TF-IDF 把文本转换成数字向量。
    # English: TF-IDF converts text into numerical vectors.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=500,
    )

    X_train_vectors = vectorizer.fit_transform(X_train)
    X_test_vectors = vectorizer.transform(X_test)

    # 中文：Logistic Regression 是一个经典分类模型。
    # English: Logistic Regression is a classic classification model.
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vectors, y_train)

    predictions = model.predict(X_test_vectors)

    print("Baseline training completed successfully.")
    print("\nAccuracy:")
    print(accuracy_score(y_test, predictions))

    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    # 中文：保存向量器和模型，后续预测时必须一起加载。
    # English: Save both vectorizer and model; prediction must load both later.
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)

    print("\nSaved files:")
    print(f"Vectorizer: {VECTORIZER_PATH}")
    print(f"Model: {MODEL_PATH}")


if __name__ == "__main__":
    main()