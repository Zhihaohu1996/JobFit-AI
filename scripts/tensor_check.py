"""
中文：Day 5 PyTorch Tensor 检查脚本。
English: Day 5 PyTorch tensor checking script.

中文：这个脚本会读取 jobfit_features.csv，
把数字特征转换成 PyTorch Tensor，并检查 shape、dtype 和样例数据。

English: This script reads jobfit_features.csv,
converts numeric features into PyTorch tensors,
and checks shape, dtype, and sample values.
"""

from pathlib import Path

import pandas as pd
import torch


# 中文：获取项目根目录。
# English: Get the project root directory.
ROOT = Path(__file__).resolve().parents[1]

# 中文：Day 5 生成的数字特征文件。
# English: Numeric feature file generated on Day 5.
FEATURE_PATH = ROOT / "data" / "jobfit_features.csv"


FEATURE_COLUMNS = [
    "resume_word_count",
    "job_word_count",
    "resume_skill_count",
    "job_skill_count",
    "common_skill_count",
    "missing_skill_count",
    "skill_overlap_score",
    "length_ratio",
]


def main():
    """
    中文：主流程：读取 feature CSV -> 转换成 Tensor -> 打印检查信息。
    English: Main workflow: read feature CSV -> convert to tensors -> print checking information.
    """

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Missing feature file: {FEATURE_PATH}. Please run scripts/features.py first."
        )

    df = pd.read_csv(FEATURE_PATH)

    missing_columns = set(FEATURE_COLUMNS + ["label"]) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # 中文：X 是模型输入特征。
    # English: X is the model input feature matrix.
    X = df[FEATURE_COLUMNS].values

    # 中文：y 是模型要预测的标签。
    # English: y is the label that the model needs to predict.
    y = df["label"].values

    # 中文：PyTorch 模型通常需要 float32 输入特征。
    # English: PyTorch models usually need float32 input features.
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # 中文：分类任务的 label 通常使用 long 类型。
    # English: Classification labels usually use long dtype.
    y_tensor = torch.tensor(y, dtype=torch.long)

    print("Tensor conversion completed successfully.")

    print("\nX_tensor shape:")
    print(X_tensor.shape)

    print("\ny_tensor shape:")
    print(y_tensor.shape)

    print("\nX_tensor dtype:")
    print(X_tensor.dtype)

    print("\ny_tensor dtype:")
    print(y_tensor.dtype)

    print("\nFirst feature row:")
    print(X_tensor[0])

    print("\nFirst label:")
    print(y_tensor[0])

    print("\nFeature columns:")
    for index, column in enumerate(FEATURE_COLUMNS):
        print(f"{index}: {column}")


if __name__ == "__main__":
    main()