"""
中文: Day 2 数据检查脚本。
English: Day 2 data checking script.

中文：这个文件用于确认 jobfit_training_data.csv 是否可以被 pandas 正常读取。
English: This file checks whether jobfit_training_data.csv can be loaded correctly by pandas.
"""

import pandas as pd
from pathlib import Path

# 中文：获取项目根目录。
# English: Get the project root directory.
ROOT = Path(__file__).resolve().parents[1]

# 中文：拼接数据文件路径。
# English: Build the dataset file path.
DATA_PATH = ROOT / "data" / "jobfit_training_data.csv"


def main():
    """
    中文：读取 CSV，并打印基本信息。
    English: Load the CSV file and print basic information.
    """

    # 中文：读取 CSV 文件。
    # English: Read the CSV file.
    df = pd.read_csv(DATA_PATH)

    # 中文：打印前 5 行，确认数据格式正确。
    # English: Print the first 5 rows to confirm the data format.
    print("First 5 rows:")
    print(df.head())

    # 中文：打印数据集形状，例如多少行、多少列。
    # English: Print dataset shape, such as number of rows and columns.
    print("\nDataset shape:")
    print(df.shape)

    # 中文：打印所有列名。
    # English: Print all column names.
    print("\nColumns:")
    print(df.columns.tolist())

    # 中文：检查 label 的分布。
    # English: Check the distribution of labels.
    print("\nLabel distribution:")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()