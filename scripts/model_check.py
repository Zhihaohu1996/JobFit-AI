"""
中文：Day 6 模型结构检查脚本。
English: Day 6 model architecture checking script.

中文：这个脚本读取 Day 5 生成的 jobfit_features.csv，
把特征转成 Tensor，然后送入 JobFitNet，检查输出 shape 是否正确。

English: This script reads jobfit_features.csv generated on Day 5,
converts features into tensors, feeds them into JobFitNet,
and checks whether the output shape is correct.
"""

from pathlib import Path

import pandas as pd
import torch

from model.jobfit_model import JobFitNet


ROOT = Path(__file__).resolve().parents[1]
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
    中文：主流程：读取特征 -> 转 Tensor -> 创建模型 -> forward pass -> 打印输出。
    English: Main workflow: load features -> convert to tensor -> create model -> forward pass -> print output.
    """

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Missing feature file: {FEATURE_PATH}. Please finish Day 5 first."
        )

    df = pd.read_csv(FEATURE_PATH)

    X = df[FEATURE_COLUMNS].values
    X_tensor = torch.tensor(X, dtype=torch.float32)

    model = JobFitNet(input_size=len(FEATURE_COLUMNS), hidden_size=16, num_classes=3)

    # 中文：模型现在还没有训练，所以输出没有真实预测意义。
    # English: The model is not trained yet, so the output is not meaningful as prediction.
    outputs = model(X_tensor)

    print("Model forward pass completed successfully.")

    print("\nInput tensor shape:")
    print(X_tensor.shape)

    print("\nOutput tensor shape:")
    print(outputs.shape)

    print("\nFirst output row logits:")
    print(outputs[0])

    print("\nExpected output shape:")
    print(f"[number_of_samples, 3] = [{len(df)}, 3]")


if __name__ == "__main__":
    main()