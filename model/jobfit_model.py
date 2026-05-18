"""
中文：Day 6 JobFit AI PyTorch 模型结构。
English: Day 6 JobFit AI PyTorch model architecture.

中文：这个文件只定义神经网络结构，不负责训练。
English: This file only defines the neural network architecture and does not train the model.
"""

import torch
import torch.nn as nn


class JobFitNet(nn.Module):
    """
    中文：JobFit AI 的前馈神经网络模型。
    English: Feed-forward neural network model for JobFit AI.

    中文：输入是 Day 5 生成的 8 个数字特征。
    English: The input is the 8 numeric features generated on Day 5.

    中文：输出是 3 个类别的 logits：
    0 = low match
    1 = medium match
    2 = high match

    English: The output is logits for 3 classes:
    0 = low match
    1 = medium match
    2 = high match
    """

    def __init__(self, input_size: int = 8, hidden_size: int = 16, num_classes: int = 3):
        """
        中文：初始化模型层。
        English: Initialize model layers.
        """

        super().__init__()

        # 中文：第一层把 8 个输入特征映射到 hidden_size 维。
        # English: The first layer maps 8 input features to hidden_size dimensions.
        self.fc1 = nn.Linear(input_size, hidden_size)

        # 中文：ReLU 激活函数增加非线性能力。
        # English: ReLU activation adds non-linear modeling ability.
        self.relu = nn.ReLU()

        # 中文：第二层把 hidden_size 映射到 3 个类别。
        # English: The second layer maps hidden_size to 3 output classes.
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        中文：定义前向传播。
        English: Define the forward pass.

        中文：输入 shape: [batch_size, 8]
        English: Input shape: [batch_size, 8]

        中文：输出 shape: [batch_size, 3]
        English: Output shape: [batch_size, 3]
        """

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x