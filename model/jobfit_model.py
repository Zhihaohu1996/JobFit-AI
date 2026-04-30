import torch
import torch.nn as nn

class JobFitNet(nn.Module):
    """中文: JobFit AI 分类模型。English: JobFit AI classification model."""
    def __init__(self, input_size: int = 5, hidden_size: int = 16, num_classes: int = 3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # 中文: 输入特征到隐藏层 / English: input features to hidden layer
            nn.ReLU(),                           # 中文: 增加非线性 / English: add non-linearity
            nn.Linear(hidden_size, num_classes)  # 中文: 输出三类匹配等级 / English: output 3 match classes
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
