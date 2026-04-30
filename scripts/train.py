import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import pandas as pd
import torch
torch.set_num_threads(1)
import torch.nn as nn
from model.jobfit_model import JobFitNet
from scripts.preprocess import build_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "jobfit_training_data.csv"
MODEL_PATH = ROOT / "model" / "jobfit_model.pt"

def main():
    df = pd.read_csv(DATA)
    X = [build_features(r.resume_text, r.job_description) for r in df.itertuples()]
    y = df["label"].tolist()
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    model = JobFitNet(input_size=X.shape[1])
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(80):
        optimizer.zero_grad()     # 中文: 清空旧梯度 / English: clear old gradients
        logits = model(X)         # 中文: 前向传播 / English: forward pass
        loss = loss_fn(logits, y)
        loss.backward()           # 中文: 反向传播 / English: backpropagation
        optimizer.step()          # 中文: 参数更新 / English: update parameters
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
if __name__ == "__main__":
    main()
