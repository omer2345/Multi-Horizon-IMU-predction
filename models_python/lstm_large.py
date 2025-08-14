# lstm_large.py
from typing import Dict
import torch
from torch import nn
from common import check_cfg

class LSTMLarge(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, horizon: int):
        super().__init__()
        self.input_size = input_size
        self.horizon = horizon
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, input_size * horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        return out.view(-1, self.horizon, self.input_size)

def build_model(input_size: int, horizon: int, cfg: Dict) -> nn.Module:
    check_cfg({"hidden_size": int, "num_layers": int}, cfg)
    if cfg["num_layers"] < 2:
        raise ValueError("LSTMLarge expects num_layers >= 2")
    return LSTMLarge(input_size, cfg["hidden_size"], cfg["num_layers"], horizon)
