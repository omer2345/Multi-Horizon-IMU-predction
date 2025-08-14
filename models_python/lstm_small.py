# lstm_small.py
from typing import Dict
import torch
from torch import nn
from common import check_cfg

class LSTMSmall(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, horizon: int):
        super().__init__()
        self.input_size = input_size
        self.horizon = horizon
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, input_size * horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])                # (B, F*horizon)
        return out.view(-1, self.horizon, self.input_size)

def build_model(input_size: int, horizon: int, cfg: Dict) -> nn.Module:
    # required keys for this model
    check_cfg({"hidden_size": int}, cfg)
    if horizon not in (1,2,3):
        raise ValueError("horizon must be 1, 2, or 3")
    return LSTMSmall(input_size=input_size,
                     hidden_size=cfg["hidden_size"],
                     horizon=horizon)