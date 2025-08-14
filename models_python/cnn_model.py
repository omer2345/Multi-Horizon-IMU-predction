# cnn_model.py
from typing import Dict
import torch
from torch import nn
from common import check_cfg

class CNN1D(nn.Module):
    def __init__(self, input_size: int, horizon: int, num_filters: int, kernel_size: int):
        super().__init__()
        self.input_size = input_size
        self.horizon = horizon
        self.conv = nn.Conv1d(in_channels=input_size,
                              out_channels=num_filters,
                              kernel_size=kernel_size,
                              padding=kernel_size//2)
        self.act  = nn.ReLU()
        self.gap  = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(num_filters, input_size * horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        x = self.act(self.conv(x))
        x = self.gap(x).squeeze(-1)           # (B, num_filters)
        x = self.fc(x)                        # (B, F*horizon)
        return x.view(-1, self.horizon, self.input_size)

def build_model(input_size: int, horizon: int, cfg: Dict) -> nn.Module:
    check_cfg({"num_filters": int, "kernel_size": int}, cfg)
    if cfg["kernel_size"] < 1:
        raise ValueError("kernel_size must be >= 1")
    return CNN1D(input_size, horizon, cfg["num_filters"], cfg["kernel_size"])
