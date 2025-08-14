# tcn_model.py
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.pad = nn.ConstantPad1d((pad, 0), 0.0)  # left-only padding for causality
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)

    def forward(self, x):  # x: (B, C, T)
        return self.conv(self.pad(x))

class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(channels, channels, kernel_size, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(channels, channels, kernel_size, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x) + x  # residual

class CausalTCN(nn.Module):
    def __init__(self, in_features=6, channels=64, layers=4, kernel_size=3, dropout=0.1, horizon=1):
        super().__init__()
        self.horizon = int(horizon)
        self.in_proj = nn.Conv1d(in_features, channels, kernel_size=1)
        blocks = []
        for i in range(layers):
            dilation = 2 ** i
            blocks.append(TCNBlock(channels, kernel_size, dilation, dropout))
        self.tcn = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),                # (B, C, 1) -> (B, C)
            nn.Linear(channels, 128),
            nn.ReLU(),
            nn.Linear(128, 3 * self.horizon)  # predict (horizon × 3 gyro)
        )

    def forward(self, x):  # x: (B, T, F)
        x = x.transpose(1, 2)          # -> (B, F, T)
        x = self.in_proj(x)            # (B, C, T)
        x = self.tcn(x)                # (B, C, T)
        x = self.pool(x)               # (B, C, 1)
        out = self.head(x)             # (B, 3*h)
        return out.view(out.size(0), self.horizon, 3)  # (B, h, 3)

def build_model(input_size: int, horizon: int, cfg: dict):
    return CausalTCN(
        in_features=input_size,
        channels=int(cfg.get("channels", 64)),
        layers=int(cfg.get("layers", 4)),
        kernel_size=int(cfg.get("kernel_size", 3)),
        dropout=float(cfg.get("dropout", 0.1)),
        horizon=int(horizon),
    )
