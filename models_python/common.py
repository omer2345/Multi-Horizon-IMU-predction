# common.py
from typing import Dict, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader

def check_cfg(required: Dict[str, type], cfg: Dict) -> None:
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing hyperparameters: {missing}")
    wrong = [k for k, t in required.items() if not isinstance(cfg[k], t)]
    if wrong:
        raise TypeError(f"Wrong types: {[(k, type(cfg[k]).__name__, t.__name__) for k,t in required.items() if k in wrong]}")

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str = "cpu") -> float:
    """Returns MSE over loader."""
    model.eval()
    crit = nn.MSELoss(reduction="sum")
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total += crit(pred, y).item()
        n += y.numel()
    return float(total / n)

def fit(model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        lr: float = 1e-3,
        device: str = "cpu") -> Tuple[list, list]:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    tr_hist, va_hist = [], []
    for _ in range(epochs):
        model.train()
        epoch_loss, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * x.size(0)
            n += x.size(0)
        tr_hist.append(epoch_loss / max(n,1))
        va_hist.append(evaluate(model, val_loader, device))
    return tr_hist, va_hist
