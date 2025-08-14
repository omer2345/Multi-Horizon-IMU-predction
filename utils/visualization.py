
"""
visualization.py
-----------------
Plot utilities for inspecting per-segment IMU prediction quality.
Works with the data structures produced by data_handeling.IMUPredictionDataset
and the (rec_idx, start, end) segments returned by train_val_test_split().
"""
from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple, Dict, Iterable

import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

# -------------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------------

def _build_segment_index(segments: List[Tuple[int,int,int]]):
    """Build per-recording index: rec_idx -> [(start, end, seg_id), ...] (sorted)."""
    per_rec = {}
    for seg_id, (ri, s, e) in enumerate(segments):
        per_rec.setdefault(ri, []).append((s, e, seg_id))
    for ri in per_rec:
        per_rec[ri].sort(key=lambda t: t[0])
    return per_rec

def _find_seg_id(rec_idx: int, sample_idx: int, seg_index) -> int:
    """Binary-search the segment containing sample_idx for a given recording; -1 if none."""
    lst = seg_index.get(rec_idx, [])
    lo, hi = 0, len(lst) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        s, e, sid = lst[mid]
        if sample_idx < s:
            hi = mid - 1
        elif sample_idx >= e:
            lo = mid + 1
        else:
            return sid
    return -1

# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------

def eval_collect_by_segment(model,
                            dataset,
                            segments: List[Tuple[int,int,int]],
                            device: str = "cuda",
                            batch_size: int = 256,
                            num_workers: int = 0,
                            horizon_ix: int = 0,
                            axis_indices: Iterable[int] = (0,1,2)):
    """Run model over dataset, group predictions/targets by segment, and compute MSE."""
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required for eval_collect_by_segment().")

    device = device if (device == "cpu" or (hasattr(torch, "cuda") and torch.cuda.is_available())) else "cpu"
    model = model.to(device).eval()

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers, pin_memory=True)

    seg_index = _build_segment_index(segments)

    seg_pred: Dict[int, list] = defaultdict(list)
    seg_targ: Dict[int, list] = defaultdict(list)

    multi_h = None

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)               # [B,T,F]
            y = batch["y"]                           # [B,F]
            rec_idx = batch["rec_idx"].cpu().numpy()
            tgt_idx = batch["target_idx"].cpu().numpy()

            out = model(x)                           # [B,F] or [B,H,F]
            if isinstance(out, (tuple, list)):
                out = out[0]

            if out.ndim == 3:
                multi_h = True
                out_np = out[:, horizon_ix, :].detach().cpu().numpy()
            else:
                multi_h = False
                out_np = out.detach().cpu().numpy()

            y_np = y.detach().cpu().numpy()

            # select axes to plot
            out_np = out_np[:, tuple(axis_indices)]
            y_np   = y_np[:,   tuple(axis_indices)]

            for i in range(out_np.shape[0]):
                sid = _find_seg_id(int(rec_idx[i]), int(tgt_idx[i]), seg_index)
                seg_pred[sid].append(out_np[i])
                seg_targ[sid].append(y_np[i])

    # convert to arrays & compute MSE per segment
    seg_arrays = {}
    seg_mse = {}
    for sid in seg_pred.keys():
        P = np.stack(seg_pred[sid]) if len(seg_pred[sid]) else np.empty((0, len(tuple(axis_indices))))
        T = np.stack(seg_targ[sid]) if len(seg_targ[sid]) else np.empty((0, len(tuple(axis_indices))))
        seg_arrays[sid] = (P, T)
        seg_mse[sid] = float(np.mean((P - T) ** 2)) if P.size else float("inf")

    meta = {"multi_horizon": bool(multi_h), "horizon_ix": int(horizon_ix), "axis_indices": tuple(axis_indices)}
    return seg_arrays, seg_mse, meta

def _plot_segment_scatter(P: np.ndarray, T: np.ndarray, title: str, axis_labels):
    """3x1 scatter plot (pred vs real) over time for selected axes."""
    steps = np.arange(len(P))
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for i, ax in enumerate(axes):
        if i >= P.shape[1]:
            ax.axis("off"); continue
        ax.scatter(steps, T[:, i], label="Real", s=12, alpha=0.7)
        ax.scatter(steps, P[:, i], label="Pred", s=12, alpha=0.7)
        ax.set_ylabel(axis_labels[i] if i < len(axis_labels) else f"axis_{i}")
        if i == 0:
            ax.set_title(title)
    axes[-1].set_xlabel("Sample index")
    axes[0].legend()
    plt.tight_layout()
    plt.show()

def plot_k_best_worst_segments(model,
                               dataset,
                               segments: List[Tuple[int,int,int]],
                               k: int = 3,
                               horizon_ix: int = 0,
                               axis_indices: Iterable[int] = (0,1,2),
                               axis_labels: Iterable[str] = ("gx","gy","gz"),
                               device: str = "cuda",
                               batch_size: int = 256,
                               num_workers: int = 0):
    """Evaluate the model, pick k best & k worst segments by MSE, and plot 3x1 scatter per segment."""
    seg_arrays, seg_mse, _ = eval_collect_by_segment(
        model, dataset, segments,
        device=device, batch_size=batch_size, num_workers=num_workers,
        horizon_ix=horizon_ix, axis_indices=axis_indices
    )

    # sort by MSE, pick best & worst k (skip sid=-1 if any)
    valid_items = [(sid, mse) for sid, mse in seg_mse.items() if sid != -1 and np.isfinite(mse)]
    if not valid_items:
        print("No valid segments to plot."); return
    valid_items.sort(key=lambda x: x[1])
    best = valid_items[:k]
    worst = valid_items[-k:] if len(valid_items) >= k else valid_items[::-1]

    # plot worst first
    for sid, mse in worst:
        P, T = seg_arrays[sid]
        _plot_segment_scatter(P, T, f"Worst seg {sid} (MSE={mse:.5f})", tuple(axis_labels))

    for sid, mse in best:
        P, T = seg_arrays[sid]
        _plot_segment_scatter(P, T, f"Best seg {sid} (MSE={mse:.5f})", tuple(axis_labels))
