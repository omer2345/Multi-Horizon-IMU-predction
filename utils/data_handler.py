
"""
data_handeling.py
-----------------
Single-file toolbox for IMU CSV loading (with calibration), segmentation by rate,
dataset windowing, and train/val/test splitting.

This file consolidates the following modules:
- calibration.py
- io.py
- segmentation.py
- dataset.py
- split.py
- __init__.py (exports)

Plus a few convenience helpers so you can use it directly in notebooks.
"""

from __future__ import annotations

import os, glob, json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Iterable

import numpy as np
import pandas as pd

# Optional torch parts guarded for environments without torch
try:
    import torch
    from torch.utils.data import Dataset
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False
    class Dataset:  # type: ignore
        pass

# ============================================================================
# Calibration
# ============================================================================

@dataclass
class Calibration:
    """Holds per-sensor scale matrices and bias vectors for gyro and acc."""
    S_gyro: np.ndarray  # (3,3)
    b_gyro: np.ndarray  # (3,)
    S_acc: np.ndarray   # (3,3)
    b_acc: np.ndarray   # (3,)

    @staticmethod
    def from_json_str(js: str) -> "Calibration":
        """Parse calibration payload tolerated for doubled quotes ("" -> ")."""
        s = js.strip()
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            s = s.replace('""', '"')
            obj = json.loads(s)
        Sg = np.array(obj["Sgyro1"], dtype=float).reshape(3,3)
        bg = np.array(obj["b_gyro1"], dtype=float).reshape(3,)
        Sa = np.array(obj["Sacc1"], dtype=float).reshape(3,3)
        ba = np.array(obj["b_acc1"], dtype=float).reshape(3,)
        return Calibration(Sg, bg, Sa, ba)


def apply_calibration_vec(S: np.ndarray, b: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Apply affine calibration (scale then bias) to vectors (1D or Nx3)."""
    raw = np.asarray(raw, dtype=float)
    if raw.ndim == 1:
        return S @ (raw - b)
    elif raw.ndim == 2 and raw.shape[1] == 3:
        return (raw - b.reshape(1,3)) @ S.T
    raise ValueError(f"Unexpected shape {raw.shape}")


# ============================================================================
# IO: loading calibrated CSVs
# ============================================================================

@dataclass
class Recording:
    """A fully calibrated IMU time series."""
    file_path: str
    timestamps_us: np.ndarray     # (N,)
    gyro: np.ndarray              # (N,3) calibrated [rad/s]
    acc: Optional[np.ndarray]     # (N,3) calibrated [m/s^2] or None
    sample_rate_hz: float
    meta: dict


_CAL_COLS = ["x", "y", "z"]  # columns that may contain the "calibration_file" token


def _extract_json_str(raw):
    """Heuristically extract a JSON object from a cell string."""
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    cands = []
    if s.startswith("{") and s.endswith("}"):
        cands = [s, s.replace('""', '"')]
    else:
        start = s.find("{"); end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            payload = s[start:end+1]
            cands = [payload, payload.replace('""','"')]
    for js in cands:
        try:
            json.loads(js); return js
        except Exception:
            pass
    return None


def _parse_calibration_row(df: pd.DataFrame) -> Optional[Calibration]:
    """Find the 'calibration_file' row and parse it into a Calibration object."""
    for _, row in df.iterrows():
        for i, col in enumerate(_CAL_COLS):
            val = row.get(col, None)
            if isinstance(val, str) and val.strip() == "calibration_file":
                if i + 1 >= len(_CAL_COLS):
                    continue
                js_cell = row.get(_CAL_COLS[i+1], None)
                js = _extract_json_str(js_cell)
                if js:
                    return Calibration.from_json_str(js)
    return None


def _coerce_numeric(df: pd.DataFrame, cols=('x','y','z')):
    """Convert columns to numeric dtype with coercion."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def _coerce_timestamp(df: pd.DataFrame):
    """Ensure 'timestamp' is int64 microseconds, dropping NaNs."""
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['timestamp'] = df['timestamp'].round().astype('int64')
    return df


def _merge_acc_gyro(df: pd.DataFrame, require_acc: bool=True):
    """
    Merge accelerometer and gyroscope by timestamp.
    If require_acc=False, returns gyro-only (unique timestamps).
    """
    df = _coerce_timestamp(df)
    df = df[df['tag'].isin(['acc','gyro'])].copy()
    df = _coerce_numeric(df, cols=['x','y','z'])
    df = df.dropna(subset=['x','y','z'], how='all')
    sort_cols = ['timestamp','tag'] + (['systemTime'] if 'systemTime' in df.columns else [])
    df = df.sort_values(sort_cols)

    acc_df = df[df['tag']=='acc'][['timestamp','x','y','z']].rename(columns={'x':'ax','y':'ay','z':'az'})
    gyr_df = df[df['tag']=='gyro'][['timestamp','x','y','z']].rename(columns={'x':'gx','y':'gy','z':'gz'})

    if require_acc:
        merged = pd.merge(gyr_df, acc_df, on='timestamp', how='inner')
        timestamps = merged['timestamp'].to_numpy(dtype=np.int64)
        gyro = merged[['gx','gy','gz']].to_numpy(dtype=float)
        acc = merged[['ax','ay','az']].to_numpy(dtype=float)
        return timestamps, gyro, acc

    gyr_df = gyr_df.drop_duplicates(subset=['timestamp'], keep='last')
    timestamps = gyr_df['timestamp'].to_numpy(dtype=np.int64)
    gyro = gyr_df[['gx','gy','gz']].to_numpy(dtype=float)
    return timestamps, gyro, None


def load_recording_csv(file_path: str, expected_hz: float, require_acc: bool=True, drop_head: int=10) -> Optional[Recording]:
    """
    Load one CSV file, parse calibration, merge streams, apply calibration,
    and drop the first few samples (sensor warmup / alignment).
    """
    df = pd.read_csv(file_path, keep_default_na=True, low_memory=False)

    for col in ['tag','timestamp']:
        if col not in df.columns:
            print(f"[WARN] {file_path} missing column '{col}', skipping.")
            return None

    calib = _parse_calibration_row(df)
    if calib is None:
        print(f"[WARN] {file_path} has no calibration_file row, skipping (calibrated-only).")
        return None

    timestamps, gyro_raw, acc_raw = _merge_acc_gyro(df, require_acc=require_acc)

    gyro_cal = apply_calibration_vec(calib.S_gyro, calib.b_gyro, gyro_raw)
    acc_cal = apply_calibration_vec(calib.S_acc,  calib.b_acc,  acc_raw) if acc_raw is not None else None

    if timestamps.shape[0] <= drop_head:
        print(f"[WARN] {file_path} too short after drop_head={drop_head}, skipping.")
        return None

    timestamps = timestamps[drop_head:]
    gyro_cal  = gyro_cal[drop_head:]
    if acc_cal is not None:
        acc_cal = acc_cal[drop_head:]

    return Recording(
        file_path=file_path,
        timestamps_us=timestamps.astype(np.int64),
        gyro=gyro_cal.astype(np.float32),
        acc=(acc_cal.astype(np.float32) if acc_cal is not None else None),
        sample_rate_hz=float(expected_hz),
        meta={}
    )


def load_recordings_from_folder(folder_path: str, expected_hz: float, require_acc: bool=True, drop_head: int=10) -> List[Recording]:
    """Load all CSV files from a flat folder (non-recursive)."""
    pattern = os.path.join(folder_path, "*.csv")
    files = sorted(glob.glob(pattern))
    recs: List[Recording] = []
    for fp in files:
        try:
            rec = load_recording_csv(fp, expected_hz=expected_hz, require_acc=require_acc, drop_head=drop_head)
            if rec is not None:
                recs.append(rec)
        except Exception as e:
            print(f"[WARN] Skipping {fp}: {e}")
    return recs


def load_two_folders(root_dir: str, folder_52: str="epsilon_52Hz", folder_104: str="epsilon_104Hz",
                     require_acc: bool=True, drop_head: int=10) -> Tuple[List[Recording], List[Recording]]:
    """Load paired 52 Hz and 104 Hz datasets."""
    f52 = os.path.join(root_dir, folder_52)
    f104 = os.path.join(root_dir, folder_104)
    recs_52  = load_recordings_from_folder(f52,  expected_hz=52.0,  require_acc=require_acc, drop_head=drop_head)
    recs_104 = load_recordings_from_folder(f104, expected_hz=104.0, require_acc=require_acc, drop_head=drop_head)
    return recs_52, recs_104


# ============================================================================
# Segmentation by expected sampling rate
# ============================================================================

def expected_dt_us(expected_hz: float) -> float:
    """Return expected delta-t in microseconds for a nominal rate."""
    return 1e6 / float(expected_hz)


def split_by_expected_rate(ts_us: np.ndarray, expected_hz: float, jump_factor: float = 1.5, min_len: int = 32) -> List[Tuple[int,int]]:
    """
    Split into continuous segments using the folder's expected rate.
    Jump if dt <= 0 or dt > jump_factor * expected_dt.
    Returns list of (start, end) index pairs (end exclusive), min_len filtered.
    """
    ts = np.asarray(ts_us, dtype=np.int64)
    n = ts.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [(0, 1)]
    dts = np.diff(ts)
    thr = expected_dt_us(expected_hz) * jump_factor
    cuts = np.where((dts <= 0) | (dts > thr))[0] + 1
    bounds = [0] + cuts.tolist() + [n]
    segs = [(bounds[i], bounds[i+1]) for i in range(len(bounds)-1)]
    segs = [s for s in segs if (s[1] - s[0]) >= min_len]
    return segs


def segment_all_recordings(recordings: List[Recording], expected_hz: float, jump_factor: float=1.5, min_len: int=32) -> List[Tuple[int,int,int]]:
    """Run segmentation over all recordings and return (rec_idx, start, end) tuples."""
    segs: List[Tuple[int,int,int]] = []
    for ri, rec in enumerate(recordings):
        local = split_by_expected_rate(rec.timestamps_us, expected_hz=expected_hz,
                                       jump_factor=jump_factor, min_len=min_len)
        segs.extend([(ri, s, e) for (s, e) in local])
    return segs


def train_val_test_split(segments: List[Tuple[int,int,int]], ratios=(0.7, 0.15, 0.15), seed: int = 42):
    """Shuffle & split segment indices to train/val/test lists with given ratios."""
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    rng = np.random.RandomState(seed)
    idx = np.arange(len(segments))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val]
    test_idx  = idx[n_train+n_val:]
    train_segs = [segments[i] for i in train_idx]
    val_segs   = [segments[i] for i in val_idx]
    test_segs  = [segments[i] for i in test_idx]
    return train_segs, val_segs, test_segs


# ============================================================================
# Dataset: fixed-length windows for prediction
# ============================================================================

@dataclass
class WindowIndex:
    """Index triple for one window/target pair within a recording."""
    rec_idx: int
    start_idx: int
    target_idx: int


class IMUPredictionDataset(Dataset):
    """
    Windows of length `history_len`; target at (history_len-1 + horizon).
    Features per step: [gx,gy,gz,(ax,ay,az)] if include_acc else [gx,gy,gz].
    Windows NEVER cross time-jump segments (you pass explicit segments).
    """
    def __init__(self, recordings: List[Recording], segments: Optional[List[Tuple[int,int,int]]]=None,
                 include_acc: bool=True, history_len: int=10, horizon: int=3):
        self.recordings = recordings
        self.segments   = segments
        self.include_acc = include_acc
        self.history_len = int(history_len)
        self.horizon     = int(horizon)
        self._windows: List[WindowIndex] = []
        self._build_index()

    def _get_arrays(self, rec: Recording):
        gyro = rec.gyro
        acc  = rec.acc if self.include_acc else None
        if self.include_acc and acc is None:
            raise ValueError(f"Accelerometer requested but missing for {rec.file_path}")
        return gyro, acc

    def _build_index(self):
        self._windows.clear()
        if self.segments is None:
            self.segments = []
            for ri, rec in enumerate(self.recordings):
                self.segments.append((ri, 0, rec.gyro.shape[0]))
        for (ri, s, e) in self.segments:
            rec = self.recordings[ri]
            gyro, acc = self._get_arrays(rec)
            max_start = e - (self.history_len + self.horizon)
            for i in range(s, max_start + 1):
                self._windows.append(WindowIndex(ri, i, i + self.history_len - 1 + self.horizon))

    def __len__(self):
        return len(self._windows)

    def __getitem__(self, idx):
        wi = self._windows[idx]
        rec = self.recordings[wi.rec_idx]
        gyro, acc = self._get_arrays(rec)
        s = wi.start_idx
        e = s + self.history_len
        x_g = gyro[s:e, :]
        x   = np.concatenate([x_g, acc[s:e, :]], axis=1) if (self.include_acc and acc is not None) else x_g
        y   = gyro[wi.target_idx, :]
        item = {
            "x": x.astype(np.float32),
            "y": y.astype(np.float32),
            "rec_idx": wi.rec_idx,
            "start_idx": s,
            "target_idx": wi.target_idx,
            "timestamps_us": rec.timestamps_us[s:e+1].copy(),
            "file_path": rec.file_path
        }
        if _TORCH_AVAILABLE:
            item["x"] = torch.from_numpy(item["x"])
            item["y"] = torch.from_numpy(item["y"])
        return item


def build_dataset_from_recordings(recordings: List[Recording], segments: Optional[List[Tuple[int,int,int]]]=None,
                                  include_acc: bool=True, history_len: int=10, horizon: int=3) -> IMUPredictionDataset:
    """Helper wrapper to construct IMUPredictionDataset."""
    return IMUPredictionDataset(recordings, segments=segments, include_acc=include_acc,
                                history_len=history_len, horizon=horizon)


# ============================================================================
# Convenience helpers to mirror simple notebook flows
# ============================================================================

def standardize(X: np.ndarray, eps: float=1e-8):
    """Return standardized array and (mu, sigma)."""
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + eps
    Xn = (X - mu) / sigma
    return Xn, mu, sigma


def make_windows(arr: np.ndarray, window: int, horizon: int):
    """
    Create sliding windows and multi-step targets like in the user's snippet.
    Returns:
      seqs: (N, window, D)
      tars: (N, horizon, D)
    """
    arr = np.asarray(arr)
    assert window >= 1, "window must be >= 1"
    assert horizon >= 1, "horizon must be >= 1"
    seqs, tars = [], []
    end = len(arr) - window - horizon + 1
    for i in range(max(end, 0)):
        seqs.append(arr[i:i+window])
        tars.append(arr[i+window:i+window+horizon])
    if len(seqs) == 0:
        return np.empty((0, window, arr.shape[-1])), np.empty((0, horizon, arr.shape[-1]))
    return np.stack(seqs), np.stack(tars)


def from_csv_for_simple_flow(csv_path: str, features: Iterable[str] = ("x","y","z"), tag: Optional[str]=None):
    """
    Simple loader that ignores calibration and just reads columns for quick experiments.
    - Filters by `tag` if provided.
    - Returns X (float32) with selected `features` (dropping NaNs).
    """
    df = pd.read_csv(csv_path, low_memory=False)
    if tag is not None and "tag" in df.columns:
        df = df[df["tag"] == tag].reset_index(drop=True)
    X = df[list(features)].dropna().to_numpy(dtype=np.float32)
    return X


# Exports similar to original __init__.py
__all__ = [
    # IO & calibration
    "Calibration", "apply_calibration_vec",
    "Recording", "load_recording_csv", "load_recordings_from_folder", "load_two_folders",
    # Segmentation & splits
    "expected_dt_us", "split_by_expected_rate", "segment_all_recordings", "train_val_test_split",
    # Dataset
    "WindowIndex", "IMUPredictionDataset", "build_dataset_from_recordings",
    # Helpers
    "standardize", "make_windows", "from_csv_for_simple_flow",
]
