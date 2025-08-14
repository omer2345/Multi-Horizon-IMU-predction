# Multi-Horizon IMU Prediction

This project predicts future Inertial Measurement Unit (IMU) readings at several short horizons in one forward pass (multi-horizon) and compares them against a direct, short-step recursive baseline. The aim is to reduce VR/AR latency by anticipating motion a few samples ahead. The repository contains end-to-end training, evaluation, and comparison pipelines with interchangeable model backbones such as CNN, LSTM, GRU, and optionally TCN.

---
<p align="center">
  <img src="docs/VrImageGif.gif" alt="IMU multi-horizon prediction demo" width="300">
  <br/>
</p>

## Repository structure

```
.
├── MultiHorizonPrediction.ipynb     # End-to-end multi-horizon training & evaluation
├── model_comparison.ipynb           # Compare models & horizons; plots & timing
│
├── models_python/
│   ├── cnn_model.py                  # 1D CNN model definition
│   ├── gru_model.py                  # GRU model definition
│   ├── lstm_large.py                 # Large LSTM model definition
│   ├── lstm_small.py                 # Small LSTM model definition
│   ├── train_utils.py                # Training loop and utility functions
│   └── __init__.py
│
├── utils/
│   ├── data_handler.py               # Load, preprocess, segment, and normalize data
│   ├── visualization.py              # Plotting functions (predictions, segments, metrics)
│   ├── metrics.py                     # Error metrics: MSE, MAE, RMSE, etc.
│   ├── timing.py                      # Inference latency measurement helpers
│   └── __init__.py
│
├── docs/                             # Figures / documentation assets
│   └── (optional images, diagrams)
└── requirements.txt                  # Python package dependencies
```

---

## Data format

The code supports CSV files containing IMU sensor readings. Two formats are common:

**A. Split by tag** (recommended)
- Columns: `timestamp, tag, x, y, z`
- `tag` is either `acc` or `gyro`.

**B. Plain 6-axis**
- Columns: `timestamp, ax, ay, az, gx, gy, gz`

**Notes:**
- Timestamps must be monotonic per stream.
- Sampling rate defines the real-time meaning of one step ahead. Example: ~52 Hz → 1 step ≈ 19.2 ms, so horizons `[1,2,3]` ≈ `[~20, ~40, ~60] ms`.

---

## Installation

```bash
# Python 3.10+ recommended
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # or CPU version
pip install numpy pandas matplotlib scikit-learn tqdm ipywidgets tabulate
```

Or, if present:
```bash
pip install -r requirements.txt
```

---

## Quickstart

1. **Prepare your dataset**: Place your CSV files in a folder, e.g., `./data/`.

2. **Run `MultiHorizonPrediction.ipynb`**:
   - Configure:
     - `DATA_PATH`: path to your CSV or folder glob
     - `WINDOW_SIZE`: input history length in steps
     - `HORIZONS`: list of future steps to predict (e.g., `[1,2,3]`)
     - Training hyperparameters: batch size, epochs, learning rate
   - The notebook will:
     - Load and segment data
     - Normalize features
     - Train a multi-output model
     - Evaluate metrics per horizon
     - Visualize predictions

3. **Compare models**:
   - Use `model_comparison.ipynb` to:
     - Evaluate CNN, LSTM, GRU (and optionally TCN)
     - Compare per-horizon MSE/MAE
     - Measure single-sample inference latency

---

## Configuration tips

- Match `HORIZONS` to your sampling rate (steps → milliseconds).
- Ensure segments are long enough for the chosen `WINDOW_SIZE`.
- Normalize on training data only to avoid leakage.
- Save normalization parameters alongside model checkpoints.

---

## Models

Available in `models_python/`:

- **CNN** (`cnn_model.py`): Fast, low-latency, good for short horizons.
- **GRU** (`gru_model.py`): Efficient recurrent baseline.
- **LSTM Small** (`lstm_small.py`): Lightweight temporal modeling.
- **LSTM Large** (`lstm_large.py`): Higher capacity version.

> You can add new architectures (e.g., `tcn_model.py`) for further comparison.

---

## Evaluation

The notebooks provide:

- **Error metrics**: MSE, MAE, RMSE per horizon
- **Comparison plots**: Per-model, per-horizon performance
- **Best/Worst segments**: Identify scenarios where models excel or fail
- **Latency benchmarks**: p50/p90 inference times (batch=1)

---

## Extending the project

- Add uncertainty estimation with quantile (pinball) loss.
- Fuse accelerometer + gyroscope channels in a single model.
- Test on different devices to evaluate real-world VR performance.

---

## License

Add a license file (e.g., MIT) if you intend others to use or modify this code.
