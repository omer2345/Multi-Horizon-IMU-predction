# 📈 Multi-Horizon IMU Prediction

<p align="center">
  <img src="docs/vr_demo.png" alt="VR headset with predicted IMU trajectory (orange) tracking ground truth (white)" width="820">
  <br/>
  <sub>Prediction (orange) tracking ground truth (white). Multi-horizon IMU forecasting for low-latency VR.</sub>
</p>

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2e.svg?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📜 Overview

**Multi-Horizon IMU Prediction** is a deep learning project for forecasting VR headset motion several milliseconds into the future using raw **Inertial Measurement Unit (IMU)** data.  
By predicting multiple time horizons (e.g., 10 ms, 20 ms, 30 ms) directly from sensor data, the model helps reduce **motion-to-photon latency** in VR/AR systems, enabling smoother and more responsive experiences.

---

## 🚀 Features

- **Multi-horizon prediction**: Directly forecasts multiple future points in a single forward pass.
- **Direct vs. Recursive models**: Compare direct multi-step forecasting to single-step models run recursively to observe prediction drift.
- **Flexible horizons**: Easily configure time offsets (in ms) for prediction targets.
- **IMU-ready preprocessing**: Handles accelerometer and gyroscope time-series inputs.
- **PyTorch implementation**: Simple to train, test, and extend.

---

## 📂 Repository Structure

├── data/ # Raw or preprocessed IMU datasets
├── notebooks/ # Jupyter notebooks for experiments
│ ├── train_multi.ipynb # Multi-horizon model training
│ ├── train_single.ipynb # Single-horizon baseline model training
│ └── evaluate.ipynb # Comparison & drift analysis
├── models/ # Saved model checkpoints
├── docs/ # Documentation & images
│ └── vr_demo.png # Demo image used in README
├── requirements.txt # Python dependencies
└── README.md # This file


## 📊 Example Output

### Direct Multi-Horizon Prediction
Predicts 10 ms, 20 ms, 30 ms ahead in a single forward pass.

Epoch 01 | Train Loss: 0.8951 | Val Loss: 55.0931 (norm MSE) | Val MAE: 0.17 | RMSE: 0.37 deg/s
Epoch 02 | Train Loss: 0.6053 | Val Loss: 51.8341 (norm MSE) | Val MAE: 0.16 | RMSE: 0.36 deg/s

### Single-Step Recursive Prediction
Predicts 1 ms ahead, repeatedly fed into itself to simulate longer horizons.  
Useful for demonstrating **prediction drift** compared to direct prediction.

---

## 🖼 Example Visualization

<p align="center">
  <img src="docs/vrDemoPicture.png" alt="IMU multi-horizon prediction demo" width="700">
  <br/>
  <sub>Orange: Model prediction trajectory. White: Ground truth trajectory.</sub>
</p>

---

## 🛠 Installation

```bash
git clone https://github.com/<your-username>/multi-horizon-imu-prediction.git
cd multi-horizon-imu-prediction
pip install -r requirements.txt

## 🛠 Installation

### 1. Prepare Data
Place your IMU CSV files in the `data/` directory.  
Files should contain **timestamp, accelerometer, and gyroscope** readings.

### 2. Train Multi-Horizon Model
```bash
jupyter notebook notebooks/train_multi.ipynb

### 3. Train Single-Horizon Model
```bash
jupyter notebook notebooks/train_single.ipynb

### 4. Evaluate & Compare Drift
jupyter notebook notebooks/evaluate.ipynb
