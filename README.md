# 📈 Multi-Horizon IMU Prediction

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A **PyTorch-based framework** for predicting future Inertial Measurement Unit (IMU) readings at **multiple time horizons**.  
Supports direct multi-step forecasting (e.g., **10ms, 20ms, 30ms**) and comparison with short-horizon recursive predictions to study **drift accumulation**.

---

## 📌 Project Overview

This project investigates **multi-horizon forecasting** for IMU sensor data, targeting applications like **VR/AR motion prediction** where reducing latency is critical.

We train multiple LSTM models:
- **Direct models** predicting 10ms, 20ms, and 30ms into the future.
- **Single-step model** predicting 1ms ahead for recursive forecasting (drift comparison).

Key features:
- 🚀 High-frequency IMU processing (1 kHz support)
- 📊 Multiple time horizons in one pipeline
- 📈 Drift visualization over long recursive predictions
- 🔍 Clean and reproducible PyTorch training loop

---

## 📂 Repository Structure

.
├── data/ # Example IMU datasets (not included in repo)
├── notebooks/
│ ├── train_multi.ipynb # Main notebook for multi-horizon models
│ ├── train_single.ipynb # Notebook for single-step drift study
├── models/ # Saved PyTorch models (.pt)
├── plots/ # Generated visualizations
├── README.md # This file
└── requirements.txt

yaml
Copy
Edit

---

## 📊 Example Results

### **Validation RMSE Across Horizons**
| Horizon | Model Type | RMSE (deg/s) |
|---------|-----------|--------------|
| 10ms    | Direct    | 0.36 |
| 20ms    | Direct    | 0.37 |
| 30ms    | Direct    | 0.39 |
| 10ms    | Recursive (1ms step) | 0.44 |
| 20ms    | Recursive (1ms step) | 0.57 |
| 30ms    | Recursive (1ms step) | 0.79 |

---

### **Drift Effect Visualization**
Direct vs. Recursive Predictions  
![Drift Comparison](plots/drift_example.png)

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/multi-horizon-imu-prediction.git
cd multi-horizon-imu-prediction

# Install dependencies
pip install -r requirements.txt
🏃‍♂️ Usage
Train Multi-Horizon Models
bash
Copy
Edit
jupyter notebook notebooks/train_multi.ipynb
Train 1ms Single-Step Model (Drift Study)
bash
Copy
Edit
jupyter notebook notebooks/train_single.ipynb
📈 Training Configuration
Parameter	Value
Target rate	100 Hz / 1 kHz
Window size	200 samples (adjustable)
Horizons	[1, 10, 20, 30] samples ahead
Batch size	256
Epochs	5
Optimizer	Adam
Learning rate	1e-3

🔬 Methodology
Data Preprocessing

Filter acc and gyro readings.

Resample to target frequency (100 Hz / 1 kHz).

Create sliding windows for each horizon.

Model Architecture

LSTM-based sequence model.

Linear layer to output prediction for each horizon.

Evaluation

RMSE, MAE, and visual drift analysis.

Comparison between direct and recursive predictions.

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
