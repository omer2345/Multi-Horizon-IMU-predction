Multi-Horizon IMU Prediction
Predicting future Inertial Measurement Unit (IMU) sensor readings across multiple time horizons to reduce motion-to-photon latency in VR/AR.

Background
The goal of this project is to explore the impact of prediction horizon length on accuracy and drift when forecasting IMU readings (gyroscope & accelerometer data).
We train separate models for short, medium, and long horizons — specifically 10 ms, 20 ms, and 30 ms — and compare them to a single-step 1 ms prediction model that is applied recursively to simulate drift over longer horizons.

Our approach involves:

Preprocessing raw IMU logs into contiguous segments.

Resampling to a target frequency (100 Hz or 1 kHz).

Creating sliding windows of past IMU readings to predict future values.

Training LSTM-based regression models for each horizon.

Visualizing and evaluating drift over time.

This setup allows us to demonstrate the trade-off between direct multi-step forecasting and recursive short-step prediction.

Prerequisites
Library	Version
Python	3.10+
torch	2.3+
numpy	1.26+
pandas	2.2+
matplotlib	3.7+
scikit-learn	1.4+

Files in the Repository
File / Folder	Purpose
data/	Raw IMU CSV files
notebooks/imu_train.ipynb	Train multi-horizon models (10 ms, 20 ms, 30 ms)
notebooks/imu_baseline_1ms.ipynb	Train single-step 1 ms prediction baseline
scripts/preprocessing.py	Segmenting, merging, and resampling IMU data
scripts/training.py	Model training and evaluation loops
scripts/visualization.py	3D trajectory plotting and drift visualization
models/	Saved PyTorch model checkpoints
docs/	Images and plots for README

Dataset
We use IMU logs containing:

Accelerometer readings: (acc_x, acc_y, acc_z) in m/s²

Gyroscope readings: (gyro_x, gyro_y, gyro_z) in °/s

Timestamp (nanoseconds)

Data is merged, segmented into continuous sequences, and resampled to the target rate before training.

Model Architecture
Base model: 2-layer LSTM

Hidden size: 64 units

Dropout: 0.1

Output layer: Fully-connected → 3D vector (gyro_x, gyro_y, gyro_z)

For each horizon h (in timesteps), the model predicts y[t+h] from a window of past IMU readings.

Results
Validation Performance (100 Hz, Gyroscope °/s)
Horizon	MAE	RMSE
10 ms	0.16	0.36
20 ms	0.17	0.36
30 ms	0.17	0.37

Example: Direct 20 ms Prediction
<img src="docs/example_20ms.png" width="500">
Drift Effect: Recursive 1 ms Model
<img src="docs/drift_effect.png" width="500">
The recursive baseline accumulates small errors at each step, leading to increasing deviation from the ground truth over time.

Training Time vs Horizon
Direct longer-horizon models require slightly more training time per epoch but avoid drift accumulation seen in recursive prediction.

How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/username/multi-horizon-imu.git
cd multi-horizon-imu
pip install -r requirements.txt
Train all horizons:

bash
Copy
Edit
python train.py --data data/session.csv
Evaluate a saved model:

bash
Copy
Edit
python eval.py --model models/model_20ms.pt
Visualize predictions:

bash
Copy
Edit
python visualize.py --model models/model_30ms.pt
Notes
Running all horizons (10 ms, 20 ms, 30 ms, plus 1 ms baseline) will take longer.

The project is tested with GPU acceleration (CUDA), but also works on CPU with slower training.

Sources
LSTM for Time Series Forecasting

Inertial Measurement Units (IMU) Basics
