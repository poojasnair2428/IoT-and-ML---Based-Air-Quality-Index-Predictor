import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score

print("=" * 60)
print("STAGE 1: GENERATING SIMULATING SENSOR DATASET")
print("=" * 60)

# 1. Generating a Synthetic Dataset matching project specs
np.random.seed(42)
data_size = 500

simulated_data = {
    'CO2': np.random.uniform(400, 1500, size=data_size),        # MQ-135 Gas Sensor ranges
    'PM2.5': np.random.uniform(10, 250, size=data_size),       # PM2.5 Dust Sensor ranges
    'Temperature': np.random.uniform(22, 42, size=data_size),  # DHT11 Temperature ranges
    'Humidity': np.random.uniform(40, 95, size=data_size)      # DHT11 Humidity ranges
}

df = pd.DataFrame(simulated_data)

# Empirical standard formulation to approximate real-world environmental AQI response curves
df['AQI'] = (df['PM2.5'] * 1.6) + (df['CO2'] * 0.04) + (df['Temperature'] * 0.25) + np.random.normal(0, 4, size=data_size)

# Save dataset to CSV as specified by the project document
df.to_csv('sensor_data.csv', index=False)
print("SUCCESS: Log dataset written to 'sensor_data.csv'.\n")

print("=" * 60)
print("STAGE 2: TRAINING ML INTELLIGENCE PIPELINE")
print("=" * 60)

# Separate independent arrays and target variables
X = df[['CO2', 'PM2.5', 'Temperature', 'Humidity']]
y = df['AQI']

# Feature Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Linear Regression & SVR
lr = LinearRegression().fit(X_scaled, y)
svr = SVR(kernel='rbf', C=10, epsilon=0.1).fit(X_scaled, y)

print("SUCCESS: Core ML models successfully trained.\n")

# Calculate metrics
y_pred_lr = lr.predict(X_scaled)
mae_lr = mean_absolute_error(y, y_pred_lr)
r2_lr = r2_score(y, y_pred_lr)

y_pred_svr = svr.predict(X_scaled)
mae_svr = mean_absolute_error(y, y_pred_svr)
r2_svr = r2_score(y, y_pred_svr)

print(f"-> Linear Regression Metrics: MAE = {mae_lr:.4f} | R² = {r2_lr:.4f}")
print(f"-> Support Vector Regressor Metrics:  MAE = {mae_svr:.4f} | R² = {r2_svr:.4f}\n")

print("=" * 60)
print("STAGE 3: PLOTTING EVALUATION GRAPH ARTIFACTS")
print("=" * 60)

# Generate performance comparison graphs for submission
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y, y_pred_lr, alpha=0.5, color='royalblue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Target AQI Values')
plt.ylabel('Predicted AQI Values')
plt.title(f'Linear Regression Performance\n(MAE: {mae_lr:.2f}, R²: {r2_lr:.2f})')

plt.subplot(1, 2, 2)
plt.scatter(y, y_pred_svr, alpha=0.5, color='seagreen')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Target AQI Values')
plt.ylabel('Predicted AQI Values')
plt.title(f'Support Vector Regression (SVR) Performance\n(MAE: {mae_svr:.2f}, R²: {r2_svr:.2f})')

plt.tight_layout()
plt.close()