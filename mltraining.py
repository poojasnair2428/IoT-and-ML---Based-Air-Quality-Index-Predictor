import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_PATH = 'sensor_data.csv'
MODEL_DIR = 'saved_models'

# Ensure the directory to save trained models exists
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("             ML MODEL TRAINING RUNTIME")
print("=" * 60)

# ============================================================
# STEP 1: LOAD AND SPLIT THE DATASET
# ============================================================
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Error: '{DATASET_PATH}' not found. Please ensure it is in the same folder.")

print(f"[*] Loading dataset from: {DATASET_PATH}...")
df = pd.read_csv(DATASET_PATH)

# Separate input features and target variable [cite: 24, 32]
X = df[['CO2', 'PM2.5', 'Temperature', 'Humidity']]
y = df['AQI']

# Split data: 80% for training patterns, 20% for unseen evaluation testing [cite: 33]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"[+] Data successfully split. Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

# ============================================================
# STEP 2: FEATURE SCALING (PREPROCESSING)
# ============================================================
print("[*] Normalizing features using StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# STEP 3: TRAINING THE REGRESSION MODELS
# ============================================================
print("[*] Training Linear Regression Model...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

print("[*] Training Support Vector Regression (SVR) Model...")
svr_model = SVR(kernel='rbf', C=10, epsilon=0.1)
svr_model.fit(X_train_scaled, y_train)
print("[+] Model training completed successfully.")

# ============================================================
# STEP 4: PERFORMANCE METRIC EVALUATION
# ============================================================
print("\n" + "=" * 40)
print("         EVALUATION METRICS")
print("=" * 40)

# Evaluate Linear Regression
y_pred_lr = lr_model.predict(X_test_scaled)
print(f"--- Linear Regression Performance ---")
print(f"    Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred_lr):.4f}")
print(f"    R² Score (Accuracy)      : {r2_score(y_test, y_pred_lr):.4f}")

# Evaluate SVR
y_pred_svr = svr_model.predict(X_test_scaled)
print(f"\n--- Support Vector Regression Performance ---")
print(f"    Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred_svr):.4f}")
print(f"    R² Score (Accuracy)      : {r2_score(y_test, y_pred_svr):.4f}")

# ============================================================
# STEP 5: SAVE ARTIFACTS FOR FUTURE HARDWARE DEPLOYMENT
# ============================================================
print("\n" + "=" * 40)
print("         EXPORTING ARTIFACTS")
print("=" * 40)

# Save the scaler, LR model, and SVR model
artifacts = {
    'scaler.pkl': scaler,
    'linear_regression_model.pkl': lr_model,
    'svr_model.pkl': svr_model
}

for filename, object_to_save in artifacts.items():
    filepath = os.path.join(MODEL_DIR, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(object_to_save, f)
    print(f"[+] Saved: {filepath}")

print("\n[✔] Setup successful! Models are fully prepared for IoT system deployment.")