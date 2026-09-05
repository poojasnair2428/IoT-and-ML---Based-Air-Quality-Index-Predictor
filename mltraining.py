import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "sensor_data.csv"
MODEL_DIR = "saved_models"

os.makedirs(MODEL_DIR, exist_ok=True)


print("=" * 60)
print("             ML MODEL TRAINING RUNTIME")
print("=" * 60)


# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Error: '{DATASET_PATH}' not found. "
        "Please ensure it is in the same folder."
    )

print(f"[*] Loading dataset from: {DATASET_PATH}...")

df = pd.read_csv(DATASET_PATH)


# ============================================================
# STEP 2: DEFINE FEATURES AND TARGET
# ============================================================

X = df[["CO2", "PM2.5", "Temperature", "Humidity"]]
y = df["AQI"]


# ============================================================
# STEP 3: TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(
    f"[+] Data successfully split. "
    f"Training samples: {len(X_train)} | "
    f"Testing samples: {len(X_test)}"
)


# ============================================================
# STEP 4: FEATURE SCALING
# ============================================================

print("[*] Normalizing features using StandardScaler...")

scaler = StandardScaler()

# Fit ONLY on training data
X_train_scaled = scaler.fit_transform(X_train)

# Use the same scaler on testing data
X_test_scaled = scaler.transform(X_test)


# ============================================================
# STEP 5: TRAIN LINEAR REGRESSION
# ============================================================

print("[*] Training Linear Regression Model...")

lr_model = LinearRegression()

lr_model.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# STEP 6: TRAIN SVR WITH RBF KERNEL
# ============================================================

print("[*] Training Support Vector Regression (SVR) Model...")

svr_model = SVR(
    kernel="rbf",
    C=10,
    epsilon=0.1
)

svr_model.fit(
    X_train_scaled,
    y_train
)

print("[+] Model training completed successfully.")


# ============================================================
# STEP 7: PREDICTION
# ============================================================

y_pred_lr = lr_model.predict(X_test_scaled)

y_pred_svr = svr_model.predict(X_test_scaled)


# ============================================================
# STEP 8: EVALUATION
# ============================================================

mae_lr = mean_absolute_error(
    y_test,
    y_pred_lr
)

r2_lr = r2_score(
    y_test,
    y_pred_lr
)


mae_svr = mean_absolute_error(
    y_test,
    y_pred_svr
)

r2_svr = r2_score(
    y_test,
    y_pred_svr
)


print("\n" + "=" * 40)
print("         EVALUATION METRICS")
print("=" * 40)


print("--- Linear Regression Performance ---")

print(
    f"    Mean Absolute Error (MAE): "
    f"{mae_lr:.4f}"
)

print(
    f"    R² Score                  : "
    f"{r2_lr:.4f}"
)


print("\n--- Support Vector Regression (SVR) Performance ---")

print(
    f"    Mean Absolute Error (MAE): "
    f"{mae_svr:.4f}"
)

print(
    f"    R² Score                  : "
    f"{r2_svr:.4f}"
)


# ============================================================
# STEP 9: SELECT BEST MODEL
# ============================================================

print("\n" + "=" * 40)
print("         BEST MODEL SELECTION")
print("=" * 40)


# Lower MAE is better
if mae_lr <= mae_svr:

    best_model = lr_model
    best_model_name = "Linear Regression"
    best_mae = mae_lr
    best_r2 = r2_lr

else:

    best_model = svr_model
    best_model_name = "Support Vector Regression (SVR)"
    best_mae = mae_svr
    best_r2 = r2_svr


print(f"    Best Model : {best_model_name}")
print(f"    Best MAE   : {best_mae:.4f}")
print(f"    Best R²    : {best_r2:.4f}")


# ============================================================
# STEP 10: SAVE TRAINED MODELS
# ============================================================

print("\n" + "=" * 40)
print("         EXPORTING ARTIFACTS")
print("=" * 40)


artifacts = {
    "scaler.pkl": scaler,
    "linear_regression_model.pkl": lr_model,
    "svr_model.pkl": svr_model,
    "best_aqi_model.pkl": best_model
}


for filename, object_to_save in artifacts.items():

    filepath = os.path.join(
        MODEL_DIR,
        filename
    )

    with open(filepath, "wb") as f:

        pickle.dump(
            object_to_save,
            f
        )

    print(f"[+] Saved: {filepath}")


# ============================================================
# STEP 11: ACTUAL VS PREDICTED GRAPH
# ============================================================

print("\n[*] Generating result graphs...")


# Linear Regression graph

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    y_pred_lr
)

plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")

plt.title(
    "Linear Regression - Actual vs Predicted AQI"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join("results", "linear_regression_result.png"),
    dpi=300
)
plt.close()


# SVR graph

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    y_pred_svr
)

plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")

plt.title(
    "SVR (RBF) - Actual vs Predicted AQI"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join("results", "svr_result.png"),
    dpi=300
)

plt.close()
MODEL_DIR = "saved_models"

print("[+] Result graphs saved successfully.")


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 60)

print(
    f"[✔] Software ML pipeline completed successfully."
)

print(
    f"[✔] Selected model: {best_model_name}"
)

print(
    "[✔] Models and scaler are ready for deployment."
)

print("=" * 60)