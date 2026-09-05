import os
import pickle
import pandas as pd


MODEL_DIR = "saved_models"

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "best_aqi_model.pkl")


# ------------------------------------------------------------
# LOAD TRAINED SCALER AND BEST MODEL
# ------------------------------------------------------------

print("=" * 60)
print("             AQI MODEL DEPLOYMENT")
print("=" * 60)

print("[*] Loading scaler...")
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

print("[+] Scaler loaded successfully.")

print("[*] Loading best AQI model...")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("[+] Best AQI model loaded successfully.")


# ------------------------------------------------------------
# NEW SENSOR DATA
# ------------------------------------------------------------

CO2 = 800
PM25 = 100
Temperature = 30
Humidity = 65


print("\n[*] Incoming sensor data:")
print(f"    CO2          : {CO2}")
print(f"    PM2.5        : {PM25}")
print(f"    Temperature  : {Temperature}")
print(f"    Humidity     : {Humidity}")


# ------------------------------------------------------------
# SCALE SENSOR DATA
# ------------------------------------------------------------

sensor_data = pd.DataFrame(
    [[CO2, PM25, Temperature, Humidity]],
    columns=["CO2", "PM2.5", "Temperature", "Humidity"]
)

sensor_data_scaled = scaler.transform(sensor_data)


# ------------------------------------------------------------
# PREDICT AQI
# ------------------------------------------------------------

predicted_aqi = model.predict(sensor_data_scaled)[0]


print("\n" + "=" * 60)
print(f"             PREDICTED AQI: {predicted_aqi:.2f}")
print("=" * 60)