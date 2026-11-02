"""
Smart Electricity Consumption Prediction System
================================================
train.py  –  Data preprocessing + LSTM model training
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import json
import matplotlib
matplotlib.use("Agg")          # headless – no GUI window needed
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
print("=" * 55)
print("  Smart Electricity Prediction – Model Training")
print("=" * 55)
print("\n[1/7] Loading dataset …")

df = pd.read_csv(
    "dataset/household_power_consumption.txt",
    sep=";",
    na_values=["?"],
    low_memory=False,
)

# ─────────────────────────────────────────────
# 2. CLEAN & CONVERT
# ─────────────────────────────────────────────
print("[2/7] Cleaning data …")

df["Global_active_power"] = pd.to_numeric(
    df["Global_active_power"], errors="coerce"
)
df = df.dropna(subset=["Global_active_power"])

# Parse datetime
df["Datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    dayfirst=True,
    errors="coerce",
)
df = df.dropna(subset=["Datetime"])
df = df.set_index("Datetime").sort_index()

# ─────────────────────────────────────────────
# 3. AGGREGATE TO DAILY kWh
# ─────────────────────────────────────────────
print("[3/7] Aggregating to daily consumption (kWh) …")

# Each reading is 1 minute → kWh = kW * (1/60)
daily_sum   = df["Global_active_power"].resample("D").sum()
daily_count = df["Global_active_power"].resample("D").count()

# Keep only days with ≥ 90 % data (1440 min/day × 90 % = 1296)
daily = (daily_sum / 60).where(daily_count >= 1296).dropna()
daily.name = "Consumption_kWh"

# Save for the Streamlit app
daily.to_csv("daily_usage.csv")
print(f"    Total usable days : {len(daily)}")
print(f"    Date range        : {daily.index[0].date()} → {daily.index[-1].date()}")
print(f"    Min / Mean / Max  : "
      f"{daily.min():.2f} / {daily.mean():.2f} / {daily.max():.2f} kWh")

# ─────────────────────────────────────────────
# 4. TRAIN / VAL / TEST SPLIT  (70 / 15 / 15)
# ─────────────────────────────────────────────
print("[4/7] Splitting dataset …")

values = daily.values.reshape(-1, 1)
n = len(values)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

# Fit scaler ONLY on training portion
scaler = MinMaxScaler()
scaler.fit(values[:train_end])
scaled = scaler.transform(values)

# ─────────────────────────────────────────────
# 5. CREATE SEQUENCES  (7-day look-back)
# ─────────────────────────────────────────────
SEQ_LEN = 7
X, y, target_indices = [], [], []
for i in range(SEQ_LEN, len(scaled)):
    X.append(scaled[i - SEQ_LEN : i])
    y.append(scaled[i])
    target_indices.append(i)

X              = np.array(X)
y              = np.array(y)
target_indices = np.array(target_indices)

train_mask = target_indices < train_end
val_mask   = (target_indices >= train_end) & (target_indices < val_end)
test_mask  = target_indices >= val_end

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"    Train : {len(X_train)} samples")
print(f"    Val   : {len(X_val)} samples")
print(f"    Test  : {len(X_test)} samples")

# ─────────────────────────────────────────────
# 6. BUILD LSTM MODEL
# ─────────────────────────────────────────────
print("[5/7] Building & training LSTM model …\n")

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(SEQ_LEN, 1)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ],
    name="LSTM_ElectricityPredictor",
)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",
    metrics=["mae"],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=7, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
    ),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    shuffle=False,
    callbacks=callbacks,
    verbose=1,
)

# ─────────────────────────────────────────────
# 7. EVALUATE
# ─────────────────────────────────────────────
print("\n[6/7] Evaluating model …")

preds_scaled = model.predict(X_test, verbose=0)
actual_kwh   = scaler.inverse_transform(y_test)
pred_kwh     = scaler.inverse_transform(preds_scaled)

mae  = mean_absolute_error(actual_kwh, pred_kwh)
rmse = np.sqrt(mean_squared_error(actual_kwh, pred_kwh))
mape = np.mean(np.abs((actual_kwh - pred_kwh) / (actual_kwh + 1e-8))) * 100

# Baseline: naïve persistence (predict today = yesterday)
naive_preds = actual_kwh[:-1]
naive_mae   = mean_absolute_error(actual_kwh[1:], naive_preds)

print(f"\n  LSTM  MAE  : {mae:.3f} kWh  (naïve baseline: {naive_mae:.3f} kWh)")
print(f"  LSTM  RMSE : {rmse:.3f} kWh")
print(f"  LSTM  MAPE : {mape:.1f} %")

# ─────────────────────────────────────────────
# 8. SAVE ARTEFACTS
# ─────────────────────────────────────────────
print("\n[7/7] Saving model, scaler, charts …")

model.save("electricity_model.keras")
joblib.dump(scaler, "scaler.pkl")

# ---- training-loss chart ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history["loss"],     label="Train Loss")
axes[0].plot(history.history["val_loss"], label="Val Loss")
axes[0].set_title("Training vs Validation Loss (MSE)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE")
axes[0].legend()
axes[0].grid(True)

# ---- actual vs predicted chart (test set) ----
test_dates = daily.index[test_mask + SEQ_LEN]
# Align lengths
n_plot = min(len(test_dates), len(actual_kwh))
axes[1].plot(test_dates[:n_plot], actual_kwh[:n_plot],   label="Actual",    linewidth=1.5)
axes[1].plot(test_dates[:n_plot], pred_kwh[:n_plot],     label="Predicted", linewidth=1.5, linestyle="--")
axes[1].set_title("Actual vs Predicted (Test Set)")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Consumption (kWh)")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("evaluation_chart.png", dpi=150)
plt.close()

# ---- save metrics ----
metrics = {
    "model": "LSTM (2-layer stacked)",
    "sequence_length_days": SEQ_LEN,
    "epochs_trained": len(history.history["loss"]),
    "test_mae_kwh":  round(float(mae),  4),
    "test_rmse_kwh": round(float(rmse), 4),
    "test_mape_pct": round(float(mape), 2),
    "naive_mae_kwh": round(float(naive_mae), 4),
    "improvement_over_naive_pct": round(
        (naive_mae - mae) / naive_mae * 100, 1
    ),
}
with open("training_history.json", "w") as fh:
    json.dump(metrics, fh, indent=4)

print("\n✅  Training complete!")
print(f"   electricity_model.keras   → saved")
print(f"   scaler.pkl                → saved")
print(f"   daily_usage.csv           → saved")
print(f"   training_history.json     → saved")
print(f"   evaluation_chart.png      → saved")
print(f"\n   Run:  streamlit run app.py")
