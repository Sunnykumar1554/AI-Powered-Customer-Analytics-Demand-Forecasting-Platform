"""
RetailPulse – Day 5: Prophet Forecasting
==========================================
Goal: 30-day demand forecast using Facebook Prophet, evaluate with MAPE & RMSE.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────
PROCESSED = os.path.join("..", "data", "processed")
MODELS    = os.path.join("..", "models")
REPORTS   = os.path.join("..", "reports", "day5")
os.makedirs(MODELS, exist_ok=True)

# ══════════════════════════════════════════════════════════
# STEP 1 – Load Daily Sales
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading Daily Sales Data")
print("=" * 60)
daily_sales = pd.read_csv(os.path.join(PROCESSED, "day4", "daily_sales.csv"), parse_dates=["Date"])
print(f"  Loaded {len(daily_sales)} days of data")

# ══════════════════════════════════════════════════════════
# STEP 2 – Prepare Prophet Format
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Preparing Prophet Format")
print("=" * 60)
prophet_df = daily_sales[["Date", "Quantity"]].copy()
prophet_df.columns = ["ds", "y"]
prophet_df = prophet_df.sort_values("ds").reset_index(drop=True)

# Fill missing dates
full_range = pd.date_range(start=prophet_df["ds"].min(), end=prophet_df["ds"].max(), freq="D")
prophet_df = prophet_df.set_index("ds").reindex(full_range, fill_value=0).reset_index()
prophet_df.columns = ["ds", "y"]

print(f"  Date range: {prophet_df['ds'].min()} – {prophet_df['ds'].max()}")
print(f"  Total data points: {len(prophet_df)}")

# ══════════════════════════════════════════════════════════
# STEP 3 – Train/Test Split
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Train/Test Split")
print("=" * 60)

# Use last 30 days as test set
train = prophet_df.iloc[:-30]
test = prophet_df.iloc[-30:]
print(f"  Training: {len(train)} days ({train['ds'].min()} – {train['ds'].max()})")
print(f"  Testing:  {len(test)} days ({test['ds'].min()} – {test['ds'].max()})")

# ══════════════════════════════════════════════════════════
# STEP 4 – Train Prophet Model
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Training Prophet Model")
print("=" * 60)

try:
    from prophet import Prophet
except ImportError:
    print("  ⚠ Prophet not installed. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "prophet"])
    from prophet import Prophet

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode="multiplicative"
)
model.fit(train)
print("  ✓ Prophet model trained successfully")

# ══════════════════════════════════════════════════════════
# STEP 5 – Make Forecast
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Making 30-Day Forecast")
print("=" * 60)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Get predictions for test period
test_forecast = forecast[forecast["ds"].isin(test["ds"])][["ds", "yhat", "yhat_lower", "yhat_upper"]]
test_forecast = test_forecast.merge(test, on="ds", how="left")

print(f"  Forecast generated: {len(forecast)} days total")
print(f"  Test period predictions: {len(test_forecast)} days")

# ══════════════════════════════════════════════════════════
# STEP 6 – Evaluate Metrics
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Evaluation Metrics")
print("=" * 60)

# MAPE
actual = test_forecast["y"].values
predicted = test_forecast["yhat"].values

# Avoid division by zero
mask = actual != 0
mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

# RMSE
rmse = np.sqrt(np.mean((actual - predicted) ** 2))

# MAE
mae = np.mean(np.abs(actual - predicted))

print(f"  MAPE: {mape:.2f}%")
print(f"  RMSE: {rmse:.2f}")
print(f"  MAE:  {mae:.2f}")

# Save metrics
metrics = pd.DataFrame({
    "Model": ["Prophet"],
    "MAPE": [round(mape, 2)],
    "RMSE": [round(rmse, 2)],
    "MAE": [round(mae, 2)]
})
os.makedirs(os.path.join(PROCESSED, "day5"), exist_ok=True)
metrics.to_csv(os.path.join(PROCESSED, "day5", "prophet_metrics.csv"), index=False)

# ══════════════════════════════════════════════════════════
# STEP 7 – Visualization
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: Forecast Visualization")
print("=" * 60)

fig, ax = plt.subplots(figsize=(14, 6))

# Actual data
ax.plot(prophet_df["ds"], prophet_df["y"], color="#3498db", alpha=0.4, linewidth=0.8, label="Actual")

# Forecast
forecast_future = forecast[forecast["ds"] > train["ds"].max()]
ax.plot(forecast_future["ds"], forecast_future["yhat"], color="#e74c3c", linewidth=2, label="Prophet Forecast")

# Confidence interval
ax.fill_between(
    forecast_future["ds"],
    forecast_future["yhat_lower"],
    forecast_future["yhat_upper"],
    alpha=0.2,
    color="#e74c3c",
    label="Confidence Interval"
)

ax.set_title(f"Prophet 30-Day Demand Forecast (MAPE: {mape:.1f}%)", fontsize=16, fontweight="bold")
ax.set_xlabel("Date", fontsize=13)
ax.set_ylabel("Quantity", fontsize=13)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "prophet_forecast.png"), dpi=150)
plt.close()
print("  ✓ Saved: reports/day5/prophet_forecast.png")

# Prophet component plots
fig = model.plot_components(forecast)
fig.savefig(os.path.join(REPORTS, "prophet_components.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved: reports/day5/prophet_components.png")

# ══════════════════════════════════════════════════════════
# STEP 8 – Save Model & Forecast
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8: Saving Model & Forecast")
print("=" * 60)

# Save Prophet model
with open(os.path.join(MODELS, "prophet_model.pkl"), "wb") as f:
    pickle.dump(model, f)
print("  ✓ Saved: models/prophet_model.pkl")

# Save forecast
forecast_save = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
forecast_save.columns = ["Date", "Forecast", "Lower_CI", "Upper_CI"]
os.makedirs(os.path.join(PROCESSED, "day5"), exist_ok=True)
forecast_save.to_csv(os.path.join(PROCESSED, "day5", "prophet_forecast.csv"), index=False)
print("  ✓ Saved: data/processed/day5/prophet_forecast.csv")

# ══════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DAY 5 COMPLETE ✓")
print("=" * 60)
print(f"""
Deliverables:
  ✓ models/prophet_model.pkl
  ✓ data/processed/prophet_forecast.csv
  ✓ data/processed/prophet_metrics.csv
  ✓ reports/day5/prophet_forecast.png
  ✓ reports/day5/prophet_components.png

Metrics:
  MAPE: {mape:.2f}%
  RMSE: {rmse:.2f}
  MAE:  {mae:.2f}
""")
