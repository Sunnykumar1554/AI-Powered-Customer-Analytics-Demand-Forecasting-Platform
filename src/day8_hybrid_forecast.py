"""
RetailPulse – Day 8: Hybrid Forecasting Model
================================================
Goal: Prophet + LSTM Ensemble forecast with weighted averaging.
      Final Forecast = 0.5 × Prophet + 0.5 × LSTM
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
REPORTS   = os.path.join("..", "reports", "day8")
MLRUNS    = os.path.join("..", "mlruns")
os.makedirs(REPORTS, exist_ok=True)

# ══════════════════════════════════════════════════════════
# STEP 1 – Load Prophet Predictions
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading Prophet Predictions")
print("=" * 60)

prophet_forecast = pd.read_csv(os.path.join(PROCESSED, "day5", "prophet_forecast.csv"))
prophet_forecast["Date"] = pd.to_datetime(prophet_forecast["Date"])
print(f"  Prophet forecast shape: {prophet_forecast.shape}")
print(f"  Columns: {prophet_forecast.columns.tolist()}")

# ══════════════════════════════════════════════════════════
# STEP 2 – Load LSTM Predictions
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Loading LSTM Predictions")
print("=" * 60)

lstm_forecast = pd.read_csv(os.path.join(PROCESSED, "day6", "lstm_forecast.csv"))
lstm_forecast["Date"] = pd.to_datetime(lstm_forecast["Date"])
print(f"  LSTM forecast shape: {lstm_forecast.shape}")
print(f"  Columns: {lstm_forecast.columns.tolist()}")

# ══════════════════════════════════════════════════════════
# STEP 3 – Align Forecasts on Common Date Range
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Aligning Forecasts")
print("=" * 60)

# LSTM forecast has 30 test-period rows with Actual & Prediction
# Prophet forecast has full range; filter to LSTM's date range
lstm_dates = lstm_forecast["Date"].values

# Get Prophet predictions for the same dates
prophet_test = prophet_forecast[prophet_forecast["Date"].isin(lstm_dates)].copy()
prophet_test = prophet_test.sort_values("Date").reset_index(drop=True)

# Align LSTM
lstm_aligned = lstm_forecast[lstm_forecast["Date"].isin(prophet_test["Date"])].copy()
lstm_aligned = lstm_aligned.sort_values("Date").reset_index(drop=True)

print(f"  Aligned period: {prophet_test['Date'].min()} – {prophet_test['Date'].max()}")
print(f"  Aligned data points: {len(prophet_test)}")

# ══════════════════════════════════════════════════════════
# STEP 4 – Create Hybrid Ensemble
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Creating Hybrid Ensemble (0.5 × Prophet + 0.5 × LSTM)")
print("=" * 60)

WEIGHT_PROPHET = 0.5
WEIGHT_LSTM = 0.5

actual = lstm_aligned["Actual"].values
prophet_pred = prophet_test["Forecast"].values
lstm_pred = lstm_aligned["Prediction"].values
hybrid_pred = WEIGHT_PROPHET * prophet_pred + WEIGHT_LSTM * lstm_pred

print(f"  Prophet weight: {WEIGHT_PROPHET}")
print(f"  LSTM weight:    {WEIGHT_LSTM}")
print(f"  Sample predictions (first 5):")
for i in range(min(5, len(actual))):
    print(f"    Date: {prophet_test['Date'].iloc[i].strftime('%Y-%m-%d')} | "
          f"Actual: {actual[i]:,.0f} | "
          f"Prophet: {prophet_pred[i]:,.0f} | "
          f"LSTM: {lstm_pred[i]:,.0f} | "
          f"Hybrid: {hybrid_pred[i]:,.0f}")

# ══════════════════════════════════════════════════════════
# STEP 5 – Evaluate All Models
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Evaluation Metrics")
print("=" * 60)

def calc_metrics(actual, predicted, model_name):
    """Calculate MAPE, RMSE, MAE for a model."""
    mask = actual != 0
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))
    return {"Model": model_name, "MAPE": round(mape, 2), "RMSE": round(rmse, 2), "MAE": round(mae, 2)}

prophet_metrics = calc_metrics(actual, prophet_pred, "Prophet")
lstm_metrics = calc_metrics(actual, lstm_pred, "LSTM")
hybrid_metrics = calc_metrics(actual, hybrid_pred, "Hybrid")

# Print comparison
print(f"\n  {'Model':<10} {'MAPE (%)':>10} {'RMSE':>12} {'MAE':>12}")
print(f"  {'-'*46}")
for m in [prophet_metrics, lstm_metrics, hybrid_metrics]:
    print(f"  {m['Model']:<10} {m['MAPE']:>10.2f} {m['RMSE']:>12.2f} {m['MAE']:>12.2f}")

os.makedirs(os.path.join(PROCESSED, "day8"), exist_ok=True)
metrics_df = pd.DataFrame([prophet_metrics, lstm_metrics, hybrid_metrics])
metrics_df.to_csv(os.path.join(PROCESSED, "day8", "forecast_metrics.csv"), index=False)
print(f"\n  ✓ Saved: data/processed/day8/forecast_metrics.csv")

# ══════════════════════════════════════════════════════════
# STEP 6 – MLflow Logging
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Logging to MLflow")
print("=" * 60)

try:
    import mlflow
    mlflow.set_tracking_uri(f"file:///{os.path.abspath(MLRUNS)}")
    mlflow.set_experiment("RetailPulse_Forecasting")

    with mlflow.start_run(run_name="Hybrid_Ensemble"):
        mlflow.log_param("model_type", "Hybrid_Ensemble")
        mlflow.log_param("weight_prophet", WEIGHT_PROPHET)
        mlflow.log_param("weight_lstm", WEIGHT_LSTM)
        mlflow.log_param("forecast_horizon", len(actual))

        mlflow.log_metric("Hybrid_MAPE", hybrid_metrics["MAPE"])
        mlflow.log_metric("Hybrid_RMSE", hybrid_metrics["RMSE"])
        mlflow.log_metric("Hybrid_MAE", hybrid_metrics["MAE"])

        # Also log individual model metrics for comparison
        mlflow.log_metric("Prophet_MAPE", prophet_metrics["MAPE"])
        mlflow.log_metric("LSTM_MAPE", lstm_metrics["MAPE"])

    print("  ✓ Hybrid ensemble logged to MLflow")
except Exception as e:
    print(f"  ⚠ MLflow logging failed: {e}")

# ══════════════════════════════════════════════════════════
# STEP 7 – Visualization
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: Forecast Comparison Plot")
print("=" * 60)

dates = prophet_test["Date"].values

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(dates, actual, label="Actual", color="#2c3e50", linewidth=2.5, marker="o", markersize=4)
ax.plot(dates, prophet_pred, label=f"Prophet (MAPE: {prophet_metrics['MAPE']:.1f}%)",
        color="#3498db", linewidth=1.5, linestyle="--", alpha=0.8)
ax.plot(dates, lstm_pred, label=f"LSTM (MAPE: {lstm_metrics['MAPE']:.1f}%)",
        color="#e74c3c", linewidth=1.5, linestyle="--", alpha=0.8)
ax.plot(dates, hybrid_pred, label=f"Hybrid (MAPE: {hybrid_metrics['MAPE']:.1f}%)",
        color="#27ae60", linewidth=2.5, marker="s", markersize=4)

ax.set_title("Hybrid Ensemble Forecast vs Individual Models",
             fontsize=16, fontweight="bold")
ax.set_xlabel("Date", fontsize=13)
ax.set_ylabel("Quantity", fontsize=13)
ax.legend(fontsize=11, loc="upper right")
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "hybrid_forecast.png"), dpi=150)
plt.close()
print("  ✓ Saved: reports/day8/hybrid_forecast.png")

# Metrics comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
models = ["Prophet", "LSTM", "Hybrid"]
colors = ["#3498db", "#e74c3c", "#27ae60"]

for idx, metric_name in enumerate(["MAPE", "RMSE", "MAE"]):
    values = [prophet_metrics[metric_name], lstm_metrics[metric_name], hybrid_metrics[metric_name]]
    bars = axes[idx].bar(models, values, color=colors, edgecolor="black", linewidth=0.5)
    axes[idx].set_title(metric_name, fontsize=14, fontweight="bold")
    axes[idx].set_ylabel(metric_name + (" (%)" if metric_name == "MAPE" else ""))
    for bar, val in zip(bars, values):
        axes[idx].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01 * max(values),
                       f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "metrics_comparison.png"), dpi=150)
plt.close()
print("  ✓ Saved: reports/day8/metrics_comparison.png")

# ══════════════════════════════════════════════════════════
# STEP 8 – Save Hybrid Model & Forecast
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8: Saving Hybrid Model & Forecast")
print("=" * 60)

# Save hybrid forecast CSV
hybrid_df = pd.DataFrame({
    "Date": dates,
    "Actual": actual,
    "Prophet_Prediction": prophet_pred,
    "LSTM_Prediction": lstm_pred,
    "Hybrid_Prediction": hybrid_pred
})
os.makedirs(os.path.join(PROCESSED, "day8"), exist_ok=True)
hybrid_df.to_csv(os.path.join(PROCESSED, "day8", "hybrid_forecast.csv"), index=False)
print("  ✓ Saved: data/processed/day8/hybrid_forecast.csv")

# Save ensemble model as pickle (weights + metadata)
ensemble_model = {
    "model_type": "Hybrid_Ensemble",
    "weight_prophet": WEIGHT_PROPHET,
    "weight_lstm": WEIGHT_LSTM,
    "prophet_model_path": "prophet_model.pkl",
    "lstm_model_path": "lstm_model.pth",
    "metrics": hybrid_metrics,
    "forecast_horizon": len(actual)
}
with open(os.path.join(MODELS, "hybrid_model.pkl"), "wb") as f:
    pickle.dump(ensemble_model, f)
print("  ✓ Saved: models/hybrid_model.pkl")

# ══════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DAY 8 COMPLETE ✓")
print("=" * 60)
print(f"""
Deliverables:
  ✓ data/processed/hybrid_forecast.csv
  ✓ data/processed/forecast_metrics.csv
  ✓ models/hybrid_model.pkl
  ✓ reports/day8/hybrid_forecast.png
  ✓ reports/day8/metrics_comparison.png

Ensemble: {WEIGHT_PROPHET} × Prophet + {WEIGHT_LSTM} × LSTM

Metrics:
  Prophet: MAPE={prophet_metrics['MAPE']}%, RMSE={prophet_metrics['RMSE']}
  LSTM:    MAPE={lstm_metrics['MAPE']}%, RMSE={lstm_metrics['RMSE']}
  Hybrid:  MAPE={hybrid_metrics['MAPE']}%, RMSE={hybrid_metrics['RMSE']}
""")
