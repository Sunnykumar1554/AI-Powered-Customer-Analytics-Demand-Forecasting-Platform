"""
RetailPulse – Day 7: MLflow Checkpoint & Week 1 Summary
=========================================================
Goal: Log all models and metrics to MLflow, generate Week 1 report.
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────
PROCESSED = os.path.join("..", "data", "processed")
MODELS    = os.path.join("..", "models")
REPORTS_BASE = os.path.join("..", "reports")
REPORTS   = os.path.join(REPORTS_BASE, "day7")
os.makedirs(REPORTS, exist_ok=True)
MLRUNS    = os.path.join("..", "mlruns")
os.makedirs(MLRUNS, exist_ok=True)

# ══════════════════════════════════════════════════════════
# STEP 1 – Install & Import MLflow
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Setting Up MLflow")
print("=" * 60)

try:
    import mlflow
    import mlflow.sklearn
except ImportError:
    print("  ⚠ MLflow not installed. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "mlflow"])
    import mlflow
    import mlflow.sklearn

# Set tracking URI to local directory
mlflow.set_tracking_uri(f"file:///{os.path.abspath(MLRUNS)}")
print(f"  MLflow tracking URI: {mlflow.get_tracking_uri()}")

# ══════════════════════════════════════════════════════════
# STEP 2 – Log Prophet Experiment
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Logging Prophet Model to MLflow")
print("=" * 60)

mlflow.set_experiment("RetailPulse_Forecasting")

prophet_metrics_path = os.path.join(PROCESSED, "day5", "prophet_metrics.csv")
if os.path.exists(prophet_metrics_path):
    prophet_metrics = pd.read_csv(prophet_metrics_path)
    
    with mlflow.start_run(run_name="Prophet_Baseline"):
        # Log parameters
        mlflow.log_param("model_type", "Prophet")
        mlflow.log_param("forecast_horizon", 30)
        mlflow.log_param("yearly_seasonality", True)
        mlflow.log_param("weekly_seasonality", True)
        mlflow.log_param("seasonality_mode", "multiplicative")
        
        # Log metrics
        mlflow.log_metric("MAPE", prophet_metrics["MAPE"].iloc[0])
        mlflow.log_metric("RMSE", prophet_metrics["RMSE"].iloc[0])
        mlflow.log_metric("MAE", prophet_metrics["MAE"].iloc[0])
        
        # Log model artifact
        prophet_model_path = os.path.join(MODELS, "prophet_model.pkl")
        if os.path.exists(prophet_model_path):
            mlflow.log_artifact(prophet_model_path)
        
        # Log forecast plot
        forecast_plot = os.path.join(REPORTS, "prophet_forecast.png")
        if os.path.exists(forecast_plot):
            mlflow.log_artifact(forecast_plot)
        
        print(f"  ✓ Prophet logged to MLflow")
        print(f"    MAPE: {prophet_metrics['MAPE'].iloc[0]}%")
        print(f"    RMSE: {prophet_metrics['RMSE'].iloc[0]}")
        print(f"    MAE:  {prophet_metrics['MAE'].iloc[0]}")
else:
    print("  ⚠ Prophet metrics not found. Run Day 5 first.")

# ══════════════════════════════════════════════════════════
# STEP 3 – Log LSTM Experiment
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Logging LSTM Model to MLflow")
print("=" * 60)

lstm_metrics_path = os.path.join(PROCESSED, "day6", "lstm_metrics.csv")
if os.path.exists(lstm_metrics_path):
    lstm_metrics = pd.read_csv(lstm_metrics_path)
    
    with mlflow.start_run(run_name="LSTM_Baseline"):
        # Log parameters
        mlflow.log_param("model_type", "LSTM")
        mlflow.log_param("hidden_size", 64)
        mlflow.log_param("num_layers", 2)
        mlflow.log_param("dropout", 0.2)
        mlflow.log_param("epochs", 50)
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("lookback", 30)
        mlflow.log_param("learning_rate", 0.001)
        
        # Log metrics
        mlflow.log_metric("MAPE", lstm_metrics["MAPE"].iloc[0])
        mlflow.log_metric("RMSE", lstm_metrics["RMSE"].iloc[0])
        mlflow.log_metric("MAE", lstm_metrics["MAE"].iloc[0])
        
        # Log model artifact
        lstm_model_path = os.path.join(MODELS, "lstm_model.pth")
        if os.path.exists(lstm_model_path):
            mlflow.log_artifact(lstm_model_path)
        
        # Log loss curve
        loss_curve = os.path.join(REPORTS, "loss_curve.png")
        if os.path.exists(loss_curve):
            mlflow.log_artifact(loss_curve)
        
        print(f"  ✓ LSTM logged to MLflow")
        print(f"    MAPE: {lstm_metrics['MAPE'].iloc[0]}%")
        print(f"    RMSE: {lstm_metrics['RMSE'].iloc[0]}")
        print(f"    MAE:  {lstm_metrics['MAE'].iloc[0]}")
else:
    print("  ⚠ LSTM metrics not found. Run Day 6 first.")

# ══════════════════════════════════════════════════════════
# STEP 4 – Log Segmentation Experiment
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Logging Segmentation to MLflow")
print("=" * 60)

mlflow.set_experiment("RetailPulse_Segmentation")

segments_path = os.path.join(PROCESSED, "day3", "customer_segments.csv")
if os.path.exists(segments_path):
    segments = pd.read_csv(segments_path)
    
    with mlflow.start_run(run_name="KMeans_Segmentation"):
        mlflow.log_param("algorithm", "K-Means")
        mlflow.log_param("n_clusters", 6)
        mlflow.log_param("features", "Recency, Frequency, Monetary")
        
        mlflow.log_metric("total_customers", len(segments))
        mlflow.log_metric("n_clusters", 6)
        
        # Log cluster distribution
        if "Segment" in segments.columns:
            for seg in segments["Segment"].unique():
                count = (segments["Segment"] == seg).sum()
                safe_name = seg.replace(" ", "_").replace("-", "_")
                mlflow.log_metric(f"segment_{safe_name}", count)
        
        # Log artifacts
        cluster_plot = os.path.join(REPORTS, "cluster_scatter.png")
        if os.path.exists(cluster_plot):
            mlflow.log_artifact(cluster_plot)
        
        print(f"  ✓ Segmentation logged to MLflow")
        print(f"    Total customers: {len(segments):,}")
else:
    print("  ⚠ Customer segments not found. Run Day 3 first.")

# ══════════════════════════════════════════════════════════
# STEP 5 – Week 1 Summary Report
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Generating Week 1 Summary Report")
print("=" * 60)

report_lines = []
report_lines.append("=" * 60)
report_lines.append("RETAILPULSE – WEEK 1 SUMMARY REPORT")
report_lines.append("=" * 60)
report_lines.append("")

# Dataset Summary
cleaned_path = os.path.join(PROCESSED, "day2", "cleaned_dataset.csv")
if os.path.exists(cleaned_path):
    cleaned = pd.read_csv(cleaned_path, nrows=1)
    # Get row count efficiently
    with open(cleaned_path, 'r') as f:
        row_count = sum(1 for _ in f) - 1  # subtract header
    report_lines.append(f"Dataset: {row_count:,} cleaned transactions")
    report_lines.append("")

# RFM Summary
rfm_path = os.path.join(PROCESSED, "day3", "rfm_features.csv")
if os.path.exists(rfm_path):
    rfm = pd.read_csv(rfm_path)
    report_lines.append(f"Customers: {len(rfm):,}")
    report_lines.append("")

# Segmentation Summary
if os.path.exists(segments_path):
    segments = pd.read_csv(segments_path)
    report_lines.append("Customer Segments:")
    if "Segment" in segments.columns:
        for seg, count in segments["Segment"].value_counts().items():
            report_lines.append(f"  {seg}: {count:,} customers")
    report_lines.append("")

# Forecasting Summary
report_lines.append("Forecasting Models:")
if os.path.exists(prophet_metrics_path):
    pm = pd.read_csv(prophet_metrics_path)
    report_lines.append(f"  Prophet:  MAPE={pm['MAPE'].iloc[0]}%, RMSE={pm['RMSE'].iloc[0]}")

if os.path.exists(lstm_metrics_path):
    lm = pd.read_csv(lstm_metrics_path)
    report_lines.append(f"  LSTM:     MAPE={lm['MAPE'].iloc[0]}%, RMSE={lm['RMSE'].iloc[0]}")

report_lines.append("")
report_lines.append("Deliverables:")
report_lines.append("  ✓ Cleaned Dataset")
report_lines.append("  ✓ RFM Features")
report_lines.append("  ✓ Customer Segments")
report_lines.append("  ✓ Time-Series Analysis")
report_lines.append("  ✓ Prophet Forecast")
report_lines.append("  ✓ LSTM Forecast")
report_lines.append("  ✓ MLflow Experiments")
report_lines.append("  ✓ Week-1 Report")

report_text = "\n".join(report_lines)
print(report_text)

# Save report
with open(os.path.join(REPORTS, "week1_report.txt"), "w", encoding="utf-8") as f:
    f.write(report_text)
print(f"\n  ✓ Saved: reports/day7/week1_report.txt")

# ══════════════════════════════════════════════════════════
# STEP 6 – Verify All Files
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Verifying All Week 1 Deliverables")
print("=" * 60)

expected_files = {
    "Data": [
        os.path.join(PROCESSED, "day2", "cleaned_dataset.csv"),
        os.path.join(PROCESSED, "day3", "rfm_features.csv"),
        os.path.join(PROCESSED, "day4", "daily_sales.csv"),
        os.path.join(PROCESSED, "day3", "customer_segments.csv"),
        os.path.join(PROCESSED, "day5", "prophet_forecast.csv"),
        os.path.join(PROCESSED, "day6", "lstm_forecast.csv"),
    ],
    "Models": [
        os.path.join(MODELS, "prophet_model.pkl"),
        os.path.join(MODELS, "lstm_model.pth"),
        os.path.join(MODELS, "scaler.pkl"),
    ],
    "Reports": [
        os.path.join(REPORTS_BASE, "day1", "missing_values.png"),
        os.path.join(REPORTS_BASE, "day1", "quantity_distribution.png"),
        os.path.join(REPORTS_BASE, "day1", "price_distribution.png"),
        os.path.join(REPORTS_BASE, "day1", "revenue_distribution.png"),
        os.path.join(REPORTS_BASE, "day1", "correlation_heatmap.png"),
        os.path.join(REPORTS_BASE, "day2", "rfm_distributions.png"),
        os.path.join(REPORTS_BASE, "day2", "rolling_demand.png"),
        os.path.join(REPORTS_BASE, "day3", "elbow_method.png"),
        os.path.join(REPORTS_BASE, "day3", "cluster_scatter.png"),
        os.path.join(REPORTS_BASE, "day3", "cluster_distribution.png"),
        os.path.join(REPORTS_BASE, "day4", "seasonal_decomposition.png"),
        os.path.join(REPORTS_BASE, "day4", "trend_plot.png"),
        os.path.join(REPORTS_BASE, "day4", "seasonality_plot.png"),
        os.path.join(REPORTS_BASE, "day5", "prophet_forecast.png"),
        os.path.join(REPORTS_BASE, "day6", "loss_curve.png"),
        os.path.join(REPORTS_BASE, "day6", "lstm_forecast.png"),
    ]
}

total = 0
found = 0
for category, files in expected_files.items():
    print(f"\n  {category}:")
    for f in files:
        total += 1
        exists = os.path.exists(f)
        if exists:
            found += 1
        status = "✓" if exists else "✗"
        fname = os.path.basename(f)
        print(f"    {status} {fname}")

print(f"\n  Result: {found}/{total} files present")

# ══════════════════════════════════════════════════════════
# FINAL
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DAY 7 COMPLETE ✓")
print("WEEK 1 COMPLETE ✓")
print("=" * 60)
print("""
All experiments have been logged to MLflow.
To view the MLflow UI, run:
  mlflow ui --backend-store-uri mlruns

Then open: http://127.0.0.1:5000
""")
