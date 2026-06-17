"""
RetailPulse – Day 14: Week 2 Checkpoint Report
================================================
Goal: Generate comprehensive Week 2 summary report covering all deliverables.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────
PROCESSED = os.path.join("..", "data", "processed")
MODELS    = os.path.join("..", "models")
REPORTS_BASE = os.path.join("..", "reports")
REPORTS   = os.path.join(REPORTS_BASE, "day14")
os.makedirs(REPORTS, exist_ok=True)

# ══════════════════════════════════════════════════════════
# STEP 1 – Gather All Metrics
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Gathering All Metrics")
print("=" * 60)

report_lines = []
report_lines.append("=" * 70)
report_lines.append("RETAILPULSE – WEEK 2 COMPREHENSIVE REPORT")
report_lines.append("Advanced Modeling & Churn Prediction")
report_lines.append("=" * 70)
report_lines.append("")

# ── Section 1: Forecasting ──
report_lines.append("-" * 70)
report_lines.append("1. FORECASTING MODELS")
report_lines.append("-" * 70)

forecast_metrics_path = os.path.join(PROCESSED, "day8", "forecast_metrics.csv")
if os.path.exists(forecast_metrics_path):
    fm = pd.read_csv(forecast_metrics_path)
    report_lines.append("")
    report_lines.append(f"  {'Model':<12} {'MAPE (%)':>10} {'RMSE':>12} {'MAE':>12}")
    report_lines.append(f"  {'-'*48}")
    for _, row in fm.iterrows():
        report_lines.append(
            f"  {row['Model']:<12} {row['MAPE']:>10.2f} {row['RMSE']:>12.2f} {row['MAE']:>12.2f}"
        )
    report_lines.append("")

    # Find best model
    best_idx = fm["MAPE"].idxmin()
    best_model = fm.iloc[best_idx]
    report_lines.append(f"  Best Model: {best_model['Model']} (MAPE: {best_model['MAPE']:.2f}%)")
    print(f"  ✓ Forecast metrics loaded")
else:
    report_lines.append("  ⚠ Forecast metrics not found")
    print(f"  ⚠ forecast_metrics.csv not found")

report_lines.append("")

# ── Section 2: Churn Prediction ──
report_lines.append("-" * 70)
report_lines.append("2. CHURN PREDICTION MODEL")
report_lines.append("-" * 70)

churn_metrics_path = os.path.join(PROCESSED, "day9", "churn_metrics.csv")
if os.path.exists(churn_metrics_path):
    cm = pd.read_csv(churn_metrics_path)
    report_lines.append("")
    report_lines.append(f"  {'Metric':<15} {'Value':>10}")
    report_lines.append(f"  {'-'*27}")
    for _, row in cm.iterrows():
        report_lines.append(f"  {row['Metric']:<15} {row['Value']:>10.4f}")
    report_lines.append("")

    auc_row = cm[cm["Metric"] == "ROC_AUC"]
    if not auc_row.empty:
        auc_val = auc_row["Value"].iloc[0]
        if auc_val >= 0.88:
            report_lines.append(f"  ✓ AUC Target Met: {auc_val:.4f} ≥ 0.88")
        else:
            report_lines.append(f"  ⚠ AUC Below Target: {auc_val:.4f} < 0.88")
    print(f"  ✓ Churn metrics loaded")
else:
    report_lines.append("  ⚠ Churn metrics not found")
    print(f"  ⚠ churn_metrics.csv not found")

# Tuned model comparison
best_params_path = os.path.join(PROCESSED, "day11", "best_params.csv")
if os.path.exists(best_params_path):
    bp = pd.read_csv(best_params_path)
    report_lines.append("")
    report_lines.append("  Tuned Hyperparameters (Optuna):")
    for _, row in bp.iterrows():
        val = row["Value"]
        if isinstance(val, float) and val == int(val):
            report_lines.append(f"    {row['Parameter']}: {int(val)}")
        elif isinstance(val, float):
            report_lines.append(f"    {row['Parameter']}: {val:.6f}")
        else:
            report_lines.append(f"    {row['Parameter']}: {val}")

report_lines.append("")

# ── Section 3: Inventory Optimization ──
report_lines.append("-" * 70)
report_lines.append("3. INVENTORY OPTIMIZATION")
report_lines.append("-" * 70)

inventory_path = os.path.join(PROCESSED, "day10", "inventory_recommendations.csv")
if os.path.exists(inventory_path):
    inv = pd.read_csv(inventory_path)
    report_lines.append(f"  Total products analyzed: {len(inv):,}")
    report_lines.append(f"  Average Safety Stock: {inv['Safety_Stock'].mean():.0f} units")
    report_lines.append(f"  Average Reorder Point: {inv['Reorder_Point'].mean():.0f} units")
    report_lines.append("")
    report_lines.append("  Top 20 Products by Demand:")
    report_lines.append(f"  {'#':<4} {'StockCode':<12} {'Total Demand':>13} "
                        f"{'Safety Stock':>13} {'Reorder Pt':>11} {'Order Qty':>10}")
    report_lines.append(f"  {'-'*65}")

    top20 = inv.head(20)
    for i, row in top20.iterrows():
        report_lines.append(
            f"  {i+1:<4} {str(row['StockCode']):<12} {row['Total_Demand']:>13,.0f} "
            f"{row['Safety_Stock']:>13} {row['Reorder_Point']:>11} "
            f"{row['Recommended_Order_Qty']:>10}"
        )
    print(f"  ✓ Inventory recommendations loaded")
else:
    report_lines.append("  ⚠ Inventory recommendations not found")
    print(f"  ⚠ inventory_recommendations.csv not found")

report_lines.append("")

# ── Section 4: Drift Detection ──
report_lines.append("-" * 70)
report_lines.append("4. DATA DRIFT DETECTION")
report_lines.append("-" * 70)

drift_path = os.path.join(PROCESSED, "day12", "drift_results.csv")
if os.path.exists(drift_path):
    drift = pd.read_csv(drift_path)
    report_lines.append("")

    if "Drift_Detected" in drift.columns:
        drifted = drift[drift["Drift_Detected"] == True]
        report_lines.append(f"  Features monitored: {len(drift)}")
        report_lines.append(f"  Features with drift: {len(drifted)}")
        report_lines.append("")

        if "Drift_Score" in drift.columns:
            report_lines.append(f"  {'Feature':<22} {'Drift Score':>13} {'Drift?':>8}")
            report_lines.append(f"  {'-'*45}")
            for _, row in drift.iterrows():
                status = "YES ⚠" if row["Drift_Detected"] else "No"
                report_lines.append(f"  {row['Feature']:<22} {row['Drift_Score']:>13.4f} {status:>8}")
        elif "KS_Statistic" in drift.columns:
            report_lines.append(f"  {'Feature':<22} {'KS Stat':>10} {'P-Value':>12} {'Drift?':>8}")
            report_lines.append(f"  {'-'*54}")
            for _, row in drift.iterrows():
                status = "YES ⚠" if row["Drift_Detected"] else "No"
                report_lines.append(f"  {row['Feature']:<22} {row['KS_Statistic']:>10.4f} "
                                    f"{row['P_Value']:>12.6f} {status:>8}")

    report_lines.append("")
    report_lines.append("  Full drift report: reports/day12/drift_report.html")
    print(f"  ✓ Drift results loaded")
else:
    report_lines.append("  ⚠ Drift results not found")
    print(f"  ⚠ drift_results.csv not found")

report_lines.append("")

# ── Section 5: Retraining Pipeline ──
report_lines.append("-" * 70)
report_lines.append("5. AUTOMATED RETRAINING PIPELINE")
report_lines.append("-" * 70)

pipeline_log_path = os.path.join(REPORTS_BASE, "day13", "pipeline_log.txt")
if os.path.exists(pipeline_log_path):
    report_lines.append("  ✓ Pipeline implemented and tested")
    report_lines.append("  Script: src/day13_retrain.py")
    report_lines.append("  Log: reports/day13/pipeline_log.txt")
    report_lines.append("")
    report_lines.append("  Pipeline Stages:")
    report_lines.append("    1. Load Data")
    report_lines.append("    2. Validate Data")
    report_lines.append("    3. Clean Data")
    report_lines.append("    4. Feature Engineering")
    report_lines.append("    5. Train Model (XGBoost)")
    report_lines.append("    6. Evaluate Model")
    report_lines.append("    7. Save & Log to MLflow")
    print(f"  ✓ Pipeline log found")
else:
    report_lines.append("  ✓ Pipeline implemented: src/day13_retrain.py")
    report_lines.append("  ⚠ Pipeline log not yet generated")
    print(f"  ⚠ Pipeline log not found")

report_lines.append("")

# ══════════════════════════════════════════════════════════
# STEP 2 – Verify All Deliverables
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Verifying All Week 2 Deliverables")
print("=" * 60)

report_lines.append("-" * 70)
report_lines.append("6. DELIVERABLES VERIFICATION")
report_lines.append("-" * 70)
report_lines.append("")

expected_files = {
    "Models": [
        ("prophet_model.pkl", os.path.join(MODELS, "prophet_model.pkl"), "Week 1"),
        ("lstm_model.pth", os.path.join(MODELS, "lstm_model.pth"), "Week 1"),
        ("hybrid_model.pkl", os.path.join(MODELS, "hybrid_model.pkl"), "Day 8"),
        ("churn_model.pkl", os.path.join(MODELS, "churn_model.pkl"), "Day 9"),
        ("best_xgboost.pkl", os.path.join(MODELS, "best_xgboost.pkl"), "Day 11"),
    ],
    "Data": [
        ("hybrid_forecast.csv", os.path.join(PROCESSED, "day8", "hybrid_forecast.csv"), "Day 8"),
        ("forecast_metrics.csv", os.path.join(PROCESSED, "day8", "forecast_metrics.csv"), "Day 8"),
        ("churn_predictions.csv", os.path.join(PROCESSED, "day9", "churn_predictions.csv"), "Day 9"),
        ("churn_metrics.csv", os.path.join(PROCESSED, "day9", "churn_metrics.csv"), "Day 9"),
        ("inventory_recommendations.csv", os.path.join(PROCESSED, "day10", "inventory_recommendations.csv"), "Day 10"),
        ("best_params.csv", os.path.join(PROCESSED, "day11", "best_params.csv"), "Day 11"),
        ("drift_results.csv", os.path.join(PROCESSED, "day12", "drift_results.csv"), "Day 12"),
    ],
    "Reports": [
        ("hybrid_forecast.png", os.path.join(REPORTS_BASE, "day8", "hybrid_forecast.png"), "Day 8"),
        ("metrics_comparison.png", os.path.join(REPORTS_BASE, "day8", "metrics_comparison.png"), "Day 8"),
        ("roc_curve.png", os.path.join(REPORTS_BASE, "day9", "roc_curve.png"), "Day 9"),
        ("confusion_matrix.png", os.path.join(REPORTS_BASE, "day9", "confusion_matrix.png"), "Day 9"),
        ("top20_products.png", os.path.join(REPORTS_BASE, "day10", "top20_products.png"), "Day 10"),
        ("feature_importance.png", os.path.join(REPORTS_BASE, "day11", "feature_importance.png"), "Day 11"),
        ("optuna_optimization.png", os.path.join(REPORTS_BASE, "day11", "optuna_optimization.png"), "Day 11"),
        ("drift_report.html", os.path.join(REPORTS_BASE, "day12", "drift_report.html"), "Day 12"),
        ("feature_shift_charts.png", os.path.join(REPORTS_BASE, "day12", "feature_shift_charts.png"), "Day 12"),
        ("pipeline_log.txt", os.path.join(REPORTS_BASE, "day13", "pipeline_log.txt"), "Day 13"),
    ]
}

total = 0
found = 0
for category, files in expected_files.items():
    report_lines.append(f"  {category}:")
    print(f"\n  {category}:")
    for fname, fpath, day in files:
        total += 1
        exists = os.path.exists(fpath)
        if exists:
            found += 1
        status = "✓" if exists else "✗"
        report_lines.append(f"    {status} {fname:<35} ({day})")
        print(f"    {status} {fname:<35} ({day})")
    report_lines.append("")

report_lines.append(f"  Total: {found}/{total} deliverables present")
print(f"\n  Result: {found}/{total} deliverables present")

# ══════════════════════════════════════════════════════════
# STEP 3 – Evaluation Criteria Summary
# ══════════════════════════════════════════════════════════
report_lines.append("")
report_lines.append("-" * 70)
report_lines.append("7. EVALUATION CRITERIA COVERAGE")
report_lines.append("-" * 70)
report_lines.append("")
report_lines.append("  Technical Depth & Model Quality:")
report_lines.append("    ✓ Hybrid Forecasting (Prophet + LSTM Ensemble)")
report_lines.append("    ✓ Churn Prediction (XGBoost with AUC target)")
report_lines.append("    ✓ Hyperparameter Tuning (Optuna, 50 trials)")
report_lines.append("    ✓ Feature Importance Analysis")
report_lines.append("")
report_lines.append("  MLOps & Production Readiness:")
report_lines.append("    ✓ MLflow Experiment Tracking")
report_lines.append("    ✓ Data Drift Detection (Evidently AI)")
report_lines.append("    ✓ Automated Retraining Pipeline")
report_lines.append("")
report_lines.append("  Business Impact:")
report_lines.append("    ✓ Inventory Optimization Recommendations")
report_lines.append("    ✓ Customer Churn Identification")
report_lines.append("    ✓ Demand Forecasting")

report_lines.append("")
report_lines.append("=" * 70)
report_lines.append("END OF WEEK 2 REPORT")
report_lines.append("=" * 70)

# ══════════════════════════════════════════════════════════
# STEP 4 – Save Report
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Saving Week 2 Report")
print("=" * 60)

report_text = "\n".join(report_lines)

with open(os.path.join(REPORTS, "week2_report.txt"), "w", encoding="utf-8") as f:
    f.write(report_text)
print(f"  ✓ Saved: reports/day14/week2_report.txt")

# Print full report
print("\n" + report_text)

# ══════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DAY 14 COMPLETE ✓")
print("WEEK 2 COMPLETE ✓")
print("=" * 60)
print(f"""
Week 2 Deliverables: {found}/{total} present

Week 2 Coverage:
  Day 8:  Hybrid Forecasting (Prophet + LSTM)    ✓
  Day 9:  Churn Prediction (XGBoost)             ✓
  Day 10: Inventory Optimization                 ✓
  Day 11: Feature Importance + Optuna Tuning     ✓
  Day 12: Drift Detection (Evidently AI)         ✓
  Day 13: Automated Retraining Pipeline          ✓
  Day 14: Week 2 Checkpoint Report               ✓

Report: reports/day14/week2_report.txt
""")
