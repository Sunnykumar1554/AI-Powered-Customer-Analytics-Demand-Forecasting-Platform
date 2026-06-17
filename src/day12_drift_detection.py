"""
RetailPulse – Day 12: Drift Detection
=======================================
Goal: Use Evidently AI to detect data drift between reference and current periods.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────
PROCESSED = os.path.join("..", "data", "processed")
REPORTS   = os.path.join("..", "reports", "day12")
os.makedirs(REPORTS, exist_ok=True)

# ══════════════════════════════════════════════════════════
# STEP 1 – Load Cleaned Dataset
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading Cleaned Dataset")
print("=" * 60)

df = pd.read_csv(os.path.join(PROCESSED, "day2", "cleaned_dataset.csv"))
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
print(f"  Loaded {len(df):,} transactions")
print(f"  Date range: {df['InvoiceDate'].min()} – {df['InvoiceDate'].max()}")

# ══════════════════════════════════════════════════════════
# STEP 2 – Split into Reference & Current Periods
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Splitting into Reference & Current Periods")
print("=" * 60)

# Split by date: first 70% = reference, last 30% = current
date_range = df["InvoiceDate"].max() - df["InvoiceDate"].min()
split_date = df["InvoiceDate"].min() + pd.Timedelta(days=int(date_range.days * 0.7))

df_reference = df[df["InvoiceDate"] <= split_date]
df_current = df[df["InvoiceDate"] > split_date]

print(f"  Split date: {split_date.strftime('%Y-%m-%d')}")
print(f"  Reference period: {df_reference['InvoiceDate'].min().strftime('%Y-%m-%d')} – "
      f"{df_reference['InvoiceDate'].max().strftime('%Y-%m-%d')} ({len(df_reference):,} transactions)")
print(f"  Current period:   {df_current['InvoiceDate'].min().strftime('%Y-%m-%d')} – "
      f"{df_current['InvoiceDate'].max().strftime('%Y-%m-%d')} ({len(df_current):,} transactions)")

# ══════════════════════════════════════════════════════════
# STEP 3 – Create Customer-Level Features for Both Periods
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Building Customer-Level Features")
print("=" * 60)

def create_customer_features(data, period_name):
    """Create customer-level aggregation features for drift detection."""
    snapshot = data["InvoiceDate"].max()

    cust_features = data.groupby("Customer ID").agg(
        Revenue=("Revenue", "sum"),
        Quantity=("Quantity", "sum"),
        Frequency=("Invoice", "nunique"),
        Avg_Price=("Price", "mean"),
        Num_Products=("StockCode", "nunique"),
        Avg_Basket_Size=("Quantity", "mean"),
    ).reset_index()

    # Monetary = Revenue (already computed)
    cust_features["Monetary"] = cust_features["Revenue"]

    # Average order value
    order_totals = data.groupby(["Customer ID", "Invoice"])["Revenue"].sum().reset_index()
    avg_order_value = order_totals.groupby("Customer ID")["Revenue"].mean().reset_index()
    avg_order_value.columns = ["Customer ID", "Avg_Order_Value"]
    cust_features = cust_features.merge(avg_order_value, on="Customer ID", how="left")

    print(f"  {period_name}: {len(cust_features):,} customers, {cust_features.shape[1]} features")
    return cust_features

ref_features = create_customer_features(df_reference, "Reference")
cur_features = create_customer_features(df_current, "Current")

# Columns to monitor for drift
monitor_cols = ["Revenue", "Quantity", "Frequency", "Monetary",
                "Avg_Price", "Num_Products", "Avg_Basket_Size", "Avg_Order_Value"]

ref_data = ref_features[monitor_cols].copy()
cur_data = cur_features[monitor_cols].copy()

print(f"\n  Monitoring columns: {monitor_cols}")

# ══════════════════════════════════════════════════════════
# STEP 4 – Evidently Data Drift Report
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Generating Evidently Drift Report")
print("=" * 60)

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    # Create drift report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_data, current_data=cur_data)

    # Save HTML report
    report_path = os.path.join(REPORTS, "drift_report.html")
    report.save_html(report_path)
    print(f"  ✓ Saved: reports/day12/drift_report.html")

    # Extract drift results
    report_dict = report.as_dict()

    # Parse drift results
    print(f"\n  Drift Detection Results:")
    print(f"  {'Feature':<22} {'Drift Detected':>15} {'Drift Score':>13}")
    print(f"  {'-'*52}")

    drift_results = []
    metrics = report_dict.get("metrics", [])
    for metric in metrics:
        result = metric.get("result", {})
        if "drift_by_columns" in result:
            for col, col_result in result["drift_by_columns"].items():
                drift_detected = col_result.get("drift_detected", False)
                drift_score = col_result.get("drift_score", 0)
                stat_test = col_result.get("stattest_name", "N/A")
                drift_results.append({
                    "Feature": col,
                    "Drift_Detected": drift_detected,
                    "Drift_Score": round(drift_score, 4),
                    "Stat_Test": stat_test
                })
                status = "⚠ YES" if drift_detected else "  No"
                print(f"  {col:<22} {status:>15} {drift_score:>13.4f}")

        if "dataset_drift" in result:
            dataset_drift = result["dataset_drift"]
            n_drifted = result.get("number_of_drifted_columns", 0)
            n_columns = result.get("number_of_columns", len(monitor_cols))
            print(f"\n  Dataset-level drift: {'YES ⚠' if dataset_drift else 'No'}")
            print(f"  Drifted columns: {n_drifted}/{n_columns}")

    # Save drift summary
    if drift_results:
        drift_df = pd.DataFrame(drift_results)
        os.makedirs(os.path.join(PROCESSED, "day12"), exist_ok=True)
        drift_df.to_csv(os.path.join(PROCESSED, "day12", "drift_results.csv"), index=False)
        print(f"\n  ✓ Saved: data/processed/day12/drift_results.csv")

except ImportError:
    print("  ⚠ Evidently not installed. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "evidently"])
    print("  Please re-run this script after installation.")
except Exception as e:
    print(f"  ⚠ Evidently report generation failed: {e}")
    print("  Generating fallback statistical drift analysis...")

    # Fallback: Manual drift detection using KS test
    from scipy import stats

    print(f"\n  Manual Drift Detection (KS Test):")
    print(f"  {'Feature':<22} {'KS Stat':>10} {'P-Value':>12} {'Drift?':>8}")
    print(f"  {'-'*54}")

    drift_results = []
    for col in monitor_cols:
        ref_vals = ref_data[col].dropna()
        cur_vals = cur_data[col].dropna()
        ks_stat, p_value = stats.ks_2samp(ref_vals, cur_vals)
        drift_detected = p_value < 0.05

        drift_results.append({
            "Feature": col,
            "KS_Statistic": round(ks_stat, 4),
            "P_Value": round(p_value, 6),
            "Drift_Detected": drift_detected
        })
        status = "YES ⚠" if drift_detected else "No"
        print(f"  {col:<22} {ks_stat:>10.4f} {p_value:>12.6f} {status:>8}")

    drift_df = pd.DataFrame(drift_results)
    os.makedirs(os.path.join(PROCESSED, "day12"), exist_ok=True)
    drift_df.to_csv(os.path.join(PROCESSED, "day12", "drift_results.csv"), index=False)

    # Generate simple HTML report
    html_content = """
    <html><head><title>Drift Detection Report - RetailPulse</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        h1 { color: #2c3e50; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #3498db; color: white; }
        .drift { background-color: #ffe6e6; }
        .no-drift { background-color: #e6ffe6; }
    </style></head><body>
    <h1>Data Drift Detection Report</h1>
    <h2>RetailPulse – Week 2</h2>
    <p>Reference period: """ + df_reference['InvoiceDate'].min().strftime('%Y-%m-%d') + " – " + df_reference['InvoiceDate'].max().strftime('%Y-%m-%d') + """</p>
    <p>Current period: """ + df_current['InvoiceDate'].min().strftime('%Y-%m-%d') + " – " + df_current['InvoiceDate'].max().strftime('%Y-%m-%d') + """</p>
    <table>
    <tr><th>Feature</th><th>KS Statistic</th><th>P-Value</th><th>Drift Detected</th></tr>
    """
    for _, row in drift_df.iterrows():
        css_class = "drift" if row["Drift_Detected"] else "no-drift"
        html_content += f'<tr class="{css_class}"><td>{row["Feature"]}</td><td>{row["KS_Statistic"]}</td><td>{row["P_Value"]}</td><td>{"Yes" if row["Drift_Detected"] else "No"}</td></tr>\n'

    html_content += "</table></body></html>"

    with open(os.path.join(REPORTS, "drift_report.html"), "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\n  ✓ Saved: reports/day12/drift_report.html (fallback)")

# ══════════════════════════════════════════════════════════
# STEP 5 – Feature Distribution Comparison Plots
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Feature Distribution Comparison Plots")
print("=" * 60)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, col in enumerate(monitor_cols):
    ax = axes[idx]
    ref_vals = ref_data[col].dropna()
    cur_vals = cur_data[col].dropna()

    # Clip outliers for visualization
    upper = max(ref_vals.quantile(0.95), cur_vals.quantile(0.95))

    ax.hist(ref_vals.clip(upper=upper), bins=50, alpha=0.5, color="#3498db",
            label="Reference", density=True, edgecolor="white")
    ax.hist(cur_vals.clip(upper=upper), bins=50, alpha=0.5, color="#e74c3c",
            label="Current", density=True, edgecolor="white")
    ax.set_title(col, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

plt.suptitle("Feature Distributions: Reference vs Current",
             fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "feature_shift_charts.png"), dpi=150)
plt.close()
print("  ✓ Saved: reports/day12/feature_shift_charts.png")

# ══════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DAY 12 COMPLETE ✓")
print("=" * 60)
print(f"""
Deliverables:
  ✓ reports/day12/drift_report.html
  ✓ reports/day12/feature_shift_charts.png
  ✓ data/processed/drift_results.csv

Drift Analysis:
  Reference: {df_reference['InvoiceDate'].min().strftime('%Y-%m-%d')} – {df_reference['InvoiceDate'].max().strftime('%Y-%m-%d')} ({len(ref_features):,} customers)
  Current:   {df_current['InvoiceDate'].min().strftime('%Y-%m-%d')} – {df_current['InvoiceDate'].max().strftime('%Y-%m-%d')} ({len(cur_features):,} customers)
  Features monitored: {len(monitor_cols)}
""")
