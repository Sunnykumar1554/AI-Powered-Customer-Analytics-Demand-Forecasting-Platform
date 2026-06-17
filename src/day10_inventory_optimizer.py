"""
RetailPulse – Day 10: Inventory Optimization
==============================================
Goal: Create inventory recommendations from forecasted demand.
      Reorder Point = Avg Daily Demand × Lead Time + Safety Stock
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────
PROCESSED = os.path.join("..", "data", "processed")
REPORTS   = os.path.join("..", "reports", "day10")
os.makedirs(REPORTS, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ══════════════════════════════════════════════════════════
# STEP 1 – Load Cleaned Dataset
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading Cleaned Dataset")
print("=" * 60)

df = pd.read_csv(os.path.join(PROCESSED, "day2", "cleaned_dataset.csv"))
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
print(f"  Loaded {len(df):,} transactions")
print(f"  Unique products (StockCode): {df['StockCode'].nunique():,}")

# ══════════════════════════════════════════════════════════
# STEP 2 – Product-Level Demand Analysis
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Product-Level Demand Analysis")
print("=" * 60)

# Calculate date range for daily demand
date_range_days = (df["InvoiceDate"].max() - df["InvoiceDate"].min()).days
print(f"  Date span: {date_range_days} days")

# Group by StockCode
product_demand = df.groupby("StockCode").agg(
    Description=("Description", "first"),
    Total_Demand=("Quantity", "sum"),
    Total_Revenue=("Revenue", "sum"),
    Num_Transactions=("Invoice", "nunique"),
    Num_Days_Sold=("InvoiceDate", lambda x: x.dt.date.nunique()),
    Demand_Std=("Quantity", "std")
).reset_index()

# Calculate average daily demand
product_demand["Avg_Daily_Demand"] = (
    product_demand["Total_Demand"] / date_range_days
).round(2)

# Fill NaN std with 0
product_demand["Demand_Std"] = product_demand["Demand_Std"].fillna(0)

# Calculate demand variability (coefficient of variation)
product_demand["Demand_CV"] = np.where(
    product_demand["Avg_Daily_Demand"] > 0,
    product_demand["Demand_Std"] / (product_demand["Total_Demand"] / product_demand["Num_Transactions"]),
    0
).round(4)

print(f"  Total products analyzed: {len(product_demand):,}")
print(f"  Products with sales > 100 units: {(product_demand['Total_Demand'] > 100).sum():,}")

# ══════════════════════════════════════════════════════════
# STEP 3 – Calculate Reorder Points
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Calculating Reorder Points")
print("=" * 60)

# Parameters
LEAD_TIME = 7          # days
Z_SCORE = 1.65         # 95% service level
print(f"  Lead Time: {LEAD_TIME} days")
print(f"  Service Level: 95% (z = {Z_SCORE})")

# Safety Stock = z × std_daily × sqrt(lead_time)
# Calculate daily std per product
daily_demand = df.groupby(["StockCode", df["InvoiceDate"].dt.date])["Quantity"].sum().reset_index()
daily_demand.columns = ["StockCode", "Date", "Daily_Quantity"]

daily_std = daily_demand.groupby("StockCode")["Daily_Quantity"].std().fillna(0)
daily_std.name = "Daily_Std"

product_demand = product_demand.merge(
    daily_std.reset_index(),
    on="StockCode",
    how="left"
)
product_demand["Daily_Std"] = product_demand["Daily_Std"].fillna(0)

# Safety Stock
product_demand["Safety_Stock"] = (
    Z_SCORE * product_demand["Daily_Std"] * np.sqrt(LEAD_TIME)
).round(0).astype(int)

# Reorder Point
product_demand["Reorder_Point"] = (
    product_demand["Avg_Daily_Demand"] * LEAD_TIME
    + product_demand["Safety_Stock"]
).round(0).astype(int)

# Recommended Order Quantity (EOQ approximation – 2 weeks supply + safety)
product_demand["Recommended_Order_Qty"] = (
    product_demand["Avg_Daily_Demand"] * 14  # 2 weeks supply
    + product_demand["Safety_Stock"]
).round(0).astype(int)

print(f"\n  Sample Calculations (Top 5 by demand):")
top5 = product_demand.nlargest(5, "Total_Demand")
print(f"  {'StockCode':<12} {'Avg Daily':>10} {'Safety Stock':>13} {'Reorder Pt':>11} {'Order Qty':>10}")
print(f"  {'-'*58}")
for _, row in top5.iterrows():
    print(f"  {str(row['StockCode']):<12} {row['Avg_Daily_Demand']:>10.1f} "
          f"{row['Safety_Stock']:>13} {row['Reorder_Point']:>11} {row['Recommended_Order_Qty']:>10}")

# ══════════════════════════════════════════════════════════
# STEP 4 – Load Forecast for Enhanced Recommendations
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Loading Hybrid Forecast for Demand Projection")
print("=" * 60)

hybrid_path = os.path.join(PROCESSED, "day8", "hybrid_forecast.csv")
if os.path.exists(hybrid_path):
    hybrid = pd.read_csv(hybrid_path)
    forecast_avg = hybrid["Hybrid_Prediction"].mean()
    actual_avg = hybrid["Actual"].mean()
    forecast_ratio = forecast_avg / actual_avg if actual_avg > 0 else 1.0
    print(f"  Forecast avg: {forecast_avg:,.0f}")
    print(f"  Actual avg:   {actual_avg:,.0f}")
    print(f"  Forecast ratio: {forecast_ratio:.2f}")

    # Adjust recommendations based on forecast trend
    product_demand["Forecast_Adj_Demand"] = (
        product_demand["Avg_Daily_Demand"] * forecast_ratio
    ).round(2)
    product_demand["Forecast_Reorder_Point"] = (
        product_demand["Forecast_Adj_Demand"] * LEAD_TIME
        + product_demand["Safety_Stock"]
    ).round(0).astype(int)
else:
    print("  ⚠ Hybrid forecast not found, using historical demand only")
    product_demand["Forecast_Adj_Demand"] = product_demand["Avg_Daily_Demand"]
    product_demand["Forecast_Reorder_Point"] = product_demand["Reorder_Point"]

# ══════════════════════════════════════════════════════════
# STEP 5 – Create Recommendation Table
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Creating Recommendation Table")
print("=" * 60)

# Sort by Total Demand descending
recommendations = product_demand.sort_values("Total_Demand", ascending=False).reset_index(drop=True)

# Select columns for output
output_cols = [
    "StockCode", "Description", "Total_Demand", "Avg_Daily_Demand",
    "Daily_Std", "Safety_Stock", "Reorder_Point",
    "Recommended_Order_Qty", "Forecast_Adj_Demand",
    "Forecast_Reorder_Point", "Total_Revenue"
]
recommendations = recommendations[output_cols]

# Top 20 products
top20 = recommendations.head(20)
print(f"\n  Top 20 Products by Demand:")
print(f"  {'#':<4} {'StockCode':<12} {'Description':<30} {'Total Demand':>13} {'Reorder Pt':>11} {'Order Qty':>10}")
print(f"  {'-'*82}")
for i, row in top20.iterrows():
    desc = str(row['Description'])[:28] if pd.notna(row['Description']) else "N/A"
    print(f"  {i+1:<4} {str(row['StockCode']):<12} {desc:<30} "
          f"{row['Total_Demand']:>13,.0f} {row['Reorder_Point']:>11} {row['Recommended_Order_Qty']:>10}")

# ══════════════════════════════════════════════════════════
# STEP 6 – Visualizations
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Visualizations")
print("=" * 60)

# Top 20 products bar chart
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Total Demand
top20_plot = top20.head(20).copy()
top20_plot["Label"] = top20_plot["StockCode"].astype(str)

axes[0].barh(
    range(len(top20_plot)),
    top20_plot["Total_Demand"],
    color="#3498db", edgecolor="black", linewidth=0.5
)
axes[0].set_yticks(range(len(top20_plot)))
axes[0].set_yticklabels(top20_plot["Label"], fontsize=9)
axes[0].set_title("Top 20 Products – Total Demand", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Total Quantity Demanded")
axes[0].invert_yaxis()

# Reorder Points vs Safety Stock
x = range(len(top20_plot))
width = 0.35
axes[1].barh([i - width/2 for i in x], top20_plot["Reorder_Point"],
             width, label="Reorder Point", color="#e74c3c", edgecolor="black", linewidth=0.5)
axes[1].barh([i + width/2 for i in x], top20_plot["Safety_Stock"],
             width, label="Safety Stock", color="#f39c12", edgecolor="black", linewidth=0.5)
axes[1].set_yticks(range(len(top20_plot)))
axes[1].set_yticklabels(top20_plot["Label"], fontsize=9)
axes[1].set_title("Reorder Point vs Safety Stock", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Units")
axes[1].legend(fontsize=11)
axes[1].invert_yaxis()

plt.suptitle("Inventory Optimization – Top 20 Products", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "top20_products.png"), dpi=150)
plt.close()
print("  ✓ Saved: reports/day10/top20_products.png")

# Demand variability scatter
fig, ax = plt.subplots(figsize=(10, 6))
scatter_data = product_demand[product_demand["Total_Demand"] > 50].copy()
scatter = ax.scatter(
    scatter_data["Avg_Daily_Demand"],
    scatter_data["Daily_Std"],
    c=scatter_data["Safety_Stock"],
    cmap="YlOrRd",
    alpha=0.6,
    s=20,
    edgecolors="black",
    linewidth=0.3
)
plt.colorbar(scatter, ax=ax, label="Safety Stock (units)")
ax.set_title("Demand vs Variability by Product", fontsize=16, fontweight="bold")
ax.set_xlabel("Average Daily Demand", fontsize=13)
ax.set_ylabel("Daily Demand Std Dev", fontsize=13)
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "demand_variability.png"), dpi=150)
plt.close()
print("  ✓ Saved: reports/day10/demand_variability.png")

# ══════════════════════════════════════════════════════════
# STEP 7 – Save Recommendations
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: Saving Inventory Recommendations")
print("=" * 60)

os.makedirs(os.path.join(PROCESSED, "day10"), exist_ok=True)
recommendations.to_csv(
    os.path.join(PROCESSED, "day10", "inventory_recommendations.csv"),
    index=False
)
print(f"  ✓ Saved: data/processed/day10/inventory_recommendations.csv")
print(f"  Total products: {len(recommendations):,}")

# ══════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DAY 10 COMPLETE ✓")
print("=" * 60)
print(f"""
Deliverables:
  ✓ data/processed/inventory_recommendations.csv
  ✓ reports/day10/top20_products.png
  ✓ reports/day10/demand_variability.png

Parameters:
  Lead Time:     {LEAD_TIME} days
  Service Level: 95% (z = {Z_SCORE})

Summary:
  Products analyzed: {len(recommendations):,}
  Avg Safety Stock:  {recommendations['Safety_Stock'].mean():.0f} units
  Avg Reorder Point: {recommendations['Reorder_Point'].mean():.0f} units
""")
