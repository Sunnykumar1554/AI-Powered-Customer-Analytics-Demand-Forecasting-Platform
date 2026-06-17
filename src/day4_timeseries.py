"""
RetailPulse – Day 4: Time Series Preparation
==============================================
Goal: Stationarity test (ADF), seasonal decomposition, trend analysis.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import os

# ── Paths ────────────────────────────────────────────────
PROCESSED = os.path.join("..", "data", "processed")
REPORTS   = os.path.join("..", "reports", "day4")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ══════════════════════════════════════════════════════════
# STEP 1 – Load Daily Sales
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading Daily Sales Data")
print("=" * 60)
daily_sales = pd.read_csv(os.path.join(PROCESSED, "day4", "daily_sales.csv"), parse_dates=["Date"])
daily_sales = daily_sales.set_index("Date").sort_index()
print(f"  Date range: {daily_sales.index.min()} – {daily_sales.index.max()}")
print(f"  Total days: {len(daily_sales)}")

# Fill missing dates with 0
full_range = pd.date_range(start=daily_sales.index.min(), end=daily_sales.index.max(), freq="D")
daily_sales = daily_sales.reindex(full_range, fill_value=0)
daily_sales.index.name = "Date"
print(f"  After filling gaps: {len(daily_sales)} days")

# ══════════════════════════════════════════════════════════
# STEP 2 – Time Series Visualization
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Time Series Visualization")
print("=" * 60)

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(daily_sales.index, daily_sales["Quantity"], color="#3498db", alpha=0.6, linewidth=0.8)
ax.set_title("Daily Quantity Sold Over Time", fontsize=16, fontweight="bold")
ax.set_xlabel("Date", fontsize=13)
ax.set_ylabel("Quantity", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "daily_demand_timeseries.png"), dpi=150)
plt.close()
print("  ✓ Saved: reports/day4/daily_demand_timeseries.png")

# ══════════════════════════════════════════════════════════
# STEP 3 – ADF Test for Stationarity
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Augmented Dickey-Fuller Test")
print("=" * 60)

adf_result = adfuller(daily_sales["Quantity"].dropna(), autolag="AIC")

print(f"  ADF Statistic:   {adf_result[0]:.6f}")
print(f"  p-value:         {adf_result[1]:.6f}")
print(f"  Lags Used:       {adf_result[2]}")
print(f"  Observations:    {adf_result[3]}")
print(f"  Critical Values:")
for key, value in adf_result[4].items():
    print(f"    {key}: {value:.4f}")

if adf_result[1] < 0.05:
    print("\n  ★ Result: Series IS stationary (p < 0.05)")
else:
    print("\n  ★ Result: Series is NOT stationary (p >= 0.05)")
    print("    → Differencing may be needed for some models")

# ══════════════════════════════════════════════════════════
# STEP 4 – Seasonal Decomposition
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Seasonal Decomposition")
print("=" * 60)

# Use period=7 for weekly seasonality
decomposition = seasonal_decompose(
    daily_sales["Quantity"],
    model="additive",
    period=7
)

# Trend Plot
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

axes[0].plot(daily_sales.index, daily_sales["Quantity"], color="#3498db", alpha=0.7, linewidth=0.8)
axes[0].set_title("Original", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Quantity")

axes[1].plot(decomposition.trend.index, decomposition.trend, color="#e74c3c", linewidth=1.5)
axes[1].set_title("Trend", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Quantity")

axes[2].plot(decomposition.seasonal.index, decomposition.seasonal, color="#2ecc71", linewidth=0.8)
axes[2].set_title("Seasonality", fontsize=13, fontweight="bold")
axes[2].set_ylabel("Quantity")

axes[3].plot(decomposition.resid.index, decomposition.resid, color="#9b59b6", alpha=0.6, linewidth=0.8)
axes[3].set_title("Residuals", fontsize=13, fontweight="bold")
axes[3].set_ylabel("Quantity")
axes[3].set_xlabel("Date")

plt.suptitle("Time Series Decomposition (Weekly Period)", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "seasonal_decomposition.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved: reports/day4/seasonal_decomposition.png")

# Save individual trend and seasonality
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(decomposition.trend.index, decomposition.trend, color="#e74c3c", linewidth=2)
ax.set_title("Demand Trend Component", fontsize=16, fontweight="bold")
ax.set_xlabel("Date", fontsize=13)
ax.set_ylabel("Trend", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "trend_plot.png"), dpi=150)
plt.close()
print("  ✓ Saved: reports/day4/trend_plot.png")

fig, ax = plt.subplots(figsize=(14, 5))
# Show only a 2-month window for clarity
season_window = decomposition.seasonal.iloc[:60]
ax.plot(season_window.index, season_window.values, color="#2ecc71", linewidth=2)
ax.set_title("Seasonality Component (First 60 Days)", fontsize=16, fontweight="bold")
ax.set_xlabel("Date", fontsize=13)
ax.set_ylabel("Seasonal Effect", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "seasonality_plot.png"), dpi=150)
plt.close()
print("  ✓ Saved: reports/day4/seasonality_plot.png")

# ══════════════════════════════════════════════════════════
# BONUS – Monthly Aggregation
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("BONUS: Monthly Demand Trend")
print("=" * 60)

monthly = daily_sales["Quantity"].resample("M").sum()
fig, ax = plt.subplots(figsize=(12, 5))
monthly.plot(kind="bar", ax=ax, color=sns.color_palette("viridis", len(monthly)), edgecolor="black", linewidth=0.3)
ax.set_title("Monthly Total Demand", fontsize=16, fontweight="bold")
ax.set_ylabel("Total Quantity", fontsize=13)
ax.set_xlabel("")
labels = [d.strftime("%b %Y") for d in monthly.index]
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "monthly_demand.png"), dpi=150)
plt.close()
print("  ✓ Saved: reports/day4/monthly_demand.png")

# ══════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DAY 4 COMPLETE ✓")
print("=" * 60)
print(f"""
Deliverables:
  ✓ reports/day4/daily_demand_timeseries.png
  ✓ reports/day4/seasonal_decomposition.png
  ✓ reports/day4/trend_plot.png
  ✓ reports/day4/seasonality_plot.png
  ✓ reports/day4/monthly_demand.png

ADF Test Result:
  Statistic: {adf_result[0]:.4f}
  p-value:   {adf_result[1]:.6f}
  Stationary: {"Yes" if adf_result[1] < 0.05 else "No"}
""")
