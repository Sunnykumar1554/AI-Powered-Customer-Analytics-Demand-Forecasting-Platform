"""
RetailPulse – Day 2: Data Cleaning + Feature Engineering
=========================================================
Goal: Clean data, create RFM features, build rolling demand features.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Paths ────────────────────────────────────────────────
RAW_DATA  = os.path.join("..", "data", "raw", "online_retail_II.xlsx")
PROCESSED = os.path.join("..", "data", "processed")
REPORTS   = os.path.join("..", "reports", "day2")
os.makedirs(PROCESSED, exist_ok=True)
os.makedirs(REPORTS, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ══════════════════════════════════════════════════════════
# STEP 1 – Load dataset
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading dataset...")
print("=" * 60)
df1 = pd.read_excel(RAW_DATA, sheet_name="Year 2009-2010")
df2 = pd.read_excel(RAW_DATA, sheet_name="Year 2010-2011")
df = pd.concat([df1, df2], ignore_index=True)
print(f"  Raw dataset: {df.shape[0]:,} rows")

# ══════════════════════════════════════════════════════════
# STEP 2 – Remove Missing Customer IDs
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Removing missing Customer IDs")
print("=" * 60)
before = len(df)
df = df.dropna(subset=["Customer ID"])
print(f"  Removed {before - len(df):,} rows with missing Customer ID")
print(f"  Remaining: {len(df):,} rows")

# ══════════════════════════════════════════════════════════
# STEP 3 – Remove Returns (negative Quantity)
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Removing returns (Quantity ≤ 0)")
print("=" * 60)
before = len(df)
df = df[df["Quantity"] > 0]
print(f"  Removed {before - len(df):,} return transactions")
print(f"  Remaining: {len(df):,} rows")

# ══════════════════════════════════════════════════════════
# STEP 4 – Remove Negative Prices
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Removing negative/zero prices")
print("=" * 60)
before = len(df)
df = df[df["Price"] > 0]
print(f"  Removed {before - len(df):,} rows with Price ≤ 0")
print(f"  Remaining: {len(df):,} rows")

# ══════════════════════════════════════════════════════════
# STEP 5 – Convert InvoiceDate & Create Revenue
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Data type conversions")
print("=" * 60)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Revenue"] = df["Quantity"] * df["Price"]
df["Customer ID"] = df["Customer ID"].astype(int)
print(f"  Date range: {df['InvoiceDate'].min()} – {df['InvoiceDate'].max()}")
print(f"  Total Revenue: £{df['Revenue'].sum():,.2f}")

# ══════════════════════════════════════════════════════════
# STEP 6 – RFM Feature Engineering
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Building RFM Features")
print("=" * 60)

snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
print(f"  Snapshot date: {snapshot_date}")

# Recency: days since last purchase
recency = (
    snapshot_date - df.groupby("Customer ID")["InvoiceDate"].max()
).dt.days

# Frequency: number of unique invoices
frequency = df.groupby("Customer ID")["Invoice"].nunique()

# Monetary: total revenue
monetary = df.groupby("Customer ID")["Revenue"].sum()

# Merge into RFM DataFrame
rfm = pd.DataFrame({
    "Recency": recency,
    "Frequency": frequency,
    "Monetary": monetary
})

print(f"  Total customers: {len(rfm):,}")
print(f"\n  RFM Statistics:")
print(rfm.describe().round(2))

# Save RFM
os.makedirs(os.path.join(PROCESSED, "day3"), exist_ok=True)
rfm.to_csv(os.path.join(PROCESSED, "day3", "rfm_features.csv"))
print(f"\n  ✓ Saved: data/processed/day3/rfm_features.csv")

# RFM Distribution Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(rfm["Recency"], bins=50, kde=True, ax=axes[0], color="#3498db", edgecolor="white")
axes[0].set_title("Recency Distribution", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Days Since Last Purchase")

sns.histplot(rfm["Frequency"].clip(upper=rfm["Frequency"].quantile(0.99)),
             bins=50, kde=True, ax=axes[1], color="#e67e22", edgecolor="white")
axes[1].set_title("Frequency Distribution", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Number of Orders")

sns.histplot(rfm["Monetary"].clip(upper=rfm["Monetary"].quantile(0.99)),
             bins=50, kde=True, ax=axes[2], color="#2ecc71", edgecolor="white")
axes[2].set_title("Monetary Distribution", fontsize=14, fontweight="bold")
axes[2].set_xlabel("Total Revenue (£)")

plt.suptitle("RFM Feature Distributions", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "rfm_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved: reports/day2/rfm_distributions.png")

# ══════════════════════════════════════════════════════════
# STEP 7 – Rolling Demand Features
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: Rolling Demand Features")
print("=" * 60)

daily_sales = (
    df.groupby(df["InvoiceDate"].dt.date)["Quantity"]
    .sum()
    .reset_index()
)
daily_sales.columns = ["Date", "Quantity"]
daily_sales["Date"] = pd.to_datetime(daily_sales["Date"])
daily_sales = daily_sales.sort_values("Date").reset_index(drop=True)

# Rolling means
daily_sales["Rolling_7d"] = daily_sales["Quantity"].rolling(window=7, min_periods=1).mean()
daily_sales["Rolling_30d"] = daily_sales["Quantity"].rolling(window=30, min_periods=1).mean()

print(f"  Date range: {daily_sales['Date'].min()} – {daily_sales['Date'].max()}")
print(f"  Total days: {len(daily_sales)}")

# Save daily sales
os.makedirs(os.path.join(PROCESSED, "day4"), exist_ok=True)
daily_sales.to_csv(os.path.join(PROCESSED, "day4", "daily_sales.csv"), index=False)
print("  ✓ Saved: data/processed/day4/daily_sales.csv")

# Plot rolling demand
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(daily_sales["Date"], daily_sales["Quantity"], alpha=0.3, color="#bdc3c7", label="Daily Demand")
ax.plot(daily_sales["Date"], daily_sales["Rolling_7d"], color="#e74c3c", linewidth=1.5, label="7-Day Rolling Avg")
ax.plot(daily_sales["Date"], daily_sales["Rolling_30d"], color="#2980b9", linewidth=2, label="30-Day Rolling Avg")
ax.set_title("Daily Demand with Rolling Averages", fontsize=16, fontweight="bold")
ax.set_xlabel("Date", fontsize=13)
ax.set_ylabel("Quantity Sold", fontsize=13)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "rolling_demand.png"), dpi=150)
plt.close()
print("  ✓ Saved: reports/day2/rolling_demand.png")

# ══════════════════════════════════════════════════════════
# Save cleaned dataset
os.makedirs(os.path.join(PROCESSED, "day2"), exist_ok=True)
df.to_csv(os.path.join(PROCESSED, "day2", "cleaned_dataset.csv"), index=False)
print(f"  ✓ Saved: data/processed/day2/cleaned_dataset.csv ({len(df):,} rows)")

# ══════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DAY 2 COMPLETE ✓")
print("=" * 60)
print(f"""
Deliverables:
  ✓ data/processed/cleaned_dataset.csv  ({len(df):,} rows)
  ✓ data/processed/rfm_features.csv     ({len(rfm):,} customers)
  ✓ data/processed/daily_sales.csv      ({len(daily_sales)} days)
  ✓ reports/day2/rfm_distributions.png
  ✓ reports/day2/rolling_demand.png
""")
