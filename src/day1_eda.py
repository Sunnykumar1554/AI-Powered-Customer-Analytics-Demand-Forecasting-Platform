"""
RetailPulse – Day 1: Dataset Understanding + Initial EDA
=========================================================
Goal: Load dataset, explore structure, analyze distributions,
      identify missing values, create correlation heatmap.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Paths ────────────────────────────────────────────────
RAW_DATA = os.path.join("..", "data", "raw", "online_retail_II.xlsx")
REPORTS  = os.path.join("..", "reports", "day1")
PROCESSED = os.path.join("..", "data", "processed")
os.makedirs(REPORTS, exist_ok=True)
os.makedirs(PROCESSED, exist_ok=True)

# ── Style ────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ══════════════════════════════════════════════════════════
# STEP 1 – Load both sheets and concatenate
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading dataset...")
print("=" * 60)

df1 = pd.read_excel(RAW_DATA, sheet_name="Year 2009-2010")
print(f"  Sheet 'Year 2009-2010': {df1.shape[0]:,} rows × {df1.shape[1]} columns")

df2 = pd.read_excel(RAW_DATA, sheet_name="Year 2010-2011")
print(f"  Sheet 'Year 2010-2011': {df2.shape[0]:,} rows × {df2.shape[1]} columns")

df = pd.concat([df1, df2], ignore_index=True)
print(f"\n  Combined dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ══════════════════════════════════════════════════════════
# STEP 2 – Basic info & statistics
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Dataset Info")
print("=" * 60)
print(df.dtypes)
print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

print("\n" + "=" * 60)
print("STEP 3: Descriptive Statistics")
print("=" * 60)
print(df.describe())

# ══════════════════════════════════════════════════════════
# STEP 4 – Missing Values Analysis
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Missing Values")
print("=" * 60)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    "Missing Count": missing,
    "Missing %": missing_pct
}).sort_values("Missing Count", ascending=False)
print(missing_df)

# Plot missing values
fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in missing.values]
missing.plot(kind="bar", ax=ax, color=colors, edgecolor="black", linewidth=0.5)
ax.set_title("Missing Values per Column", fontsize=16, fontweight="bold")
ax.set_ylabel("Count", fontsize=13)
ax.set_xlabel("")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "missing_values.png"), dpi=150)
plt.close()
print("  ✓ Saved: reports/day1/missing_values.png")

# ══════════════════════════════════════════════════════════
# STEP 5 – Distribution Analysis
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Distribution Analysis")
print("=" * 60)

# Quantity Distribution
fig, ax = plt.subplots(figsize=(10, 6))
# Clip extreme outliers for better visualization
q_clip = df["Quantity"].clip(
    lower=df["Quantity"].quantile(0.01),
    upper=df["Quantity"].quantile(0.99)
)
sns.histplot(q_clip, bins=80, kde=True, ax=ax, color="#3498db", edgecolor="white")
ax.set_title("Quantity Distribution (1st–99th Percentile)", fontsize=16, fontweight="bold")
ax.set_xlabel("Quantity", fontsize=13)
ax.set_ylabel("Frequency", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "quantity_distribution.png"), dpi=150)
plt.close()
print("  ✓ Saved: reports/day1/quantity_distribution.png")

# Price Distribution
fig, ax = plt.subplots(figsize=(10, 6))
p_clip = df["Price"].clip(
    lower=df["Price"].quantile(0.01),
    upper=df["Price"].quantile(0.99)
)
sns.histplot(p_clip, bins=80, kde=True, ax=ax, color="#e67e22", edgecolor="white")
ax.set_title("Price Distribution (1st–99th Percentile)", fontsize=16, fontweight="bold")
ax.set_xlabel("Price (£)", fontsize=13)
ax.set_ylabel("Frequency", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "price_distribution.png"), dpi=150)
plt.close()
print("  ✓ Saved: reports/day1/price_distribution.png")

# ══════════════════════════════════════════════════════════
# STEP 6 – Create Revenue Column
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Creating Revenue Column")
print("=" * 60)
df["Revenue"] = df["Quantity"] * df["Price"]
print(f"  Revenue range: £{df['Revenue'].min():,.2f} – £{df['Revenue'].max():,.2f}")
print(f"  Total Revenue: £{df['Revenue'].sum():,.2f}")

# Revenue Distribution
fig, ax = plt.subplots(figsize=(10, 6))
r_clip = df["Revenue"].clip(
    lower=df["Revenue"].quantile(0.01),
    upper=df["Revenue"].quantile(0.99)
)
sns.histplot(r_clip, bins=80, kde=True, ax=ax, color="#2ecc71", edgecolor="white")
ax.set_title("Revenue Distribution (1st–99th Percentile)", fontsize=16, fontweight="bold")
ax.set_xlabel("Revenue (£)", fontsize=13)
ax.set_ylabel("Frequency", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "revenue_distribution.png"), dpi=150)
plt.close()
print("  ✓ Saved: reports/day1/revenue_distribution.png")

# ══════════════════════════════════════════════════════════
# STEP 7 – Correlation Heatmap
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: Correlation Heatmap")
print("=" * 60)
corr_cols = ["Quantity", "Price", "Revenue"]
corr_matrix = df[corr_cols].corr()
print(corr_matrix)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    center=0,
    linewidths=1,
    linecolor="white",
    ax=ax,
    square=True,
    cbar_kws={"shrink": 0.8}
)
ax.set_title("Correlation Heatmap", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "correlation_heatmap.png"), dpi=150)
plt.close()
print("  ✓ Saved: reports/day1/correlation_heatmap.png")

# ══════════════════════════════════════════════════════════
# BONUS – Top Countries & Monthly Revenue Trend
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("BONUS: Top 10 Countries by Revenue")
print("=" * 60)
top_countries = (
    df.groupby("Country")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
print(top_countries)

fig, ax = plt.subplots(figsize=(12, 6))
top_countries.plot(kind="bar", ax=ax, color=sns.color_palette("viridis", 10), edgecolor="black", linewidth=0.5)
ax.set_title("Top 10 Countries by Revenue", fontsize=16, fontweight="bold")
ax.set_ylabel("Revenue (£)", fontsize=13)
ax.set_xlabel("")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "top_countries.png"), dpi=150)
plt.close()
print("  ✓ Saved: reports/day1/top_countries.png")

# ══════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DAY 1 COMPLETE ✓")
print("=" * 60)
print(f"""
Deliverables:
  ✓ reports/day1/missing_values.png
  ✓ reports/day1/quantity_distribution.png
  ✓ reports/day1/price_distribution.png
  ✓ reports/day1/revenue_distribution.png
  ✓ reports/day1/correlation_heatmap.png
  ✓ reports/day1/top_countries.png

Dataset Summary:
  Rows:     {df.shape[0]:,}
  Columns:  {df.shape[1]}
  Revenue:  £{df['Revenue'].sum():,.2f}
""")
