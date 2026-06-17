"""
RetailPulse – Day 3: Customer Segmentation
============================================
Goal: K-Means clustering, DBSCAN comparison, business interpretation.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import os

# ── Paths ────────────────────────────────────────────────
PROCESSED = os.path.join("..", "data", "processed")
REPORTS   = os.path.join("..", "reports", "day3")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ══════════════════════════════════════════════════════════
# STEP 1 – Load RFM Features
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading RFM Features")
print("=" * 60)
rfm = pd.read_csv(os.path.join(PROCESSED, "day3", "rfm_features.csv"), index_col="Customer ID")
print(f"  Loaded {len(rfm):,} customers")
print(rfm.head())

# ══════════════════════════════════════════════════════════
# STEP 2 – Scale RFM
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Scaling RFM Features")
print("=" * 60)
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)
print(f"  Scaled shape: {rfm_scaled.shape}")

# ══════════════════════════════════════════════════════════
# STEP 3 – Elbow Method for Optimal K
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Elbow Method")
print("=" * 60)
inertias = []
sil_scores = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(rfm_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(rfm_scaled, labels)
    sil_scores.append(sil)
    print(f"  K={k}: Inertia={km.inertia_:.0f}, Silhouette={sil:.4f}")

# Elbow Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(K_range, inertias, "bo-", linewidth=2, markersize=8)
axes[0].set_title("Elbow Method", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia")

axes[1].plot(K_range, sil_scores, "ro-", linewidth=2, markersize=8)
axes[1].set_title("Silhouette Score", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")

plt.suptitle("Optimal Cluster Selection", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "elbow_method.png"), dpi=150)
plt.close()
print("  ✓ Saved: reports/day3/elbow_method.png")

# ══════════════════════════════════════════════════════════
# STEP 4 – K-Means Clustering (K=6)
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: K-Means Clustering (K=6)")
print("=" * 60)
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

# Cluster summary
cluster_summary = rfm.groupby("Cluster").agg({
    "Recency": "mean",
    "Frequency": "mean",
    "Monetary": "mean"
}).round(2)

# Assign business labels based on RFM characteristics
# Sort clusters by Monetary descending to assign labels
cluster_summary_sorted = cluster_summary.sort_values("Monetary", ascending=False)
segment_labels = {
    cluster_summary_sorted.index[0]: "VIP Customers",
    cluster_summary_sorted.index[1]: "Loyal Customers",
    cluster_summary_sorted.index[2]: "Potential Loyalists",
    cluster_summary_sorted.index[3]: "At-Risk Customers",
    cluster_summary_sorted.index[4]: "Occasional Buyers",
    cluster_summary_sorted.index[5]: "Lost Customers",
}
rfm["Segment"] = rfm["Cluster"].map(segment_labels)

print("\n  Cluster Summary:")
for cluster_id, row in cluster_summary.iterrows():
    label = segment_labels[cluster_id]
    count = (rfm["Cluster"] == cluster_id).sum()
    print(f"  Cluster {cluster_id} ({label}): "
          f"R={row['Recency']:.0f}, F={row['Frequency']:.1f}, "
          f"M=£{row['Monetary']:.0f}, Count={count:,}")

# ══════════════════════════════════════════════════════════
# STEP 5 – Cluster Visualizations
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Cluster Visualizations")
print("=" * 60)

# Scatter: Recency vs Monetary (colored by cluster)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

colors = sns.color_palette("husl", 6)
for cluster_id in range(6):
    mask = rfm["Cluster"] == cluster_id
    label = segment_labels[cluster_id]
    axes[0].scatter(
        rfm.loc[mask, "Recency"],
        rfm.loc[mask, "Monetary"].clip(upper=rfm["Monetary"].quantile(0.99)),
        label=label,
        alpha=0.5,
        s=20,
        color=colors[cluster_id]
    )
axes[0].set_title("Recency vs Monetary", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Recency (Days)")
axes[0].set_ylabel("Monetary (£)")
axes[0].legend(fontsize=8, loc="upper right")

# Cluster distribution pie
cluster_counts = rfm["Segment"].value_counts()
axes[1].pie(
    cluster_counts.values,
    labels=cluster_counts.index,
    autopct="%1.1f%%",
    colors=colors,
    startangle=140,
    textprops={"fontsize": 9}
)
axes[1].set_title("Customer Segment Distribution", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "cluster_scatter.png"), dpi=150)
plt.close()
print("  ✓ Saved: reports/day3/cluster_scatter.png")

# Cluster bar chart
fig, ax = plt.subplots(figsize=(10, 6))
segment_revenue = rfm.groupby("Segment")["Monetary"].sum().sort_values(ascending=False)
segment_revenue.plot(kind="bar", ax=ax, color=colors, edgecolor="black", linewidth=0.5)
ax.set_title("Revenue by Customer Segment", fontsize=16, fontweight="bold")
ax.set_ylabel("Total Revenue (£)", fontsize=13)
ax.set_xlabel("")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "cluster_distribution.png"), dpi=150)
plt.close()
print("  ✓ Saved: reports/day3/cluster_distribution.png")

# ══════════════════════════════════════════════════════════
# STEP 6 – DBSCAN Comparison
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: DBSCAN Clustering")
print("=" * 60)
db = DBSCAN(eps=0.5, min_samples=5)
db_labels = db.fit_predict(rfm_scaled)

n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = (db_labels == -1).sum()
print(f"  DBSCAN Clusters: {n_clusters_db}")
print(f"  Noise Points:    {n_noise:,}")

if n_clusters_db > 1:
    db_sil = silhouette_score(rfm_scaled[db_labels != -1], db_labels[db_labels != -1])
    print(f"  Silhouette Score: {db_sil:.4f}")

# ══════════════════════════════════════════════════════════
# STEP 7 – Save Results
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: Saving Results")
print("=" * 60)
os.makedirs(os.path.join(PROCESSED, "day3"), exist_ok=True)
rfm.to_csv(os.path.join(PROCESSED, "day3", "customer_segments.csv"))
print(f"  ✓ Saved: data/processed/day3/customer_segments.csv")

# ══════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DAY 3 COMPLETE ✓")
print("=" * 60)
print(f"""
Deliverables:
  ✓ data/processed/customer_segments.csv
  ✓ reports/day3/elbow_method.png
  ✓ reports/day3/cluster_scatter.png
  ✓ reports/day3/cluster_distribution.png

K-Means (K=6):
  Silhouette Score: {silhouette_score(rfm_scaled, rfm['Cluster']):.4f}

DBSCAN:
  Clusters Found: {n_clusters_db}
  Noise Points:   {n_noise:,}
""")
