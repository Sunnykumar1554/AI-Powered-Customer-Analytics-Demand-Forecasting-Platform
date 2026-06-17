"""
RetailPulse – Day 9: Churn Prediction Model
=============================================
Goal: Build XGBoost churn classifier with AUC ≥ 0.88.
      Churn = No purchase in last 90 days.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report,
    confusion_matrix, roc_curve
)
from xgboost import XGBClassifier
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────
PROCESSED = os.path.join("..", "data", "processed")
MODELS    = os.path.join("..", "models")
REPORTS   = os.path.join("..", "reports", "day9")
MLRUNS    = os.path.join("..", "mlruns")
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
print(f"  Customers: {df['Customer ID'].nunique():,}")
print(f"  Date range: {df['InvoiceDate'].min()} – {df['InvoiceDate'].max()}")

# ══════════════════════════════════════════════════════════
# STEP 2 – Create Churn Label
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Creating Churn Label (90-day threshold)")
print("=" * 60)

snapshot_date = df["InvoiceDate"].max()
print(f"  Snapshot date: {snapshot_date}")

# Last purchase date per customer
last_purchase = df.groupby("Customer ID")["InvoiceDate"].max()

# Days since last purchase
days_since_last = (snapshot_date - last_purchase).dt.days

# Churn label: 1 if no purchase in last 90 days
CHURN_THRESHOLD = 90
churn = (days_since_last > CHURN_THRESHOLD).astype(int)

print(f"  Churn threshold: {CHURN_THRESHOLD} days")
print(f"  Churned customers:     {churn.sum():,} ({churn.mean()*100:.1f}%)")
print(f"  Active customers:      {(1-churn).sum():,} ({(1-churn.mean())*100:.1f}%)")

# ══════════════════════════════════════════════════════════
# STEP 3 – Feature Engineering
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Building Churn Features")
print("=" * 60)

# Recency: days since last purchase
recency = days_since_last

# Frequency: number of unique invoices
frequency = df.groupby("Customer ID")["Invoice"].nunique()

# Monetary: total revenue
monetary = df.groupby("Customer ID")["Revenue"].sum()

# Average Basket Size: average quantity per order
avg_basket = df.groupby(["Customer ID", "Invoice"])["Quantity"].sum().groupby("Customer ID").mean()

# Average Revenue per order
avg_revenue = df.groupby(["Customer ID", "Invoice"])["Revenue"].sum().groupby("Customer ID").mean()

# Purchase Count: total number of items purchased
purchase_count = df.groupby("Customer ID")["Quantity"].sum()

# Days as customer (tenure)
first_purchase = df.groupby("Customer ID")["InvoiceDate"].min()
tenure = (snapshot_date - first_purchase).dt.days

# Number of unique products purchased
unique_products = df.groupby("Customer ID")["StockCode"].nunique()

# Number of unique countries
unique_countries = df.groupby("Customer ID")["Country"].nunique()

# Average days between purchases
def avg_days_between(group):
    dates = group.sort_values()
    if len(dates) < 2:
        return 0
    diffs = dates.diff().dropna().dt.days
    return diffs.mean() if len(diffs) > 0 else 0

avg_purchase_gap = df.groupby("Customer ID")["InvoiceDate"].apply(avg_days_between)

# Combine features
features = pd.DataFrame({
    "Recency": recency,
    "Frequency": frequency,
    "Monetary": monetary,
    "Avg_Basket_Size": avg_basket,
    "Avg_Revenue": avg_revenue,
    "Purchase_Count": purchase_count,
    "Tenure": tenure,
    "Unique_Products": unique_products,
    "Unique_Countries": unique_countries,
    "Avg_Purchase_Gap": avg_purchase_gap,
    "Churn": churn
})

# Drop NaN rows
features = features.dropna()
print(f"  Total features: {features.shape[1] - 1}")
print(f"  Total customers: {len(features):,}")
print(f"\n  Feature summary:")
print(features.describe().round(2))

# ══════════════════════════════════════════════════════════
# STEP 4 – Train/Test Split
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Train/Test Split (80/20, Stratified)")
print("=" * 60)

X = features.drop("Churn", axis=1)
y = features["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Training set: {len(X_train):,} ({y_train.mean()*100:.1f}% churn)")
print(f"  Test set:     {len(X_test):,} ({y_test.mean()*100:.1f}% churn)")

# ══════════════════════════════════════════════════════════
# STEP 5 – Train XGBoost Classifier
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Training XGBoost Classifier")
print("=" * 60)

# Handle class imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="auc",
    use_label_encoder=False
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
print("  ✓ XGBoost model trained successfully")

# ══════════════════════════════════════════════════════════
# STEP 6 – Evaluate Model
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Model Evaluation")
print("=" * 60)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n  {'Metric':<15} {'Value':>10}")
print(f"  {'-'*27}")
print(f"  {'Accuracy':<15} {accuracy:>10.4f}")
print(f"  {'Precision':<15} {precision:>10.4f}")
print(f"  {'Recall':<15} {recall:>10.4f}")
print(f"  {'F1 Score':<15} {f1:>10.4f}")
print(f"  {'ROC AUC':<15} {roc_auc:>10.4f}")

if roc_auc >= 0.88:
    print(f"\n  ✓ AUC target met! ({roc_auc:.4f} ≥ 0.88)")
else:
    print(f"\n  ⚠ AUC below target ({roc_auc:.4f} < 0.88)")

print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Active", "Churned"]))

# ══════════════════════════════════════════════════════════
# STEP 7 – Visualizations
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 7: Visualizations")
print("=" * 60)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color="#e74c3c", linewidth=2.5, label=f"XGBoost (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], color="#95a5a6", linewidth=1, linestyle="--", label="Random (AUC = 0.50)")
ax.fill_between(fpr, tpr, alpha=0.1, color="#e74c3c")
ax.set_title("ROC Curve – Churn Prediction", fontsize=16, fontweight="bold")
ax.set_xlabel("False Positive Rate", fontsize=13)
ax.set_ylabel("True Positive Rate", fontsize=13)
ax.legend(fontsize=12, loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "roc_curve.png"), dpi=150)
plt.close()
print("  ✓ Saved: reports/day9/roc_curve.png")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Active", "Churned"],
            yticklabels=["Active", "Churned"],
            annot_kws={"size": 16})
ax.set_title("Confusion Matrix – Churn Prediction", fontsize=16, fontweight="bold")
ax.set_xlabel("Predicted", fontsize=13)
ax.set_ylabel("Actual", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "confusion_matrix.png"), dpi=150)
plt.close()
print("  ✓ Saved: reports/day9/confusion_matrix.png")

# ══════════════════════════════════════════════════════════
# STEP 8 – MLflow Logging
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8: Logging to MLflow")
print("=" * 60)

try:
    import mlflow
    mlflow.set_tracking_uri(f"file:///{os.path.abspath(MLRUNS)}")
    mlflow.set_experiment("RetailPulse_Churn")

    with mlflow.start_run(run_name="XGBoost_Churn_Baseline"):
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("churn_threshold_days", CHURN_THRESHOLD)
        mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 2))

        mlflow.log_metric("Accuracy", round(accuracy, 4))
        mlflow.log_metric("Precision", round(precision, 4))
        mlflow.log_metric("Recall", round(recall, 4))
        mlflow.log_metric("F1_Score", round(f1, 4))
        mlflow.log_metric("ROC_AUC", round(roc_auc, 4))

        mlflow.log_artifact(os.path.join(REPORTS, "roc_curve.png"))
        mlflow.log_artifact(os.path.join(REPORTS, "confusion_matrix.png"))

    print("  ✓ Churn model logged to MLflow")
except Exception as e:
    print(f"  ⚠ MLflow logging failed: {e}")

# ══════════════════════════════════════════════════════════
# STEP 9 – Save Model & Predictions
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 9: Saving Model & Predictions")
print("=" * 60)

# Save model
with open(os.path.join(MODELS, "churn_model.pkl"), "wb") as f:
    pickle.dump(model, f)
print("  ✓ Saved: models/churn_model.pkl")

# Save predictions
predictions_df = pd.DataFrame({
    "Customer_ID": X_test.index,
    "Churn_Actual": y_test.values,
    "Churn_Predicted": y_pred,
    "Churn_Probability": y_pred_proba.round(4)
})
os.makedirs(os.path.join(PROCESSED, "day9"), exist_ok=True)
predictions_df.to_csv(os.path.join(PROCESSED, "day9", "churn_predictions.csv"), index=False)
print("  ✓ Saved: data/processed/day9/churn_predictions.csv")

# Save churn metrics
churn_metrics = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"],
    "Value": [round(accuracy, 4), round(precision, 4), round(recall, 4),
              round(f1, 4), round(roc_auc, 4)]
})
os.makedirs(os.path.join(PROCESSED, "day9"), exist_ok=True)
churn_metrics.to_csv(os.path.join(PROCESSED, "day9", "churn_metrics.csv"), index=False)
print("  ✓ Saved: data/processed/day9/churn_metrics.csv")

# Save features for later use (Day 11 tuning)
os.makedirs(os.path.join(PROCESSED, "day9"), exist_ok=True)
features.to_csv(os.path.join(PROCESSED, "day9", "churn_features.csv"))
print("  ✓ Saved: data/processed/day9/churn_features.csv")

# ══════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DAY 9 COMPLETE ✓")
print("=" * 60)
print(f"""
Deliverables:
  ✓ models/churn_model.pkl
  ✓ data/processed/churn_predictions.csv
  ✓ data/processed/churn_metrics.csv
  ✓ data/processed/churn_features.csv
  ✓ reports/day9/roc_curve.png
  ✓ reports/day9/confusion_matrix.png

Churn Definition: No purchase in {CHURN_THRESHOLD} days
Churned: {churn.sum():,} / {len(churn):,} ({churn.mean()*100:.1f}%)

Performance:
  Accuracy:  {accuracy:.4f}
  Precision: {precision:.4f}
  Recall:    {recall:.4f}
  F1 Score:  {f1:.4f}
  ROC AUC:   {roc_auc:.4f}
""")
