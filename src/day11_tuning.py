"""
RetailPulse – Day 11: Feature Importance + Hyperparameter Tuning
==================================================================
Goal: Optuna hyperparameter tuning for XGBoost churn model,
      extract top 10 feature importances.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# Fix scikit-learn 1.6+ compatibility with XGBoost MRO
original_tags = XGBClassifier.__sklearn_tags__
XGBClassifier.__sklearn_tags__ = lambda self: (t := original_tags(self), setattr(t, 'estimator_type', 'classifier'), t)[2]

# Suppress Optuna logs during optimization
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths ────────────────────────────────────────────────
PROCESSED = os.path.join("..", "data", "processed")
MODELS    = os.path.join("..", "models")
REPORTS   = os.path.join("..", "reports", "day11")
MLRUNS    = os.path.join("..", "mlruns")
os.makedirs(REPORTS, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ══════════════════════════════════════════════════════════
# STEP 1 – Load Churn Features
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading Churn Features")
print("=" * 60)

features_path = os.path.join(PROCESSED, "day9", "churn_features.csv")
if os.path.exists(features_path):
    features = pd.read_csv(features_path, index_col=0)
    print(f"  Loaded from churn_features.csv: {features.shape}")
else:
    # Fallback: recreate features from cleaned dataset
    print("  ⚠ churn_features.csv not found. Recreating from cleaned dataset...")
    df = pd.read_csv(os.path.join(PROCESSED, "day2", "cleaned_dataset.csv"))
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    snapshot_date = df["InvoiceDate"].max()
    last_purchase = df.groupby("Customer ID")["InvoiceDate"].max()
    days_since_last = (snapshot_date - last_purchase).dt.days
    churn = (days_since_last > 90).astype(int)
    recency = days_since_last
    frequency = df.groupby("Customer ID")["Invoice"].nunique()
    monetary = df.groupby("Customer ID")["Revenue"].sum()
    avg_basket = df.groupby(["Customer ID", "Invoice"])["Quantity"].sum().groupby("Customer ID").mean()
    avg_revenue = df.groupby(["Customer ID", "Invoice"])["Revenue"].sum().groupby("Customer ID").mean()
    purchase_count = df.groupby("Customer ID")["Quantity"].sum()
    first_purchase = df.groupby("Customer ID")["InvoiceDate"].min()
    tenure = (snapshot_date - first_purchase).dt.days
    unique_products = df.groupby("Customer ID")["StockCode"].nunique()
    unique_countries = df.groupby("Customer ID")["Country"].nunique()

    def avg_days_between(group):
        dates = group.sort_values()
        if len(dates) < 2:
            return 0
        diffs = dates.diff().dropna().dt.days
        return diffs.mean() if len(diffs) > 0 else 0

    avg_purchase_gap = df.groupby("Customer ID")["InvoiceDate"].apply(avg_days_between)

    features = pd.DataFrame({
        "Recency": recency, "Frequency": frequency, "Monetary": monetary,
        "Avg_Basket_Size": avg_basket, "Avg_Revenue": avg_revenue,
        "Purchase_Count": purchase_count, "Tenure": tenure,
        "Unique_Products": unique_products, "Unique_Countries": unique_countries,
        "Avg_Purchase_Gap": avg_purchase_gap, "Churn": churn
    }).dropna()
    print(f"  Recreated features: {features.shape}")

X = features.drop("Churn", axis=1)
y = features["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"  Training: {len(X_train):,}, Test: {len(X_test):,}")
print(f"  Churn rate: {y.mean()*100:.1f}%")
print(f"  Features: {list(X.columns)}")

# ══════════════════════════════════════════════════════════
# STEP 2 – Optuna Hyperparameter Tuning
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Optuna Hyperparameter Tuning (50 trials)")
print("=" * 60)

def objective(trial):
    """Optuna objective function for XGBoost hyperparameter tuning."""
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    model = XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="auc",
        use_label_encoder=False
    )

    # 3-fold cross-validation
    scores = cross_val_score(
        model, X_train, y_train,
        cv=3, scoring="roc_auc", n_jobs=1
    )

    return scores.mean()


# Run optimization
study = optuna.create_study(direction="maximize", study_name="XGBoost_Churn_Tuning")
study.optimize(objective, n_trials=50, show_progress_bar=True)

best_params = study.best_params
best_score = study.best_value

print(f"\n  Best ROC AUC (CV): {best_score:.4f}")
print(f"\n  Best Parameters:")
for param, value in best_params.items():
    if isinstance(value, float):
        print(f"    {param}: {value:.6f}")
    else:
        print(f"    {param}: {value}")

# ══════════════════════════════════════════════════════════
# STEP 3 – Train Best Model
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Training Best Model")
print("=" * 60)

best_model = XGBClassifier(
    **best_params,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="auc",
    use_label_encoder=False
)

best_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# Evaluate
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n  Tuned Model Performance:")
print(f"  {'Metric':<15} {'Value':>10}")
print(f"  {'-'*27}")
print(f"  {'Accuracy':<15} {accuracy:>10.4f}")
print(f"  {'Precision':<15} {precision:>10.4f}")
print(f"  {'Recall':<15} {recall:>10.4f}")
print(f"  {'F1 Score':<15} {f1:>10.4f}")
print(f"  {'ROC AUC':<15} {roc_auc:>10.4f}")

# Compare with baseline
baseline_metrics_path = os.path.join(PROCESSED, "day9", "churn_metrics.csv")
if os.path.exists(baseline_metrics_path):
    baseline = pd.read_csv(baseline_metrics_path)
    baseline_auc = baseline[baseline["Metric"] == "ROC_AUC"]["Value"].iloc[0]
    improvement = roc_auc - baseline_auc
    print(f"\n  Baseline AUC: {baseline_auc:.4f}")
    print(f"  Tuned AUC:    {roc_auc:.4f}")
    print(f"  Improvement:  {improvement:+.4f}")

# ══════════════════════════════════════════════════════════
# STEP 4 – Feature Importance
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Feature Importance Analysis")
print("=" * 60)

importances = best_model.feature_importances_
feature_imp = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print(f"\n  Top 10 Important Features:")
print(f"  {'#':<4} {'Feature':<22} {'Importance':>12}")
print(f"  {'-'*40}")
for i, (_, row) in enumerate(feature_imp.head(10).iterrows()):
    print(f"  {i+1:<4} {row['Feature']:<22} {row['Importance']:>12.4f}")

# Feature Importance Plot
fig, ax = plt.subplots(figsize=(10, 7))
top10 = feature_imp.head(10).sort_values("Importance", ascending=True)

colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(top10)))
bars = ax.barh(
    top10["Feature"], top10["Importance"],
    color=colors, edgecolor="black", linewidth=0.5
)

# Add value labels
for bar, val in zip(bars, top10["Importance"]):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2.,
            f'{val:.4f}', ha='left', va='center', fontweight='bold', fontsize=10)

ax.set_title("Top 10 Feature Importances (XGBoost – Tuned)",
             fontsize=16, fontweight="bold")
ax.set_xlabel("Importance Score", fontsize=13)
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "feature_importance.png"), dpi=150)
plt.close()
print("\n  ✓ Saved: reports/day11/feature_importance.png")

# ══════════════════════════════════════════════════════════
# STEP 5 – Optuna Visualization
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Optuna Optimization Plots")
print("=" * 60)

# Optimization history (matplotlib version)
fig, ax = plt.subplots(figsize=(12, 5))
trials = range(1, len(study.trials) + 1)
values = [t.value for t in study.trials]
best_values = [max(values[:i+1]) for i in range(len(values))]

ax.scatter(trials, values, color="#3498db", alpha=0.5, s=30, label="Trial Score")
ax.plot(trials, best_values, color="#e74c3c", linewidth=2, label="Best Score")
ax.axhline(y=best_score, color="#27ae60", linestyle="--", alpha=0.7,
           label=f"Best: {best_score:.4f}")
ax.set_title("Optuna Optimization History", fontsize=16, fontweight="bold")
ax.set_xlabel("Trial Number", fontsize=13)
ax.set_ylabel("ROC AUC (CV)", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "optuna_optimization.png"), dpi=150)
plt.close()
print("  ✓ Saved: reports/day11/optuna_optimization.png")

# Parameter importance plot
fig, ax = plt.subplots(figsize=(10, 6))
param_imp = optuna.importance.get_param_importances(study)
params_sorted = dict(sorted(param_imp.items(), key=lambda x: x[1], reverse=False))
colors_param = plt.cm.viridis(np.linspace(0.3, 0.9, len(params_sorted)))
ax.barh(list(params_sorted.keys()), list(params_sorted.values()),
        color=colors_param, edgecolor="black", linewidth=0.5)
ax.set_title("Hyperparameter Importance", fontsize=16, fontweight="bold")
ax.set_xlabel("Importance", fontsize=13)
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "param_importance.png"), dpi=150)
plt.close()
print("  ✓ Saved: reports/day11/param_importance.png")

# ══════════════════════════════════════════════════════════
# STEP 6 – MLflow Logging
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Logging to MLflow")
print("=" * 60)

try:
    import mlflow
    mlflow.set_tracking_uri(f"file:///{os.path.abspath(MLRUNS)}")
    mlflow.set_experiment("RetailPulse_Churn")

    with mlflow.start_run(run_name="XGBoost_Churn_Tuned"):
        # Log best parameters
        for param, value in best_params.items():
            mlflow.log_param(param, value)

        mlflow.log_param("tuning_method", "Optuna")
        mlflow.log_param("n_trials", 50)
        mlflow.log_param("cv_folds", 3)

        # Log metrics
        mlflow.log_metric("Accuracy", round(accuracy, 4))
        mlflow.log_metric("Precision", round(precision, 4))
        mlflow.log_metric("Recall", round(recall, 4))
        mlflow.log_metric("F1_Score", round(f1, 4))
        mlflow.log_metric("ROC_AUC", round(roc_auc, 4))
        mlflow.log_metric("Best_CV_AUC", round(best_score, 4))

        # Log artifacts
        mlflow.log_artifact(os.path.join(REPORTS, "feature_importance.png"))
        mlflow.log_artifact(os.path.join(REPORTS, "optuna_optimization.png"))

    print("  ✓ Tuned model logged to MLflow")
except Exception as e:
    print(f"  ⚠ MLflow logging failed: {e}")

# ══════════════════════════════════════════════════════════
# STEP 7 – Save Best Model
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: Saving Best Model")
print("=" * 60)

with open(os.path.join(MODELS, "best_xgboost.pkl"), "wb") as f:
    pickle.dump(best_model, f)
print("  ✓ Saved: models/best_xgboost.pkl")

os.makedirs(os.path.join(PROCESSED, "day11"), exist_ok=True)
best_params_df.to_csv(os.path.join(PROCESSED, "day11", "best_params.csv"), index=False)
print("  ✓ Saved: data/processed/day11/best_params.csv")

# ══════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DAY 11 COMPLETE ✓")
print("=" * 60)
print(f"""
Deliverables:
  ✓ models/best_xgboost.pkl
  ✓ data/processed/best_params.csv
  ✓ reports/day11/feature_importance.png
  ✓ reports/day11/optuna_optimization.png
  ✓ reports/day11/param_importance.png

Optuna Tuning:
  Trials: 50
  Best CV AUC: {best_score:.4f}
  Test AUC:    {roc_auc:.4f}

Top 3 Features:
  1. {feature_imp.iloc[0]['Feature']}: {feature_imp.iloc[0]['Importance']:.4f}
  2. {feature_imp.iloc[1]['Feature']}: {feature_imp.iloc[1]['Importance']:.4f}
  3. {feature_imp.iloc[2]['Feature']}: {feature_imp.iloc[2]['Importance']:.4f}
""")
