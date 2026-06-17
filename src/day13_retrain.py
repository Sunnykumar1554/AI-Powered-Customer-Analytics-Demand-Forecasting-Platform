"""
RetailPulse – Day 13: Automated Retraining Pipeline
=====================================================
Goal: Lightweight retraining pipeline for the churn prediction model.
      Pipeline: Load → Validate → Feature Engineer → Train → Evaluate → Log → Save
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score
)
from xgboost import XGBClassifier
import pickle
import os
import sys
import logging
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────
PROCESSED = os.path.join("..", "data", "processed")
MODELS    = os.path.join("..", "models")
REPORTS   = os.path.join("..", "reports", "day13")
MLRUNS    = os.path.join("..", "mlruns")
os.makedirs(REPORTS, exist_ok=True)

# ── Logging Setup ────────────────────────────────────────
LOG_FILE = os.path.join(REPORTS, "pipeline_log.txt")

# Set up dual logging (console + file)
logger = logging.getLogger("RetailPulse_Pipeline")
logger.setLevel(logging.INFO)

# Clear existing handlers
logger.handlers = []

# File handler
fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
fh.setLevel(logging.INFO)

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)


# ══════════════════════════════════════════════════════════
# PIPELINE FUNCTIONS
# ══════════════════════════════════════════════════════════

def load_data(data_path):
    """
    Load the cleaned dataset.

    Args:
        data_path: Path to cleaned_dataset.csv

    Returns:
        DataFrame with loaded data
    """
    logger.info("=" * 50)
    logger.info("STAGE 1: LOADING DATA")
    logger.info("=" * 50)

    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    logger.info(f"Loaded {len(df):,} transactions")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Date range: {df['InvoiceDate'].min()} – {df['InvoiceDate'].max()}")

    return df


def validate_data(df):
    """
    Validate data quality and integrity.

    Args:
        df: DataFrame to validate

    Returns:
        Validated DataFrame (bad rows removed)
    """
    logger.info("=" * 50)
    logger.info("STAGE 2: VALIDATING DATA")
    logger.info("=" * 50)

    initial_rows = len(df)
    issues = []

    # Check required columns
    required_cols = ["Invoice", "StockCode", "Quantity", "InvoiceDate",
                     "Price", "Customer ID", "Revenue"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"Missing columns: {missing_cols}")

    # Check for nulls in critical columns
    null_counts = df[required_cols].isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            issues.append(f"{col}: {count:,} null values")

    # Remove rows with null Customer ID
    df = df.dropna(subset=["Customer ID"])

    # Remove negative quantities and prices
    df = df[df["Quantity"] > 0]
    df = df[df["Price"] > 0]

    removed = initial_rows - len(df)
    logger.info(f"Initial rows: {initial_rows:,}")
    logger.info(f"Removed rows: {removed:,}")
    logger.info(f"Clean rows:   {len(df):,}")

    if issues:
        for issue in issues:
            logger.warning(f"Data issue: {issue}")

    logger.info("✓ Data validation passed")
    return df


def clean_data(df):
    """
    Apply additional cleaning rules.

    Args:
        df: Validated DataFrame

    Returns:
        Cleaned DataFrame
    """
    logger.info("=" * 50)
    logger.info("STAGE 3: CLEANING DATA")
    logger.info("=" * 50)

    # Ensure Revenue column exists
    if "Revenue" not in df.columns:
        df["Revenue"] = df["Quantity"] * df["Price"]
        logger.info("Created Revenue column")

    # Convert Customer ID to int
    df["Customer ID"] = df["Customer ID"].astype(int)

    logger.info(f"Customers: {df['Customer ID'].nunique():,}")
    logger.info(f"Products:  {df['StockCode'].nunique():,}")
    logger.info("✓ Data cleaning complete")

    return df


def create_features(df):
    """
    Create churn prediction features.

    Args:
        df: Cleaned DataFrame

    Returns:
        X (features), y (churn labels)
    """
    logger.info("=" * 50)
    logger.info("STAGE 4: FEATURE ENGINEERING")
    logger.info("=" * 50)

    snapshot_date = df["InvoiceDate"].max()
    logger.info(f"Snapshot date: {snapshot_date}")

    # Core features
    last_purchase = df.groupby("Customer ID")["InvoiceDate"].max()
    days_since_last = (snapshot_date - last_purchase).dt.days

    # Churn label
    CHURN_THRESHOLD = 90
    churn = (days_since_last > CHURN_THRESHOLD).astype(int)

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
    }).dropna()

    X = features.drop("Churn", axis=1)
    y = features["Churn"]

    logger.info(f"Features created: {X.shape[1]}")
    logger.info(f"Samples: {len(X):,}")
    logger.info(f"Churn rate: {y.mean()*100:.1f}%")
    logger.info("✓ Feature engineering complete")

    return X, y


def train_model(X_train, y_train):
    """
    Train XGBoost model with best parameters from Day 11.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        Trained model
    """
    logger.info("=" * 50)
    logger.info("STAGE 5: TRAINING MODEL")
    logger.info("=" * 50)

    # Try to load best params from Day 11
    best_params_path = os.path.join(PROCESSED, "day11", "best_params.csv")
    if os.path.exists(best_params_path):
        params_df = pd.read_csv(best_params_path)
        params = {}
        for _, row in params_df.iterrows():
            key = row["Parameter"]
            val = row["Value"]
            # Type conversion
            if key in ["max_depth", "n_estimators", "min_child_weight"]:
                params[key] = int(float(val))
            else:
                params[key] = float(val)
        logger.info(f"Loaded best params from Day 11: {params}")
    else:
        params = {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 300,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
        logger.info(f"Using default params: {params}")

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="auc",
        use_label_encoder=False
    )

    model.fit(X_train, y_train, verbose=False)
    logger.info("✓ Model training complete")

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary of metrics
    """
    logger.info("=" * 50)
    logger.info("STAGE 6: EVALUATING MODEL")
    logger.info("=" * 50)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1_Score": round(f1_score(y_test, y_pred), 4),
        "ROC_AUC": round(roc_auc_score(y_test, y_pred_proba), 4)
    }

    logger.info("Model Performance:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value}")

    if metrics["ROC_AUC"] >= 0.88:
        logger.info(f"✓ AUC target met ({metrics['ROC_AUC']} ≥ 0.88)")
    else:
        logger.warning(f"⚠ AUC below target ({metrics['ROC_AUC']} < 0.88)")

    return metrics


def save_model(model, metrics):
    """
    Save model and log to MLflow.

    Args:
        model: Trained model
        metrics: Performance metrics dictionary
    """
    logger.info("=" * 50)
    logger.info("STAGE 7: SAVING MODEL & LOGGING")
    logger.info("=" * 50)

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODELS, "churn_model_retrained.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"✓ Saved model: {model_path}")

    # Log to MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri(f"file:///{os.path.abspath(MLRUNS)}")
        mlflow.set_experiment("RetailPulse_Churn")

        with mlflow.start_run(run_name=f"Retrain_{timestamp}"):
            mlflow.log_param("pipeline_version", "1.0")
            mlflow.log_param("retrain_timestamp", timestamp)

            for metric, value in metrics.items():
                mlflow.log_metric(metric, value)

            mlflow.log_artifact(model_path)
            mlflow.log_artifact(LOG_FILE)

        logger.info("✓ Logged to MLflow")
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")

    logger.info("✓ Pipeline complete")


# ══════════════════════════════════════════════════════════
# MAIN PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════
def run_pipeline():
    """
    Execute the full retraining pipeline.
    """
    start_time = datetime.now()

    logger.info("=" * 60)
    logger.info("RETAILPULSE – AUTOMATED RETRAINING PIPELINE")
    logger.info(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    try:
        # Stage 1: Load
        data_path = os.path.join(PROCESSED, "day2", "cleaned_dataset.csv")
        df = load_data(data_path)

        # Stage 2: Validate
        df = validate_data(df)

        # Stage 3: Clean
        df = clean_data(df)

        # Stage 4: Feature Engineering
        X, y = create_features(df)

        # Stage 5: Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

        # Stage 6: Train
        model = train_model(X_train, y_train)

        # Stage 7: Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        # Stage 8: Save & Log
        save_model(model, metrics)

        # Pipeline timing
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info("")
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY ✓")
        logger.info(f"Duration: {elapsed:.1f} seconds")
        logger.info("=" * 60)

        return model, metrics

    except Exception as e:
        logger.error(f"Pipeline FAILED: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


# ══════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("RetailPulse – Day 13: Automated Retraining Pipeline")
    print("=" * 60)

    model, metrics = run_pipeline()

    print(f"\n  ✓ Pipeline log saved: reports/day13/pipeline_log.txt")
    print(f"  ✓ Retrained model saved: models/churn_model_retrained.pkl")
    print(f"\n  To schedule automated retraining:")
    print(f"    python retrain.py")
    print(f"    # or set up a cron job / Windows Task Scheduler")

    print("\n" + "=" * 60)
    print("DAY 13 COMPLETE ✓")
    print("=" * 60)
    print(f"""
Deliverables:
  ✓ src/day13_retrain.py (this pipeline)
  ✓ reports/day13/pipeline_log.txt
  ✓ models/churn_model_retrained.pkl

Pipeline Stages:
  1. Load Data
  2. Validate Data
  3. Clean Data
  4. Feature Engineering
  5. Train Model
  6. Evaluate Model
  7. Save & Log to MLflow
""")
