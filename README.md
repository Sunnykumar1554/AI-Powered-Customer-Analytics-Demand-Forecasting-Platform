# RetailPulse 🛒📊

**AI-Powered Customer Analytics & Demand Forecasting Platform**

## 📌 Project Overview
RetailPulse is an end-to-end data science project that leverages machine learning and deep learning to provide actionable retail insights including customer segmentation, demand forecasting, churn prediction, and inventory optimization.

## 🛠 Technology Stack
- **Language**: Python 3.11
- **Data**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **ML/DL**: Scikit-Learn, Prophet, PyTorch
- **MLOps**: MLflow
- **Dashboard**: Streamlit
- **Deployment**: Docker, Kubernetes

## 📂 Project Structure
```
RetailPulse/
├── data/
│   ├── raw/              # Original dataset
│   └── processed/        # Cleaned & feature-engineered data
├── notebooks/            # Jupyter notebooks
├── src/                  # Python scripts (Day 1-7)
├── models/               # Trained models
├── reports/              # Visualizations & reports
├── mlruns/               # MLflow experiments
├── requirements.txt
└── README.md
```

## 📊 Dataset
**Online Retail II** – UCI Machine Learning Repository
- ~1M transactions from a UK-based online retailer
- Period: Dec 2009 – Dec 2011

## 🚀 Getting Started
```bash
pip install -r requirements.txt
cd src
python day1_eda.py
python day2_cleaning.py
python day3_segmentation.py
python day4_timeseries.py
python day5_prophet.py
python day6_lstm.py
python day7_mlflow.py
python day8_hybrid_forecast.py
python day9_churn_prediction.py
python day10_inventory_optimizer.py
python day11_tuning.py
python day12_drift_detection.py
python day13_retrain.py
python day14_week2_report.py
python day21_week3_report.py
```

## 📊 Running the Streamlit Dashboard
To run the full multi-page dashboard, execute:
```bash
streamlit run streamlit_app.py
```

## 📅 Day-Wise Project Roadmap

### 📅 Week 1 – Foundations & Modeling Baseline
*   **Day 1 – Exploratory Data Analysis (`day1_eda.py`):** Visualizes transactions, profile sales, and generate initial statistics reports.
*   **Day 2 – Data Cleaning (`day2_cleaning.py`):** Filters outliers, handles missing descriptors, and generates the baseline [cleaned_dataset.csv](file:///c:/Users/sunny/Desktop/Internship_2/RetailPulse/data/processed/day2/cleaned_dataset.csv).
*   **Day 3 – Customer Segmentation (`day3_segmentation.py`):** Performs K-Means & DBSCAN clustering on customer RFM metrics, saving [customer_segments.csv](file:///c:/Users/sunny/Desktop/Internship_2/RetailPulse/data/processed/day3/customer_segments.csv).
*   **Day 4 – Time Series Decomposition (`day4_timeseries.py`):** Evaluates overall seasonal components and trend components of daily sales.
*   **Day 5 – Prophet Forecasting (`day5_prophet.py`):** Builds a baseline Prophet forecasting model to predict sales for the next 30 days.
*   **Day 6 – LSTM Deep Learning (`day6_lstm.py`):** Implements a PyTorch LSTM sequence model to capture complex, non-linear sales behaviors.
*   **Day 7 – MLOps tracking (`day7_mlflow.py`):** Integrates MLflow tracking to log baseline model hyperparameter runs and metrics.

### 📅 Week 2 – Advanced Modeling & Drift Control
*   **Day 8 – Hybrid Forecast Ensemble (`day8_hybrid_forecast.py`):** Combines predictions (0.5 × Prophet + 0.5 × LSTM) to form the hybrid forecast (Test set MAPE: 24.08%).
*   **Day 9 – Churn Risk Classifier (`day9_churn_prediction.py`):** Builds an XGBoost binary classifier (target AUC >= 0.88 met) to identify customer attrition indicators.
*   **Day 10 – Inventory Safety Optimization (`day10_inventory_optimizer.py`):** Computes Safety Stocks and Reorder Points for all active SKUs.
*   **Day 11 – Hyperparameter Tuning (`day11_tuning.py`):** Utilizes Optuna to tune model parameters and generates feature importance benchmarks.
*   **Day 12 – Data Drift Analysis (`day12_drift_detection.py`):** Automatically monitors RFM dataset inputs using Evidently reports to detect feature drift.
*   **Day 13 – Retraining Pipeline (`day13_retrain.py`):** Automates run triggers, logging models to MLflow registries.
*   **Day 14 – Week 2 Briefing Report (`day14_week2_report.py`):** Compiles predictive performance KPIs across all modeling components.

### 📅 Week 3 – Streamlit Dashboard & Analytics Layer
*   **Day 15 – Executive Layout & Home View (`day15_home.py`):** Implements dashboard structure, executive KPI cards (revenue, orders, active customers), and real-time operations alert bars.
*   **Day 16 – Demand Forecasting Simulator (`day16_forecasting.py`):** Embeds Prophet vs LSTM vs Hybrid graphs and adds a What-If promo demand slider.
*   **Day 17 – Customer Clusters Drilldown (`day17_segmentation.py`):** Features interactive 3D RFM scatter plots and VIP customer search panels.
*   **Day 18 – Churn Diagnostics Panel (`day18_churn.py`):** Visualizes risk score distribution histograms alongside ROC and Confusion Matrix tabs.
*   **Day 19 – Safety Stock Reorder Actions (`day19_inventory.py`):** Groups catalog inventory into healthy, reorder, or critical status tables.
*   **Day 20 – PDF Reports Generator (`day20_reports.py`):** Integrates CSV data downloads and compiles dynamically formatted ReportLab briefing PDFs.
*   **Day 21 – Dashboard Validation Checkpoint (`day21_week3_report.py`):** Programmatically validates page health structure, package imports, and saves the final [week3_report.pdf](file:///c:/Users/sunny/Desktop/Internship_2/RetailPulse/reports/week3_report.pdf) report.
