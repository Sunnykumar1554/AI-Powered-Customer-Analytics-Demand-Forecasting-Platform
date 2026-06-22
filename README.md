# RetailPulse 🛒📊

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.14-0194E2?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-00C853?style=flat)](LICENSE)

**AI-Powered Customer Analytics & Demand Forecasting Platform**

> An end-to-end data science project combining machine learning, deep learning, and MLOps to deliver actionable retail insights through an executive-grade Streamlit dashboard.

---

## 📌 Project Overview

RetailPulse is a production-ready analytics platform that processes **1M+ retail transactions** to provide:

- **Demand Forecasting** — Hybrid Prophet + LSTM ensemble for accurate sales prediction
- **Customer Segmentation** — RFM-based K-Means & DBSCAN clustering
- **Churn Prediction** — XGBoost classifier with ROC AUC ≥ 0.94
- **Inventory Optimization** — Automated safety stock & reorder point calculations
- **Data Drift Detection** — Evidently-based feature drift monitoring
- **MLOps Pipeline** — Full MLflow experiment tracking with automated retraining

Built as part of the **Zidio Development Internship Program**.

---

## ✨ Key Features

| Module | Description | Technology |
|--------|-------------|------------|
| 📈 **Forecasting** | 30-day sales prediction with what-if promo simulation | Prophet, LSTM, Hybrid Ensemble |
| 👥 **Segmentation** | Interactive 3D RFM scatter plots & VIP customer search | K-Means, DBSCAN |
| 🔮 **Churn** | Risk score distributions, ROC curves, confusion matrix | XGBoost, Optuna |
| 📦 **Inventory** | Real-time stock alerts with healthy/reorder/critical status | Safety Stock Model |
| 🔍 **Drift Detection** | Automated RFM feature distribution monitoring | Evidently |
| ⚙️ **MLOps** | Experiment tracking, model registry, retraining pipeline | MLflow |
| 📊 **Dashboard** | Executive KPI portal with glassmorphism dark theme | Streamlit, Plotly |
| 📡 **Monitoring** | Application metrics with Prometheus + Grafana | prometheus-client |

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                         │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐ │
│  │  Streamlit   │  │   Netlify   │  │    Grafana           │ │
│  │  Dashboard   │  │  Landing    │  │    Monitoring        │ │
│  │  (5 Pages)   │  │  Page       │  │    Dashboard         │ │
│  └──────┬───────┘  └─────────────┘  └──────────┬───────────┘ │
├─────────┼──────────────────────────────────────┼─────────────┤
│         │           ANALYTICS LAYER            │             │
│  ┌──────▼───────────────────────────────────────▼──────────┐ │
│  │  Prophet │ LSTM │ XGBoost │ K-Means │ Safety Stock     │ │
│  │  Hybrid Ensemble │ Optuna Tuning │ Evidently Drift    │ │
│  └──────┬────────────────────────────────────────┬────────┘ │
├─────────┼────────────────────────────────────────┼──────────┤
│         │          OPERATIONS LAYER              │          │
│  ┌──────▼──────┐  ┌──────────────┐  ┌────────────▼───────┐ │
│  │   MLflow    │  │   Docker     │  │   Prometheus       │ │
│  │   Tracking  │  │   Container  │  │   Metrics          │ │
│  └─────────────┘  └──────────────┘  └────────────────────┘ │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │ Kubernetes  │  │ GitHub       │  │   Locust           │ │
│  │ Deployment  │  │ Actions CI   │  │   Load Testing     │ │
│  └─────────────┘  └──────────────┘  └────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                      DATA LAYER                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Online Retail II Dataset (UCI) — ~1M Transactions     │ │
│  │  Dec 2009 – Dec 2011 │ UK-based Online Retailer        │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
RetailPulse/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI/CD pipeline
├── data/
│   ├── raw/                       # Original UCI dataset
│   └── processed/                 # Cleaned & feature-engineered data
├── k8s/
│   ├── deployment.yaml            # Kubernetes Deployment (2 replicas)
│   └── service.yaml               # Kubernetes LoadBalancer Service
├── landing/
│   ├── index.html                 # Netlify landing page
│   └── style.css                  # Landing page styles
├── models/
│   ├── prophet_model.pkl          # Prophet forecasting model
│   ├── lstm_model.pth             # PyTorch LSTM model
│   ├── churn_model.pkl            # Churn prediction model
│   ├── best_xgboost.pkl           # Tuned XGBoost model
│   └── hybrid_model.pkl           # Hybrid ensemble config
├── monitoring/
│   ├── grafana_dashboard.json     # Grafana dashboard config
│   └── prometheus.yml             # Prometheus scrape config
├── pages/
│   ├── 1_Forecasting.py           # Demand forecasting simulator
│   ├── 2_Segmentation.py          # Customer clusters drilldown
│   ├── 3_Churn.py                 # Churn diagnostics panel
│   ├── 4_Inventory.py             # Safety stock reorder actions
│   └── 5_Reports.py               # PDF report generator
├── reports/
│   ├── final_metrics.csv          # Consolidated model metrics
│   └── week3_report.pdf           # Week 3 validation report
├── src/
│   ├── day1_eda.py                # Exploratory Data Analysis
│   ├── day2_cleaning.py           # Data cleaning pipeline
│   ├── day3_segmentation.py       # Customer segmentation
│   ├── day4_timeseries.py         # Time series decomposition
│   ├── day5_prophet.py            # Prophet forecasting
│   ├── day6_lstm.py               # LSTM deep learning
│   ├── day7_mlflow.py             # MLflow tracking
│   ├── day8_hybrid_forecast.py    # Hybrid ensemble
│   ├── day9_churn_prediction.py   # Churn classifier
│   ├── day10_inventory_optimizer.py # Inventory optimization
│   ├── day11_tuning.py            # Hyperparameter tuning
│   ├── day12_drift_detection.py   # Data drift analysis
│   ├── day13_retrain.py           # Retraining pipeline
│   ├── day14_week2_report.py      # Week 2 report
│   ├── day15_home.py → day21_week3_report.py  # Dashboard scripts
│   └── metrics.py                 # Prometheus metrics module
├── tests/
│   └── test_app.py                # Smoke tests
├── .dockerignore
├── Dockerfile                     # Production container
├── locustfile.py                  # Load testing config
├── requirements.txt               # Python dependencies
├── streamlit_app.py               # Main dashboard entry point
└── README.md
```

---

## 📊 Dataset

**Online Retail II** — [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II)

| Property | Value |
|----------|-------|
| Transactions | ~1,000,000 |
| Period | December 2009 – December 2011 |
| Source | UK-based online retailer |
| Features | Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Docker (optional, for containerized deployment)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/Sunnykumar1554/AI-Powered-Customer-Analytics-Demand-Forecasting-Platform.git
cd AI-Powered-Customer-Analytics-Demand-Forecasting-Platform

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Data Pipeline (Weeks 1–2)

```bash
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

### Launch Dashboard

```bash
streamlit run streamlit_app.py
```

---

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t retailpulse .

# Run the container
docker run -p 8501:8501 retailpulse

# Access at http://localhost:8501
```

### Kubernetes Deployment

```bash
# Apply manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Check status
kubectl get pods -l app=retailpulse
kubectl get svc retailpulse-service
```

---

## 📡 Monitoring

### Prometheus Metrics

```bash
# Start the metrics server (in your app or standalone)
python src/metrics.py
# Metrics available at http://localhost:8000/metrics
```

### Grafana Dashboard

Import `monitoring/grafana_dashboard.json` into your Grafana instance to visualize:
- Request rates per page
- Model inference latency (p50, p95, p99)
- Forecast model usage breakdown
- Churn prediction volume

### Load Testing

```bash
# Run Locust load tests
locust -f locustfile.py --host http://localhost:8501

# Open http://localhost:8089 to configure and start the test
```

---

## 📈 Model Performance Results

### Demand Forecasting

| Model | MAPE |
|-------|------|
| Prophet | 28.3% |
| LSTM | 26.1% |
| **Hybrid Ensemble** | **24.1%** |

### Churn Prediction

| Metric | Value |
|--------|-------|
| Accuracy | 91.2% |
| Precision | 89.4% |
| Recall | 87.8% |
| **ROC AUC** | **0.94** |

### Customer Segmentation

| Algorithm | Silhouette Score |
|-----------|-----------------|
| K-Means | 0.62 |
| DBSCAN | 0.55 |

---

## 🛠 Technology Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.11 |
| **Data** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **ML/DL** | Scikit-Learn, Prophet, PyTorch, XGBoost |
| **Optimization** | Optuna |
| **Drift Detection** | Evidently |
| **MLOps** | MLflow |
| **Dashboard** | Streamlit |
| **Containerization** | Docker |
| **Orchestration** | Kubernetes |
| **CI/CD** | GitHub Actions |
| **Monitoring** | Prometheus, Grafana |
| **Load Testing** | Locust |
| **Hosting** | Streamlit Cloud, Netlify |

---

## 🌐 Live Demo

| Service | URL |
|---------|-----|
| 📊 **Dashboard** | [retailpulse-9q8k.onrender.com](https://retailpulse-9q8k.onrender.com) |
| 🌐 **Landing Page** | [retailpulse.netlify.app](https://retailpulse.netlify.app) |
| 💻 **Repository** | [GitHub](https://github.com/Sunnykumar1554/AI-Powered-Customer-Analytics-Demand-Forecasting-Platform) |

---

## 📅 Development Roadmap

### Week 1 — Foundations & Modeling Baseline
Days 1–7: EDA → Data Cleaning → Segmentation → Time Series → Prophet → LSTM → MLflow

### Week 2 — Advanced Modeling & Drift Control
Days 8–14: Hybrid Forecast → Churn Prediction → Inventory Optimization → Tuning → Drift Detection → Retraining → Report

### Week 3 — Streamlit Dashboard & Analytics Layer
Days 15–21: Executive Home → Forecasting → Segmentation → Churn → Inventory → Reports → Validation

### Week 4 — Deployment & Production Polish
Days 22–28: Docker → Kubernetes → CI/CD → Cloud Deploy → Monitoring → Load Testing → Final Documentation

---

## 🔮 Future Work

- [ ] Real-time streaming data ingestion with Apache Kafka
- [ ] A/B testing framework for marketing campaign optimization
- [ ] Multi-store support with geographic analytics
- [ ] Advanced NLP for product description analysis
- [ ] Automated anomaly detection in sales patterns
- [ ] Mobile-responsive PWA dashboard

---

## 👤 Author

**Sunny Kumar**

Zidio Development Internship Project — 2025

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
