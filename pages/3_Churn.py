import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

# ── Page Configuration ──
st.set_page_config(
    page_title="Churn Analysis | RetailPulse",
    page_icon="🎯",
    layout="wide"
)

# ── Dynamic Path Resolution ──
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(current_dir) == "pages":
    ROOT_DIR = os.path.abspath(os.path.join(current_dir, ".."))
else:
    ROOT_DIR = current_dir

CHURN_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "day9", "churn_predictions.csv")
METRICS_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "day9", "churn_metrics.csv")

CONF_MATRIX_IMG = os.path.join(ROOT_DIR, "reports", "day9", "confusion_matrix.png")
ROC_CURVE_IMG = os.path.join(ROOT_DIR, "reports", "day9", "roc_curve.png")
FEAT_IMPORTANCE_IMG = os.path.join(ROOT_DIR, "reports", "day11", "feature_importance.png")

# ── Custom CSS ──
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    .kpi-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #8a8f98;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
    }
    .kpi-subtext {
        font-size: 0.8rem;
        color: #ff2a5f;
        margin-top: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎯 Churn Analysis & Risk Management")
st.markdown("Monitor predicted customer churn probabilities and identify high-risk accounts for retention campaigns.")

# ── Load Datasets ──
@st.cache_data
def load_churn_data():
    if not os.path.exists(CHURN_DATA_PATH):
        return None
    return pd.read_csv(CHURN_DATA_PATH)

@st.cache_data
def load_metrics():
    if not os.path.exists(METRICS_DATA_PATH):
        return None
    return pd.read_csv(METRICS_DATA_PATH)

df_churn = load_churn_data()
df_metrics = load_metrics()

if df_churn is not None:
    # Calculations
    total_customers = len(df_churn)
    actual_churn_rate = (df_churn["Churn_Actual"].sum() / total_customers) * 100
    predicted_churn_rate = (df_churn["Churn_Predicted"].sum() / total_customers) * 100
    at_risk_customers = len(df_churn[df_churn["Churn_Probability"] > 0.75])
    
    # ── KPIs Row ──
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Historical Churn Rate</div>
                <div class='kpi-value'>{actual_churn_rate:.2f}%</div>
                <div class='kpi-subtext' style='color: #8a8f98;'>Baseline customer turnover</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col2:
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Predicted Churn Rate</div>
                <div class='kpi-value'>{predicted_churn_rate:.2f}%</div>
                <div class='kpi-subtext'>Based on XGBoost predictions</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col3:
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-label'>High-Risk Customers (>75%)</div>
                <div class='kpi-value'>{at_risk_customers:,}</div>
                <div class='kpi-subtext'>Immediate intervention recommended</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ── Visualizations Row ──
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.markdown("### 📊 Churn Probability Distribution")
        
        # Plotly histogram of Churn_Probability
        fig_hist = px.histogram(
            df_churn, 
            x="Churn_Probability", 
            nbins=30,
            title="Density of Customer Churn Risk Scores",
            color_discrete_sequence=["#ff2a5f"],
            template="plotly_dark",
            labels={"Churn_Probability": "Churn Probability Score"}
        )
        fig_hist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            title_font=dict(size=14, family="Outfit"),
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col_right:
        st.markdown("### 🎯 At-Risk Customers List (Top 50)")
        df_at_risk = df_churn[df_churn["Churn_Probability"] > 0.75].sort_values("Churn_Probability", ascending=False).head(50)
        df_at_risk.columns = ["Customer ID", "Actual Churn", "Predicted Churn", "Risk Score"]
        
        st.dataframe(
            df_at_risk.style.format({
                "Risk Score": "{:.2%}"
            }),
            use_container_width=True,
            hide_index=True
        )
        
    # ── Model Diagnostics & Explainability Tabs ──
    st.write("---")
    st.markdown("### 🔍 Model Performance & Feature Explainability")
    
    tab1, tab2, tab3 = st.tabs(["💡 Churn Drivers (Feature Importance)", "📉 ROC Curve", "📋 Confusion Matrix"])
    
    with tab1:
        st.markdown("#### Feature Importance Profile")
        st.write("Shows which customer behaviors have the most significant impact on predicted churn probability.")
        if os.path.exists(FEAT_IMPORTANCE_IMG):
            st.image(FEAT_IMPORTANCE_IMG, use_container_width=True)
        else:
            st.info("Feature importance plot not found. Run tuning script to generate.")
            
    with tab2:
        st.markdown("#### ROC Curve")
        st.write("Receiver Operating Characteristic (ROC) measures the model's ability to distinguish between churners and loyalists.")
        if os.path.exists(ROC_CURVE_IMG):
            st.image(ROC_CURVE_IMG, width=700)
        else:
            st.info("ROC Curve plot not found.")
            
    with tab3:
        st.markdown("#### Confusion Matrix")
        st.write("Confusion matrix comparing model predictions with actual outcomes on the test set.")
        if os.path.exists(CONF_MATRIX_IMG):
            st.image(CONF_MATRIX_IMG, width=600)
        else:
            st.info("Confusion matrix plot not found.")
            
else:
    st.warning("⚠ Churn prediction data not found. Please verify churn prediction scripts are executed under data/processed/day9/.")
