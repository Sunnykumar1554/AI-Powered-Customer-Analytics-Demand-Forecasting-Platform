import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

# ── Page Configuration ──
st.set_page_config(
    page_title="Demand Forecasting | RetailPulse",
    page_icon="📈",
    layout="wide"
)

# ── Dynamic Path Resolution ──
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(current_dir) in ["pages", "src"]:
    ROOT_DIR = os.path.abspath(os.path.join(current_dir, ".."))
else:
    ROOT_DIR = current_dir

FORECAST_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "day8", "hybrid_forecast.csv")
METRICS_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "day8", "forecast_metrics.csv")

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
        color: #00f5d4;
        margin-top: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📈 Demand Forecasting & What-If Simulation")
st.markdown("Compare model predictions and simulate the business impact of demand shifts.")

# ── Load Datasets ──
@st.cache_data
def load_forecast_data():
    if not os.path.exists(FORECAST_DATA_PATH):
        return None
    df = pd.read_csv(FORECAST_DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_data
def load_metrics():
    if not os.path.exists(METRICS_DATA_PATH):
        return None
    return pd.read_csv(METRICS_DATA_PATH)

df_forecast = load_forecast_data()
df_metrics = load_metrics()

if df_forecast is not None:
    # ── Sidebar Control Panel ──
    with st.sidebar:
        st.markdown("### 🛠 What-If Demand Controls")
        st.write("Adjust the projected demand to simulate the impact of upcoming sales promotions or supply events.")
        demand_adj = st.slider("Projected Demand Adjustment (%)", min_value=0, max_value=50, value=0, step=5)
        st.write("---")
        if df_metrics is not None:
            st.markdown("### 🏆 Model Comparison")
            st.write("Test set validation metrics:")
            st.dataframe(df_metrics, hide_index=True)
    
    # Calculate base KPIs
    base_forecast_revenue = df_forecast["Hybrid_Prediction"].sum()
    base_actual_revenue = df_forecast["Actual"].sum()
    avg_daily_forecast = df_forecast["Hybrid_Prediction"].mean()
    
    # Calculate adjusted KPIs
    multiplier = 1 + (demand_adj / 100.0)
    adj_forecast_revenue = base_forecast_revenue * multiplier
    incremental_revenue = adj_forecast_revenue - base_forecast_revenue
    adj_daily_forecast = avg_daily_forecast * multiplier
    
    # Create copies for plotting
    df_plot = df_forecast.copy()
    df_plot["Adjusted_Hybrid_Prediction"] = df_plot["Hybrid_Prediction"] * multiplier
    
    # ── KPIs Row ──
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Total Forecast Revenue (30-day)</div>
                <div class='kpi-value'>${base_forecast_revenue:,.2f}</div>
                <div class='kpi-subtext' style='color: #8a8f98;'>Ensemble baseline estimate</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col2:
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Adjusted Forecast Revenue</div>
                <div class='kpi-value'>${adj_forecast_revenue:,.2f}</div>
                <div class='kpi-subtext'>▲ +{demand_adj}% simulated demand increase</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col3:
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Incremental Simulated Gain</div>
                <div class='kpi-value'>+${incremental_revenue:,.2f}</div>
                <div class='kpi-subtext' style='color: #00c6ff;'>Value added by promotions</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # ── Forecast Chart ──
    st.markdown("### 📊 Interactive Forecast Comparison")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_plot["Date"], y=df_plot["Actual"],
        name="Actual Sales",
        line=dict(color="#ffffff", width=2, dash="dash"),
        mode="lines+markers"
    ))
    
    fig.add_trace(go.Scatter(
        x=df_plot["Date"], y=df_plot["Prophet_Prediction"],
        name="Prophet Forecast",
        line=dict(color="#ff9f1c", width=1.5),
        opacity=0.6
    ))
    
    fig.add_trace(go.Scatter(
        x=df_plot["Date"], y=df_plot["LSTM_Prediction"],
        name="LSTM Forecast",
        line=dict(color="#9d4edd", width=1.5),
        opacity=0.6
    ))
    
    fig.add_trace(go.Scatter(
        x=df_plot["Date"], y=df_plot["Hybrid_Prediction"],
        name="Hybrid Ensemble Forecast",
        line=dict(color="#00c6ff", width=3)
    ))
    
    if demand_adj > 0:
        fig.add_trace(go.Scatter(
            x=df_plot["Date"], y=df_plot["Adjusted_Hybrid_Prediction"],
            name=f"Adjusted Forecast (+{demand_adj}%)",
            line=dict(color="#00f5d4", width=2.5, dash="dot")
        ))
        
    fig.update_layout(
        title="Actual vs Predicted Sales Revenue",
        xaxis_title="Date",
        yaxis_title="Revenue ($)",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        margin=dict(l=40, r=40, t=65, b=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ── Detail Table ──
    st.markdown("### 📋 Forecast Schedule Details")
    df_present = df_plot.copy()
    df_present["Date"] = df_present["Date"].dt.strftime("%Y-%m-%d")
    df_present.columns = ["Date", "Actual Sales ($)", "Prophet Pred ($)", "LSTM Pred ($)", "Hybrid Pred ($)", "Adjusted Hybrid Pred ($)"]
    st.dataframe(df_present, use_container_width=True, hide_index=True)
    
else:
    st.warning("⚠ Forecast data not found. Please verify hybrid forecast results are generated under data/processed/day8/.")
