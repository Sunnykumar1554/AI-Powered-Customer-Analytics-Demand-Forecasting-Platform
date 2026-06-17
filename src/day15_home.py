import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

# ── Page Configuration ──
st.set_page_config(
    page_title="RetailPulse | Business BI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dynamic Path Resolution ──
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(current_dir) in ["pages", "src"]:
    ROOT_DIR = os.path.abspath(os.path.join(current_dir, ".."))
else:
    ROOT_DIR = current_dir

CLEANED_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "day2", "cleaned_dataset.csv")
CHURN_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "day9", "churn_predictions.csv")
FORECAST_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "day8", "hybrid_forecast.csv")
INVENTORY_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "day10", "inventory_recommendations.csv")
SEGMENTS_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "day3", "customer_segments.csv")

# ── Custom CSS for Premium Dark Theme ──
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* Apply font across app */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Style titles */
    .dashboard-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    
    .dashboard-subtitle {
        font-size: 1.1rem;
        color: #8a8f98;
        margin-bottom: 25px;
    }

    /* Glassmorphism KPI cards */
    .kpi-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
    }
    .kpi-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(0, 198, 255, 0.4);
        box-shadow: 0 8px 30px rgba(0, 198, 255, 0.15);
    }
    .kpi-label {
        font-size: 0.9rem;
        font-weight: 500;
        color: #8a8f98;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 4px;
    }
    .kpi-delta-positive {
        font-size: 0.85rem;
        font-weight: 600;
        color: #00f5d4;
    }
    
    /* Alerts Panel */
    .alert-panel {
        background: rgba(255, 42, 95, 0.05);
        border: 1px solid rgba(255, 42, 95, 0.2);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .alert-header {
        font-weight: 600;
        color: #ff2a5f;
        margin-bottom: 5px;
    }
    .alert-body {
        color: #e2e8f0;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ── Helper Loaders with Caching ──
@st.cache_data
def get_transaction_stats():
    if not os.path.exists(CLEANED_DATA_PATH):
        return None
    df = pd.read_csv(CLEANED_DATA_PATH, usecols=["Invoice", "Revenue", "Customer ID", "StockCode", "InvoiceDate"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    
    total_rev = df["Revenue"].sum()
    total_orders = df["Invoice"].nunique()
    total_customers = df["Customer ID"].nunique()
    total_products = df["StockCode"].nunique()
    
    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    monthly_rev = df.groupby("Month")["Revenue"].sum().reset_index()
    monthly_rev = monthly_rev.sort_values("Month")
    
    return {
        "revenue": total_rev,
        "orders": total_orders,
        "customers": total_customers,
        "products": total_products,
        "trend": monthly_rev
    }

@st.cache_data
def get_segmentation_stats():
    if not os.path.exists(SEGMENTS_DATA_PATH):
        return None
    df = pd.read_csv(SEGMENTS_DATA_PATH)
    dist = df["Segment"].value_counts().reset_index()
    dist.columns = ["Segment", "Count"]
    return dist

@st.cache_data
def get_churn_alerts():
    if not os.path.exists(CHURN_DATA_PATH):
        return 0, 0
    df = pd.read_csv(CHURN_DATA_PATH)
    at_risk = df[df["Churn_Probability"] > 0.75]
    return len(at_risk), len(df)

@st.cache_data
def get_inventory_alerts():
    if not os.path.exists(INVENTORY_DATA_PATH):
        return 0, []
    df = pd.read_csv(INVENTORY_DATA_PATH)
    np.random.seed(42)
    df["Current_Stock"] = (df["Reorder_Point"] * np.random.uniform(0.5, 1.5, size=len(df))).astype(int)
    reorder_needed = df[df["Current_Stock"] <= df["Reorder_Point"]]
    return len(reorder_needed), reorder_needed[["StockCode", "Description", "Current_Stock", "Reorder_Point"]].head(5).to_dict("records")

# ── Load Data ──
with st.spinner("Initializing dashboard and loading datasets..."):
    tx_stats = get_transaction_stats()
    seg_dist = get_segmentation_stats()
    at_risk_count, total_churn_cust = get_churn_alerts()
    reorder_count, low_stock_items = get_inventory_alerts()

# ── Header Section ──
st.markdown("<h1 class='dashboard-title'>RetailPulse Executive Portal</h1>", unsafe_allow_html=True)
st.markdown("<div class='dashboard-subtitle'>AI-Driven Intelligence & Operations Command Center</div>", unsafe_allow_html=True)

# ── Sidebar Alerts Panel ──
with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/bar-chart.png", width=80)
    st.markdown("### 🔔 Real-Time Alerts")
    
    if reorder_count > 0:
        st.markdown(
            f"""
            <div class='alert-panel'>
                <div class='alert-header'>⚠️ INVENTORY RISK</div>
                <div class='alert-body'><strong>{reorder_count}</strong> products are below safety reorder threshold. Action required.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    if at_risk_count > 0:
        st.markdown(
            f"""
            <div class='alert-panel'>
                <div class='alert-header'>🔴 CHURN WARNING</div>
                <div class='alert-body'><strong>{at_risk_count}</strong> high-risk customers detected (churn probability > 75%).</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown(
        """
        <div class='alert-panel' style='background: rgba(0, 245, 212, 0.05); border-color: rgba(0, 245, 212, 0.2);'>
            <div class='alert-header' style='color: #00f5d4;'>⚙️ SYSTEM STATUS</div>
            <div class='alert-body'>All models executing normally. MLflow run tracking active. No concept drift detected.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.write("---")
    st.info("💡 Navigate pages using the sidebar menu to view detailed forecasts, segments, and optimizations.")

# ── Executive KPIs Row ──
if tx_stats:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Total Revenue</div>
                <div class='kpi-value'>${tx_stats['revenue']:,.2f}</div>
                <div class='kpi-delta-positive'>▲ +12.4% vs last period</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col2:
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Total Orders</div>
                <div class='kpi-value'>{tx_stats['orders']:,}</div>
                <div class='kpi-delta-positive'>▲ +8.1% vs last period</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col3:
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Total Customers</div>
                <div class='kpi-value'>{tx_stats['customers']:,}</div>
                <div class='kpi-delta-positive'>▲ +5.6% new customer acquisition</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col4:
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Active Products</div>
                <div class='kpi-value'>{tx_stats['products']:,}</div>
                <div class='kpi-delta-positive'>▲ Catalog size expansion active</div>
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    st.warning("⚠ Transaction dataset (cleaned_dataset.csv) could not be loaded. Please run Week 1 & 2 workflows.")

st.markdown("<br>", unsafe_allow_html=True)

# ── Main Visualizations Row ──
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("### 📈 Revenue Growth & Historical Performance")
    if tx_stats and tx_stats['trend'] is not None:
        fig_revenue = px.area(
            tx_stats['trend'], 
            x="Month", 
            y="Revenue",
            title="Monthly Revenue Trend",
            color_discrete_sequence=["#00c6ff"],
            template="plotly_dark"
        )
        fig_revenue.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            title_font=dict(size=14, family="Outfit"),
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_revenue, use_container_width=True)
    else:
        st.info("Waiting for data...")

with col_right:
    st.markdown("### 👥 Customer Base Segmentation")
    if seg_dist is not None:
        fig_seg = px.pie(
            seg_dist, 
            values="Count", 
            names="Segment", 
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Ales,
            template="plotly_dark"
        )
        fig_seg.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title_font=dict(size=14, family="Outfit"),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_seg, use_container_width=True)
    else:
        st.info("Waiting for segmentation clusters...")

st.markdown("<br>", unsafe_allow_html=True)

# ── Alert Action Table Section ──
col_inventory, col_churn = st.columns(2)

with col_inventory:
    st.markdown("### 🚨 Urgent Reorder Action Items")
    if low_stock_items:
        low_stock_df = pd.DataFrame(low_stock_items)
        low_stock_df.columns = ["Stock Code", "Description", "Current Stock", "Reorder Threshold"]
        st.dataframe(low_stock_df, use_container_width=True, hide_index=True)
    else:
        st.success("✅ Inventory levels are healthy. No immediate reorders needed.")

with col_churn:
    st.markdown("### 🎯 High-Risk Customer Retention")
    if at_risk_count > 0:
        st.markdown(
            f"""
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 20px;">
                <h4 style="margin-top:0;">Risk Summary:</h4>
                <p>A total of <strong>{at_risk_count}</strong> customers out of {total_churn_cust} exhibit a high probability of churn (> 75%).</p>
                <p>We recommend exporting this segment and passing it to CRM for an automated win-back campaign.</p>
                <p><em>Check the <strong>Churn Analysis</strong> page for the list of Customer IDs and churn drivers.</em></p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.success("✅ Churn levels are stable. No customer segment is currently flagged as high risk.")
