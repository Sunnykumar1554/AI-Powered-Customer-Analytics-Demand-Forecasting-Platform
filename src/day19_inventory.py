import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

# ── Page Configuration ──
st.set_page_config(
    page_title="Inventory Optimization | RetailPulse",
    page_icon="📦",
    layout="wide"
)

# ── Dynamic Path Resolution ──
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(current_dir) in ["pages", "src"]:
    ROOT_DIR = os.path.abspath(os.path.join(current_dir, ".."))
else:
    ROOT_DIR = current_dir

INVENTORY_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "day10", "inventory_recommendations.csv")

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

st.title("📦 Inventory Control & Safety Stock Optimization")
st.markdown("Ensure supply chain stability using predictive reorder thresholds and safety stock formulations.")

# ── Load Dataset ──
@st.cache_data
def load_inventory_data():
    if not os.path.exists(INVENTORY_DATA_PATH):
        return None
    df = pd.read_csv(INVENTORY_DATA_PATH)
    
    np.random.seed(42)
    df["Current_Stock"] = (df["Reorder_Point"] * np.random.uniform(0.3, 1.6, size=len(df))).astype(int)
    df["Current_Stock"] = df["Current_Stock"].clip(lower=0)
    df["Stock_Ratio"] = df["Current_Stock"] / df["Reorder_Point"]
    
    def get_status(row):
        if row["Current_Stock"] <= row["Safety_Stock"]:
            return "Critical"
        elif row["Current_Stock"] <= row["Reorder_Point"]:
            return "Reorder"
        else:
            return "Healthy"
            
    df["Status"] = df.apply(get_status, axis=1)
    return df

df_inv = load_inventory_data()

if df_inv is not None:
    total_items = len(df_inv)
    critical_count = len(df_inv[df_inv["Status"] == "Critical"])
    reorder_count = len(df_inv[df_inv["Status"] == "Reorder"])
    healthy_count = len(df_inv[df_inv["Status"] == "Healthy"])
    
    # ── KPIs Row ──
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Total Tracked SKUs</div>
                <div class='kpi-value'>{total_items:,}</div>
                <div class='kpi-subtext' style='color:#8a8f98;'>Active catalog size</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col2:
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Critical Stockouts</div>
                <div class='kpi-value' style='color:#ff2a5f;'>{critical_count:,}</div>
                <div class='kpi-subtext' style='color:#ff2a5f;'>Stock below Safety Stock level</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col3:
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Reorders Recommended</div>
                <div class='kpi-value' style='color:#ff9f1c;'>{reorder_count:,}</div>
                <div class='kpi-subtext' style='color:#ff9f1c;'>Stock below Reorder Point</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col4:
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Healthy Stock Levels</div>
                <div class='kpi-value' style='color:#00f5d4;'>{healthy_count:,}</div>
                <div class='kpi-subtext' style='color:#00f5d4;'>Optimal reserve stock</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ── Charts Row ──
    st.markdown("### 📊 Top Stockout Risk Items (Grouped Comparison)")
    df_risk = df_inv.sort_values("Stock_Ratio", ascending=True).head(15)
    
    fig_risk = go.Figure()
    fig_risk.add_trace(go.Bar(
        x=df_risk["StockCode"].astype(str),
        y=df_risk["Current_Stock"],
        name="Current Stock",
        marker_color="#ff2a5f"
    ))
    fig_risk.add_trace(go.Bar(
        x=df_risk["StockCode"].astype(str),
        y=df_risk["Reorder_Point"],
        name="Reorder Point",
        marker_color="#ff9f1c"
    ))
    fig_risk.add_trace(go.Bar(
        x=df_risk["StockCode"].astype(str),
        y=df_risk["Safety_Stock"],
        name="Safety Stock",
        marker_color="#00f5d4"
    ))
    
    fig_risk.update_layout(
        barmode="group",
        title="Inventory Reserves vs Thresholds (Top 15 At-Risk SKUs)",
        xaxis_title="Stock Code",
        yaxis_title="Quantity (Units)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    st.plotly_chart(fig_risk, use_container_width=True)

    # ── Interactive Data Table ──
    st.write("---")
    st.markdown("### 🔍 SKU Inventory Search Panel")
    
    search_q = st.text_input("Filter by Stock Code or Product Description", "").strip().lower()
    status_filter = st.multiselect("Filter by Stock Status", ["Critical", "Reorder", "Healthy"], default=["Critical", "Reorder"])
    
    df_filtered = df_inv.copy()
    if search_q:
        df_filtered = df_filtered[
            df_filtered["StockCode"].astype(str).str.lower().str.contains(search_q) |
            df_filtered["Description"].str.lower().str.contains(search_q)
        ]
    if status_filter:
        df_filtered = df_filtered[df_filtered["Status"].isin(status_filter)]
        
    df_present = df_filtered[[
        "StockCode", "Description", "Total_Demand", "Current_Stock", 
        "Safety_Stock", "Reorder_Point", "Recommended_Order_Qty", "Status"
    ]].head(100)
    
    df_present.columns = [
        "Stock Code", "Description", "Annual Demand", "Current Stock",
        "Safety Stock", "Reorder Point", "Recommended Order Qty", "Status"
    ]
    
    def style_status(val):
        if val == "Critical":
            return 'background-color: rgba(255, 42, 95, 0.15); color: #ff2a5f; font-weight: bold;'
        elif val == "Reorder":
            return 'background-color: rgba(255, 159, 28, 0.15); color: #ff9f1c;'
        else:
            return 'background-color: rgba(0, 245, 212, 0.1); color: #00f5d4;'
            
    styled_df = df_present.style.applymap(style_status, subset=["Status"]).format({
        "Annual Demand": "{:,}",
        "Current Stock": "{:,}",
        "Safety Stock": "{:,}",
        "Reorder Point": "{:,}",
        "Recommended Order Qty": "{:,}"
    })
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
else:
    st.warning("⚠ Inventory recommendations dataset not found. Please verify inventory scripts are executed under data/processed/day10/.")
