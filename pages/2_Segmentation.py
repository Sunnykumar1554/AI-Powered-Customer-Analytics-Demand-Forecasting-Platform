import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

# ── Page Configuration ──
st.set_page_config(
    page_title="Customer Segmentation | RetailPulse",
    page_icon="👥",
    layout="wide"
)

# ── Dynamic Path Resolution ──
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(current_dir) == "pages":
    ROOT_DIR = os.path.abspath(os.path.join(current_dir, ".."))
else:
    ROOT_DIR = current_dir

SEGMENTS_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "day3", "customer_segments.csv")

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

st.title("👥 Customer Segmentation (RFM Analysis)")
st.markdown("Explore AI-generated customer clusters based on Recency, Frequency, and Monetary parameters.")

# ── Load Dataset ──
@st.cache_data
def load_segment_data():
    if not os.path.exists(SEGMENTS_DATA_PATH):
        return None
    return pd.read_csv(SEGMENTS_DATA_PATH)

df_seg = load_segment_data()

if df_seg is not None:
    # Rename columns to ensure standard access
    df_seg.columns = [c.strip() for c in df_seg.columns]
    
    # Calculate global metrics
    total_customers = len(df_seg)
    total_monetary = df_seg["Monetary"].sum()
    
    # Calculate group contributions
    segment_stats = df_seg.groupby("Segment").agg(
        Count=("Customer ID", "count"),
        Total_Monetary=("Monetary", "sum"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean")
    ).reset_index()
    
    segment_stats["Pct_Customers"] = (segment_stats["Count"] / total_customers) * 100
    segment_stats["Pct_Revenue"] = (segment_stats["Total_Monetary"] / total_monetary) * 100
    
    # Find VIP/Loyal stats
    loyal_vip = segment_stats[segment_stats["Segment"].isin(["Loyal Customers", "Potential Loyalists"])]
    if not loyal_vip.empty:
        loyal_vip_cust_pct = loyal_vip["Pct_Customers"].sum()
        loyal_vip_rev_pct = loyal_vip["Pct_Revenue"].sum()
    else:
        loyal_vip_cust_pct = 0.0
        loyal_vip_rev_pct = 0.0
        
    # ── KPIs Row ──
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Total Segmented Customer Base</div>
                <div class='kpi-value'>{total_customers:,}</div>
                <div class='kpi-subtext' style='color: #8a8f98;'>Active buyers cataloged</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col2:
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Loyal/VIP Base Contribution</div>
                <div class='kpi-value'>{loyal_vip_cust_pct:.1f}%</div>
                <div class='kpi-subtext'>Represent customer core strength</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col3:
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Loyal/VIP Revenue Share</div>
                <div class='kpi-value'>{loyal_vip_rev_pct:.1f}%</div>
                <div class='kpi-subtext' style='color: #00c6ff;'>Value share of total spend</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ── Business Insights Box ──
    if loyal_vip_cust_pct > 0:
        st.markdown(
            f"""
            <div style="background: rgba(0, 245, 212, 0.03); border: 1px solid rgba(0, 245, 212, 0.15); border-radius: 12px; padding: 20px; margin-bottom: 25px;">
                <h4 style="margin-top:0;color:#00f5d4;">💡 Strategic Insights</h4>
                <p style="margin-bottom:0;">The data reveals that <strong>{loyal_vip_cust_pct:.1f}%</strong> of your customer base (Loyal & Potential Loyalists) contributes to <strong>{loyal_vip_rev_pct:.1f}%</strong> of total monetary revenue. Focus loyalty programs and premium rewards here to maximize Customer Lifetime Value (LTV).</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # ── Visualizations Row ──
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### 📊 Segment Distribution vs Revenue Share")
        
        # We can construct a grouped/side-by-side bar chart of % customer base vs % revenue share
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=segment_stats["Segment"],
            y=segment_stats["Pct_Customers"],
            name="% of Customer Base",
            marker_color="#9d4edd"
        ))
        fig_bar.add_trace(go.Bar(
            x=segment_stats["Segment"],
            y=segment_stats["Pct_Revenue"],
            name="% of Total Revenue",
            marker_color="#00c6ff"
        ))
        fig_bar.update_layout(
            barmode="group",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col_right:
        st.markdown("### 🌌 3D RFM Cluster Visualization")
        # Sample data if it is huge, otherwise render full
        df_sample = df_seg if len(df_seg) <= 2000 else df_seg.sample(2000, random_state=42)
        
        fig_3d = px.scatter_3d(
            df_sample, 
            x="Recency", 
            y="Frequency", 
            z="Monetary",
            color="Segment",
            log_z=True, # Log scale for Monetary due to huge range differences
            title="3D Spatial Distribution (Recency, Frequency, log(Monetary))",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            template="plotly_dark"
        )
        fig_3d.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=40, b=0),
            scene=dict(
                xaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
                yaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
                zaxis=dict(backgroundcolor="rgba(0,0,0,0)")
            )
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        
    # ── Segment Stats Table ──
    st.markdown("### 📋 Customer Segments Summary Table")
    df_present = segment_stats.copy()
    df_present.columns = ["Customer Segment", "Count", "Total Spend ($)", "Avg Recency (Days)", "Avg Frequency (Orders)", "% of Customer Base", "% of Revenue Contribution"]
    st.dataframe(
        df_present.style.format({
            "Total Spend ($)": "${:,.2f}",
            "Avg Recency (Days)": "{:.1f}",
            "Avg Frequency (Orders)": "{:.1f}",
            "% of Customer Base": "{:.1f}%",
            "% of Revenue Contribution": "{:.1f}%"
        }),
        use_container_width=True,
        hide_index=True
    )
    
    # ── Customer Drilldown Selector ──
    st.write("---")
    st.markdown("### 🔍 Segment Customer Lookup")
    selected_seg = st.selectbox("Select Segment to Filter Customers", df_seg["Segment"].unique())
    
    seg_cust = df_seg[df_seg["Segment"] == selected_seg].sort_values("Monetary", ascending=False).head(50)
    seg_cust.columns = ["Customer ID", "Recency (Days)", "Frequency (Orders)", "Monetary Spend ($)", "Cluster ID", "Segment Name"]
    
    st.dataframe(
        seg_cust.style.format({
            "Monetary Spend ($)": "${:,.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )
    
else:
    st.warning("⚠ Segmentation data not found. Please verify customer segments are generated under data/processed/day3/.")
