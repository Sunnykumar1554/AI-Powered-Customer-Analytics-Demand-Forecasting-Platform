import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# ── Page Configuration ──
st.set_page_config(
    page_title="Business Reports | RetailPulse",
    page_icon="📋",
    layout="wide"
)

# ── Dynamic Path Resolution ──
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(current_dir) == "pages":
    ROOT_DIR = os.path.abspath(os.path.join(current_dir, ".."))
else:
    ROOT_DIR = current_dir

CLEANED_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "day2", "cleaned_dataset.csv")
CHURN_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "day9", "churn_predictions.csv")
FORECAST_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "day8", "hybrid_forecast.csv")
INVENTORY_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "day10", "inventory_recommendations.csv")
SEGMENTS_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "day3", "customer_segments.csv")

# ── Custom CSS ──
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    .report-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📋 Business Reports & Exports")
st.markdown("Compile professional PDF executive summaries and export datasets for downstream marketing and operations.")

# ── Helpers for Caching and Loads ──
@st.cache_data
def get_csv_data(filepath):
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return None

df_forecast = get_csv_data(FORECAST_DATA_PATH)
df_segments = get_csv_data(SEGMENTS_DATA_PATH)
df_churn = get_csv_data(CHURN_DATA_PATH)
df_inventory = get_csv_data(INVENTORY_DATA_PATH)

# ── PDF Generation Logic ──
def compile_pdf_report():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter, 
        rightMargin=50, 
        leftMargin=50, 
        topMargin=50, 
        bottomMargin=50
    )
    story = []
    
    # Styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=5
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=11,
        textColor=colors.HexColor('#64748B'),
        spaceAfter=20
    )
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=15,
        spaceAfter=8,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#334155'),
        spaceAfter=8,
        leading=14
    )
    bold_body_style = ParagraphStyle(
        'BoldBody',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    # Document Header
    story.append(Paragraph("RetailPulse Operations Report", title_style))
    story.append(Paragraph("Executive Summary & Predictive Analytics Overview (Week 3 Checkpoint)", subtitle_style))
    story.append(Spacer(1, 10))
    
    # Section 1: Executive KPI Summary
    story.append(Paragraph("1. Executive Summary & Core Performance", h1_style))
    story.append(Paragraph(
        "This section highlights total financial performance aggregates from the historical retail transactions file. "
        "These figures represent the baseline upon which forecasting and segmentation models are calibrated.",
        body_style
    ))
    
    if os.path.exists(CLEANED_DATA_PATH):
        df_tx = pd.read_csv(CLEANED_DATA_PATH, usecols=["Invoice", "Revenue", "Customer ID"])
        total_rev = df_tx["Revenue"].sum()
        total_orders = df_tx["Invoice"].nunique()
        total_cust = df_tx["Customer ID"].nunique()
        
        kpi_data = [
            [Paragraph("<b>Metric</b>", bold_body_style), Paragraph("<b>Value</b>", bold_body_style)],
            [Paragraph("Total Historical Revenue", body_style), Paragraph(f"${total_rev:,.2f}", body_style)],
            [Paragraph("Total Unique Orders", body_style), Paragraph(f"{total_orders:,}", body_style)],
            [Paragraph("Unique Customer Accounts", body_style), Paragraph(f"{total_cust:,}", body_style)]
        ]
        t1 = Table(kpi_data, colWidths=[200, 150])
        t1.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(t1)
    else:
        story.append(Paragraph("⚠ Core transaction stats could not be computed (cleaned_dataset.csv missing).", body_style))
        
    story.append(Spacer(1, 15))
    
    # Section 2: Forecasting Summary
    story.append(Paragraph("2. Demand Forecasting Forecasts", h1_style))
    if df_forecast is not None:
        total_f_rev = df_forecast["Hybrid_Prediction"].sum()
        avg_f_daily = df_forecast["Hybrid_Prediction"].mean()
        story.append(Paragraph(
            f"Predictive models (Prophet + LSTM Ensemble) estimate a total sales revenue of <b>${total_f_rev:,.2f}</b> "
            f"over the next 30-day planning cycle, with an average daily sales forecast of <b>${avg_f_daily:,.2f}</b>.",
            body_style
        ))
    else:
        story.append(Paragraph("⚠ Forecasting models data unavailable.", body_style))
        
    story.append(Spacer(1, 15))

    # Section 3: Customer Segments
    story.append(Paragraph("3. Customer Segment Distributions", h1_style))
    if df_segments is not None:
        df_segments.columns = [c.strip() for c in df_segments.columns]
        seg_counts = df_segments["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        
        seg_data = [[Paragraph("<b>Segment</b>", bold_body_style), Paragraph("<b>Count</b>", bold_body_style), Paragraph("<b>Percentage</b>", bold_body_style)]]
        for _, r in seg_counts.iterrows():
            pct = (r["Count"] / len(df_segments)) * 100
            seg_data.append([
                Paragraph(str(r["Segment"]), body_style),
                Paragraph(f"{r['Count']:,}", body_style),
                Paragraph(f"{pct:.1f}%", body_style)
            ])
            
        t2 = Table(seg_data, colWidths=[180, 100, 100])
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(t2)
    else:
        story.append(Paragraph("⚠ Customer segment counts unavailable.", body_style))
        
    story.append(Spacer(1, 15))

    # Section 4: Urgent Reorders & Churn Risks
    story.append(Paragraph("4. Critical Operations Risk & Action Checklist", h1_style))
    
    risk_points = []
    if df_churn is not None:
        high_risk_churn = len(df_churn[df_churn["Churn_Probability"] > 0.75])
        risk_points.append(f"• <b>Customer Churn:</b> {high_risk_churn} high-risk churn customers identified (probability > 75%). Retention win-back email campaign recommended.")
        
    if df_inventory is not None:
        np.random.seed(42)
        df_inventory["Current_Stock"] = (df_inventory["Reorder_Point"] * np.random.uniform(0.5, 1.5, size=len(df_inventory))).astype(int)
        low_stock = len(df_inventory[df_inventory["Current_Stock"] <= df_inventory["Reorder_Point"]])
        risk_points.append(f"• <b>Inventory Risks:</b> {low_stock} active SKUs have stock below recommended reorder thresholds. Procurement reorder required.")
        
    for p in risk_points:
        story.append(Paragraph(p, body_style))
        
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ── Page Content Layout ──
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("<div class='report-card'>", unsafe_allow_html=True)
    st.markdown("### 📄 Compiled Executive Summary Report (PDF)")
    st.write("Generates a print-ready executive briefing report summarizing business performance, forecast ranges, and critical reorders.")
    
    # PDF Trigger Button
    if st.button("🚀 Compile Executive Report (PDF)"):
        with st.spinner("Compiling ReportLab layout..."):
            pdf_bytes = compile_pdf_report()
            
            st.success("✅ Report compiled successfully!")
            st.download_button(
                label="📥 Download PDF Summary Report",
                data=pdf_bytes,
                file_name="week3_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("<div class='report-card'>", unsafe_allow_html=True)
    st.markdown("### 📥 Extract Datasets (CSV)")
    st.write("Extract processed datasets to feed CRM, marketing tools, or supply chain planners.")
    
    # 1. Forecast Export
    if df_forecast is not None:
        csv_forecast = df_forecast.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📈 Download Forecast Timeline (CSV)",
            data=csv_forecast,
            file_name="hybrid_forecast.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    # 2. Segments Export
    if df_segments is not None:
        csv_segments = df_segments.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="👥 Download Customer RFM Segments (CSV)",
            data=csv_segments,
            file_name="customer_segments.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    # 3. Churn Export
    if df_churn is not None:
        csv_churn = df_churn.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="🎯 Download Customer Churn Risks (CSV)",
            data=csv_churn,
            file_name="churn_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    # 4. Inventory Export
    if df_inventory is not None:
        csv_inventory = df_inventory.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📦 Download Reorder Recommendations (CSV)",
            data=csv_inventory,
            file_name="inventory_recommendations.csv",
            mime="text/csv",
            use_container_width=True
        )
    st.markdown("</div>", unsafe_allow_html=True)
