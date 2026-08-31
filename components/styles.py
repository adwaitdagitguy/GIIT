import streamlit as st

def apply_custom_css():
    """
    Applies custom CSS for the dashboard and the sidebar node inspector.
    """
    st.markdown("""
    <style>
    /* ── GLOBAL STYLES ── */
    body {
        background: linear-gradient(135deg, #eef2f7 0%, #f8fafc 100%);
    }
    .block-container {
        padding-top: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    .section-card {
        background-color: white;
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
        margin-bottom: 30px;
    }
    [data-testid="stMetric"] {
        background-color: #f8fafc;
        color: #000000 !important;
        padding: 20px;
        border-radius: 12px;
    }
    [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        color: #1f2937 !important;
        font-weight: 600;
    }
    [data-testid="stSidebar"] {
        transition: transform 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        background-color: #f8fafc !important;
    }

    /* ── SIDEBAR TEXT OVERRIDES ── */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] div {
        color: #1e293b !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
        color: #1e293b !important;
    }

    [data-testid="stSidebar"] strong, 
    [data-testid="stSidebar"] b {
        color: #0f172a !important;
        font-weight: 700;
    }

    /* ── NODE INSPECTOR COMPONENTS ── */
    .node-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        background: #dbeafe;
        color: #1d4ed8 !important;
        margin-bottom: 14px;
        border: 1px solid #bfdbfe;
    }

    .node-type-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        margin-left: 8px;
        margin-bottom: 14px;
    }

    [data-testid="stSidebar"] .formula-box {
        background: #1d4ed8 !important;
        color: #facc15 !important;
        padding: 14px 18px;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
        font-size: 15px;
        letter-spacing: 1.5px;
        margin: 10px 0;
        border: 1px solid #1e40af;
        display: block;
        word-break: break-all;
    }

    [data-testid="stSidebar"] .highlight-term {
        background: linear-gradient(90deg, #fef9c3, #fde68a) !important;
        color: #78350f !important;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-family: 'Courier New', monospace;
        font-size: 15px;
        display: inline-block;
        margin: 6px 0;
        border: 1px solid #f59e0b;
    }

    .var-card {
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 8px;
        background: #e2e8f0 !important;
        border-left: 4px solid #3b82f6;
        font-size: 14px;
        color: #1e293b !important;
    }

    .var-symbol {
        font-family: 'Courier New', monospace;
        font-weight: 700;
        color: #1d4ed8 !important;
        font-size: 15px;
    }

    .var-desc {
        color: #374151 !important;
        font-size: 13px;
    }

    .info-section-title {
        font-size: 13px;
        font-weight: 700;
        color: #374151 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 14px 0 6px 0;
        padding-bottom: 4px;
        border-bottom: 1px solid #e2e8f0;
    }

    .node-card-outer {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px;
        margin-top: 8px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }

    .desc-box {
        background: #f1f5f9;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 14px;
        color: #334155 !important;
        line-height: 1.6;
        border-left: 3px solid #94a3b8;
    }

    .domain-tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 10px;
        font-size: 11px;
        background: #f0fdf4;
        color: #166534 !important;
        border: 1px solid #bbf7d0;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
