"""CSS theme and styling for the ICD Code Mapper Streamlit frontend."""

MAIN_CSS = """
<style>
    /* ── Global ──────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Page background ─────────────────────────────────────── */
    .stApp {
        background: #f0f4f8;
    }

    /* ── Header ──────────────────────────────────────────────── */
    .app-header {
        background: linear-gradient(135deg, #0f4c81 0%, #1a7fc1 50%, #0ea5e9 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(15, 76, 129, 0.3);
    }
    .app-header h1 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .app-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        margin: 0.4rem 0 0 0;
    }

    /* ── Status badge ─────────────────────────────────────────── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.75rem;
    }
    .status-online {
        background: rgba(34, 197, 94, 0.2);
        color: #16a34a;
        border: 1px solid rgba(34, 197, 94, 0.4);
    }
    .status-offline {
        background: rgba(239, 68, 68, 0.15);
        color: #dc2626;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* ── Query type pill ─────────────────────────────────────── */
    .query-type-pill {
        display: inline-block;
        padding: 3px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: capitalize;
    }
    .pill-diagnosis {
        background: #dbeafe;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
    }
    .pill-procedure {
        background: #ede9fe;
        color: #6d28d9;
        border: 1px solid #ddd6fe;
    }

    /* ── Result cards ─────────────────────────────────────────── */
    .result-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        transition: box-shadow 0.2s ease;
        position: relative;
    }
    .result-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    .result-rank {
        position: absolute;
        top: 1rem;
        right: 1.1rem;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: #f1f5f9;
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .result-code {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f4c81;
        font-family: 'Courier New', monospace;
        letter-spacing: 1px;
    }
    .result-description {
        font-size: 0.95rem;
        color: #1e293b;
        margin: 0.3rem 0;
        font-weight: 500;
    }
    .result-category {
        font-size: 0.8rem;
        color: #64748b;
        margin-bottom: 0.6rem;
    }
    .result-category span {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        padding: 1px 6px;
    }

    /* ── Confidence badges ────────────────────────────────────── */
    .conf-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .conf-high {
        background: #dcfce7;
        color: #15803d;
        border: 1px solid #bbf7d0;
    }
    .conf-medium {
        background: #fef3c7;
        color: #b45309;
        border: 1px solid #fde68a;
    }
    .conf-low {
        background: #fee2e2;
        color: #b91c1c;
        border: 1px solid #fecaca;
    }

    /* ── Score bar ────────────────────────────────────────────── */
    .score-bar-container {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 0.5rem;
    }
    .score-bar-bg {
        flex: 1;
        height: 6px;
        background: #e2e8f0;
        border-radius: 3px;
        overflow: hidden;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.4s ease;
    }
    .score-fill-high   { background: #22c55e; }
    .score-fill-medium { background: #f59e0b; }
    .score-fill-low    { background: #ef4444; }
    .score-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #475569;
        min-width: 38px;
        text-align: right;
    }

    /* ── Section titles ───────────────────────────────────────── */
    .section-title {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
        margin: 1.2rem 0 0.6rem 0;
    }

    /* ── Latency row ──────────────────────────────────────────── */
    .latency-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .latency-card .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0f4c81;
    }
    .latency-card .metric-label {
        font-size: 0.72rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 2px;
    }
    .latency-total .metric-value {
        color: #0ea5e9;
    }

    /* ── Input card ───────────────────────────────────────────── */
    .input-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }

    /* ── Empty state ──────────────────────────────────────────── */
    .empty-state {
        text-align: center;
        padding: 3rem 2rem;
        color: #94a3b8;
    }
    .empty-state .icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .empty-state p {
        font-size: 1rem;
        margin: 0;
    }

    /* ── Sidebar styling ──────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #1e293b !important;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stTextInput input {
        background: #334155 !important;
        border: 1px solid #475569 !important;
        color: #f1f5f9 !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: #0ea5e9 !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #0284c7 !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .streamlit-expanderHeader,
    [data-testid="stSidebar"] .streamlit-expanderHeader:hover,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
        background: #334155 !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
        border: 1px solid #475569 !important;
    }
    [data-testid="stSidebar"] .streamlit-expanderContent,
    [data-testid="stSidebar"] [data-testid="stExpander"] > div {
        background: #263548 !important;
        border: 1px solid #334155 !important;
        border-top: none !important;
        color: #cbd5e1 !important;
    }

    /* ── Streamlit button override ────────────────────────────── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0f4c81, #1a7fc1);
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.55rem 2rem;
        font-size: 0.95rem;
        transition: opacity 0.2s;
    }
    .stButton > button[kind="primary"]:hover {
        opacity: 0.9;
    }

    /* ── Hide Streamlit default chrome ───────────────────────── */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
</style>
"""
