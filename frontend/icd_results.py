"""Result cards and results-section components for the ICD Code Mapper frontend."""

from __future__ import annotations

import streamlit as st

from models import ICDCodeResult, ICDMapResponse
from latency_metrics import render_latency_row



_CONF_CLASS = {"high": "conf-high",       "medium": "conf-medium",       "low": "conf-low"}



def _confidence_badge(confidence: str) -> str:
    cls = _CONF_CLASS.get(confidence, "conf-low")
    return f'<span class="conf-badge {cls}">{confidence}</span>'


def _score_bar(score: float, confidence: str) -> str:
    colors = {"high": "#22c55e", "medium": "#f59e0b", "low": "#ef4444"}
    color = colors.get(confidence, "#ef4444")
    pct = round(score * 100)
    return (
        f'<span style="display:inline-flex;align-items:center;gap:8px;flex:1;">'
        f'<span style="flex:1;height:6px;background:#e2e8f0;border-radius:3px;display:inline-block;vertical-align:middle;">'
        f'<span style="display:inline-block;width:{pct}%;height:6px;background:{color};border-radius:3px;"></span>'
        f'</span>'
        f'<span style="font-size:0.8rem;font-weight:600;color:#475569;min-width:36px;text-align:right;">{pct}%</span>'
        f'</span>'
    )



def render_result_card(result: ICDCodeResult, rank: int) -> None:
    badge = _confidence_badge(result.confidence)
    bar   = _score_bar(result.score, result.confidence)

    st.markdown(
        f"""
        <div class="result-card">
            <span class="result-rank">#{rank}</span>
            <div class="result-code">{result.code_dotted}</div>
            <div class="result-description">{result.long_description}</div>
            <div class="result-category">
                <span>{result.category_code}</span>&nbsp;·&nbsp;{result.category_title}
            </div>
            <div style="display:flex; align-items:center; gap:10px;">
                {badge}
                {bar}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(message: str = "Enter a medical term above to get started.") -> None:
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="icon">🔍</div>
            <p>{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results(response: ICDMapResponse) -> None:
    # Detected query-type pill
    pill_cls = "pill-diagnosis" if response.query_type == "diagnosis" else "pill-procedure"
    st.markdown(
        f"""
        <p class="section-title">
            Detected type:&nbsp;
            <span class="query-type-pill {pill_cls}">{response.query_type}</span>
        </p>
        """,
        unsafe_allow_html=True,
    )

    render_latency_row(response.latencies)

    st.markdown('<p class="section-title">Results</p>', unsafe_allow_html=True)

    if not response.results:
        render_empty_state("No ICD codes found for this query.")
        return

    for rank, result in enumerate(response.results, start=1):
        render_result_card(result, rank)
