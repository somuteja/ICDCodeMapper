"""Latency metrics row component for the ICD Code Mapper frontend."""

from __future__ import annotations

import streamlit as st

from models import Latencies


def render_latency_row(lat: Latencies) -> None:
    st.markdown('<p class="section-title">Pipeline Latencies</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    def _card(
        col: st.delta_generator.DeltaGenerator,
        label: str,
        value_ms: float | None,
        extra_cls: str = "",
    ) -> None:
        display = f"{value_ms:.0f} ms" if value_ms is not None else "—"
        col.markdown(
            f"""
            <div class="latency-card {extra_cls}">
                <div class="metric-value">{display}</div>
                <div class="metric-label">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    _card(col1, "Type Detection",     lat.type_detection_ms)
    _card(col2, "Hybrid Search",      lat.hybrid_search_ms)
    _card(col3, "Confidence Scoring", lat.confidence_scoring_ms)
    _card(col4, "Total",              lat.total_ms, extra_cls="latency-total")
