"""Page header component for the ICD Code Mapper frontend."""

from __future__ import annotations

import streamlit as st


def render_header(api_online: bool) -> None:
    status_cls  = "status-online" if api_online else "status-offline"
    status_text = "API Online"    if api_online else "API Offline"

    st.markdown(
        f"""
        <div class="app-header">
            <h1>🏥 ICD Code Mapper</h1>
            <p>Map medical text to ICD-10 codes using hybrid vector search + LLM confidence scoring</p>
            <span class="status-badge {status_cls}">● {status_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
