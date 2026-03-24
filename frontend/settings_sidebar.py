"""Sidebar component for the ICD Code Mapper frontend."""

from __future__ import annotations

import streamlit as st


def render_sidebar(default_url: str) -> tuple[str, bool]:
    """
    Render the settings sidebar.

    Returns:
        (api_base_url, check_health_clicked)
    """
    st.sidebar.markdown("## ⚙️ Settings")

    base_url = st.sidebar.text_input(
        "API Base URL",
        value=default_url,
        key="api_url",
    )

    check_clicked = st.sidebar.button("Check API Health", use_container_width=True)

    st.sidebar.markdown("---")

    with st.sidebar.expander("How it works", expanded=False):
        st.markdown(
            """
**4-Stage Pipeline:**

1. **Type Detection** — LLM classifies your query as a *diagnosis* or *procedure* (only in Auto mode).

2. **Hybrid Search** — Dense (semantic) + Sparse (BM25) vector search over 149k+ ICD-10 codes (71k diagnosis + 78k procedure), fused with Reciprocal Rank Fusion (RRF).

3. **Re-ranking** — Cross-encoder model re-scores the top candidates for higher precision.

4. **Confidence Scoring** — LLM evaluates each candidate and assigns a confidence score (High / Medium / Low).
            """
        )

    st.sidebar.markdown("---")
    st.sidebar.caption("ICD Code Mapper v1.0.0")

    return base_url, check_clicked
