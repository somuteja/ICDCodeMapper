"""
ICD Code Mapper — Streamlit Frontend
Run with:  streamlit run frontend/app.py
"""

from __future__ import annotations

import streamlit as st

from api_client import ICDMapperClient
from page_header import render_header
from icd_results import render_empty_state, render_results
from settings_sidebar import render_sidebar
from theme import MAIN_CSS



st.set_page_config(
    page_title="ICD Code Mapper",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)



st.markdown(MAIN_CSS, unsafe_allow_html=True)


if "response" not in st.session_state:
    st.session_state.response = None
if "error_msg" not in st.session_state:
    st.session_state.error_msg = None
if "api_online" not in st.session_state:
    st.session_state.api_online = False


base_url, check_health_clicked = render_sidebar(default_url="http://localhost:8000")
client = ICDMapperClient(base_url=base_url)


if "last_checked_url" not in st.session_state or st.session_state.last_checked_url != base_url:
    health = client.health()
    st.session_state.api_online = health.online
    st.session_state.last_checked_url = base_url

if check_health_clicked:
    health = client.health()
    st.session_state.api_online = health.online
    if health.online:
        st.sidebar.success(f"✓ {health.service} {health.version}")
    else:
        st.sidebar.error(f"✗ {health.error}")


render_header(st.session_state.api_online)


with st.container(border=True):
    query_col, options_col = st.columns([3, 1], gap="large")

    with query_col:
        query_text = st.text_area(
            "Medical query",
            placeholder="e.g. type 2 diabetes,  laparoscopic appendectomy,  hypertensive heart disease…",
            height=120,
            key="query_text",
        )

    with options_col:
        query_type = st.selectbox(
            "Query type",
            options=["auto", "diagnosis", "procedure"],
            index=0,
            format_func=lambda x: x.capitalize(),
            help="**Auto** lets the AI decide. Choose **Diagnosis** or **Procedure** to skip type detection.",
        )
        top_k = st.slider(
            "Top results",
            min_value=1,
            max_value=20,
            value=5,
            help="Maximum number of ICD codes to return.",
        )

    submit_col, _ = st.columns([1, 4])
    with submit_col:
        submit = st.button(
            "Map ICD Codes",
            type="primary",
            use_container_width=True,
            disabled=not st.session_state.api_online,
        )


if submit:
    text = query_text.strip()
    if not text:
        st.warning("Please enter a medical query before submitting.")
    elif not st.session_state.api_online:
        st.error("The API is offline. Check the URL in the sidebar and try again.")
    else:
        st.session_state.error_msg = None
        st.session_state.response = None

        with st.spinner("Mapping ICD codes — this may take a few seconds…"):
            try:
                st.session_state.response = client.map_icd(
                    query_text=text,
                    query_type=query_type,
                    top_k=top_k,
                )
            except Exception as exc:  # noqa: BLE001
                st.session_state.error_msg = str(exc)


if st.session_state.error_msg:
    st.error(
        f"**Request failed:** {st.session_state.error_msg}\n\n"
        "Make sure the FastAPI server is running (`python src/main.py`) "
        "and the URL in the sidebar is correct."
    )
elif st.session_state.response is not None:
    render_results(st.session_state.response)
else:
    render_empty_state()
