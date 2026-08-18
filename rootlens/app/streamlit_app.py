#!/usr/bin/env python3
"""RootLensAI technical operations console."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rootlens.app.ui_components import (  # noqa: E402
    APP_CSS,
    header_html,
    probabilities_html,
    service_status_html,
    status_html,
    telemetry_style,
    topology_svg,
)
from rootlens.demo.fault_controller import (  # noqa: E402
    FaultControllerError,
    get_fault_state,
    inject_fault,
    restore_fault,
    supported_faults,
)
from rootlens.inference.live_telemetry import run_live_rca  # noqa: E402
from rootlens.inference.rca_inference import RCAInference  # noqa: E402


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("rootlens.mvp")
TOPOLOGY_PATH = REPO_ROOT / "rootlens/config/service_graph_v1.yaml"

st.set_page_config(
    page_title="RootLensAI",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(APP_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_engine() -> RCAInference:
    return RCAInference(device="cpu")


def section(title: str) -> None:
    st.markdown(f'<div class="rl-section-label">{title}</div>', unsafe_allow_html=True)


def render_model_details(result: dict) -> None:
    with st.expander("MODEL DETAILS"):
        left, right = st.columns(2)
        with left:
            st.markdown(
                """
                **Model**

                `GraphSAGE RCA v2 node-preserving`

                **Dataset**

                `RootLens Dataset v2`

                **Parameters**

                `61,445`

                **Input**

                `12 services × 7 telemetry features`
                """
            )
        with right:
            st.markdown(
                f"""
                **Topology**

                `44 message-passing edges`

                **MLflow Run**

                `c5b8c95fd91d4ce2ab27df44e846ddec`

                **Graph Shape**

                `{result['graph_shape']}`

                **Runtime**

                `{result['device']} · window {result['promql_window']}`
                """
            )


def render_result(result: dict) -> None:
    predicted = str(result["predicted_root_cause"])
    services = list(result["service_metrics"])
    st.markdown(status_html(result), unsafe_allow_html=True)

    overview, topology = st.columns([.92, 1.35], gap="large")
    with overview:
        section("Service status")
        st.markdown(service_status_html(services, predicted), unsafe_allow_html=True)
    with topology:
        section("Service topology · model-indicated root cause")
        st.markdown(topology_svg(TOPOLOGY_PATH, predicted), unsafe_allow_html=True)

    section("Model output · class probabilities")
    st.markdown(
        probabilities_html(result["probabilities"], predicted),
        unsafe_allow_html=True,
    )

    section("Live service telemetry")
    table = pd.DataFrame.from_dict(result["service_metrics"], orient="index")
    table.index.name = "service"
    table = table.reset_index()[
        [
            "service", "cpu", "memory", "request_rate", "has_requests",
            "latency_ms", "error_rps", "error_rate",
        ]
    ]
    st.dataframe(
        telemetry_style(table, predicted),
        hide_index=True,
        width="stretch",
        height=455,
    )

    render_model_details(result)
    inferred_at = datetime.fromisoformat(result["timestamp"].replace("Z", "+00:00"))
    source = "DEMO INPUT" if result["demo_mode"] else "PROMETHEUS CONNECTED"
    st.markdown(
        '<div class="rl-footer">MODEL READY &nbsp;|&nbsp; GRAPH [12×7] '
        f'&nbsp;|&nbsp; {source} &nbsp;|&nbsp; LAST INFERENCE '
        f'{inferred_at.strftime("%H:%M:%S UTC")}</div>',
        unsafe_allow_html=True,
    )


def render_demo_controls() -> None:
    section("Demo controls · fault injection")
    st.markdown(
        '<div class="rl-separation-note"><b>FAULT INJECTION</b> changes the testbed only. '
        'Injected fault metadata is not passed to the RCA model.</div>',
        unsafe_allow_html=True,
    )
    choices = supported_faults()
    control_left, control_right = st.columns(2)
    service = control_left.selectbox(
        "Target service", list(choices), format_func=lambda value: value.replace("_", " ").title()
    )
    fault_type = control_right.selectbox("Fault type", choices[service], format_func=str.title)
    state = get_fault_state()
    if state["active"]:
        state_text = f'{state["service"].replace("_", " ").upper()} · {state["fault_type"].upper()} · {state["display_value"]}'
        state_class = "active"
    else:
        state_text = "NONE"
        state_class = ""
    st.markdown(
        f'<div class="rl-injection-state {state_class}"><span>INJECTION STATE</span><b>{state_text}</b></div>',
        unsafe_allow_html=True,
    )
    notice = st.session_state.pop("fault_notice", None)
    if notice:
        st.success(notice)
    inject_column, restore_column, _ = st.columns([1, 1, 2])
    if inject_column.button("INJECT FAULT", width="stretch"):
        try:
            with st.spinner("Applying allow-listed fault · recreating target service · verifying runtime control"):
                inject_fault(service, fault_type)
            st.session_state.fault_notice = (
                "Fault injected and verified. Allow telemetry to update, then run Refresh RCA "
                "(recommended wait: 30–45 seconds)."
            )
            st.rerun()
        except FaultControllerError as exc:
            LOGGER.exception("Fault injection failed")
            st.error(f"Fault injection failed: {exc}")
    if restore_column.button("RESTORE SYSTEM", width="stretch"):
        try:
            with st.spinner("Restoring all demo controls · verifying healthy values"):
                restore_fault()
            st.session_state.fault_notice = (
                "All supported fault controls restored and verified at healthy values."
            )
            st.rerun()
        except FaultControllerError as exc:
            LOGGER.exception("System restoration failed")
            st.error(f"System restoration failed: {exc}")

default_demo = os.getenv("ROOTLENS_DEMO_MODE", "false").lower() in {
    "1", "true", "yes", "on",
}
mode_column, refresh_column, spacer = st.columns([1.1, 1.1, 3.8])
demo_mode = mode_column.toggle("Demo mode", value=default_demo)
refresh = refresh_column.button("Refresh RCA", type="primary", width="stretch")

st.markdown(header_html(demo_mode), unsafe_allow_html=True)
if demo_mode:
    st.markdown('<div class="rl-mode-note">DEMO TELEMETRY INPUT</div>', unsafe_allow_html=True)

render_demo_controls()

section("RCA result · telemetry-derived inference")

stored_result = st.session_state.get("rca_result")
mode_changed = stored_result is not None and bool(stored_result.get("demo_mode")) != demo_mode
if stored_result is None or refresh or mode_changed:
    try:
        with st.spinner("Collecting telemetry · constructing graph · running inference"):
            st.session_state.rca_result = run_live_rca(
                demo_mode=demo_mode,
                inference=load_engine(),
            )
        st.session_state.rca_error = None
    except Exception as exc:
        LOGGER.exception("RootLens RCA refresh failed")
        st.session_state.rca_error = str(exc)

if st.session_state.get("rca_error"):
    st.error(f'RCA unavailable: {st.session_state.rca_error}')
elif st.session_state.get("rca_result"):
    render_result(st.session_state.rca_result)
